from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException

logger = logging.getLogger(__name__)


class ModelManager:
    """Ensure models are loaded, wrapped, and configured for inference."""

    def __init__(self, context: Any):
        self.context = context

    def ensure_model_ready(self, model_id: str):
        """Return an LTTS-wrapped model, loading and wrapping if required."""
        model_instance = self.context.models.get_model(model_id)

        if (
            model_instance is None
            or not model_instance.loaded
            or model_instance.model is None
        ):
            logger.info(
                "🔄 INFERENCE: Model '%s' not loaded, loading automatically...",
                model_id,
            )
            try:
                engine = "transformers"
                logger.info(
                    "Using default engine '%s' for automatic model loading", engine
                )
                model, _ = self.context.models.load_model(model_id, engine)
                if model is not None:
                    self.context.ltts_integration.wrap_model(model, model_id)
                    logger.info(
                        "Successfully loaded and wrapped model '%s'", model_id
                    )
                model_instance = self.context.models.get_model(model_id)
                logger.info(
                    "✅ INFERENCE: Model '%s' loaded successfully with engine '%s'",
                    model_id,
                    engine,
                )
            except Exception as load_error:
                logger.error(
                    "❌ INFERENCE: Failed to load model '%s': %s", model_id, load_error
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load model '{model_id}': {str(load_error)}",
                )

        ltts_model = self.context.ltts_integration.wrapped_models.get(model_id)
        if not ltts_model:
            ltts_model = self.context.ltts_integration.wrap_model(
                model_instance.model, model_id
            )

        return ltts_model

    @staticmethod
    def _emits_attention_weights(base_model) -> bool:
        """Run a two-token forward and report whether attentions come back.

        Only used to rescue a run the implementation-string check would reject,
        so the cost is paid once and never on the happy path. Any failure here
        means "cannot confirm", which keeps the original rejection.
        """
        import torch

        try:
            device = next(base_model.parameters()).device
            input_ids = torch.zeros((1, 2), dtype=torch.long, device=device)
            kwargs = {"output_attentions": True, "return_dict": True}
            if getattr(base_model.config, "is_encoder_decoder", False):
                kwargs["decoder_input_ids"] = input_ids
            was_training = base_model.training
            base_model.eval()
            try:
                with torch.no_grad():
                    outputs = base_model(input_ids=input_ids, **kwargs)
            finally:
                base_model.train(was_training)

            for key in ("attentions", "encoder_attentions", "decoder_attentions"):
                weights = getattr(outputs, key, None)
                if weights:
                    return True
            return False
        except Exception as probe_error:
            logger.debug("Attention probe failed: %s", probe_error)
            return False

    def configure_attention_outputs(
        self, ltts_model, *, force_eager: bool = True
    ) -> dict[str, Any]:
        """Enable attention outputs and return an actionable capability result."""
        base_model = getattr(ltts_model, "original_model", None)
        if base_model is None or not hasattr(base_model, "config"):
            logger.debug(
                "base_model or config not found - base_model: %s",
                base_model is not None,
            )
            return {
                "supported": False,
                "implementation": None,
                "detail": "Backing model has no configurable attention interface",
            }

        try:
            old_impl = getattr(base_model.config, "_attn_implementation", None)
            saved_state = getattr(ltts_model, "_attention_config_saved", None)
            if saved_state is None:
                # Save once per configure so the original values survive for restore.
                ltts_model._attention_config_saved = {
                    "_attn_implementation": old_impl,
                    "attn_implementation_attr": getattr(
                        base_model, "_attn_implementation", None
                    ),
                    "output_attentions": getattr(
                        base_model.config, "output_attentions", None
                    ),
                    "return_dict": getattr(base_model.config, "return_dict", None),
                    "return_dict_in_generate": getattr(
                        base_model.config, "return_dict_in_generate", None
                    ),
                }
            if force_eager:
                setter = getattr(base_model, "set_attn_implementation", None)
                if callable(setter):
                    setter("eager")
                # The setter can decline without raising - BERT-family models
                # log a refusal because they predate AttentionInterface - so
                # check the result and fall back to assigning the config value.
                if (
                    getattr(base_model.config, "_attn_implementation", None) != "eager"
                    and hasattr(base_model.config, "_attn_implementation")
                ):
                    base_model.config._attn_implementation = "eager"
                if hasattr(base_model, "_attn_implementation"):
                    base_model._attn_implementation = "eager"

            current_impl = getattr(
                base_model.config, "_attn_implementation", old_impl
            )
            if force_eager and current_impl not in {
                None,
                "eager",
                "eager_paged",
                "flex_attention",
            }:
                # Some architectures refuse the switch (BERT's
                # set_attn_implementation logs a refusal rather than raising,
                # because it predates AttentionInterface) yet still return real
                # attention weights: their sdpa path falls back to manual
                # attention whenever output_attentions is requested. Rejecting
                # on the implementation string alone blocks runs that work, so
                # ask the model instead of trusting the label.
                if not self._emits_attention_weights(base_model):
                    return {
                        "supported": False,
                        "implementation": current_impl,
                        "detail": f"Attention implementation remained '{current_impl}'",
                    }
                logger.info(
                    "Implementation stayed '%s' but the model returns attention "
                    "weights for output_attentions=True; proceeding.",
                    current_impl,
                )

            # Set output flags only after switching implementation. Recent
            # Transformers versions validate this assignment immediately.
            base_model.config.output_attentions = True
            base_model.config.return_dict = True
            if hasattr(base_model.config, "return_dict_in_generate"):
                base_model.config.return_dict_in_generate = True

            if hasattr(base_model, "model") and hasattr(base_model.model, "config"):
                base_model.model.config.output_attentions = True

            logger.info(
                "Attention capture enabled with implementation '%s'",
                current_impl or "model default",
            )
            return {
                "supported": True,
                "implementation": current_impl,
                "detail": "Attention outputs enabled",
            }
        except Exception as attention_error:
            logger.info("Attention outputs are unavailable: %s", attention_error)
            return {
                "supported": False,
                "implementation": getattr(
                    base_model.config, "_attn_implementation", None
                ),
                "detail": str(attention_error),
            }

    @staticmethod
    def restore_attention_outputs(ltts_model) -> None:
        """Restore attention-related config values saved by configure_attention_outputs."""
        saved_state = getattr(ltts_model, "_attention_config_saved", None)
        if not saved_state:
            return
        base_model = getattr(ltts_model, "original_model", None)
        if base_model is None or not hasattr(base_model, "config"):
            return
        try:
            config = base_model.config
            impl = saved_state.get("_attn_implementation")
            if impl is not None:
                setter = getattr(base_model, "set_attn_implementation", None)
                if callable(setter):
                    setter(impl)
                elif hasattr(config, "_attn_implementation"):
                    config._attn_implementation = impl
            impl_attr = saved_state.get("attn_implementation_attr")
            if impl_attr is not None and hasattr(base_model, "_attn_implementation"):
                base_model._attn_implementation = impl_attr
            if saved_state.get("output_attentions") is not None:
                config.output_attentions = saved_state["output_attentions"]
            if saved_state.get("return_dict") is not None:
                config.return_dict = saved_state["return_dict"]
            if (
                saved_state.get("return_dict_in_generate") is not None
                and hasattr(config, "return_dict_in_generate")
            ):
                config.return_dict_in_generate = saved_state["return_dict_in_generate"]
            if hasattr(base_model, "model") and hasattr(base_model.model, "config"):
                base_model.model.config.output_attentions = getattr(
                    config, "output_attentions", None
                )
        except Exception as restore_error:
            logger.debug("Failed to restore attention config: %s", restore_error)
        finally:
            ltts_model._attention_config_saved = None

    @staticmethod
    def force_attention_extraction(ltts_model, enabled: bool = True) -> None:
        """Flag whether the LTTS model should capture attention this run.

        Pass the run's requires_attention_outputs so the model only requests
        output_attentions (and materializes [batch, heads, seq, seq] tensors)
        when a module actually needs attention.
        """
        try:
            setattr(ltts_model, "_force_extract_attention", bool(enabled))
        except Exception as e:
            logger.debug("Failed to set _force_extract_attention: %s", e)
