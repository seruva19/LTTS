import logging
from typing import Any, Dict, Optional, Callable

import torch.nn as nn

from .ltts_event import LTTSContext

logger = logging.getLogger(__name__)


class LTTSLayer:
    def __init__(
        self,
        layer: nn.Module,
        layer_path: str,
        event_dispatcher: Optional[Callable] = None,
        module_descriptor: Optional[Dict[str, Any]] = None,
    ):
        self.original_layer = layer
        self.layer_path = layer_path
        self.module_descriptor = dict(module_descriptor or {})
        self.module_role = self.module_descriptor.get("role") or "other"
        self.architecture_stage = self.module_descriptor.get("stage") or "model"
        self.layer_index = self.module_descriptor.get("layer_index")
        self.attention_kind = self.module_descriptor.get("attention_kind")
        self.layer_type = self._identify_layer_type()
        self.event_dispatcher = event_dispatcher
        self._hook_handles = []
        self._setup_hooks()

    def _identify_layer_type(self) -> str:
        role_mapping = {
            "block": "transformer_block",
            "self_attention": "attention",
            "cross_attention": "attention",
            "mlp": "mlp",
            "moe_router": "moe",
            "moe_expert": "moe",
            "normalization": "normalization",
            "embedding": "embedding",
            "positional_embedding": "embedding",
            "activation": "activation",
        }
        if self.module_role in role_mapping:
            return role_mapping[self.module_role]

        layer_class = type(self.original_layer).__name__.lower()

        if "attention" in layer_class or "attn" in layer_class:
            return "attention"
        elif (
            "mlp" in layer_class or "feedforward" in layer_class or "ffn" in layer_class
        ):
            return "mlp"
        elif "moe" in layer_class or "expert" in layer_class:
            return "moe"
        elif (
            "norm" in layer_class
            or "layernorm" in layer_class
            or "rmsnorm" in layer_class
        ):
            return "normalization"
        elif "embedding" in layer_class or "embed" in layer_class:
            return "embedding"
        else:
            return "unknown"

    def _setup_hooks(self):
        def pre_hook(_module, inputs):
            if self.event_dispatcher:
                return self._dispatch_before_event(inputs)

        def post_hook(_module, inputs, outputs):
            if self.event_dispatcher:
                return self._dispatch_after_event(inputs, outputs)

        self._hook_handles.extend(
            (
                self.original_layer.register_forward_pre_hook(pre_hook),
                self.original_layer.register_forward_hook(post_hook),
            )
        )

    def close(self) -> None:
        """Remove instrumentation hooks owned by this wrapper."""

        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self.event_dispatcher = None

    def _dispatch_before_event(self, inputs):
        if not self.event_dispatcher:
            return None
        context = self._create_layer_context("before", inputs=inputs)
        self.event_dispatcher("layer_before", context)

        # Intervention API: a module may set analysis_results["modified_inputs"]
        # to replace the layer's inputs (must match the original tuple structure).
        modified = context.analysis_results.get("modified_inputs")
        if modified is not None:
            logger.debug(
                "LTTS_LAYER: inputs of %s modified by a module", self.layer_path
            )
            return modified
        return None

    def _dispatch_after_event(self, inputs, outputs):
        if not self.event_dispatcher:
            return None
        context = self._create_layer_context("after", inputs=inputs, outputs=outputs)

        self.event_dispatcher("layer_after", context)

        layer_specific_event = f"layer_after_{self.layer_path.replace('.', '_')}"
        self.event_dispatcher(layer_specific_event, context)

        # Intervention API: a module may set analysis_results["modified_outputs"]
        # to replace the layer's outputs (must match the original structure:
        # tensor stays tensor, tuple stays tuple of the same arity).
        modified = context.analysis_results.get("modified_outputs")
        if modified is not None:
            logger.debug(
                "LTTS_LAYER: outputs of %s modified by a module", self.layer_path
            )
            return modified
        return None

    def _create_layer_context(self, event_phase: str, **kwargs) -> LTTSContext:
        context = LTTSContext(
            module_path=self.layer_path,
            module_class=type(self.original_layer).__name__,
            layer_type=self.layer_type,
            module_role=self.module_role,
            architecture_stage=self.architecture_stage,
            layer_index=self.layer_index,
            attention_kind=self.attention_kind,
            layer_state=self._get_layer_state(),
            **kwargs,
        )
        context.analysis_results["event_phase"] = event_phase
        return context

    def _get_layer_state(self) -> Dict[str, Any]:
        try:
            parameters = {}
            buffers = {}

            for name, param in self.original_layer.named_parameters():
                try:
                    parameters[name] = list(param.shape)
                except Exception:
                    parameters[name] = "unavailable"

            for name, buffer in self.original_layer.named_buffers():
                try:
                    buffers[name] = list(buffer.shape)
                except Exception:
                    buffers[name] = "unavailable"

            return {
                "parameters": parameters,
                "buffers": buffers,
                "training": self.original_layer.training,
                "module_descriptor": dict(self.module_descriptor),
                "device": (
                    str(next(self.original_layer.parameters()).device)
                    if list(self.original_layer.parameters())
                    else "unknown"
                ),
            }
        except Exception as e:
            logger.warning(
                "Error extracting layer state for %s: %s", self.layer_path, e
            )
            return {"error": str(e)}

    def forward(self, *args, **kwargs):
        return self.original_layer(*args, **kwargs)
