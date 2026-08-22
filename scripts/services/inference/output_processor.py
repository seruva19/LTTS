from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from scripts.core.emission_controller import update_expected_passes
from scripts.services.inference.attachments import AttachmentSpec

logger = logging.getLogger(__name__)


class OutputProcessor:
    """Interpret model outputs and coordinate generation flows."""

    def interpret_outputs(
        self,
        ltts_model,
        inputs: Dict[str, Any],
        outputs: Any,
        tokenizer: Optional[Any],
        attached_specs: List[AttachmentSpec],
        inference_input,
        requires_attention_outputs: bool,
    ) -> str:
        model_task = str(getattr(ltts_model, "_model_task", "causal_lm"))
        if model_task == "question_answering" and hasattr(outputs, "start_logits"):
            return self._interpret_question_answering(
                inputs, outputs, tokenizer
            )
        if hasattr(outputs, "logits"):
            if model_task == "masked_lm":
                return self._interpret_masked_lm(inputs, outputs.logits, tokenizer)
            if model_task == "sequence_classification":
                return self._interpret_sequence_classification(
                    ltts_model, outputs.logits
                )
            if model_task == "token_classification":
                return self._interpret_token_classification(
                    ltts_model, inputs, outputs.logits, tokenizer
                )
            return self._handle_generation_outputs(
                ltts_model,
                inputs,
                outputs,
                tokenizer,
                attached_specs,
                inference_input,
                requires_attention_outputs,
            )
        if isinstance(outputs, torch.Tensor):
            return f"Output tensor shape: {outputs.shape}"
        if hasattr(outputs, "last_hidden_state"):
            return f"Hidden states shape: {tuple(outputs.last_hidden_state.shape)}"
        return str(outputs)

    @staticmethod
    def _interpret_masked_lm(
        inputs: Dict[str, Any], logits: torch.Tensor, tokenizer: Optional[Any]
    ) -> str:
        if tokenizer is None or logits.ndim != 3:
            return f"Masked-LM logits shape: {tuple(logits.shape)}"

        input_ids = inputs["input_ids"][0]
        mask_token_id = getattr(tokenizer, "mask_token_id", None)
        positions = []
        if mask_token_id is not None:
            positions = (
                (input_ids == int(mask_token_id)).nonzero(as_tuple=False).flatten().tolist()
            )
        if not positions:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                last_position = max(0, int(attention_mask[0].sum().item()) - 1)
            else:
                last_position = max(0, input_ids.shape[0] - 1)
            positions = [last_position]

        summaries = []
        for position in positions[:8]:
            probabilities = torch.softmax(logits[0, position].float(), dim=-1)
            values, indices = torch.topk(probabilities, k=min(5, probabilities.shape[-1]))
            predictions = []
            for value, token_id in zip(values.tolist(), indices.tolist()):
                label = tokenizer.decode([token_id], skip_special_tokens=False).strip()
                predictions.append(f"{label or f'<{token_id}>'} ({value:.4f})")
            summaries.append(f"position {position}: " + ", ".join(predictions))
        return "Masked-token predictions — " + " | ".join(summaries)

    @staticmethod
    def _interpret_sequence_classification(
        ltts_model, logits: torch.Tensor
    ) -> str:
        if logits.ndim < 2:
            return f"Classification logits shape: {tuple(logits.shape)}"
        probabilities = torch.softmax(logits[0].float(), dim=-1)
        value, index = torch.max(probabilities, dim=-1)
        config = getattr(getattr(ltts_model, "original_model", None), "config", None)
        id2label = getattr(config, "id2label", None) or {}
        label = id2label.get(int(index), id2label.get(str(int(index)), str(int(index))))
        return f"Predicted class: {label} ({float(value):.4f})"

    @staticmethod
    def _interpret_token_classification(
        ltts_model,
        inputs: Dict[str, Any],
        logits: torch.Tensor,
        tokenizer: Optional[Any],
    ) -> str:
        if logits.ndim != 3:
            return f"Token-classification logits shape: {tuple(logits.shape)}"
        config = getattr(getattr(ltts_model, "original_model", None), "config", None)
        id2label = getattr(config, "id2label", None) or {}
        predicted = torch.argmax(logits[0], dim=-1).tolist()
        token_ids = inputs["input_ids"][0].tolist()
        pairs = []
        for position, (token_id, label_id) in enumerate(zip(token_ids, predicted)):
            token = (
                tokenizer.decode([token_id], skip_special_tokens=False).strip()
                if tokenizer is not None
                else str(token_id)
            )
            label = id2label.get(label_id, id2label.get(str(label_id), str(label_id)))
            pairs.append(f"{position}:{token or f'<{token_id}>'}→{label}")
        return "Token labels — " + ", ".join(pairs[:32])

    @staticmethod
    def _interpret_question_answering(
        inputs: Dict[str, Any], outputs: Any, tokenizer: Optional[Any]
    ) -> str:
        start = int(torch.argmax(outputs.start_logits[0]).item())
        end = int(torch.argmax(outputs.end_logits[0]).item())
        if end < start:
            end = start
        if tokenizer is None:
            return f"Answer span: tokens {start}–{end}"
        token_ids = inputs["input_ids"][0, start : end + 1]
        answer = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return f"Answer span {start}–{end}: {answer!r}"

    def _handle_generation_outputs(
        self,
        ltts_model,
        inputs: Dict[str, Any],
        outputs: Any,
        tokenizer: Optional[Any],
        attached_specs: List[AttachmentSpec],
        inference_input,
        requires_attention_outputs: bool,
    ) -> str:
        if tokenizer is None:
            return f"Generated logits shape: {outputs.logits.shape}"

        try:
            last_token_logits = outputs.logits[0, -1, :]
            next_token_id = torch.argmax(last_token_logits, dim=-1)
            next_token = tokenizer.decode([next_token_id], skip_special_tokens=True)
            input_text = tokenizer.decode(
                inputs["input_ids"][0], skip_special_tokens=True
            )
            result_text = f"{input_text} → Next token: '{next_token}'"

            params = inference_input.parameters or {}
            generation_params: Dict[str, Any] = {}
            finalize_specs: List[AttachmentSpec] = []

            requested_max_tokens = (
                params.get("max_new_tokens")
                or params.get("max_tokens")
                or params.get("maxTokens")
                or inference_input.max_tokens
            )

            def _parse_int(value):
                if value is None or isinstance(value, bool):
                    return None
                if isinstance(value, (int, float)):
                    return int(value)
                if isinstance(value, str):
                    try:
                        return int(value)
                    except ValueError:
                        return None
                return None

            max_new_tokens = _parse_int(requested_max_tokens) or 20
            max_new_tokens = max(1, min(512, max_new_tokens))
            generation_params["max_new_tokens"] = max_new_tokens

            for spec in attached_specs:
                spec_params = spec.parameters or {}
                module_instance = spec.module
                target_layer = spec.layer_name
                if spec_params.get("emit_mode") == "final":
                    try:
                        if hasattr(module_instance, "set_runtime_config"):
                            module_instance.set_runtime_config(
                                target_layer,
                                {"expected_generation_passes": max_new_tokens},
                            )
                        update_expected_passes(
                            spec.module_name,
                            target_layer,
                            max_new_tokens,
                        )
                        logger.debug(
                            "INFERENCE: Set expected_generation_passes=%s for %s on %s",
                            max_new_tokens,
                            spec.module_name,
                            target_layer,
                        )
                    except Exception as config_error:
                        logger.warning(
                            "INFERENCE: Failed to set expected_generation_passes for %s on %s: %s",
                            spec.module_name,
                            target_layer,
                            config_error,
                        )
                    if hasattr(module_instance, "finalize_layer"):
                        finalize_specs.append(spec)

            def _maybe_add_param(source_key, target_key, caster):
                if source_key not in params:
                    return
                value = params.get(source_key)
                if value is None:
                    return
                try:
                    generation_params[target_key] = caster(value)
                except (TypeError, ValueError):
                    logger.warning(
                        "Invalid generation parameter '%s': %s", source_key, value
                    )

            _maybe_add_param("temperature", "temperature", float)
            _maybe_add_param("top_p", "top_p", float)
            _maybe_add_param("top_k", "top_k", int)
            _maybe_add_param("repetition_penalty", "repetition_penalty", float)
            _maybe_add_param("num_return_sequences", "num_return_sequences", int)

            if "do_sample" in params:
                generation_params["do_sample"] = bool(params["do_sample"])
            else:
                generation_params["do_sample"] = False

            generation_params["pad_token_id"] = tokenizer.eos_token_id
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                generation_params["attention_mask"] = attention_mask

            if requires_attention_outputs:
                generation_params.setdefault("output_attentions", True)
                generation_params.setdefault("return_dict_in_generate", True)

            with torch.no_grad():
                generated_ids = ltts_model.generate(
                    inputs["input_ids"], **generation_params
                )

            if hasattr(generated_ids, "sequences"):
                generated_ids = generated_ids.sequences

            if generated_ids.dim() == 1:
                generated_ids = generated_ids.unsqueeze(0)

            base_model = getattr(ltts_model, "original_model", ltts_model)
            is_encoder_decoder = bool(
                getattr(getattr(base_model, "config", None), "is_encoder_decoder", False)
            )
            input_length = inputs["input_ids"].shape[-1]
            decoded_sequences = []
            for sequence in generated_ids:
                new_tokens = sequence if is_encoder_decoder else sequence[input_length:]
                decoded_text = tokenizer.decode(
                    new_tokens, skip_special_tokens=True
                ).strip()
                if decoded_text:
                    decoded_sequences.append(decoded_text)

            if generation_params.get("num_return_sequences", 1) > 1 and decoded_sequences:
                result_text = "\n".join(decoded_sequences)
            elif decoded_sequences:
                result_text = decoded_sequences[0]
            else:
                result_text = tokenizer.decode(
                    generated_ids[0], skip_special_tokens=True
                ).strip()

            return result_text

        except Exception as gen_error:
            logger.warning(
                "Generate method failed: %s, using next token prediction", gen_error
            )
            try:
                return (
                    f"{tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)}"
                    f" → Next token: '{tokenizer.decode([torch.argmax(outputs.logits[0, -1, :])], skip_special_tokens=True)}'"
                )
            except Exception:
                return f"Generated logits shape: {outputs.logits.shape}"

    def finalize_emit_mode_modules(self, attached_specs: List[AttachmentSpec]) -> None:
        for spec in attached_specs:
            spec_params = spec.parameters or {}
            if spec_params.get("emit_mode") == "final":
                module_instance = spec.module
                target_layer = spec.layer_name
                if hasattr(module_instance, "finalize_layer"):
                    try:
                        module_instance.finalize_layer(target_layer)
                        logger.debug(
                            "🔧 INFERENCE: Finalized deferred emissions for %s on %s",
                            spec.module_name,
                            target_layer,
                        )
                    except Exception as finalize_error:
                        logger.warning(
                            "⚠ INFERENCE: Deferred emission finalize failed for %s on %s: %s",
                            spec.module_name,
                            target_layer,
                            finalize_error,
                        )
