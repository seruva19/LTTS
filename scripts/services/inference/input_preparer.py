from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from transformers import AutoTokenizer

from scripts.utils.hf_auth import get_hf_token

logger = logging.getLogger(__name__)


class InputPreparer:
    """Prepare tokenizer inputs for inference while recording prompt metadata."""

    def prepare_inputs(
        self,
        ltts_model,
        model_id: str,
        prompt: str,
        requires_attention_outputs: bool,
    ) -> Tuple[Dict[str, Any], Optional[Any]]:
        tokenizer = None

        try:
            tokenizer = getattr(ltts_model, "_tokenizer", None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    cache_dir="models",
                    token=get_hf_token(),
                )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            device = next(ltts_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            base_model = getattr(ltts_model, "original_model", ltts_model)
            config = getattr(base_model, "config", None)
            if bool(getattr(config, "is_encoder_decoder", False)):
                decoder_start_token_id = getattr(
                    config, "decoder_start_token_id", None
                )
                if decoder_start_token_id is None:
                    decoder_start_token_id = (
                        getattr(config, "bos_token_id", None)
                        or tokenizer.bos_token_id
                        or tokenizer.pad_token_id
                        or tokenizer.eos_token_id
                    )
                if decoder_start_token_id is None:
                    raise ValueError(
                        "Encoder-decoder model has no decoder start, BOS, PAD, or EOS token"
                    )
                batch_size = inputs["input_ids"].shape[0]
                inputs["decoder_input_ids"] = inputs["input_ids"].new_full(
                    (batch_size, 1), int(decoder_start_token_id)
                )

            try:
                setattr(ltts_model, "_tokenizer", tokenizer)
                if "input_ids" in inputs:
                    ids_tensor = inputs["input_ids"][0]
                    ids = ids_tensor.detach().cpu().tolist()
                    tokens = []
                    for token_id in ids:
                        try:
                            decoded_token = tokenizer.decode(
                                [token_id], skip_special_tokens=False
                            )
                            cleaned_token = decoded_token.strip()
                            if not cleaned_token:
                                cleaned_token = f"<{token_id}>"
                            tokens.append(cleaned_token)
                        except Exception:
                            tokens.append(f"<{token_id}>")
                    setattr(ltts_model, "_last_input_tokens", tokens)
                    setattr(ltts_model, "_last_input_text", prompt)
                    setattr(ltts_model, "_last_input_ids", ids)
                    if "attention_mask" in inputs:
                        mask_tensor = inputs["attention_mask"][0]
                        setattr(
                            ltts_model,
                            "_last_attention_mask",
                            mask_tensor.detach().cpu().tolist(),
                        )
            except Exception as token_error:  # pragma: no cover - best effort
                logger.debug("Token attachment failed: %s", token_error)

        except Exception as tokenizer_error:
            logger.warning(
                "Could not load tokenizer for %s: %s", model_id, tokenizer_error
            )
            raise ValueError(
                f"Tokenizer load failed for '{model_id}'. Check model access and configuration."
            ) from tokenizer_error

        if requires_attention_outputs:
            inputs["output_attentions"] = True

        return inputs, tokenizer
