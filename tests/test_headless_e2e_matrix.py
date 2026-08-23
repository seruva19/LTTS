"""Offline end-to-end matrix: model hooks -> module -> renderable artifact.

The matrix builds tiny random Transformers from config, so it needs no server,
browser, model download, tokenizer download, or network access.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
    MixtralConfig,
    MixtralForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)

from scripts.core.ltts_model import LTTSModel
from scripts.messages.message_bus import MessageBus
from scripts.router import context
from scripts.services.inference.attachments import ModuleAttachmentManager


RENDERABLE_TYPES = {"scalar", "chart", "table", "heatmap", "text", "image"}


def _decoder():
    return GPT2LMHeadModel(
        GPT2Config(vocab_size=32, n_embd=16, n_layer=1, n_head=2, n_positions=16)
    ), {"input_ids": torch.tensor([[1, 2, 3, 4]])}


def _encoder():
    return BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=16,
        )
    ), {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def _encoder_decoder():
    return T5ForConditionalGeneration(
        T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
    ), {
        "input_ids": torch.tensor([[2, 3, 4]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "decoder_input_ids": torch.tensor([[0, 5]]),
    }


def _moe():
    try:
        model = MixtralForCausalLM(
            MixtralConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                num_local_experts=2,
                num_experts_per_tok=1,
                max_position_embeddings=16,
            )
        )
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"Mixtral unavailable: {exc}")
    return model, {"input_ids": torch.tensor([[1, 2, 3, 4]])}


CASES = {
    "decoder_only": (_decoder, {"block", "self_attention"}),
    "encoder_only": (_encoder, {"block", "self_attention"}),
    "encoder_decoder": (
        _encoder_decoder,
        {"block", "self_attention", "cross_attention"},
    ),
    "decoder_moe": (_moe, {"block", "self_attention", "moe_router", "moe_expert"}),
}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_headless_architecture_emits_a_real_module_artifact(case_name):
    """Exercise a shipped module through discovery, hooks, dispatch, and output."""
    builder, required_roles = CASES[case_name]
    model, inputs = builder()
    model.eval()
    wrapped = LTTSModel(model, message_bus=context.message_bus)
    wrapped._last_input_tokens = ["a", "b", "c", "d"]

    roles = {item.get("role") for item in wrapped.module_descriptors.values()}
    assert required_roles <= roles
    blocks = [
        path
        for path, descriptor in wrapped.module_descriptors.items()
        if descriptor.get("role") == "block"
    ]
    assert blocks
    assert "layer_statistics" in context.module_loader.get_all_modules()

    captured = []
    original = MessageBus.publish_from_thread
    MessageBus.publish_from_thread = (
        lambda _self, _token, message: captured.append(message)
    )
    try:
        manager = ModuleAttachmentManager(context, wrapped)
        manager.reset_attachments()
        specs, _capabilities = manager.attach_modules(
            {
                blocks[0]: [
                    {
                        "name": "layer_statistics",
                        "parameters": {"emit_mode": "all"},
                    }
                ]
            }
        )
        assert specs
        with torch.no_grad():
            output = wrapped(**inputs)
        assert output is not None
    finally:
        MessageBus.publish_from_thread = original
        wrapped.module_registry.detach_all_modules()

    artifacts = [
        message
        for message in captured
        if isinstance(message, dict)
        and message.get("type") == "visualization"
        and message.get("module") == "layer_statistics"
    ]
    assert artifacts, f"{case_name}: layer_statistics emitted no artifact"
    for message in artifacts:
        payload = message.get("data")
        assert isinstance(payload, dict), f"{case_name}: non-object artifact payload"
        assert payload.get("type") in RENDERABLE_TYPES, (
            f"{case_name}: unsupported artifact type {payload.get('type')!r}"
        )
