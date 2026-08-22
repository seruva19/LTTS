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


def _run_wrapped(model, **inputs):
    model.eval()
    wrapped = LTTSModel(model, message_bus=None)
    assert hasattr(wrapped.message_bus, "publish")
    with torch.no_grad():
        output = wrapped(**inputs)
    return wrapped, output


def test_real_gpt2_decoder_only_forward_and_hooks():
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=16,
            n_embd=16,
            n_layer=1,
            n_head=2,
        )
    )

    wrapped, output = _run_wrapped(
        model, input_ids=torch.tensor([[1, 2, 3, 4]])
    )

    assert wrapped.architecture["topology"] == "decoder_only"
    assert wrapped.architecture["family"] == "gpt2"
    assert output.logits.shape == (1, 4, 32)
    assert any(
        descriptor["role"] == "block"
        for descriptor in wrapped.module_descriptors.values()
    )
    assert any(
        descriptor["role"] == "self_attention"
        for descriptor in wrapped.module_descriptors.values()
    )


def test_real_bert_encoder_only_forward_and_hooks():
    model = BertForMaskedLM(
        BertConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )

    wrapped, output = _run_wrapped(
        model,
        input_ids=torch.tensor([[1, 2, 3, 4]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
    )

    assert wrapped.architecture["topology"] == "encoder_only"
    assert wrapped.architecture["capabilities"]["masked_lm"] is True
    assert output.logits.shape == (1, 4, 32)
    assert {
        descriptor["stage"] for descriptor in wrapped.module_descriptors.values()
    } == {"encoder"}


def test_real_t5_encoder_decoder_forward_and_cross_attention_hooks():
    model = T5ForConditionalGeneration(
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
    )

    wrapped, output = _run_wrapped(
        model,
        input_ids=torch.tensor([[2, 3, 4]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        decoder_input_ids=torch.tensor([[0, 5]]),
    )

    assert wrapped.architecture["topology"] == "encoder_decoder"
    assert wrapped.architecture["capabilities"]["seq2seq_lm"] is True
    assert wrapped.architecture["capabilities"]["cross_attention"] is True
    assert output.logits.shape == (1, 2, 32)
    roles = {
        descriptor["role"] for descriptor in wrapped.module_descriptors.values()
    }
    assert "self_attention" in roles
    assert "cross_attention" in roles


def test_real_mixtral_moe_forward_and_router_expert_hooks():
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
        pytest.skip(f"Mixtral implementation unavailable in this transformers build: {exc}")

    wrapped, output = _run_wrapped(
        model, input_ids=torch.tensor([[1, 2, 3, 4]])
    )

    assert wrapped.architecture["topology"] == "decoder_only"
    assert wrapped.architecture["capabilities"]["moe"] is True
    assert output.logits.shape == (1, 4, 32)
    roles = {
        descriptor["role"] for descriptor in wrapped.module_descriptors.values()
    }
    assert "moe_router" in roles
    assert "moe_expert" in roles


def test_eager_attention_preflight_accepts_every_topology():
    """Preflight tests attention capability rather than the config label."""
    import types

    from scripts.services.inference.model_manager import ModelManager

    manager = ModelManager.__new__(ModelManager)
    models = {
        "encoder_only": BertForMaskedLM(
            BertConfig(
                vocab_size=64,
                hidden_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                intermediate_size=64,
            )
        ),
        "decoder_only": GPT2LMHeadModel(
            GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=2)
        ),
        "encoder_decoder": T5ForConditionalGeneration(
            T5Config(
                vocab_size=64,
                d_model=32,
                num_layers=2,
                num_heads=2,
                d_ff=64,
                d_kv=16,
            )
        ),
    }

    for topology, model in models.items():
        status = manager.configure_attention_outputs(
            types.SimpleNamespace(original_model=model), force_eager=True
        )
        assert status["supported"] is True, f"{topology}: {status['detail']}"
