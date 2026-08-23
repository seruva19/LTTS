"""Offline end-to-end matrix: model hooks -> module -> renderable artifact.

The matrix builds tiny random Transformers from config, so it needs no server,
browser, model download, tokenizer download, or network access.
"""

from __future__ import annotations

import pytest
import torch
from pathlib import Path
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
from ltts_engine import HeadlessEngine, load_module

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


def _external_module(name):
    path = Path(__file__).parents[2] / "LTTS-modules" / name / "init.py"
    if not path.is_file():
        pytest.skip(f"sibling module checkout missing: {name}")
    return load_module(path, name=f"headless_matrix_{name}")


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
    MessageBus.publish_from_thread = lambda _self, _token, message: captured.append(
        message
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
        assert (
            payload.get("type") in RENDERABLE_TYPES
        ), f"{case_name}: unsupported artifact type {payload.get('type')!r}"


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_anisotropy_diagnostic_runs_on_every_base_topology(case_name):
    builder, _required_roles = CASES[case_name]
    model, inputs = builder()
    module = _external_module("activation_anisotropy")

    with HeadlessEngine().instrument(model.eval()) as session:
        block = session.layers("block")[0]
        session.attach(
            module,
            layer=block,
            config={"emit_mode": "all", "max_vectors": 32},
        )
        with torch.no_grad():
            session.run(**inputs)
        artifacts = [
            artifact
            for artifact in session.artifacts
            if artifact.module == "activation_anisotropy"
        ]

    assert artifacts, f"{case_name}: anisotropy diagnostic emitted nothing"
    assert {artifact.data.get("type") for artifact in artifacts} >= {
        "scalar",
        "table",
        "chart",
    }


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_activation_steering_changes_real_model_logits(case_name):
    builder, _required_roles = CASES[case_name]
    model, inputs = builder()
    model.eval()
    module = _external_module("neuron_activation_steering")

    with HeadlessEngine().instrument(model) as session:
        with torch.no_grad():
            baseline = session.run(**inputs).logits.detach().clone()
        session.clear_artifacts()
        block = session.layers("block")[0]
        session.attach(
            module,
            layer=block,
            config={
                "enabled": True,
                "neuron_indices": "0-3",
                "operation": "add",
                "value": 5.0,
                "positions": "all",
                "emit_mode": "all",
            },
        )
        with torch.no_grad():
            steered = session.run(**inputs).logits.detach().clone()
        artifacts = [
            artifact
            for artifact in session.artifacts
            if artifact.module == "neuron_activation_steering"
        ]

    assert not torch.allclose(
        baseline, steered
    ), f"{case_name}: steering did not change logits"
    assert artifacts and artifacts[-1].data.get("type") == "table"


@pytest.mark.parametrize(
    ("operation", "parameters"),
    [
        ("add", {"value": 3.0}),
        ("multiply", {"value": 0.0}),
        ("set", {"value": 2.0}),
        ("clamp", {"clamp_min": 0.0, "clamp_max": 0.0}),
    ],
)
def test_every_activation_steering_operation_changes_logits(operation, parameters):
    model, inputs = _decoder()
    model.eval()
    module = _external_module("neuron_activation_steering")

    with HeadlessEngine().instrument(model) as session:
        with torch.no_grad():
            baseline = session.run(**inputs).logits.detach().clone()
        session.attach(
            module,
            layer=session.layers("block")[0],
            config={
                "enabled": True,
                "neuron_indices": "0-3",
                "operation": operation,
                "positions": "all",
                "emit_mode": "all",
                **parameters,
            },
        )
        with torch.no_grad():
            steered = session.run(**inputs).logits.detach().clone()

    assert not torch.allclose(
        baseline, steered
    ), f"{operation}: intervention did not change logits"
