from __future__ import annotations

import types

import torch

from ltts_modules.gradient_importance.noise_methods import (
    attribution_sanity_report,
    noise_ensemble_attribution,
)
from scripts.core.attention_capture import AttentionCaptureManager
from scripts.services.inference.model_manager import ModelManager
from scripts.services.inference_service import InferenceService
from scripts.services.model_compatibility import (
    apply_capability_override,
    build_compatibility_report,
    module_attachment_contract,
    module_required_capabilities,
    module_target_roles,
)
from scripts.services.run_export import build_run_export


class _AttentionConfig:
    def __init__(self):
        self._attn_implementation = "sdpa"
        self.return_dict = False
        self.return_dict_in_generate = False
        self._output_attentions = False

    @property
    def output_attentions(self):
        return self._output_attentions

    @output_attentions.setter
    def output_attentions(self, value):
        if value and self._attn_implementation != "eager":
            raise ValueError("attention outputs require eager")
        self._output_attentions = value


class _CompatibleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _AttentionConfig()
        self.config.output_hidden_states = False
        self.proj = torch.nn.Linear(4, 8)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)

    def set_attn_implementation(self, implementation):
        self.config._attn_implementation = implementation

    def forward(self, inputs, output_hidden_states=False):
        return self.proj(inputs)


def test_attention_configuration_switches_before_enabling_outputs():
    model = _CompatibleModel()
    ltts_model = types.SimpleNamespace(original_model=model)

    status = ModelManager(types.SimpleNamespace()).configure_attention_outputs(
        ltts_model
    )

    assert status["supported"] is True
    assert status["implementation"] == "eager"
    assert model.config.output_attentions is True


def test_attention_capture_does_not_mutate_model_configuration_on_init():
    model = _CompatibleModel()

    manager = AttentionCaptureManager(model)

    assert manager.enabled is False
    assert model.config.output_attentions is False


def test_module_requirements_do_not_force_attention_for_every_module():
    assert "eager_attention" in module_required_capabilities(
        "attention_head_grid"
    )
    assert "eager_attention" not in module_required_capabilities(
        "residual_stream_norm", {"requires": ["hidden_states"]}
    )
    assert module_required_capabilities(
        "gradient_importance", {"requires": ["hidden_states", "gradients"]}
    ) == {"hidden_states", "gradients"}


def test_module_attachment_contracts_distinguish_architecture_roles():
    assert module_target_roles("attention_rollout") == {"self_attention"}
    assert module_target_roles("attention_head_grid") == {
        "self_attention",
        "cross_attention",
    }
    assert module_target_roles("residual_stream_norm") == {"block"}
    assert module_target_roles("neuron_ablation") == {
        "block",
        "mlp",
        "activation",
    }
    assert module_target_roles("neuron_activation_steering") == {
        "block",
        "mlp",
        "activation",
    }
    assert module_target_roles("memory_monitor") == {"model"}

    contract = module_attachment_contract("sae_features")
    assert contract["target_roles"] == ["block"]
    assert contract["supported_topologies"] == ["decoder_only"]
    assert contract["optional_dependencies"][0]["package"] == "sae_lens"

    control_contract = module_attachment_contract("model_control_vector")
    assert control_contract["target_roles"] == ["model"]
    assert control_contract["supported_topologies"] == ["decoder_only"]
    assert control_contract["optional_dependencies"][0]["package"] == "repeng"


def test_control_vector_visualization_reports_real_norm():
    from ltts_modules.model_control_vector import init as control_vector

    vector = types.SimpleNamespace(
        directions={
            0: torch.tensor([3.0, 4.0]).numpy(),
            1: torch.tensor([0.0, 2.0]).numpy(),
        }
    )

    payload = control_vector.generate_control_vector_visualization_data(
        vector_strength=-2.0,
        trained_vector=vector,
    )

    assert payload["trained"] is True
    assert payload["layers"] == 2
    assert payload["coefficient"] == -2.0
    assert payload["value"] == 7.0


def test_sae_features_accepts_semantic_block_role(monkeypatch):
    from ltts_modules.sae_features import init as sae_features

    class CapturingSender:
        def __init__(self):
            self.context = None
            self.tables = []

        def set_context(self, context):
            self.context = context

        def send_table(self, headers, rows, **kwargs):
            self.tables.append((headers, rows, kwargs))
            return True

        def send_scalar(self, *_args, **_kwargs):
            return True

    class FakeSAE:
        cfg = types.SimpleNamespace(d_in=4)
        device = torch.device("cpu")

        @staticmethod
        def parameters():
            yield torch.nn.Parameter(torch.zeros(1))

        @staticmethod
        def encode(vector):
            return torch.tensor([[0.0, 2.0, 0.5, 1.0]])

    sender = CapturingSender()
    monkeypatch.setattr(sae_features, "_SENDER", sender)
    monkeypatch.setattr(sae_features, "SAE_AVAILABLE", True)
    monkeypatch.setattr(
        sae_features, "_get_sae", lambda *_args, **_kwargs: FakeSAE()
    )

    event = types.SimpleNamespace(
        event_type="layer_after",
        context=types.SimpleNamespace(
            module_path="model.layers.0",
            module_class="LlamaDecoderLayer",
            module_role="block",
            layer_type="block",
            outputs=torch.randn(1, 3, 4),
        ),
        module_state={"enabled": True, "emit_mode": "all"},
        environment={},
        should_emit=lambda *_args, **_kwargs: {"emit": True},
        mark_emission_finalized=lambda *_args, **_kwargs: None,
    )

    result = sae_features.process_event(event)

    assert result["status"] == "ok"
    assert result["top_features"][0] == 1
    assert sender.context is event.context
    assert len(sender.tables) == 1


def test_compatibility_report_covers_each_selected_module():
    model = _CompatibleModel()
    ltts_model = types.SimpleNamespace(
        original_model=model,
        _tokenizer=types.SimpleNamespace(decode=lambda ids: str(ids[0])),
        layer_wrappers={"model.layers.0": object()},
    )
    modules = {
        "model.layers.0": [
            {"name": "residual_stream_norm", "parameters": {}},
            {"name": "attention_head_grid", "parameters": {}},
        ]
    }
    available = {
        "residual_stream_norm": {
            "metadata": {"requires": ["hidden_states"]}
        },
        "attention_head_grid": {
            "metadata": {"requires": ["hidden_states", "eager_attention"]}
        },
    }
    context = types.SimpleNamespace(
        module_loader=types.SimpleNamespace(get_all_modules=lambda: available)
    )

    report = build_compatibility_report(
        context, ltts_model, "test/model", modules
    )

    assert report["compatible"] is True
    assert {item["name"] for item in report["modules"]} == {
        "attention_head_grid",
        "residual_stream_norm",
    }
    assert all(item["compatible"] for item in report["modules"])

    apply_capability_override(
        report,
        "eager_attention",
        {"supported": False, "detail": "active probe failed"},
    )
    attention_report = next(
        item for item in report["modules"] if item["name"] == "attention_head_grid"
    )
    residual_report = next(
        item for item in report["modules"] if item["name"] == "residual_stream_norm"
    )
    assert attention_report["compatible"] is False
    assert residual_report["compatible"] is True
    assert report["compatible"] is False


def test_compatibility_report_rejects_semantically_wrong_layer_role():
    model = _CompatibleModel()
    ltts_model = types.SimpleNamespace(
        original_model=model,
        architecture={
            "topology": "decoder_only",
            "capabilities": {"hidden_states": True},
        },
        module_descriptors={
            "model.layers.0.self_attn": {"role": "self_attention"}
        },
        layer_wrappers={"model.layers.0.self_attn": object()},
    )
    context = types.SimpleNamespace(
        module_loader=types.SimpleNamespace(
            get_all_modules=lambda: {
                "residual_stream_norm": {
                    "metadata": {"requires": ["hidden_states"]}
                }
            }
        )
    )

    report = build_compatibility_report(
        context,
        ltts_model,
        "test/model",
        {
            "model.layers.0.self_attn": [
                {"name": "residual_stream_norm", "parameters": {}}
            ]
        },
    )

    module_report = report["modules"][0]
    assert report["compatible"] is False
    assert module_report["role_mismatches"] == [
        {
            "layer": "model.layers.0.self_attn",
            "resolved_layer": "model.layers.0.self_attn",
            "role": "self_attention",
            "allowed_roles": ["block"],
        }
    ]


def test_compatibility_report_rejects_missing_optional_runtime(monkeypatch):
    import scripts.services.model_compatibility as compatibility

    monkeypatch.setattr(
        compatibility.importlib.util,
        "find_spec",
        lambda package: None if package == "sae_lens" else object(),
    )
    model = _CompatibleModel()
    ltts_model = types.SimpleNamespace(
        original_model=model,
        architecture={
            "topology": "decoder_only",
            "capabilities": {"hidden_states": True},
        },
        module_descriptors={"model.layers.0": {"role": "block"}},
        layer_wrappers={"model.layers.0": object()},
    )
    context = types.SimpleNamespace(
        module_loader=types.SimpleNamespace(
            get_all_modules=lambda: {
                "sae_features": {
                    "metadata": {
                        "requires": ["hidden_states"],
                        "target_roles": ["block"],
                        "supported_topologies": ["decoder_only"],
                    }
                }
            }
        )
    )

    report = build_compatibility_report(
        context,
        ltts_model,
        "test/model",
        {"model.layers.0": [{"name": "sae_features"}]},
    )

    assert report["compatible"] is False
    assert report["modules"][0]["missing_optional_dependencies"] == ["sae_lens"]


def test_noise_methods_are_deterministic_and_vargrad_is_nonnegative():
    outputs = torch.linspace(-1.0, 1.0, 24).reshape(1, 3, 8)
    first = noise_ensemble_attribution(
        outputs, "smoothgrad", samples=6, noise_std=0.2, seed=7
    )
    second = noise_ensemble_attribution(
        outputs, "smoothgrad", samples=6, noise_std=0.2, seed=7
    )
    vargrad = noise_ensemble_attribution(
        outputs, "vargrad", samples=6, noise_std=0.2, seed=7
    )

    assert torch.equal(first, second)
    assert torch.all(vargrad >= 0)
    assert first.shape == (1, 3)


def test_attribution_sanity_report_is_finite_and_explicit_about_scope():
    outputs = torch.randn(1, 4, 6)
    report = attribution_sanity_report(outputs, samples=4, seed=3)

    assert report["finite_fraction"] == 1.0
    assert 0.0 <= report["smoothgrad_similarity"] <= 1.0
    assert report["scope"] == "activation-space perturbation checks"


def test_gradient_importance_defers_until_real_target_backward():
    from ltts_modules.gradient_importance import init as gradient_module
    from scripts.core.ltts_event import LTTSReactor

    activation = torch.randn(1, 3, 4, requires_grad=True)

    class Event:
        event_type = "layer_after"
        context = types.SimpleNamespace(
            outputs=activation,
            module_path="model.layers.0",
            ltts_model=types.SimpleNamespace(),
        )
        module_state = {
            "method": "saliency",
            "visualization_type": "gradient_heatmap",
            "emit_mode": "all",
        }
        reactor = LTTSReactor(message_bus=None)

        @staticmethod
        def should_emit(*_args, **_kwargs):
            return {"emit": True}

    gradient_module.cleanup()
    event = Event()
    pending = gradient_module.process_event(event)
    assert pending["status"] == "pending_backward"
    assert event.reactor.signals == []

    weights = torch.tensor(
        [[[1.0, -2.0, 0.5, 0.0], [0.5, 0.5, 0.5, 0.5], [3.0, 0.0, 0.0, 4.0]]]
    )
    target = (activation * weights).sum()
    target.backward()
    finalized = gradient_module.finalize_gradients(
        target_info={"kind": "test_score", "score": float(target.detach())}
    )

    assert finalized["status"] == "emitted"
    payload = event.reactor.signals[0]["data"]
    assert payload["target"]["kind"] == "test_score"
    values = torch.tensor(payload["data"]["values"])
    expected = torch.linalg.vector_norm(weights.abs(), dim=-1)
    expected = (expected - expected.min()) / (expected.max() - expected.min())
    assert torch.allclose(values, expected)
    gradient_module.cleanup()


def test_inference_gradient_target_is_selected_from_model_logits():
    logits = torch.tensor(
        [[[0.1, 0.2, 0.9], [1.5, -1.0, 0.3]]], requires_grad=True
    )
    outputs = types.SimpleNamespace(logits=logits)
    ltts_model = types.SimpleNamespace(
        _model_task="causal_lm",
        original_model=types.SimpleNamespace(config=types.SimpleNamespace()),
    )

    target, info = InferenceService._select_gradient_target(
        ltts_model,
        {"input_ids": torch.tensor([[1, 2]])},
        outputs,
        tokenizer=None,
        attached_specs=[],
    )

    assert info["kind"] == "token_logit"
    assert info["position"] == 1
    assert info["target_id"] == 0
    assert torch.equal(target, logits[:, 1, 0].sum())


def test_run_export_contains_reproducibility_fields():
    compatibility = {
        "model_revision": "abc123",
        "model_class": "ExampleModel",
        "compatible": True,
    }
    exported = build_run_export(
        model_id="org/model",
        prompt="hello",
        parameters={"temperature": 0.2},
        modules={"layer.0": [{"name": "residual_stream_norm"}]},
        result="world",
        output_info={"type": "ModelOutput"},
        compatibility=compatibility,
    )

    assert exported["schema_version"] == 1
    assert exported["model"]["revision"] == "abc123"
    assert exported["input"]["prompt"] == "hello"
    assert exported["output"]["result"] == "world"
    assert exported["environment"]["packages"]["torch"]
