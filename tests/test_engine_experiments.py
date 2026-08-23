from __future__ import annotations

from types import ModuleType

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import (
    ExperimentCase,
    ExperimentRunner,
    HeadlessEngine,
    ModuleAttachment,
)


def _model():
    torch.manual_seed(19)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def _intervention_module() -> ModuleType:
    module = ModuleType("experiment_intervention")
    module.METADATA = {
        "name": "experiment_intervention",
        "event_types": ["layer_after"],
    }
    module.cleanup_calls = 0

    def initialize_module(context, **_config):
        module.context = context

    def cleanup_module():
        module.cleanup_calls += 1

    def interceptor(event):
        outputs = event.context.outputs
        hidden = outputs[0] if isinstance(outputs, tuple) else outputs
        changed = hidden.clone()
        changed[..., 0] += float(event.module_state.get("amount", 0.0))
        replacement = (changed, *outputs[1:]) if isinstance(outputs, tuple) else changed
        event.context.analysis_results["modified_outputs"] = replacement
        event.reactor.signal(
            "visualization",
            {
                "module": "experiment_intervention",
                "subtype": "scalar",
                "layer_path": event.context.module_path,
                "data": {
                    "type": "scalar",
                    "value": float(changed[..., 0].mean().detach()),
                },
            },
        )
        return {"status": "ok", "emitted": True}

    module.initialize_module = initialize_module
    module.cleanup_module = cleanup_module
    module.interceptor = interceptor
    return module


def _score(output):
    return output.logits[0, -1, 0].detach().item()


def test_experiment_runner_compares_baseline_control_and_intervention():
    model = _model()
    module = _intervention_module()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    with HeadlessEngine().instrument(model) as session:
        layer = session.layers("block")[0]
        result = ExperimentRunner(session).run(
            scorer=_score,
            kwargs=inputs,
            seed=31,
            cases=(
                ExperimentCase(name="empty-control", kind="control"),
                ExperimentCase(
                    name="steered",
                    attachments=(
                        ModuleAttachment(
                            module=module,
                            layer=layer,
                            config={"amount": 20.0},
                        ),
                    ),
                ),
            ),
            metadata={"question": "does channel zero affect logit zero?"},
        )

        assert result.by_name("empty-control").score == pytest.approx(
            result.baseline.score
        )
        steered = result.by_name("steered")
        assert steered.delta_from_baseline != pytest.approx(0.0, abs=1e-7)
        assert steered.artifacts[0].module == "experiment_intervention"
        assert not session.model.module_registry.attached_modules
        assert not session.artifacts

    assert result.to_dict()["schema_version"] == 1
    assert module.cleanup_calls == 1


def test_causal_sweep_isolates_layers_and_does_not_accumulate_hooks():
    model = _model()
    module = _intervention_module()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    blocks = list(model.transformer.h)
    hooks_before = [len(block._forward_hooks) for block in blocks]

    session = HeadlessEngine().instrument(model)
    hooks_instrumented = [len(block._forward_hooks) for block in blocks]
    assert hooks_instrumented == [value + 1 for value in hooks_before]
    layers = session.layers("block")
    result = ExperimentRunner(session).causal_sweep(
        module=module,
        layers=layers,
        scorer=_score,
        config={"amount": 20.0},
        kwargs=inputs,
        seed=5,
    )

    assert [point.layer for point in result.points] == layers
    assert all(
        point.delta_from_baseline != pytest.approx(0.0, abs=1e-7)
        for point in result.points
    )
    assert [len(block._forward_hooks) for block in blocks] == hooks_instrumented
    assert len({point.artifacts[0].layer_path for point in result.points}) == len(
        layers
    )

    session.close()
    assert [len(block._forward_hooks) for block in blocks] == hooks_before
