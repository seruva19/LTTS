from __future__ import annotations

from types import ModuleType

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import (
    ExperimentCase,
    GenerationExperimentRunner,
    HeadlessEngine,
    ModuleAttachment,
)


def _model():
    torch.manual_seed(41)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
            eos_token_id=1,
            pad_token_id=0,
        )
    ).eval()


def _probe_module():
    module = ModuleType("generation_schedule_probe")
    module.METADATA = {
        "name": "generation_schedule_probe",
        "event_types": ["layer_after"],
    }

    def interceptor(event):
        step = event.context.generation_step
        event.reactor.signal(
            "visualization",
            {
                "module": "generation_schedule_probe",
                "subtype": "scalar",
                "layer_path": event.context.module_path,
                "data": {"type": "scalar", "value": step},
            },
        )
        return {"status": "ok", "step": step}

    module.interceptor = interceptor
    return module


def _force_token_module(target_vector):
    module = ModuleType("force_generation_token")
    module.METADATA = {
        "name": "force_generation_token",
        "event_types": ["layer_after"],
    }

    def interceptor(event):
        hidden = event.context.outputs
        changed = torch.zeros_like(hidden)
        changed[...] = target_vector.to(device=hidden.device, dtype=hidden.dtype)
        event.context.analysis_results["modified_outputs"] = changed
        return {"status": "ok"}

    module.interceptor = interceptor
    return module


def _position_probe_module():
    module = ModuleType("generation_position_probe")
    module.METADATA = {
        "name": "generation_position_probe",
        "event_types": ["layer_after"],
    }

    def interceptor(event):
        event.reactor.signal(
            "visualization",
            {
                "module": "generation_position_probe",
                "subtype": "json",
                "layer_path": event.context.module_path,
                "data": {
                    "step": event.context.generation_step,
                    "token_positions": event.environment["token_positions"],
                    "output_token_position": event.environment[
                        "output_token_position"
                    ],
                    "attachment_schedule": event.environment[
                        "attachment_schedule"
                    ],
                },
            },
        )
        return {"status": "ok"}

    module.interceptor = interceptor
    return module


def test_attachment_runs_only_on_selected_generation_step():
    model = _model()
    module = _probe_module()
    with HeadlessEngine().instrument(model) as session:
        session.attach(
            module,
            layer=session.layers("block")[0],
            schedule={"generation_steps": [1]},
        )
        session.run(input_ids=torch.tensor([[1, 2, 3]]))
        assert not session.artifacts

        generated = session.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=3,
            do_sample=False,
        )
        artifacts = session.artifacts

    assert generated.shape == (1, 6)
    assert len(artifacts) == 1
    assert artifacts[0].data == {"type": "scalar", "value": 1}


def test_model_after_attachment_retains_generation_context():
    model = _model()
    module = _probe_module()
    with HeadlessEngine().instrument(model) as session:
        session.attach(
            module,
            event_type="model_after",
            schedule={"generation_only": True},
        )
        session.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=2,
            do_sample=False,
        )
        artifacts = session.artifacts

    assert len(artifacts) == 1
    assert artifacts[0].layer_path == "model"
    assert artifacts[0].data == {"type": "scalar", "value": 1}


def test_generation_runner_saves_schedules_and_changed_token_trajectories():
    model = _model()
    module = _force_token_module(model.lm_head.weight[31].detach())
    with HeadlessEngine().instrument(model) as session:
        result = GenerationExperimentRunner(session).run(
            cases=(
                ExperimentCase(
                    name="force-31",
                    attachments=(
                        ModuleAttachment(
                            module=module,
                            layer="transformer.ln_f",
                            schedule={"generation_steps": [0, 1, 2]},
                        ),
                    ),
                ),
            ),
            kwargs={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "max_new_tokens": 3,
                "do_sample": False,
            },
            seed=7,
            decoder=lambda tokens: " ".join(str(token) for token in tokens),
        )

    trial = result.trials[0]
    assert trial.generated_tokens == ((31, 31, 31),)
    assert trial.generated_tokens != result.baseline.generated_tokens
    assert trial.changed_positions[0]
    assert trial.schedules[0]["schedule"] == {"generation_steps": [0, 1, 2]}
    assert trial.schedules[0]["module"] == "force_generation_token"
    assert trial.text == ("1 2 3 31 31 31",)


def test_generation_schedules_select_input_and_output_token_positions():
    model = _model()
    module = _position_probe_module()
    with HeadlessEngine().instrument(model) as session:
        layer = session.layers("block")[0]
        session.attach(module, layer=layer, schedule={"token_positions": [3]})
        session.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=3,
            do_sample=False,
        )
        input_position_artifacts = session.artifacts

    assert len(input_position_artifacts) == 1
    assert input_position_artifacts[0].data == {
        "step": 1,
        "token_positions": [3],
        "output_token_position": 4,
        "attachment_schedule": {"token_positions": [3]},
    }

    model = _model()
    module = _position_probe_module()
    with HeadlessEngine().instrument(model) as session:
        session.attach(
            module,
            layer=session.layers("block")[0],
            schedule={"output_token_positions": [3]},
        )
        session.generate(
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=2,
            do_sample=False,
        )
        output_position_artifacts = session.artifacts

    assert len(output_position_artifacts) == 1
    assert output_position_artifacts[0].data["step"] == 0
    assert output_position_artifacts[0].data["token_positions"] == [0, 1, 2]
    assert output_position_artifacts[0].data["output_token_position"] == 3
