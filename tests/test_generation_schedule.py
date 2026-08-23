from __future__ import annotations

from types import ModuleType

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import HeadlessEngine


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
