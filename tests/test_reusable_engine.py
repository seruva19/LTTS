from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import HeadlessEngine, load_module


def _tiny_model():
    torch.manual_seed(7)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def test_engine_import_does_not_load_web_framework():
    root = Path(__file__).parents[1]
    code = (
        "import json,sys,ltts_engine; "
        "print(json.dumps({'fastapi': 'fastapi' in sys.modules, "
        "'router': 'scripts.router' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(completed.stdout.strip())
    assert state == {"fastapi": False, "router": False}


def test_headless_consumer_loads_attaches_runs_and_collects_artifact(tmp_path):
    module_file = tmp_path / "demo_module" / "init.py"
    module_file.parent.mkdir()
    module_file.write_text(
        """
METADATA = {"name": "consumer_probe", "event_types": ["layer_after"]}
def initialize_module(context, **config):
    return {"status": "initialized"}
def cleanup_module():
    return {"status": "cleaned_up"}
def interceptor(event):
    event.reactor.signal("visualization", {
        "module": "consumer_probe",
        "subtype": "scalar",
        "layer_path": event.context.module_path,
        "data": {"type": "scalar", "value": 1.0},
    })
    return {"status": "ok", "emitted": True}
""".strip(),
        encoding="utf-8",
    )
    module = load_module(module_file)
    inputs = torch.tensor([[1, 2, 3, 4]])

    with HeadlessEngine().instrument(_tiny_model()) as session:
        block = session.layers("block")[0]
        assert session.attach(module, layer=block) == block
        output = session.run(input_ids=inputs)

        assert output.logits.shape == (1, 4, 32)
        assert session.architecture["topology"] == "decoder_only"
        assert len(session.artifacts) == 1
        artifact = session.artifacts[0]
        assert artifact.module == "consumer_probe"
        assert artifact.layer_path == block
        assert artifact.data == {"type": "scalar", "value": 1.0}
