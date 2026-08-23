"""Optional executable GPU smoke test for the reusable engine."""

import os
from pathlib import Path

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import HeadlessEngine, load_module


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_gpu_steering_changes_logits_and_keeps_outputs_on_device():
    torch.manual_seed(11)
    model = (
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                n_positions=16,
            )
        )
        .eval()
        .cuda()
    )
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]], device="cuda")}
    modules_dir = Path(
        os.getenv("LTTS_MODULES_DIR", Path(__file__).parents[2] / "LTTS-modules")
    )
    module_path = modules_dir / "neuron_activation_steering" / "init.py"
    if not module_path.is_file():
        pytest.skip("sibling LTTS-modules checkout is unavailable")
    module = load_module(module_path, name="gpu_neuron_activation_steering")

    with HeadlessEngine().instrument(model) as session:
        with torch.no_grad():
            baseline = session.run(**inputs).logits.detach().clone()
        session.attach(
            module,
            layer=session.layers("block")[0],
            config={
                "enabled": True,
                "neuron_indices": "0-3",
                "operation": "add",
                "value": 5.0,
                "emit_mode": "all",
            },
        )
        with torch.no_grad():
            steered = session.run(**inputs).logits.detach().clone()

        assert baseline.is_cuda and steered.is_cuda
        assert not torch.allclose(baseline, steered)
        assert any(
            artifact.module == "neuron_activation_steering"
            and artifact.data.get("type") == "table"
            for artifact in session.artifacts
        )
