from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from ltts_engine import (
    RecirculationRunner,
    RecirculationSpec,
    discover_recirculation_layers,
    mix_recirculated_residual,
    recirculation_sweep_heatmap,
)


class _CausalBlock(nn.Module):
    def __init__(self, width: int, scale: float) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.projection.weight)
        self.projection.weight.data.mul_(scale)

    def forward(self, hidden):
        positions = torch.arange(1, hidden.shape[1] + 1, device=hidden.device)
        context = hidden.cumsum(dim=1) / positions.view(1, -1, 1)
        return (hidden + torch.tanh(self.projection(context)),)


class _TinyCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(3)
        self.embedding = nn.Embedding(9, 4)
        self.layers = nn.ModuleList([_CausalBlock(4, 0.3), _CausalBlock(4, 1.1)])
        self.head = nn.Linear(4, 9, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.head(hidden))


def test_l2_mixing_matches_destination_norm_and_paper_equation():
    source = torch.tensor([[[3.0, 4.0]]])
    destination = torch.tensor([[[0.0, 10.0]]])

    mixed = mix_recirculated_residual(
        source, destination, alpha=0.25, beta=0.75, normalization="l2"
    )

    assert torch.allclose(mixed, torch.tensor([[[1.5, 9.5]]]))


def test_reference_runner_is_paired_serializable_and_leaves_no_hooks():
    model = _TinyCausalLM()
    runner = RecirculationRunner(model)
    spec = RecirculationSpec(
        source_layer="layers.1",
        destination_layer="layers.0",
        alpha=0.2,
        ramp_steps=2,
    )

    result = runner.evaluate(
        torch.tensor([[1, 2, 3, 4, 5]]), spec, metadata={"model": "tiny"}
    )

    assert result.token_count == 4
    assert result.baseline_token_nll != result.recirculation_token_nll
    assert result.metadata["model"] == "tiny"
    assert result.spec.alpha_at(0) == 0.0
    assert result.spec.beta_at(0) == 1.0
    assert result.spec.alpha_at(2) == 0.2
    assert result.spec.beta_at(2) == 0.8
    assert not model.layers[0]._forward_hooks
    assert not model.layers[1]._forward_hooks
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1


def test_zero_alpha_convex_recirculation_matches_ordinary_execution():
    model = _TinyCausalLM()
    result = RecirculationRunner(model).evaluate(
        torch.tensor([[1, 2, 3, 4]]),
        RecirculationSpec(
            source_layer="layers.1", destination_layer="layers.0", alpha=0.0
        ),
    )

    assert result.recirculation_token_nll == pytest.approx(
        result.baseline_token_nll, abs=1e-6
    )
    assert result.perplexity_change_percent == pytest.approx(0.0, abs=1e-5)


def test_sweep_is_paired_selects_lowest_perplexity_and_rejects_duplicates():
    runner = RecirculationRunner(_TinyCausalLM())
    input_ids = torch.tensor([[1, 2, 3, 4]])
    specs = (
        RecirculationSpec("layers.1", "layers.0", alpha=0.0),
        RecirculationSpec("layers.1", "layers.0", alpha=0.2),
    )

    result = runner.sweep(input_ids, specs, metadata={"dataset": "tiny"})

    assert len(result.evaluations) == 2
    assert result.best.recirculation_perplexity == min(
        item.recirculation_perplexity for item in result.evaluations
    )
    assert result.evaluations[0].baseline_token_nll == pytest.approx(
        result.evaluations[1].baseline_token_nll
    )
    assert result.metadata["dataset"] == "tiny"
    assert json.loads(json.dumps(result.to_dict()))["best_index"] == result.best_index

    with pytest.raises(ValueError, match="unique"):
        runner.sweep(input_ids, (specs[0], specs[0]))


def test_dataset_sweep_uses_token_weighted_aggregation():
    runner = RecirculationRunner(_TinyCausalLM())
    examples = (torch.tensor([[1, 2, 3]]), torch.tensor([[4, 5, 6, 7, 8]]))
    specs = (
        RecirculationSpec("layers.1", "layers.0", alpha=0.0),
        RecirculationSpec("layers.1", "layers.0", alpha=0.2),
    )

    result = runner.sweep_dataset(examples, specs, metadata={"split": "validation"})

    assert result.evaluations[0].token_count == 6
    expected = sum(
        item.baseline_mean_nll * item.token_count
        for item in result.evaluations[0].examples
    ) / 6
    assert result.evaluations[0].baseline_mean_nll == pytest.approx(expected)
    assert result.best.recirculation_perplexity == min(
        item.recirculation_perplexity for item in result.evaluations
    )
    assert result.metadata["split"] == "validation"
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1


def test_sweep_builds_existing_renderer_heatmap_payload():
    runner = RecirculationRunner(_TinyCausalLM())
    base = dict(alpha=0.1, ramp_steps=2)
    specs = (
        RecirculationSpec("layers.1", "layers.0", **base),
    )
    result = runner.sweep(torch.tensor([[1, 2, 3]]), specs)

    heatmap = recirculation_sweep_heatmap(result)

    assert heatmap["type"] == "heatmap"
    assert heatmap["dimensions"] == {"rows": 1, "cols": 1}
    assert heatmap["row_labels"] == ["layers.1"]
    assert heatmap["column_labels"] == ["layers.0"]
    assert heatmap["color_scale"] == "diverging_zero"
    assert json.loads(json.dumps(heatmap))["metric"] == "perplexity_change_percent"


def test_paired_greedy_generation_is_serializable_and_restores_hooks():
    model = _TinyCausalLM()
    result = RecirculationRunner(model).generate(
        torch.tensor([[1, 2, 3]]),
        RecirculationSpec("layers.1", "layers.0", alpha=0.6, beta=1.0),
        max_new_tokens=4,
        metadata={"seed": 0},
    )

    assert result.prompt_length == 3
    assert len(result.baseline_sequence) == 7
    assert len(result.recirculation_sequence) == 7
    assert all(position >= 3 for position in result.changed_positions)
    assert result.metadata["seed"] == 0
    assert not model.layers[0]._forward_hooks
    assert not model.layers[1]._forward_hooks
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1


def test_zero_alpha_generation_matches_baseline():
    result = RecirculationRunner(_TinyCausalLM()).generate(
        torch.tensor([[1, 2, 3]]),
        RecirculationSpec("layers.1", "layers.0", alpha=0.0),
        max_new_tokens=3,
    )

    assert result.baseline_sequence == result.recirculation_sequence
    assert result.changed_positions == ()


def test_recirculation_rejects_invalid_configuration_and_inputs():
    with pytest.raises(ValueError, match="different"):
        RecirculationSpec("layers.0", "layers.0")
    with pytest.raises(ValueError, match="after"):
        RecirculationRunner(_TinyCausalLM()).evaluate(
            torch.tensor([[1, 2]]),
            RecirculationSpec("layers.0", "layers.1"),
        )
    with pytest.raises(ValueError, match="shapes"):
        mix_recirculated_residual(
            torch.zeros(1, 1, 2),
            torch.zeros(1, 1, 3),
            alpha=0.1,
            beta=0.9,
        )


def test_reference_runner_supports_real_transformers_decoder_contract():
    transformers = pytest.importorskip("transformers")
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            vocab_size=17,
            n_positions=8,
            n_embd=8,
            n_layer=3,
            n_head=2,
        )
    )
    result = RecirculationRunner(model).evaluate(
        torch.tensor([[1, 2, 3]]),
        RecirculationSpec("transformer.h.2", "transformer.h.0", alpha=0.1),
    )

    assert result.token_count == 2
    assert result.baseline_perplexity > 0
    assert result.recirculation_perplexity > 0
    assert not model.transformer.h[0]._forward_hooks
    assert not model.transformer.h[2]._forward_hooks

    compatibility = discover_recirculation_layers(model)
    assert compatibility.supported
    assert compatibility.container_path == "transformer.h"
    assert compatibility.layer_paths == (
        "transformer.h.0",
        "transformer.h.1",
        "transformer.h.2",
    )
    assert json.loads(json.dumps(compatibility.to_dict()))["schema_version"] == 1
