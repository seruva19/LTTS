from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ltts_engine import (
    decompose_attention_heads,
    decompose_mlp_neurons,
    graph_from_component_contributions,
)


def _scores(contribution):
    return {item.token_id: item.score for item in contribution.promoted}


def test_attention_head_decomposition_reconstructs_projected_logits():
    torch.manual_seed(41)
    heads = torch.randn(1, 3, 8)
    output_projection = nn.Linear(8, 6, bias=False)
    output_embedding = nn.Linear(6, 11, bias=False)

    contributions = decompose_attention_heads(
        heads,
        output_projection=output_projection,
        output_embedding=output_embedding,
        num_heads=2,
        top_k=11,
        decoder=lambda token_id: f"token-{token_id}",
    )

    assert [item.index for item in contributions] == [0, 1]
    assert all(item.kind == "attention_head" for item in contributions)
    expected = output_embedding(output_projection(heads[0, -1]))
    reconstructed = torch.tensor(
        [
            sum(_scores(item)[token_id] for item in contributions)
            for token_id in range(11)
        ]
    )
    assert torch.allclose(reconstructed, expected, atol=1e-5)

    graph = graph_from_component_contributions(
        contributions, layer_path="transformer.h.0.attn"
    )
    assert len([node for node in graph.nodes if node.kind == "attention_head"]) == 2
    assert all(
        edge.metadata["layer_path"] == "transformer.h.0.attn" for edge in graph.edges
    )
    assert json.loads(json.dumps(graph.to_dict()))["schema_version"] == 1


def test_mlp_neuron_decomposition_reconstructs_selected_neuron_effects():
    torch.manual_seed(43)
    activations = torch.randn(1, 2, 9)
    output_projection = nn.Linear(9, 6, bias=False)
    output_embedding = nn.Linear(6, 7, bias=False)

    contributions = decompose_mlp_neurons(
        activations,
        output_projection=output_projection,
        output_embedding=output_embedding,
        neuron_ids=[1, 4],
        top_k=7,
    )

    selected = torch.zeros(9)
    selected[1] = activations[0, -1, 1]
    selected[4] = activations[0, -1, 4]
    expected = output_embedding(output_projection(selected))
    reconstructed = torch.tensor(
        [
            sum(_scores(item)[token_id] for item in contributions)
            for token_id in range(7)
        ]
    )
    assert torch.allclose(reconstructed, expected, atol=1e-5)
    assert [item.index for item in contributions] == [1, 4]
    assert all(item.kind == "mlp_neuron" for item in contributions)


def test_component_decomposition_rejects_inexact_boundaries():
    with pytest.raises(ValueError, match="divisible"):
        decompose_attention_heads(
            torch.zeros(1, 2, 7),
            output_projection=nn.Linear(7, 4, bias=False),
            output_embedding=nn.Linear(4, 5, bias=False),
            num_heads=2,
        )
