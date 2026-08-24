from __future__ import annotations

import json

import pytest

from ltts_engine import (
    AttributionEdge,
    AttributionGraph,
    AttributionNode,
    InterventionTargetRef,
    discover_circuit,
)


def _graph() -> AttributionGraph:
    return AttributionGraph(
        nodes=(
            AttributionNode(
                id="strong-source",
                kind="attention_head",
                label="head 1",
                intervention_target=InterventionTargetRef(
                    "transformer.h.0.attn", component_kind="head", component_index=1
                ),
            ),
            AttributionNode(
                id="weak-source",
                kind="neuron",
                label="neuron 3",
                intervention_target=InterventionTargetRef(
                    "transformer.h.0.mlp", component_kind="neuron", component_index=3
                ),
            ),
            AttributionNode(
                id="middle",
                kind="component",
                label="residual",
                intervention_target=InterventionTargetRef("transformer.h.1"),
            ),
            AttributionNode(id="target", kind="metric", label="answer score"),
            AttributionNode(id="unrelated", kind="component", label="unrelated"),
        ),
        edges=(
            AttributionEdge("strong-source", "middle", -0.8, "patching"),
            AttributionEdge("weak-source", "middle", 0.2, "patching"),
            AttributionEdge("middle", "target", 0.9, "patching"),
            AttributionEdge("unrelated", "unrelated", 50.0, "correlation"),
        ),
        metadata={"run_id": "run-7"},
    )


def test_discovery_returns_strong_connected_signed_circuit():
    result = discover_circuit(_graph(), target_node_id="target", max_nodes=3)

    assert {node.id for node in result.graph.nodes} == {
        "strong-source",
        "middle",
        "target",
    }
    assert [edge.score for edge in result.graph.edges] == [-0.8, 0.9]
    assert result.selected_edge_indices == (0, 2)
    assert result.candidate_edge_count == 3
    assert result.total_absolute_score == pytest.approx(1.9)
    assert result.retained_absolute_score == pytest.approx(1.7)
    assert result.estimated_retained_fraction == pytest.approx(1.7 / 1.9)
    assert result.graph.metadata["run_id"] == "run-7"
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1


def test_discovery_validates_full_and_compact_intervention_sets_separately():
    calls = []

    def validate(targets):
        calls.append(targets)
        return float(len(targets))

    result = discover_circuit(
        _graph(), target_node_id="target", max_nodes=2, validator=validate
    )

    assert [len(call) for call in calls] == [3, 1]
    assert result.validation.full_effect == 3.0
    assert result.validation.circuit_effect == 1.0
    assert result.validation.fidelity == pytest.approx(1 / 3)
    assert result.intervention_targets[0].layer_path == "transformer.h.1"


def test_discovery_is_deterministic_and_obeys_threshold_and_budget():
    graph = _graph()
    first = discover_circuit(
        graph, target_node_id="target", max_nodes=1, minimum_absolute_score=0.5
    )
    second = discover_circuit(
        graph, target_node_id="target", max_nodes=1, minimum_absolute_score=0.5
    )

    assert first.to_dict() == second.to_dict()
    assert [node.id for node in first.graph.nodes] == ["target"]
    assert first.graph.edges == ()
    assert first.candidate_edge_count == 2


def test_discovery_rejects_invalid_limits_scores_and_targets():
    with pytest.raises(ValueError, match="max_nodes"):
        discover_circuit(_graph(), target_node_id="target", max_nodes=0)
    with pytest.raises(KeyError):
        discover_circuit(_graph(), target_node_id="missing", max_nodes=2)

    graph = AttributionGraph(
        nodes=(
            AttributionNode(id="source", kind="layer", label="source"),
            AttributionNode(id="target", kind="metric", label="target"),
        ),
        edges=(AttributionEdge("source", "target", float("nan"), "test"),),
    )
    with pytest.raises(ValueError, match="finite"):
        discover_circuit(graph, target_node_id="target", max_nodes=2)
