from __future__ import annotations

import json

import pytest

from ltts_engine import (
    Artifact,
    attribution_from_gradient_artifact,
    attribution_from_occlusion_artifact,
    attribution_from_token_contributor_artifact,
    component_attribution_from_gradient_artifact,
    graph_from_component_attribution,
    graph_from_token_attribution,
)


def test_occlusion_artifact_becomes_signed_token_graph():
    artifact = Artifact(
        module="occlusion_attribution",
        subtype="token_attribution",
        layer_path="model",
        data={
            "type": "chart",
            "chart_type": "bar",
            "data": {
                "positions": [0, 1],
                "tokens": ["The", "sun"],
                "attribution": [0.2, -0.05],
            },
            "baseline_top_token": " shines",
            "baseline_probability": 0.7,
        },
        raw={"type": "visualization"},
    )

    result = attribution_from_occlusion_artifact(artifact)
    assert result.score_semantics == "signed_effect"
    assert result.target.label == " shines"
    assert [point.label for point in result.points] == ["The", "sun"]
    graph = graph_from_token_attribution(result)
    assert [edge.score for edge in graph.edges] == [0.2, -0.05]
    assert graph.node("token:1").position == 1
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1


def test_gradient_heatmap_aggregates_batches_and_retains_target():
    artifact = Artifact(
        module="gradient_importance",
        subtype="gradient_heatmap",
        layer_path="transformer.h.0",
        data={
            "type": "heatmap",
            "values": [[1.0, -2.0, 3.0], [3.0, 4.0, -1.0]],
            "dimensions": {"rows": 2, "cols": 3},
            "target": {"token_id": 7, "token": " sun", "position": 2},
            "method": "input_x_gradient",
            "quantity": "absolute activation × target gradient",
        },
        raw={"type": "visualization"},
    )

    result = attribution_from_gradient_artifact(
        artifact, token_labels=["The", "bright", "sun"]
    )
    assert [point.score for point in result.points] == [2.0, 3.0, 2.0]
    assert result.target.token_id == 7
    assert result.target.position == 2
    graph = graph_from_token_attribution(result)
    assert graph.node("target").kind == "output_logit"
    assert graph.metadata["score_semantics"] == "absolute_importance"


def test_token_contributor_chart_and_invalid_schema_are_explicit():
    artifact = Artifact(
        module="token_contributor",
        subtype="chart",
        layer_path="transformer.h.0",
        data={
            "type": "chart",
            "chart_type": "line",
            "data": {"positions": [0, 1], "importance": [0.4, 0.1]},
        },
        raw={},
    )
    result = attribution_from_token_contributor_artifact(artifact)
    assert result.method == "token_contributor"
    assert result.layer_path == "transformer.h.0"

    bad = Artifact("other", "chart", artifact.data, None, {})
    with pytest.raises(ValueError, match="token_contributor"):
        attribution_from_token_contributor_artifact(bad)


def test_gradient_neuron_table_becomes_component_graph():
    artifact = Artifact(
        module="gradient_importance",
        subtype="neuron_importance",
        layer_path="transformer.h.0.mlp",
        data={
            "type": "table",
            "headers": ["Neuron", "Absolute attribution"],
            "rows": [[7, 0.8], [2, 0.3]],
            "target": {"token_id": 11, "label": " bright", "position": 3},
            "method": "input_x_gradient",
        },
        raw={},
    )

    result = component_attribution_from_gradient_artifact(artifact)
    assert result.component_kind == "neuron"
    assert [point.index for point in result.points] == [7, 2]
    graph = graph_from_component_attribution(result)
    assert graph.node("neuron:7").layer_path == "transformer.h.0.mlp"
    assert graph.node("target").metadata["token_id"] == 11
    assert [edge.score for edge in graph.edges] == [0.8, 0.3]
