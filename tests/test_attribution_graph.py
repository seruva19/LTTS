from __future__ import annotations

import json

import pytest

from ltts_engine import (
    Artifact,
    AttributionEdge,
    AttributionGraph,
    AttributionGraphBuilder,
    AttributionNode,
    CausalSweepPoint,
    CausalSweepResult,
    graph_from_causal_sweep,
    fuse_attribution_graphs,
    graph_from_attention_matrix,
    graph_from_dla_artifact,
    graph_from_sae_artifact,
)


def _sweep():
    artifact = Artifact(
        module="steering",
        subtype="table",
        data={"type": "table", "rows": []},
        layer_path="blocks.0",
        raw={},
    )
    return CausalSweepResult(
        baseline_score=2.0,
        seed=17,
        points=(
            CausalSweepPoint("blocks.0", 2.5, 0.5, (artifact,)),
            CausalSweepPoint("blocks.1", 1.99, -0.01, ()),
        ),
    )


def test_causal_sweep_graph_is_serializable_and_preserves_provenance():
    graph = graph_from_causal_sweep(_sweep(), target_label="Paris logit")

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2
    assert graph.node("target").label == "Paris logit"
    assert graph.edges[0].score == 0.5
    assert graph.edges[0].metadata["baseline_score"] == 2.0
    assert graph.edges[0].metadata["artifact_types"] == ["table"]
    assert json.loads(json.dumps(graph.to_dict()))["schema_version"] == 1

    pruned = graph.prune(0.1)
    assert len(pruned.edges) == 1
    assert {node.id for node in pruned.nodes} == {"layer:0", "target"}


def test_attribution_graph_rejects_unknown_nodes_and_conflicting_ids():
    node = AttributionNode(id="known", kind="layer", label="block")
    with pytest.raises(ValueError, match="unknown node"):
        AttributionGraph(
            nodes=(node,),
            edges=(
                AttributionEdge(
                    source="known", target="missing", score=1.0, method="test"
                ),
            ),
        )

    with pytest.raises(ValueError, match="unique"):
        AttributionGraph(nodes=(node, node), edges=())


def test_graph_fusion_namespaces_components_and_shares_output_logits():
    def graph(component_id, score):
        builder = AttributionGraphBuilder()
        builder.add_node(
            AttributionNode(id=component_id, kind="component", label=component_id)
        )
        builder.add_node(
            AttributionNode(
                id="local-target",
                kind="output_logit",
                label="Paris",
                metadata={"token_id": 7},
            )
        )
        builder.add_edge(
            AttributionEdge(
                source=component_id,
                target="local-target",
                score=score,
                method="test",
            )
        )
        return builder.build()

    fused = fuse_attribution_graphs(
        {"patching": graph("layer-0", 0.4), "dla": graph("head-2", -0.2)}
    )

    assert len(fused.nodes) == 3
    assert len(fused.edges) == 2
    assert fused.edges[0].target == fused.edges[1].target
    shared = fused.node(fused.edges[0].target)
    assert shared.kind == "output_logit"
    assert shared.metadata["graph_sources"] == ["patching", "dla"]
    assert {edge.metadata["graph_source"] for edge in fused.edges} == {
        "patching",
        "dla",
    }


def test_real_module_table_artifacts_convert_to_provenance_graphs():
    dla = Artifact(
        module="direct_logit_attribution",
        subtype="table",
        data={
            "type": "table",
            "headers": ["Layer", "Path", "Label", "Delta logit"],
            "rows": [
                [0, "transformer.h.0", "block 0", 0.25],
                [1, "transformer.h.1", "block 1", -0.1],
            ],
        },
        layer_path="transformer.h.1",
        raw={"type": "visualization"},
    )
    graph = graph_from_dla_artifact(dla, target_label=" Paris", target_token_id=7)
    assert graph.node("target").metadata["token_id"] == 7
    assert [edge.score for edge in graph.edges] == [0.25, -0.1]
    assert graph.edges[0].metadata["artifact_row"] == 0

    sae = Artifact(
        module="sae_features",
        subtype="table",
        data={
            "type": "table",
            "headers": ["Rank", "Feature", "Activation"],
            "rows": [[1, "feature 42", 3.5], [2, "feature 9", 1.25]],
        },
        layer_path="transformer.h.0",
        raw={"type": "visualization"},
    )
    graph = graph_from_sae_artifact(sae)
    assert graph.node("sae_feature:42").metadata["feature_id"] == 42
    assert graph.node("sae_feature:42").layer_path == "transformer.h.0"
    assert [edge.score for edge in graph.edges] == [3.5, 1.25]


def test_attention_matrix_graph_preserves_direction_and_head_provenance():
    graph = graph_from_attention_matrix(
        [[1.0, 0.0], [0.3, 0.7]],
        layer_path="transformer.h.0.attn",
        head_index=1,
        token_labels=("The", "sun"),
        minimum_weight=0.2,
    )

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 3
    cross = next(edge for edge in graph.edges if edge.score == 0.3)
    assert (cross.source, cross.target) == ("token:0", "token:1")
    assert cross.metadata["head_index"] == 1
