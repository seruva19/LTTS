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
