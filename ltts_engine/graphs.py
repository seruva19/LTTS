"""Versioned attribution-graph schemas shared by research components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .experiments import CausalSweepResult
from .inspection import ComponentComparison

NODE_KINDS = {
    "input_token",
    "layer",
    "component",
    "attention_head",
    "neuron",
    "sae_feature",
    "output_logit",
    "metric",
}


@dataclass(frozen=True)
class AttributionNode:
    id: str
    kind: str
    label: str
    layer_path: Optional[str] = None
    position: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Attribution node id cannot be empty")
        if self.kind not in NODE_KINDS:
            raise ValueError(f"Unsupported attribution node kind: {self.kind}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "layer_path": self.layer_path,
            "position": self.position,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AttributionEdge:
    source: str
    target: str
    score: float
    method: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "score": self.score,
            "method": self.method,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AttributionGraph:
    """A transport-neutral causal/attribution graph with provenance."""

    nodes: Tuple[AttributionNode, ...]
    edges: Tuple[AttributionEdge, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Attribution node ids must be unique")
        known = set(node_ids)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"Attribution edge references an unknown node: "
                    f"{edge.source} -> {edge.target}"
                )

    def node(self, node_id: str) -> AttributionNode:
        for item in self.nodes:
            if item.id == node_id:
                return item
        raise KeyError(node_id)

    def prune(self, minimum_absolute_score: float) -> "AttributionGraph":
        """Drop weak edges and disconnected non-target nodes."""

        threshold = abs(float(minimum_absolute_score))
        edges = tuple(edge for edge in self.edges if abs(edge.score) >= threshold)
        connected = {
            endpoint for edge in edges for endpoint in (edge.source, edge.target)
        }
        nodes = tuple(
            node
            for node in self.nodes
            if node.id in connected or node.kind in {"metric", "output_logit"}
        )
        return AttributionGraph(
            nodes=nodes,
            edges=edges,
            metadata={**self.metadata, "minimum_absolute_score": threshold},
            schema_version=self.schema_version,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metadata": dict(self.metadata),
        }


class AttributionGraphBuilder:
    """Validated mutable builder with an immutable graph result."""

    def __init__(self, *, metadata: Optional[Mapping[str, Any]] = None) -> None:
        self._nodes: Dict[str, AttributionNode] = {}
        self._edges = []
        self._metadata = dict(metadata or {})

    def add_node(self, node: AttributionNode) -> "AttributionGraphBuilder":
        existing = self._nodes.get(node.id)
        if existing is not None and existing != node:
            raise ValueError(f"Conflicting attribution node id: {node.id}")
        self._nodes[node.id] = node
        return self

    def add_edge(self, edge: AttributionEdge) -> "AttributionGraphBuilder":
        self._edges.append(edge)
        return self

    def get_node(self, node_id: str) -> Optional[AttributionNode]:
        return self._nodes.get(node_id)

    def replace_node(self, node: AttributionNode) -> "AttributionGraphBuilder":
        if node.id not in self._nodes:
            raise KeyError(f"Cannot replace missing attribution node: {node.id}")
        self._nodes[node.id] = node
        return self

    def build(self) -> AttributionGraph:
        return AttributionGraph(
            nodes=tuple(self._nodes.values()),
            edges=tuple(self._edges),
            metadata=self._metadata,
        )


def graph_from_causal_sweep(
    sweep: CausalSweepResult,
    *,
    target_label: str,
    target_kind: str = "metric",
    method: str = "causal_intervention",
) -> AttributionGraph:
    """Represent layer-wise intervention effects as a causal graph."""

    target_id = "target"
    builder = AttributionGraphBuilder(
        metadata={
            "source": "causal_sweep",
            "seed": sweep.seed,
            "baseline_score": sweep.baseline_score,
        }
    )
    builder.add_node(
        AttributionNode(id=target_id, kind=target_kind, label=target_label)
    )
    for index, point in enumerate(sweep.points):
        node_id = f"layer:{index}"
        artifact_types = sorted(
            {artifact.subtype for artifact in point.artifacts if artifact.subtype}
        )
        builder.add_node(
            AttributionNode(
                id=node_id,
                kind="layer",
                label=point.layer,
                layer_path=point.layer,
                metadata={"sweep_index": index},
            )
        )
        builder.add_edge(
            AttributionEdge(
                source=node_id,
                target=target_id,
                score=point.delta_from_baseline,
                method=method,
                metadata={
                    "intervention_score": point.score,
                    "baseline_score": sweep.baseline_score,
                    "artifact_types": artifact_types,
                },
            )
        )
    return builder.build()


def graph_from_component_comparison(
    comparison: ComponentComparison,
) -> AttributionGraph:
    """Build signed component-to-logit edges from a vocabulary drill-down."""

    component_id = "component"
    builder = AttributionGraphBuilder(
        metadata={
            "source": "component_vocabulary_comparison",
            "reference_layer": comparison.reference_layer,
            "position": comparison.position,
        }
    )
    builder.add_node(
        AttributionNode(
            id=component_id,
            kind="component",
            label=comparison.component_layer,
            layer_path=comparison.component_layer,
            position=comparison.position,
        )
    )
    scores = {item.token_id: item for item in comparison.suppressed}
    scores.update({item.token_id: item for item in comparison.promoted})
    for token_id, item in sorted(scores.items()):
        node_id = f"logit:{token_id}"
        builder.add_node(
            AttributionNode(
                id=node_id,
                kind="output_logit",
                label=item.token or str(token_id),
                metadata={"token_id": token_id},
            )
        )
        builder.add_edge(
            AttributionEdge(
                source=component_id,
                target=node_id,
                score=item.score,
                method="vocabulary_projection_delta",
                metadata={"reference_layer": comparison.reference_layer},
            )
        )
    return builder.build()


def _shared_node_key(node: AttributionNode) -> Optional[Tuple[Any, ...]]:
    if node.kind == "output_logit" and "token_id" in node.metadata:
        return (node.kind, int(node.metadata["token_id"]))
    if node.kind == "input_token" and node.position is not None:
        return (node.kind, node.position, node.label)
    if node.kind == "metric":
        return (node.kind, node.label)
    return None


def fuse_attribution_graphs(
    graphs: Mapping[str, AttributionGraph],
) -> AttributionGraph:
    """Fuse graphs while sharing compatible token/logit/metric target nodes.

    Component-like nodes are namespaced by source. Shared targets use semantic
    keys, preventing unrelated local ids such as ``target`` from colliding.
    """

    if not graphs:
        raise ValueError("At least one attribution graph is required")
    builder = AttributionGraphBuilder(
        metadata={
            "source": "graph_fusion",
            "graphs": list(graphs),
            "schema_versions": {
                name: graph.schema_version for name, graph in graphs.items()
            },
        }
    )
    remapped: Dict[Tuple[str, str], str] = {}
    shared_ids: Dict[Tuple[Any, ...], str] = {}
    for source_name, graph in graphs.items():
        if not source_name:
            raise ValueError("Attribution graph source names cannot be empty")
        for node in graph.nodes:
            shared_key = _shared_node_key(node)
            if shared_key is not None:
                node_id = shared_ids.setdefault(
                    shared_key, f"shared:{len(shared_ids)}"
                )
            else:
                node_id = f"{source_name}:{node.id}"
            remapped[(source_name, node.id)] = node_id
            metadata = dict(node.metadata)
            metadata.setdefault("graph_sources", [])
            sources = list(metadata["graph_sources"])
            if source_name not in sources:
                sources.append(source_name)
            metadata["graph_sources"] = sources
            candidate = AttributionNode(
                id=node_id,
                kind=node.kind,
                label=node.label,
                layer_path=node.layer_path,
                position=node.position,
                metadata=metadata,
            )
            existing = builder.get_node(node_id)
            if existing is not None:
                merged_metadata = dict(existing.metadata)
                merged_sources = list(merged_metadata.get("graph_sources", []))
                for item in sources:
                    if item not in merged_sources:
                        merged_sources.append(item)
                merged_metadata["graph_sources"] = merged_sources
                candidate = AttributionNode(
                    id=existing.id,
                    kind=existing.kind,
                    label=existing.label,
                    layer_path=existing.layer_path,
                    position=existing.position,
                    metadata=merged_metadata,
                )
            if existing is None:
                builder.add_node(candidate)
            else:
                builder.replace_node(candidate)

    for source_name, graph in graphs.items():
        for edge in graph.edges:
            builder.add_edge(
                AttributionEdge(
                    source=remapped[(source_name, edge.source)],
                    target=remapped[(source_name, edge.target)],
                    score=edge.score,
                    method=edge.method,
                    metadata={**edge.metadata, "graph_source": source_name},
                )
            )
    return builder.build()
