"""Compact, connected circuit discovery for attribution graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .graphs import AttributionEdge, AttributionGraph, InterventionTargetRef


CircuitValidator = Callable[[Tuple[InterventionTargetRef, ...]], float]


@dataclass(frozen=True)
class CircuitValidation:
    """Optional intervention measurement kept separate from graph estimates."""

    full_effect: float
    circuit_effect: float
    fidelity: Optional[float]

    def __post_init__(self) -> None:
        for value in (self.full_effect, self.circuit_effect):
            if not math.isfinite(value):
                raise ValueError("Circuit validation effects must be finite")
        if self.fidelity is not None and not math.isfinite(self.fidelity):
            raise ValueError("Circuit validation fidelity must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "full_effect": self.full_effect,
            "circuit_effect": self.circuit_effect,
            "fidelity": self.fidelity,
        }


@dataclass(frozen=True)
class CircuitDiscoveryResult:
    """Versioned, transport-neutral result of connected circuit selection."""

    target_node_id: str
    graph: AttributionGraph
    selected_edge_indices: Tuple[int, ...]
    candidate_edge_count: int
    total_absolute_score: float
    retained_absolute_score: float
    estimated_retained_fraction: float
    intervention_targets: Tuple[InterventionTargetRef, ...]
    validation: Optional[CircuitValidation] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_node_id": self.target_node_id,
            "graph": self.graph.to_dict(),
            "selected_edge_indices": list(self.selected_edge_indices),
            "candidate_edge_count": self.candidate_edge_count,
            "total_absolute_score": self.total_absolute_score,
            "retained_absolute_score": self.retained_absolute_score,
            "estimated_retained_fraction": self.estimated_retained_fraction,
            "intervention_targets": [item.to_dict() for item in self.intervention_targets],
            "validation": self.validation.to_dict() if self.validation else None,
            "metadata": dict(self.metadata),
        }


def discover_circuit(
    graph: AttributionGraph,
    *,
    target_node_id: str,
    max_nodes: int,
    minimum_absolute_score: float = 0.0,
    validator: Optional[CircuitValidator] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> CircuitDiscoveryResult:
    """Select a strong, compact subgraph whose every edge leads to the target.

    Selection expands backwards from the target. An edge is eligible only when
    its target is already connected, so disconnected collections of large edges
    cannot be reported as a circuit. Optional validation receives actionable
    intervention coordinates for the full candidate set and selected circuit.
    """

    graph.node(target_node_id)
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least 1")
    threshold = abs(float(minimum_absolute_score))
    if not math.isfinite(threshold):
        raise ValueError("minimum_absolute_score must be finite")
    for edge in graph.edges:
        if not math.isfinite(edge.score):
            raise ValueError("Attribution edge scores must be finite")

    # First restrict the score denominator to edges that can reach the target.
    reachable = {target_node_id}
    changed = True
    while changed:
        changed = False
        for edge in graph.edges:
            if edge.target in reachable and edge.source not in reachable:
                reachable.add(edge.source)
                changed = True

    candidates = tuple(
        (index, edge)
        for index, edge in enumerate(graph.edges)
        if edge.source in reachable
        and edge.target in reachable
        and abs(edge.score) >= threshold
    )
    total = sum(abs(edge.score) for _, edge in candidates)

    connected = {target_node_id}
    selected = []
    while len(connected) < max_nodes:
        eligible = [
            (index, edge)
            for index, edge in candidates
            if index not in selected
            and edge.target in connected
            and edge.source not in connected
        ]
        if not eligible:
            break
        index, edge = min(
            eligible,
            key=lambda item: (
                -abs(item[1].score),
                item[1].source,
                item[1].target,
                item[0],
            ),
        )
        selected.append(index)
        connected.add(edge.source)

    # Include additional edges between selected nodes without spending node budget.
    selected_set = set(selected)
    for index, edge in candidates:
        if edge.source in connected and edge.target in connected:
            selected_set.add(index)
    selected_indices = tuple(sorted(selected_set))
    selected_edges = tuple(graph.edges[index] for index in selected_indices)
    selected_nodes = tuple(node for node in graph.nodes if node.id in connected)
    retained = sum(abs(edge.score) for edge in selected_edges)
    fraction = retained / total if total else 0.0

    targets = _targets_for_nodes(selected_nodes)
    validation = None
    if validator is not None:
        full_targets = _targets_for_nodes(
            tuple(graph.node(node_id) for node_id in reachable)
        )
        full_effect = float(validator(full_targets))
        circuit_effect = float(validator(targets))
        fidelity = circuit_effect / full_effect if full_effect != 0.0 else None
        validation = CircuitValidation(full_effect, circuit_effect, fidelity)

    circuit_graph = AttributionGraph(
        nodes=selected_nodes,
        edges=selected_edges,
        metadata={
            **graph.metadata,
            "circuit_target": target_node_id,
            "circuit_max_nodes": max_nodes,
            "circuit_minimum_absolute_score": threshold,
        },
        schema_version=graph.schema_version,
    )
    return CircuitDiscoveryResult(
        target_node_id=target_node_id,
        graph=circuit_graph,
        selected_edge_indices=selected_indices,
        candidate_edge_count=len(candidates),
        total_absolute_score=total,
        retained_absolute_score=retained,
        estimated_retained_fraction=fraction,
        intervention_targets=targets,
        validation=validation,
        metadata=dict(metadata or {}),
    )


def _targets_for_nodes(nodes: Sequence[Any]) -> Tuple[InterventionTargetRef, ...]:
    unique = {}
    for node in nodes:
        target = node.intervention_target
        if target is None:
            continue
        key = (
            target.layer_path,
            target.position,
            target.component_kind,
            target.component_index,
            target.feature_id,
        )
        unique.setdefault(key, target)
    return tuple(unique[key] for key in sorted(unique, key=lambda item: repr(item)))
