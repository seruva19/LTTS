"""Structured token/component attribution records and module artifact adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .runtime import Artifact


@dataclass(frozen=True)
class AttributionTarget:
    label: str
    token_id: Optional[int] = None
    position: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "token_id": self.token_id,
            "position": self.position,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AttributionPoint:
    position: int
    score: float
    label: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.position < 0:
            raise ValueError("Attribution position cannot be negative")
        if not math.isfinite(self.score):
            raise ValueError("Attribution score must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "score": self.score,
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TokenAttributionResult:
    method: str
    points: Tuple[AttributionPoint, ...]
    target: AttributionTarget
    layer_path: Optional[str] = None
    score_semantics: str = "signed_effect"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ValueError("Attribution method cannot be empty")
        if not self.points:
            raise ValueError("Token attribution requires at least one point")
        positions = [point.position for point in self.points]
        if len(positions) != len(set(positions)):
            raise ValueError("Token attribution positions must be unique")
        if self.score_semantics not in {
            "signed_effect",
            "absolute_importance",
            "normalized_importance",
        }:
            raise ValueError(f"Unsupported score semantics: {self.score_semantics}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "layer_path": self.layer_path,
            "score_semantics": self.score_semantics,
            "target": self.target.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class ComponentAttributionPoint:
    index: int
    score: float
    label: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("Component attribution index cannot be negative")
        if not math.isfinite(self.score):
            raise ValueError("Component attribution score must be finite")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "score": self.score,
            "label": self.label,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ComponentAttributionResult:
    method: str
    component_kind: str
    points: Tuple[ComponentAttributionPoint, ...]
    target: AttributionTarget
    layer_path: Optional[str] = None
    score_semantics: str = "absolute_importance"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.component_kind not in {
            "attention_head",
            "neuron",
            "sae_feature",
            "layer",
        }:
            raise ValueError(f"Unsupported component kind: {self.component_kind}")
        if not self.points:
            raise ValueError("Component attribution requires at least one point")
        indices = [point.index for point in self.points]
        if len(indices) != len(set(indices)):
            raise ValueError("Component attribution indices must be unique")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "component_kind": self.component_kind,
            "layer_path": self.layer_path,
            "score_semantics": self.score_semantics,
            "target": self.target.to_dict(),
            "points": [point.to_dict() for point in self.points],
            "provenance": dict(self.provenance),
        }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _parallel_points(
    positions: Sequence[Any], scores: Sequence[Any], labels: Optional[Sequence[Any]] = None
) -> Tuple[AttributionPoint, ...]:
    if len(positions) != len(scores) or (labels is not None and len(labels) != len(scores)):
        raise ValueError("Attribution positions, labels, and scores must have equal length")
    return tuple(
        AttributionPoint(
            position=int(position),
            score=float(score),
            label=None if labels is None else str(labels[index]),
        )
        for index, (position, score) in enumerate(zip(positions, scores))
    )


def attribution_from_occlusion_artifact(artifact: Artifact) -> TokenAttributionResult:
    """Parse the structured chart emitted by occlusion_attribution."""

    if artifact.module != "occlusion_attribution" or artifact.subtype != "token_attribution":
        raise ValueError("Artifact is not an occlusion_attribution token result")
    envelope = _mapping(artifact.data, "Occlusion artifact data")
    chart = _mapping(envelope.get("data"), "Occlusion chart data")
    points = _parallel_points(
        chart.get("positions") or (),
        chart.get("attribution") or (),
        chart.get("tokens") or (),
    )
    return TokenAttributionResult(
        method="token_occlusion_probability_drop",
        points=points,
        target=AttributionTarget(
            label=str(envelope.get("baseline_top_token") or "baseline top token"),
            metadata={"baseline_probability": envelope.get("baseline_probability")},
        ),
        layer_path=artifact.layer_path,
        score_semantics="signed_effect",
        provenance={"module": artifact.module, "subtype": artifact.subtype},
    )


def attribution_from_token_contributor_artifact(
    artifact: Artifact,
    *,
    target_label: str = "layer token contribution",
) -> TokenAttributionResult:
    """Parse the token_contributor position/importance chart."""

    if artifact.module != "token_contributor" or artifact.subtype != "chart":
        raise ValueError("Artifact is not a token_contributor chart")
    envelope = _mapping(artifact.data, "Token contributor artifact data")
    chart = _mapping(envelope.get("data"), "Token contributor chart data")
    scores = chart.get("importance")
    if not isinstance(scores, (list, tuple)):
        raise ValueError("Token contributor chart is missing importance values")
    positions = chart.get("positions") or range(len(scores))
    return TokenAttributionResult(
        method="token_contributor",
        points=_parallel_points(tuple(positions), scores),
        target=AttributionTarget(label=target_label),
        layer_path=artifact.layer_path,
        score_semantics="absolute_importance",
        provenance={"module": artifact.module, "subtype": artifact.subtype},
    )


def attribution_from_gradient_artifact(
    artifact: Artifact,
    *,
    token_labels: Optional[Sequence[str]] = None,
) -> TokenAttributionResult:
    """Parse gradient_importance token heatmaps into one score per position."""

    if artifact.module != "gradient_importance":
        raise ValueError("Artifact is not from gradient_importance")
    data = _mapping(artifact.data, "Gradient artifact data")
    if data.get("type") != "heatmap":
        raise ValueError("Gradient token attribution requires a heatmap artifact")
    rows = data.get("values")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Gradient heatmap has no values")
    numeric_rows = tuple(tuple(float(item) for item in row) for row in rows)
    width = len(numeric_rows[0])
    if width == 0 or any(len(row) != width for row in numeric_rows):
        raise ValueError("Gradient heatmap rows must have equal non-zero width")
    scores = tuple(
        sum(abs(row[position]) for row in numeric_rows) / len(numeric_rows)
        for position in range(width)
    )
    labels = tuple(token_labels) if token_labels is not None else None
    target_data = data.get("target")
    target = dict(target_data) if isinstance(target_data, Mapping) else {}
    token_id = target.get("token_id")
    method = str(data.get("method") or "gradient")
    return TokenAttributionResult(
        method=method,
        points=_parallel_points(range(width), scores, labels),
        target=AttributionTarget(
            label=str(target.get("label") or target.get("token") or "model target"),
            token_id=int(token_id) if token_id is not None else None,
            position=(
                int(target["position"]) if target.get("position") is not None else None
            ),
            metadata={key: value for key, value in target.items() if key not in {"label", "token", "token_id", "position"}},
        ),
        layer_path=artifact.layer_path,
        score_semantics="absolute_importance",
        provenance={
            "module": artifact.module,
            "subtype": artifact.subtype,
            "quantity": data.get("quantity"),
        },
    )


def component_attribution_from_gradient_artifact(
    artifact: Artifact,
) -> ComponentAttributionResult:
    """Parse the neuron-importance table emitted by gradient_importance."""

    if artifact.module != "gradient_importance":
        raise ValueError("Artifact is not from gradient_importance")
    data = _mapping(artifact.data, "Gradient artifact data")
    if data.get("type") != "table":
        raise ValueError("Gradient component attribution requires a table artifact")
    headers = tuple(str(item).strip().lower() for item in data.get("headers") or ())
    try:
        neuron_column = headers.index("neuron")
        score_column = headers.index("absolute attribution")
    except ValueError as exc:
        raise ValueError("Gradient table is not a neuron attribution table") from exc
    rows = data.get("rows")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Gradient neuron table has no rows")
    points = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) != len(headers):
            raise ValueError(f"Gradient table row {row_index} has the wrong width")
        index = int(row[neuron_column])
        points.append(
            ComponentAttributionPoint(
                index=index,
                score=float(row[score_column]),
                label=f"neuron {index}",
                metadata={"artifact_row": row_index},
            )
        )
    target_data = data.get("target")
    target = dict(target_data) if isinstance(target_data, Mapping) else {}
    token_id = target.get("token_id")
    return ComponentAttributionResult(
        method=str(data.get("method") or "gradient"),
        component_kind="neuron",
        points=tuple(points),
        target=AttributionTarget(
            label=str(target.get("label") or target.get("token") or "model target"),
            token_id=int(token_id) if token_id is not None else None,
            position=(
                int(target["position"]) if target.get("position") is not None else None
            ),
        ),
        layer_path=artifact.layer_path,
        score_semantics="absolute_importance",
        provenance={"module": artifact.module, "subtype": artifact.subtype},
    )
