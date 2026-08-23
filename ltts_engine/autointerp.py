"""Provider-neutral feature interpretation and held-out explanation scoring."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, Tuple


@dataclass(frozen=True)
class FeatureEvidence:
    """One human-readable example paired with a measured feature activation."""

    example_id: str
    text: str
    activation: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "text": self.text,
            "activation": self.activation,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class InterpretationDraft:
    """Provider output before independent evaluation."""

    label: str
    explanation: str
    notes: Tuple[str, ...] = ()


class AutointerpProvider(Protocol):
    """Minimal interface implemented by local or hosted explanation providers."""

    def explain(
        self, feature_id: int, evidence: Sequence[FeatureEvidence]
    ) -> InterpretationDraft: ...


EvidencePredictor = Callable[[InterpretationDraft, FeatureEvidence], float]


@dataclass(frozen=True)
class ExplanationScore:
    examples: int
    correlation: float
    mean_absolute_error: float
    top_bottom_separation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "examples": self.examples,
            "correlation": self.correlation,
            "mean_absolute_error": self.mean_absolute_error,
            "top_bottom_separation": self.top_bottom_separation,
        }


@dataclass(frozen=True)
class FeatureInterpretation:
    feature_id: int
    label: str
    explanation: str
    notes: Tuple[str, ...]
    training_examples: Tuple[str, ...]
    held_out_score: ExplanationScore
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_id": self.feature_id,
            "label": self.label,
            "explanation": self.explanation,
            "notes": list(self.notes),
            "training_examples": list(self.training_examples),
            "held_out_score": self.held_out_score.to_dict(),
        }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    count = len(left)
    left_mean = sum(left) / count
    right_mean = sum(right) / count
    numerator = sum(
        (left_item - left_mean) * (right_item - right_mean)
        for left_item, right_item in zip(left, right)
    )
    left_scale = math.sqrt(sum((item - left_mean) ** 2 for item in left))
    right_scale = math.sqrt(sum((item - right_mean) ** 2 for item in right))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return max(-1.0, min(1.0, numerator / (left_scale * right_scale)))


def score_explanation(
    draft: InterpretationDraft,
    held_out: Sequence[FeatureEvidence],
    *,
    predictor: EvidencePredictor,
) -> ExplanationScore:
    """Evaluate an explanation on examples not shown to its provider.

    Predictions and activations are normalized independently to [0, 1] for the
    absolute-error metric. Correlation remains scale invariant. The separation
    metric compares predicted scores in the upper and lower activation halves.
    """

    if len(held_out) < 2:
        raise ValueError("Held-out scoring requires at least two examples")
    actual = [float(item.activation) for item in held_out]
    predicted = [float(predictor(draft, item)) for item in held_out]
    if not all(math.isfinite(item) for item in actual + predicted):
        raise ValueError("Activations and predictions must be finite")

    def normalized(values: Sequence[float]) -> Sequence[float]:
        low, high = min(values), max(values)
        if high == low:
            return [0.0 for _ in values]
        return [(item - low) / (high - low) for item in values]

    actual_normalized = normalized(actual)
    predicted_normalized = normalized(predicted)
    mae = sum(
        abs(left - right)
        for left, right in zip(actual_normalized, predicted_normalized)
    ) / len(actual)
    order = sorted(range(len(actual)), key=actual.__getitem__)
    split = len(order) // 2
    lower = order[:split]
    upper = order[-split:]
    separation = sum(predicted_normalized[index] for index in upper) / len(upper)
    separation -= sum(predicted_normalized[index] for index in lower) / len(lower)
    return ExplanationScore(
        examples=len(held_out),
        correlation=_pearson(actual, predicted),
        mean_absolute_error=float(mae),
        top_bottom_separation=float(separation),
    )


class FeatureAutointerpreter:
    """Generate an explanation from training evidence and score it independently."""

    def __init__(self, provider: AutointerpProvider) -> None:
        self.provider = provider

    def interpret(
        self,
        feature_id: int,
        *,
        training: Sequence[FeatureEvidence],
        held_out: Sequence[FeatureEvidence],
        predictor: EvidencePredictor,
    ) -> FeatureInterpretation:
        if not training:
            raise ValueError("Interpretation requires training evidence")
        training_ids = {item.example_id for item in training}
        overlap = sorted(
            training_ids.intersection(item.example_id for item in held_out)
        )
        if overlap:
            raise ValueError(
                "Training and held-out evidence must be disjoint; overlap: "
                + ", ".join(overlap)
            )
        draft = self.provider.explain(int(feature_id), tuple(training))
        if not isinstance(draft, InterpretationDraft):
            raise TypeError("Autointerp provider must return InterpretationDraft")
        if not draft.label.strip() or not draft.explanation.strip():
            raise ValueError("Interpretation label and explanation cannot be empty")
        score = score_explanation(draft, held_out, predictor=predictor)
        return FeatureInterpretation(
            feature_id=int(feature_id),
            label=draft.label.strip(),
            explanation=draft.explanation.strip(),
            notes=tuple(draft.notes),
            training_examples=tuple(item.example_id for item in training),
            held_out_score=score,
        )
