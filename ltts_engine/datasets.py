"""Bounded-memory dataset scans over internal activations or SAE features."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch

from .inspection import ComponentInspector
from .runtime import HeadlessSession

FeatureExtractor = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class DatasetExample:
    id: str
    inputs: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TopActivation:
    example_id: str
    value: float
    position: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "value": self.value,
            "position": self.position,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FeatureStatistics:
    feature_id: int
    count: int
    mean: float
    variance: float
    maximum: float
    activation_frequency: float
    top_activations: Tuple[TopActivation, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "maximum": self.maximum,
            "activation_frequency": self.activation_frequency,
            "top_activations": [item.to_dict() for item in self.top_activations],
        }


@dataclass(frozen=True)
class FeatureCorrelation:
    left_feature_id: int
    right_feature_id: int
    correlation: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left_feature_id": self.left_feature_id,
            "right_feature_id": self.right_feature_id,
            "correlation": self.correlation,
        }


@dataclass(frozen=True)
class DatasetScanResult:
    layer: str
    examples_seen: int
    vectors_seen: int
    features: Tuple[FeatureStatistics, ...]
    activation_threshold: float
    correlations: Tuple[FeatureCorrelation, ...] = ()
    schema_version: int = 1

    def feature(self, feature_id: int) -> FeatureStatistics:
        for item in self.features:
            if item.feature_id == feature_id:
                return item
        raise KeyError(feature_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "layer": self.layer,
            "examples_seen": self.examples_seen,
            "vectors_seen": self.vectors_seen,
            "activation_threshold": self.activation_threshold,
            "features": [item.to_dict() for item in self.features],
            "correlations": [item.to_dict() for item in self.correlations],
        }


class ActivationDatasetScanner:
    """Aggregate feature statistics without retaining dataset activations."""

    def __init__(self, session: HeadlessSession) -> None:
        self.session = session

    def scan(
        self,
        examples: Iterable[DatasetExample],
        *,
        layer: str,
        feature_extractor: FeatureExtractor,
        feature_ids: Optional[Sequence[int]] = None,
        top_k: int = 10,
        activation_threshold: float = 0.0,
        max_examples: Optional[int] = None,
        compute_correlations: bool = False,
    ) -> DatasetScanResult:
        """Scan examples and retain only running moments and bounded top-k heaps."""

        if top_k < 1:
            raise ValueError("top_k must be at least one")
        requested = tuple(dict.fromkeys(int(item) for item in feature_ids or ()))
        sums = None
        sums_squared = None
        maxima = None
        active_counts = None
        selected = None
        heaps: Dict[int, list] = {}
        cross_products = None
        examples_seen = 0
        vectors_seen = 0
        inspector = ComponentInspector(self.session)

        for example in examples:
            if max_examples is not None and examples_seen >= max_examples:
                break
            _output, captures = inspector.capture([layer], kwargs=example.inputs)
            features = feature_extractor(captures[layer])
            if not isinstance(features, torch.Tensor) or features.ndim < 2:
                raise TypeError(
                    "feature_extractor must return [..., feature] tensor output"
                )
            flat = features.detach().float().reshape(-1, features.shape[-1]).cpu()
            if selected is None:
                selected = requested or tuple(range(flat.shape[-1]))
                invalid = [item for item in selected if not 0 <= item < flat.shape[-1]]
                if invalid:
                    raise IndexError(
                        f"Feature ids outside width {flat.shape[-1]}: {invalid}"
                    )
                width = len(selected)
                sums = torch.zeros(width, dtype=torch.float64)
                sums_squared = torch.zeros(width, dtype=torch.float64)
                maxima = torch.full((width,), -torch.inf, dtype=torch.float64)
                active_counts = torch.zeros(width, dtype=torch.int64)
                heaps = {feature_id: [] for feature_id in selected}
                if compute_correlations:
                    if width > 512:
                        raise ValueError(
                            "Correlation scans are limited to 512 selected features"
                        )
                    cross_products = torch.zeros(
                        (width, width), dtype=torch.float64
                    )
            elif flat.shape[-1] <= max(selected):
                raise ValueError("Feature width changed during the dataset scan")

            chosen = flat[:, list(selected)].to(torch.float64)
            sums += chosen.sum(dim=0)
            sums_squared += chosen.square().sum(dim=0)
            maxima = torch.maximum(maxima, chosen.max(dim=0).values)
            active_counts += (chosen > activation_threshold).sum(dim=0)
            if cross_products is not None:
                cross_products += chosen.T @ chosen
            vectors_seen += int(chosen.shape[0])

            per_feature_values, per_feature_positions = chosen.max(dim=0)
            for offset, feature_id in enumerate(selected):
                activation = TopActivation(
                    example_id=str(example.id),
                    value=float(per_feature_values[offset]),
                    position=int(per_feature_positions[offset]),
                    metadata=dict(example.metadata),
                )
                heap = heaps[feature_id]
                entry = (activation.value, examples_seen, activation)
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif entry[:2] > heap[0][:2]:
                    heapq.heapreplace(heap, entry)
            examples_seen += 1

        if examples_seen == 0 or selected is None:
            raise ValueError("Dataset scan received no examples")
        assert sums is not None
        assert sums_squared is not None
        assert maxima is not None
        assert active_counts is not None
        mean = sums / vectors_seen
        variance = torch.clamp(sums_squared / vectors_seen - mean.square(), min=0.0)
        statistics = []
        for offset, feature_id in enumerate(selected):
            top = tuple(
                item[2]
                for item in sorted(
                    heaps[feature_id], key=lambda item: item[:2], reverse=True
                )
            )
            statistics.append(
                FeatureStatistics(
                    feature_id=feature_id,
                    count=vectors_seen,
                    mean=float(mean[offset]),
                    variance=float(variance[offset]),
                    maximum=float(maxima[offset]),
                    activation_frequency=float(active_counts[offset]) / vectors_seen,
                    top_activations=top,
                )
            )
        correlations = []
        if cross_products is not None:
            covariance = cross_products / vectors_seen - torch.outer(mean, mean)
            standard_deviation = variance.sqrt()
            denominator = torch.outer(standard_deviation, standard_deviation)
            matrix = torch.where(
                denominator > 0,
                covariance / denominator.clamp_min(1e-12),
                torch.zeros_like(covariance),
            ).clamp(-1.0, 1.0)
            for left in range(len(selected)):
                for right in range(left + 1, len(selected)):
                    correlations.append(
                        FeatureCorrelation(
                            left_feature_id=selected[left],
                            right_feature_id=selected[right],
                            correlation=float(matrix[left, right]),
                        )
                    )
        return DatasetScanResult(
            layer=layer,
            examples_seen=examples_seen,
            vectors_seen=vectors_seen,
            features=tuple(statistics),
            activation_threshold=float(activation_threshold),
            correlations=tuple(correlations),
        )
