"""Typed, portable observations and comparisons for research studies."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .research_io import ResearchRunBundle, bundle_result
from .studies import ResearchStudyResult


def _required(value: Any, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} cannot be empty")
    return text


def _optional(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class StudyArtifactRecord:
    """One numeric observation with enough provenance to align later runs."""

    artifact_id: str
    metric: str
    value: float
    module: str
    subtype: str
    layer_path: Optional[str] = None
    component: Optional[str] = None
    token_index: Optional[int] = None
    token: Optional[str] = None
    seed: Optional[int] = None
    example_id: Optional[str] = None
    stage: Optional[str] = None
    run_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _required(self.artifact_id, "Artifact id"))
        object.__setattr__(self, "metric", _required(self.metric, "Artifact metric"))
        object.__setattr__(self, "module", _required(self.module, "Artifact module"))
        object.__setattr__(self, "subtype", _required(self.subtype, "Artifact subtype"))
        object.__setattr__(self, "value", _finite(self.value, "Artifact value"))
        for name in ("layer_path", "component", "token", "example_id", "stage", "run_id"):
            object.__setattr__(self, name, _optional(getattr(self, name)))
        if self.token_index is not None and int(self.token_index) < 0:
            raise ValueError("Artifact token index cannot be negative")
        if self.token_index is not None:
            object.__setattr__(self, "token_index", int(self.token_index))
        if self.seed is not None:
            seed = int(self.seed)
            if seed < 0 or seed > 0xFFFFFFFF:
                raise ValueError("Artifact seed must be a uint32")
            object.__setattr__(self, "seed", seed)
        if int(self.schema_version) != 1:
            raise ValueError(f"Unsupported study artifact schema: {self.schema_version}")
        # Reuse the bundle serializer as the single finite-JSON validation path.
        bundle_result("study_artifact_metadata", dict(self.metadata))

    @property
    def comparison_key(self) -> Tuple[Any, ...]:
        """Stable semantic identity; trial-specific provenance is excluded."""

        return (
            self.metric,
            self.module,
            self.subtype,
            self.layer_path,
            self.component,
            self.token_index,
            self.token,
            self.stage,
        )

    @property
    def trial_key(self) -> Tuple[Optional[str], Optional[int]]:
        return self.example_id, self.seed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "metric": self.metric,
            "value": self.value,
            "module": self.module,
            "subtype": self.subtype,
            "layer_path": self.layer_path,
            "component": self.component,
            "token_index": self.token_index,
            "token": self.token,
            "seed": self.seed,
            "example_id": self.example_id,
            "stage": self.stage,
            "run_id": self.run_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StudyArtifactRecord":
        if not isinstance(value, Mapping):
            raise TypeError("Study artifact must be an object")
        return cls(
            artifact_id=value.get("artifact_id", ""),
            metric=value.get("metric", ""),
            value=value.get("value"),
            module=value.get("module", ""),
            subtype=value.get("subtype", ""),
            layer_path=value.get("layer_path"),
            component=value.get("component"),
            token_index=value.get("token_index"),
            token=value.get("token"),
            seed=value.get("seed"),
            example_id=value.get("example_id"),
            stage=value.get("stage"),
            run_id=value.get("run_id"),
            metadata=value.get("metadata") or {},
            schema_version=value.get("schema_version", 0),
        )


@dataclass(frozen=True)
class StudyArtifactCollection:
    name: str
    records: Tuple[StudyArtifactRecord, ...]
    manifest_fingerprint: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _required(self.name, "Artifact collection name"))
        object.__setattr__(self, "manifest_fingerprint", _optional(self.manifest_fingerprint))
        if int(self.schema_version) != 1:
            raise ValueError(
                f"Unsupported study artifact collection schema: {self.schema_version}"
            )
        identifiers = [record.artifact_id for record in self.records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Study artifact ids must be unique within a collection")
        bundle_result("study_artifact_collection_metadata", dict(self.metadata))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "manifest_fingerprint": self.manifest_fingerprint,
            "metadata": dict(self.metadata),
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StudyArtifactCollection":
        if not isinstance(value, Mapping):
            raise TypeError("Study artifact collection must be an object")
        records = value.get("records")
        if not isinstance(records, list):
            raise ValueError("Study artifact collection records must be a list")
        return cls(
            name=value.get("name", ""),
            records=tuple(StudyArtifactRecord.from_dict(item) for item in records),
            manifest_fingerprint=value.get("manifest_fingerprint"),
            metadata=value.get("metadata") or {},
            schema_version=value.get("schema_version", 0),
        )


@dataclass(frozen=True)
class ResearchStudyArchive:
    """A complete study payload carried inside the standard run bundle."""

    study: Mapping[str, Any]
    artifacts: StudyArtifactCollection
    manifest: Optional[Mapping[str, Any]] = None
    trials: Tuple[Mapping[str, Any], ...] = ()
    graphs: Tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if int(self.schema_version) != 1:
            raise ValueError(f"Unsupported research study archive schema: {self.schema_version}")
        if not isinstance(self.study, Mapping) or not self.study:
            raise ValueError("Research study archive requires a study object")
        if self.manifest is not None and not isinstance(self.manifest, Mapping):
            raise TypeError("Research study archive manifest must be an object")
        for label, items in (("trials", self.trials), ("graphs", self.graphs)):
            if not all(isinstance(item, Mapping) for item in items):
                raise TypeError(f"Research study archive {label} must contain objects")
        bundle_result("research_study_archive_validation", self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "study": dict(self.study),
            "manifest": None if self.manifest is None else dict(self.manifest),
            "trials": [dict(item) for item in self.trials],
            "graphs": [dict(item) for item in self.graphs],
            "artifacts": self.artifacts.to_dict(),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResearchStudyArchive":
        if not isinstance(value, Mapping):
            raise TypeError("Research study archive must be an object")
        trials = value.get("trials")
        graphs = value.get("graphs")
        if not isinstance(trials, list) or not isinstance(graphs, list):
            raise ValueError("Research study archive trials and graphs must be lists")
        return cls(
            study=value.get("study") or {},
            manifest=value.get("manifest"),
            trials=tuple(trials),
            graphs=tuple(graphs),
            artifacts=StudyArtifactCollection.from_dict(value.get("artifacts") or {}),
            metadata=value.get("metadata") or {},
            schema_version=value.get("schema_version", 0),
        )


def archive_research_study(
    study: ResearchStudyResult,
    artifacts: StudyArtifactCollection,
    *,
    manifest: Optional[Any] = None,
    trials: Sequence[Any] = (),
    graphs: Sequence[Any] = (),
    metadata: Optional[Mapping[str, Any]] = None,
) -> ResearchStudyArchive:
    """Normalize a typed study and its supporting results into one archive."""

    if not isinstance(study, ResearchStudyResult):
        raise TypeError("study must be ResearchStudyResult")

    def mapping(value: Any, label: str) -> Mapping[str, Any]:
        converter = getattr(value, "to_dict", None)
        result = converter() if callable(converter) else value
        if not isinstance(result, Mapping):
            raise TypeError(f"{label} must be an object or expose to_dict()")
        return dict(result)

    return ResearchStudyArchive(
        study=study.to_dict(),
        artifacts=artifacts,
        manifest=None if manifest is None else mapping(manifest, "manifest"),
        trials=tuple(mapping(item, "trial") for item in trials),
        graphs=tuple(mapping(item, "graph") for item in graphs),
        metadata=dict(metadata or {}),
    )


def bundle_study_archive(archive: ResearchStudyArchive) -> ResearchRunBundle:
    if not isinstance(archive, ResearchStudyArchive):
        raise TypeError("archive must be ResearchStudyArchive")
    return bundle_result("research_study_archive", archive)


def study_archive_from_bundle(bundle: ResearchRunBundle) -> ResearchStudyArchive:
    if not isinstance(bundle, ResearchRunBundle):
        raise TypeError("bundle must be ResearchRunBundle")
    if bundle.kind != "research_study_archive":
        raise ValueError(f"Expected research_study_archive bundle, got {bundle.kind!r}")
    return ResearchStudyArchive.from_dict(bundle.payload)


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


@dataclass(frozen=True)
class StudyComparisonPoint:
    metric: str
    module: str
    subtype: str
    layer_path: Optional[str]
    component: Optional[str]
    token_index: Optional[int]
    token: Optional[str]
    stage: Optional[str]
    status: str
    baseline_mean: Optional[float]
    baseline_std: Optional[float]
    baseline_count: int
    candidate_mean: Optional[float]
    candidate_std: Optional[float]
    candidate_count: int
    delta: Optional[float]
    matched_trials: int
    paired_delta_mean: Optional[float]
    paired_delta_std: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ResearchStudyComparison:
    baseline_name: str
    candidate_name: str
    points: Tuple[StudyComparisonPoint, ...]
    manifest_match: Optional[bool]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_name": self.baseline_name,
            "candidate_name": self.candidate_name,
            "manifest_match": self.manifest_match,
            "metadata": dict(self.metadata),
            "points": [point.to_dict() for point in self.points],
        }


def compare_study_artifacts(
    baseline: StudyArtifactCollection,
    candidate: StudyArtifactCollection,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ResearchStudyComparison:
    """Compare aligned effects and paired trial stability without a web app."""

    if not isinstance(baseline, StudyArtifactCollection) or not isinstance(
        candidate, StudyArtifactCollection
    ):
        raise TypeError("baseline and candidate must be StudyArtifactCollection")

    def grouped(collection: StudyArtifactCollection) -> Dict[Tuple[Any, ...], list]:
        result: Dict[Tuple[Any, ...], list] = {}
        for record in collection.records:
            result.setdefault(record.comparison_key, []).append(record)
        return result

    baseline_groups = grouped(baseline)
    candidate_groups = grouped(candidate)
    points = []
    for key in sorted(set(baseline_groups).union(candidate_groups), key=repr):
        left = baseline_groups.get(key, [])
        right = candidate_groups.get(key, [])
        left_values = [record.value for record in left]
        right_values = [record.value for record in right]
        left_mean = _mean(left_values) if left_values else None
        right_mean = _mean(right_values) if right_values else None
        if left and right:
            status = "matched"
            delta = right_mean - left_mean
        elif right:
            status = "appeared"
            delta = None
        else:
            status = "disappeared"
            delta = None

        def by_trial(records: Iterable[StudyArtifactRecord]) -> Dict[Tuple[Any, ...], float]:
            grouped_trials: Dict[Tuple[Any, ...], list] = {}
            for record in records:
                if record.example_id is None or record.seed is None:
                    continue
                grouped_trials.setdefault(record.trial_key, []).append(record.value)
            return {trial: _mean(values) for trial, values in grouped_trials.items()}

        left_trials = by_trial(left)
        right_trials = by_trial(right)
        paired = [
            right_trials[trial] - left_trials[trial]
            for trial in sorted(set(left_trials).intersection(right_trials), key=repr)
        ]
        points.append(
            StudyComparisonPoint(
                metric=key[0],
                module=key[1],
                subtype=key[2],
                layer_path=key[3],
                component=key[4],
                token_index=key[5],
                token=key[6],
                stage=key[7],
                status=status,
                baseline_mean=left_mean,
                baseline_std=_std(left_values) if left_values else None,
                baseline_count=len(left_values),
                candidate_mean=right_mean,
                candidate_std=_std(right_values) if right_values else None,
                candidate_count=len(right_values),
                delta=delta,
                matched_trials=len(paired),
                paired_delta_mean=_mean(paired) if paired else None,
                paired_delta_std=_std(paired) if paired else None,
            )
        )

    left_fingerprint = baseline.manifest_fingerprint
    right_fingerprint = candidate.manifest_fingerprint
    manifest_match = (
        None
        if left_fingerprint is None or right_fingerprint is None
        else left_fingerprint == right_fingerprint
    )
    comparison = ResearchStudyComparison(
        baseline_name=baseline.name,
        candidate_name=candidate.name,
        points=tuple(points),
        manifest_match=manifest_match,
        metadata=dict(metadata or {}),
    )
    bundle_result("research_study_comparison_validation", comparison)
    return comparison
