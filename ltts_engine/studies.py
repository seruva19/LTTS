"""Composable, serializable DAGs of headless research stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .autointerp import (
    AutointerpProvider,
    EvidencePredictor,
    FeatureAutointerpreter,
    FeatureEvidence,
    partition_feature_evidence,
)
from .datasets import (
    ActivationDatasetScanner,
    DatasetExample,
    DatasetManifest,
    FeatureExtractor,
    examples_for_split,
)
from .experiments import ExperimentRunner
from .graphs import graph_from_causal_sweep
from .runtime import HeadlessSession


@dataclass(frozen=True)
class ResearchStudyStage:
    name: str
    kind: str
    result: Mapping[str, Any]
    depends_on: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.kind.strip():
            raise ValueError("Study stage name and kind are required")
        if self.name in self.depends_on:
            raise ValueError("Study stage cannot depend on itself")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "metadata": dict(self.metadata),
            "result": dict(self.result),
        }


@dataclass(frozen=True)
class ResearchStudyResult:
    name: str
    stages: Tuple[ResearchStudyStage, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Research study name is required")
        if not self.stages:
            raise ValueError("Research study requires at least one stage")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("Research study stage names must be unique")
        known = set(names)
        for stage in self.stages:
            missing = sorted(set(stage.depends_on).difference(known))
            if missing:
                raise ValueError(
                    f"Study stage {stage.name!r} has unknown dependencies: "
                    + ", ".join(missing)
                )
        self._validate_acyclic()

    def _validate_acyclic(self) -> None:
        dependencies = {stage.name: stage.depends_on for stage in self.stages}
        visiting = set()
        visited = set()

        def visit(name: str) -> None:
            if name in visiting:
                raise ValueError(f"Research study contains a dependency cycle at {name!r}")
            if name in visited:
                return
            visiting.add(name)
            for dependency in dependencies[name]:
                visit(dependency)
            visiting.remove(name)
            visited.add(name)

        for stage_name in dependencies:
            visit(stage_name)

    def stage(self, name: str) -> ResearchStudyStage:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "metadata": dict(self.metadata),
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _result_mapping(result: Any) -> Mapping[str, Any]:
    converter = getattr(result, "to_dict", None)
    value = converter() if callable(converter) else result
    if not isinstance(value, Mapping):
        raise TypeError("Study stage result must be a mapping or expose to_dict()")
    return dict(value)


class ResearchStudyBuilder:
    """Build a validated study in execution order without retaining tensors."""

    def __init__(
        self, name: str, *, metadata: Optional[Mapping[str, Any]] = None
    ) -> None:
        self.name = str(name)
        self.metadata = dict(metadata or {})
        self._stages = []

    def add_stage(
        self,
        name: str,
        kind: str,
        result: Any,
        *,
        depends_on: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ResearchStudyBuilder":
        stage_name = str(name)
        if any(stage.name == stage_name for stage in self._stages):
            raise ValueError(f"Duplicate research study stage: {stage_name}")
        known = {stage.name for stage in self._stages}
        missing = sorted(set(depends_on).difference(known))
        if missing:
            raise ValueError(
                "Study stages must be added after their dependencies; missing: "
                + ", ".join(missing)
            )
        self._stages.append(
            ResearchStudyStage(
                name=stage_name,
                kind=str(kind),
                result=_result_mapping(result),
                depends_on=tuple(str(item) for item in depends_on),
                metadata=dict(metadata or {}),
            )
        )
        return self

    def build(self) -> ResearchStudyResult:
        return ResearchStudyResult(
            name=self.name,
            stages=tuple(self._stages),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class ResearchStudyPlanStep:
    name: str
    kind: str
    run: Callable[[Mapping[str, Mapping[str, Any]]], Any]
    depends_on: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ResearchStudyRunner:
    """Execute a dependency plan and retain only serializable stage results."""

    def run(
        self,
        name: str,
        steps: Sequence[ResearchStudyPlanStep],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ResearchStudyResult:
        if not steps:
            raise ValueError("Research study plan requires at least one step")
        step_by_name = {step.name: step for step in steps}
        if len(step_by_name) != len(steps):
            raise ValueError("Research study plan step names must be unique")
        for step in steps:
            if not step.name.strip() or not step.kind.strip() or not callable(step.run):
                raise ValueError("Study plan steps require name, kind, and callable run")
            missing = sorted(set(step.depends_on).difference(step_by_name))
            if missing:
                raise ValueError(
                    f"Study plan step {step.name!r} has unknown dependencies: "
                    + ", ".join(missing)
                )

        builder = ResearchStudyBuilder(name, metadata=metadata)
        results: Dict[str, Mapping[str, Any]] = {}
        pending = dict(step_by_name)
        while pending:
            ready = [
                step
                for step in steps
                if step.name in pending
                and all(dependency in results for dependency in step.depends_on)
            ]
            if not ready:
                raise ValueError("Research study plan contains a dependency cycle")
            for step in ready:
                dependency_results = {
                    dependency: results[dependency] for dependency in step.depends_on
                }
                result = _result_mapping(step.run(dependency_results))
                builder.add_stage(
                    step.name,
                    step.kind,
                    result,
                    depends_on=step.depends_on,
                    metadata=step.metadata,
                )
                results[step.name] = result
                del pending[step.name]
        return builder.build()


class CausalStudyRunner:
    """Execute a layer intervention sweep and preserve its attribution graph."""

    def __init__(self, session: HeadlessSession) -> None:
        self.session = session

    def run(
        self,
        name: str,
        *,
        module: Any,
        layers: Sequence[str],
        scorer: Callable[[Any], float],
        target_label: str,
        config: Optional[Mapping[str, Any]] = None,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        seed: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ResearchStudyResult:
        sweep = ExperimentRunner(self.session).causal_sweep(
            module=module,
            layers=layers,
            scorer=scorer,
            config=config,
            args=args,
            kwargs=kwargs,
            seed=seed,
        )
        graph = graph_from_causal_sweep(sweep, target_label=target_label)
        return (
            ResearchStudyBuilder(name, metadata=metadata)
            .add_stage(
                "causal_sweep",
                "causal_sweep",
                sweep,
                metadata={"seed": seed, "target_label": target_label},
            )
            .add_stage(
                "attribution_graph",
                "attribution_graph",
                graph,
                depends_on=("causal_sweep",),
            )
            .build()
        )


class FeatureStudyRunner:
    """Run a manifest-bound feature scan and independently scored explanation."""

    def __init__(self, session: HeadlessSession) -> None:
        self.session = session

    def run(
        self,
        name: str,
        *,
        examples: Sequence[DatasetExample],
        manifest: DatasetManifest,
        layer: str,
        feature_extractor: FeatureExtractor,
        feature_id: int,
        evidence: Sequence[FeatureEvidence],
        provider: AutointerpProvider,
        predictor: EvidencePredictor,
        top_k: int = 10,
        activation_threshold: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ResearchStudyResult:
        example_ids = [str(example.id) for example in examples]
        if len(example_ids) != len(set(example_ids)):
            raise ValueError("Feature study examples contain duplicate ids")
        expected_ids = set(manifest.ids())
        actual_ids = set(example_ids)
        if actual_ids != expected_ids:
            missing = sorted(expected_ids.difference(actual_ids))
            extra = sorted(actual_ids.difference(expected_ids))
            raise ValueError(
                "Feature study examples do not match manifest "
                f"(missing={missing[:10]}, extra={extra[:10]})"
            )
        training_examples = tuple(examples_for_split(examples, manifest, "train"))
        scan = ActivationDatasetScanner(self.session).scan(
            training_examples,
            layer=layer,
            feature_extractor=feature_extractor,
            feature_ids=(int(feature_id),),
            top_k=top_k,
            activation_threshold=activation_threshold,
        )
        training_evidence, held_out_evidence = partition_feature_evidence(
            evidence, manifest
        )
        interpretation = FeatureAutointerpreter(provider).interpret(
            int(feature_id),
            training=training_evidence,
            held_out=held_out_evidence,
            predictor=predictor,
        )
        return (
            ResearchStudyBuilder(name, metadata=metadata)
            .add_stage("dataset", "dataset_manifest", manifest)
            .add_stage(
                "feature_scan",
                "feature_scan",
                scan,
                depends_on=("dataset",),
                metadata={"split": "train", "feature_id": int(feature_id)},
            )
            .add_stage(
                "feature_interpretation",
                "feature_interpretation",
                interpretation,
                depends_on=("dataset", "feature_scan"),
                metadata={"held_out_split": "test"},
            )
            .build()
        )
