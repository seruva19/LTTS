"""Composable, UI-independent experiment execution for LTTS.

The runner deliberately reuses one :class:`HeadlessSession`. Re-wrapping the
same PyTorch model for every trial would register another set of forward hooks;
resetting module attachments keeps trials isolated without accumulating hooks.
"""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

import torch

from .runtime import Artifact, HeadlessSession

ScoreFunction = Callable[[Any], float]


@dataclass(frozen=True)
class ModuleAttachment:
    """One module binding used by an experiment trial."""

    module: ModuleType
    layer: Optional[str] = None
    config: Mapping[str, Any] = field(default_factory=dict)
    event_type: Optional[str] = None
    schedule: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentCase:
    """A named baseline, control, or intervention trial."""

    name: str
    attachments: Tuple[ModuleAttachment, ...] = ()
    kind: str = "intervention"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialResult:
    """Compact result of one trial; model tensors are not retained."""

    name: str
    kind: str
    score: float
    delta_from_baseline: float
    duration_seconds: float
    artifacts: Tuple[Artifact, ...]
    metadata: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "score": self.score,
            "delta_from_baseline": self.delta_from_baseline,
            "duration_seconds": self.duration_seconds,
            "artifacts": [
                {
                    "module": artifact.module,
                    "subtype": artifact.subtype,
                    "data": artifact.data,
                    "layer_path": artifact.layer_path,
                }
                for artifact in self.artifacts
            ],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Versioned collection of comparable experiment trials."""

    baseline: TrialResult
    trials: Tuple[TrialResult, ...]
    seed: int
    schema_version: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def by_name(self, name: str) -> TrialResult:
        for trial in (self.baseline, *self.trials):
            if trial.name == name:
                return trial
        raise KeyError(name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "metadata": dict(self.metadata),
            "baseline": self.baseline.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
        }


@dataclass(frozen=True)
class CausalSweepPoint:
    layer: str
    score: float
    delta_from_baseline: float
    artifacts: Tuple[Artifact, ...]


@dataclass(frozen=True)
class CausalSweepResult:
    """Layer-wise causal effect measurements for one intervention."""

    baseline_score: float
    points: Tuple[CausalSweepPoint, ...]
    seed: int
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "baseline_score": self.baseline_score,
            "points": [
                {
                    "layer": point.layer,
                    "score": point.score,
                    "delta_from_baseline": point.delta_from_baseline,
                    "artifacts": [
                        {
                            "module": artifact.module,
                            "subtype": artifact.subtype,
                            "data": artifact.data,
                            "layer_path": artifact.layer_path,
                        }
                        for artifact in point.artifacts
                    ],
                }
                for point in self.points
            ],
        }


@contextmanager
def _reproducible_rng(seed: int, devices: Sequence[int]) -> Iterator[None]:
    """Use the same RNG state per trial and restore the caller's state."""

    python_state = random.getstate()
    cuda_devices = list(dict.fromkeys(int(device) for device in devices))
    with torch.random.fork_rng(devices=cuda_devices):
        random.seed(seed)
        torch.manual_seed(seed)
        for device in cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)


class ExperimentRunner:
    """Execute isolated, comparable trials on one instrumented model."""

    def __init__(self, session: HeadlessSession) -> None:
        self.session = session

    def _cuda_devices(self) -> Tuple[int, ...]:
        devices = set()
        for parameter in self.session.model.original_model.parameters():
            if parameter.device.type == "cuda":
                devices.add(
                    parameter.device.index
                    if parameter.device.index is not None
                    else torch.cuda.current_device()
                )
        return tuple(sorted(devices))

    def run(
        self,
        *,
        scorer: ScoreFunction,
        cases: Sequence[ExperimentCase],
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        baseline: Optional[ExperimentCase] = None,
        seed: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ExperimentResult:
        """Run a baseline followed by controls/interventions with equal RNG."""

        baseline_case = baseline or ExperimentCase(name="baseline", kind="baseline")
        names = [baseline_case.name, *(case.name for case in cases)]
        if len(names) != len(set(names)):
            raise ValueError("Experiment case names must be unique")

        baseline_result = self._run_case(
            baseline_case, scorer, args, kwargs or {}, seed, baseline_score=None
        )
        trials = tuple(
            self._run_case(
                case,
                scorer,
                args,
                kwargs or {},
                seed,
                baseline_score=baseline_result.score,
            )
            for case in cases
        )
        self.session.reset_modules()
        return ExperimentResult(
            baseline=baseline_result,
            trials=trials,
            seed=seed,
            metadata=dict(metadata or {}),
        )

    def causal_sweep(
        self,
        *,
        module: ModuleType,
        layers: Sequence[str],
        scorer: ScoreFunction,
        config: Optional[Mapping[str, Any]] = None,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        seed: int = 0,
    ) -> CausalSweepResult:
        """Attach the same intervention at each layer and measure its effect."""

        if not layers:
            raise ValueError("Causal sweep requires at least one layer")
        cases = tuple(
            ExperimentCase(
                name=f"layer:{layer}",
                attachments=(
                    ModuleAttachment(module=module, layer=layer, config=config or {}),
                ),
                metadata={"layer": layer},
            )
            for layer in layers
        )
        result = self.run(
            scorer=scorer,
            cases=cases,
            args=args,
            kwargs=kwargs,
            seed=seed,
            metadata={"kind": "causal_layer_sweep"},
        )
        return CausalSweepResult(
            baseline_score=result.baseline.score,
            points=tuple(
                CausalSweepPoint(
                    layer=str(trial.metadata["layer"]),
                    score=trial.score,
                    delta_from_baseline=trial.delta_from_baseline,
                    artifacts=trial.artifacts,
                )
                for trial in result.trials
            ),
            seed=seed,
        )

    def _run_case(
        self,
        case: ExperimentCase,
        scorer: ScoreFunction,
        args: Sequence[Any],
        kwargs: Mapping[str, Any],
        seed: int,
        baseline_score: Optional[float],
    ) -> TrialResult:
        self.session.reset_modules()
        try:
            for attachment in case.attachments:
                self.session.attach(
                    attachment.module,
                    layer=attachment.layer,
                    config=dict(attachment.config),
                    event_type=attachment.event_type,
                    schedule=dict(attachment.schedule),
                )
            started = time.perf_counter()
            with _reproducible_rng(seed, self._cuda_devices()):
                output = self.session.run(*tuple(args), **dict(kwargs))
            duration = time.perf_counter() - started
            score = float(scorer(output))
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError(f"Trial {case.name!r} produced a non-finite score")
            return TrialResult(
                name=case.name,
                kind=case.kind,
                score=score,
                delta_from_baseline=(
                    0.0 if baseline_score is None else score - baseline_score
                ),
                duration_seconds=duration,
                artifacts=tuple(self.session.artifacts),
                metadata=dict(case.metadata),
            )
        finally:
            self.session.reset_modules()
