from __future__ import annotations

import copy
import math
import statistics
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Sequence

from fastapi import HTTPException

from scripts.services.research_study import (
    CapturedTrial,
    ResearchStudyService,
    _artifact_metric_score,
)


def _module_name(spec: Any) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        return str(spec.get("name") or "")
    return ""


def _without_module(
    modules: Mapping[str, Sequence[Any]] | None, module_name: str
) -> Dict[str, list[Any]]:
    result: Dict[str, list[Any]] = {}
    for path, specs in (modules or {}).items():
        retained = [copy.deepcopy(spec) for spec in specs if _module_name(spec) != module_name]
        if retained:
            result[str(path)] = retained
    return result


def _with_intervention(
    baseline_modules: Mapping[str, Sequence[Any]],
    layer: str,
    intervention: Mapping[str, Any],
) -> Dict[str, list[Any]]:
    result = {path: [copy.deepcopy(spec) for spec in specs] for path, specs in baseline_modules.items()}
    spec = copy.deepcopy(dict(intervention))
    parameters = dict(spec.get("parameters") or {})
    parameters["layer_path"] = layer
    spec["parameters"] = parameters
    result.setdefault(layer, []).append(spec)
    return result


def _trial_metric(
    trial: CapturedTrial, *, metric: str, measurement_module: str
) -> float:
    values = []
    for artifact in trial.artifacts:
        if str(artifact.get("module") or "") != measurement_module:
            continue
        metric_score = _artifact_metric_score(artifact)
        if metric_score is None or metric_score[0] != metric:
            continue
        values.append(metric_score[1])
    if not values:
        raise HTTPException(
            status_code=422,
            detail=(
                f'Measurement module "{measurement_module}" did not emit a finite '
                f'"{metric}" metric. Run the attached analysis once and choose a '
                "numeric metric it emits."
            ),
        )
    return statistics.fmean(values)


class CausalScanService:
    """Execute paired, seeded intervention sweeps through the production runtime."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.study = ResearchStudyService(context)

    async def run(
        self,
        *,
        model_id: str,
        prompt: str,
        layers: Sequence[str],
        seeds: Sequence[int],
        parameters: Mapping[str, Any],
        modules: Mapping[str, Sequence[Any]] | None,
        intervention: Mapping[str, Any],
        measurement_module: str,
        metric: str,
        client_token: str,
    ) -> Dict[str, Any]:
        intervention_name = _module_name(intervention)
        if not intervention_name:
            raise HTTPException(status_code=422, detail="Intervention module name is required")
        if intervention_name == measurement_module:
            raise HTTPException(
                status_code=422,
                detail="Intervention and measurement modules must be different",
            )

        baseline_modules = _without_module(modules, intervention_name)
        scan_id = str(uuid.uuid4())
        baseline_trials = []
        trial_summaries = []
        baseline_by_seed: Dict[int, float] = {}
        for seed in seeds:
            trial = await self.study._run_trial(
                model_id=model_id,
                example_id="current-prompt",
                prompt=prompt,
                seed=int(seed),
                parameters=parameters,
                modules=baseline_modules,
                parent_token=client_token,
            )
            baseline_trials.append(trial)
            trial_summaries.append(
                {"kind": "baseline", "layer": None, "seed": int(seed), "status": trial.response.get("status")}
            )
            baseline_by_seed[int(seed)] = _trial_metric(
                trial, metric=metric, measurement_module=measurement_module
            )

        points = []
        for layer in layers:
            scores = []
            deltas = []
            for seed in seeds:
                trial = await self.study._run_trial(
                    model_id=model_id,
                    example_id="current-prompt",
                    prompt=prompt,
                    seed=int(seed),
                    parameters=parameters,
                    modules=_with_intervention(baseline_modules, layer, intervention),
                    parent_token=client_token,
                )
                trial_summaries.append(
                    {
                        "kind": "intervention",
                        "layer": layer,
                        "seed": int(seed),
                        "status": trial.response.get("status"),
                    }
                )
                score = _trial_metric(
                    trial, metric=metric, measurement_module=measurement_module
                )
                scores.append(score)
                deltas.append(score - baseline_by_seed[int(seed)])
            points.append(
                {
                    "layer": layer,
                    "score": statistics.fmean(scores),
                    "delta_from_baseline": statistics.fmean(deltas),
                    "delta_std": statistics.pstdev(deltas) if len(deltas) > 1 else 0.0,
                    "observations": len(deltas),
                    "seed_deltas": [
                        {"seed": int(seed), "delta": delta}
                        for seed, delta in zip(seeds, deltas)
                    ],
                }
            )

        created_at = datetime.now(timezone.utc).isoformat()
        result = {
            "schema_version": 1,
            "scan_id": scan_id,
            "created_at": created_at,
            "model_id": model_id,
            "metric": metric,
            "measurement_module": measurement_module,
            "intervention_module": intervention_name,
            "seeds": [int(seed) for seed in seeds],
            "baseline": {
                "mean": statistics.fmean(baseline_by_seed.values()),
                "std": statistics.pstdev(baseline_by_seed.values())
                if len(baseline_by_seed) > 1
                else 0.0,
                "scores": [
                    {"seed": seed, "score": score}
                    for seed, score in baseline_by_seed.items()
                ],
            },
            "points": points,
            "trials": trial_summaries,
        }
        values = [
            [next(item["delta"] for item in point["seed_deltas"] if item["seed"] == seed) for seed in seeds]
            for point in points
        ]
        if not all(math.isfinite(value) for row in values for value in row):
            raise HTTPException(status_code=422, detail="Causal scan produced a non-finite effect")
        await self.context.message_bus.publish(
            client_token,
            {
                "type": "visualization",
                "module": "causal_scan",
                "subtype": "heatmap",
                "data": {
                    "type": "heatmap",
                    "values": values,
                    "dimensions": {"rows": len(points), "cols": len(seeds)},
                    "row_labels": [point["layer"] for point in points],
                    "column_labels": [f"seed {seed}" for seed in seeds],
                    "color_scale": "diverging_zero",
                    "metric": metric,
                    "scan_id": scan_id,
                },
                "timestamp": created_at,
            },
        )
        return result
