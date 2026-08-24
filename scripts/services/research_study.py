from __future__ import annotations

import asyncio
import math
import statistics
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from ltts_engine.study_artifacts import StudyArtifactCollection, StudyArtifactRecord
from scripts.services.inference.contracts import InferenceInput
from scripts.services.inference_service import InferenceService


_SCORE_KEYS = ("score", "effect", "delta", "attribution", "value")


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _artifact_metric_score(message: Mapping[str, Any]) -> tuple[str, float] | None:
    payload = message.get("data")
    if isinstance(payload, Mapping) and isinstance(payload.get("visualization"), Mapping):
        payload = payload["visualization"]
    if not isinstance(payload, Mapping):
        return None
    for key in _SCORE_KEYS:
        score = _finite_number(payload.get(key))
        if score is not None:
            return key, score
    stats = payload.get("stats")
    if isinstance(stats, Mapping):
        score = _finite_number(stats.get("mean"))
        if score is not None:
            return "mean", score
    return None


def _artifact_score(message: Mapping[str, Any]) -> float | None:
    metric_score = _artifact_metric_score(message)
    return None if metric_score is None else metric_score[1]


def _artifact_layer(message: Mapping[str, Any]) -> str | None:
    for key in ("layer_path", "module_path", "layer_name"):
        value = message.get(key)
        if value:
            return str(value)
    payload = message.get("data")
    if isinstance(payload, Mapping):
        layer_info = payload.get("layer_info")
        if isinstance(layer_info, Mapping) and layer_info.get("path"):
            return str(layer_info["path"])
    return None


@dataclass(frozen=True)
class CapturedTrial:
    example_id: str
    seed: int
    response: Mapping[str, Any]
    artifacts: Sequence[Mapping[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "example_id": self.example_id,
            "seed": self.seed,
            "status": self.response.get("status"),
            "result": self.response.get("result"),
            "run_export": self.response.get("run_export"),
            "artifacts": [dict(item) for item in self.artifacts],
        }


class ResearchStudyService:
    """Run reproducible inference trials and aggregate real module artifacts."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.inference = InferenceService(context)

    async def _run_trial(
        self,
        *,
        model_id: str,
        example_id: str,
        prompt: str,
        seed: int,
        parameters: Mapping[str, Any],
        modules: Mapping[str, Any] | None,
        parent_token: str,
    ) -> CapturedTrial:
        token = f"{parent_token}:study:{uuid.uuid4().hex}"
        messages: List[Mapping[str, Any]] = []
        completed = asyncio.Event()

        async def collect(message: Any) -> None:
            if not isinstance(message, Mapping):
                return
            messages.append(dict(message))
            if message.get("type") in {
                "inference_complete",
                "inference_cancelled",
                "inference_error",
            }:
                completed.set()

        self.context.message_bus.subscribe(token, collect)
        try:
            response = await self.inference.run_inference(
                InferenceInput(
                    model_id=model_id,
                    prompt=prompt,
                    modules=dict(modules) if modules else None,
                    client_token=token,
                    parameters={**dict(parameters), "seed": int(seed)},
                )
            )
            if self.context.message_bus.running:
                try:
                    await asyncio.wait_for(completed.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    # The inference response remains authoritative. A delayed
                    # message must not turn a successful trial into a failure.
                    pass
        finally:
            self.context.message_bus.unsubscribe(token, collect)

        artifacts = tuple(message for message in messages if message.get("type") == "visualization")
        return CapturedTrial(
            example_id=str(example_id),
            seed=int(seed),
            response=response,
            artifacts=artifacts,
        )

    @staticmethod
    def _aggregate(trials: Iterable[CapturedTrial]) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for trial in trials:
            for artifact in trial.artifacts:
                layer = _artifact_layer(artifact)
                score = _artifact_score(artifact)
                if not layer or score is None:
                    continue
                grouped.setdefault(layer, []).append(
                    {
                        "score": score,
                        "seed": trial.seed,
                        "example_id": trial.example_id,
                        "module": artifact.get("module"),
                        "artifact_type": artifact.get("subtype") or artifact.get("type"),
                    }
                )

        rows = []
        for layer, observations in grouped.items():
            scores = [item["score"] for item in observations]
            rows.append(
                {
                    "layer": layer,
                    "mean_effect": statistics.fmean(scores),
                    "std_effect": statistics.pstdev(scores) if len(scores) > 1 else 0.0,
                    "min_effect": min(scores),
                    "max_effect": max(scores),
                    "observations": len(scores),
                    "provenance": observations,
                }
            )
        rows.sort(key=lambda row: abs(row["mean_effect"]), reverse=True)
        return rows

    @staticmethod
    def _artifact_collection(
        name: str,
        study_id: str,
        trials: Sequence[CapturedTrial],
        manifest: Mapping[str, Any],
    ) -> StudyArtifactCollection:
        records = []
        for trial_index, trial in enumerate(trials):
            for artifact_index, artifact in enumerate(trial.artifacts):
                layer = _artifact_layer(artifact)
                metric_score = _artifact_metric_score(artifact)
                if not layer or metric_score is None:
                    continue
                metric, score = metric_score
                payload = artifact.get("data")
                details = payload if isinstance(payload, Mapping) else {}
                records.append(
                    StudyArtifactRecord(
                        artifact_id=f"{study_id}:{trial_index}:{artifact_index}",
                        metric=metric,
                        value=score,
                        module=str(artifact.get("module") or "unknown"),
                        subtype=str(artifact.get("subtype") or "generic"),
                        layer_path=layer,
                        component=details.get("component"),
                        token_index=details.get("token_index"),
                        token=details.get("token"),
                        seed=trial.seed,
                        example_id=trial.example_id,
                        stage="scan",
                        run_id=study_id,
                        metadata={
                            "source": "web_research_study",
                            "artifact_type": artifact.get("subtype")
                            or artifact.get("type"),
                        },
                    )
                )
        return StudyArtifactCollection(
            name=name,
            records=tuple(records),
            manifest_fingerprint=manifest.get("fingerprint"),
            metadata={"study_id": study_id},
        )

    async def run(
        self,
        *,
        name: str,
        model_id: str,
        manifest: Mapping[str, Any],
        examples: Sequence[Mapping[str, str]],
        seeds: Sequence[int],
        parameters: Mapping[str, Any],
        modules: Mapping[str, Any] | None,
        client_token: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        study_id = str(uuid.uuid4())
        trials: List[CapturedTrial] = []
        for example in examples:
            for seed in seeds:
                trials.append(
                    await self._run_trial(
                        model_id=model_id,
                        example_id=str(example["id"]),
                        prompt=str(example["prompt"]),
                        seed=int(seed),
                        parameters=parameters,
                        modules=modules,
                        parent_token=client_token,
                    )
                )

        candidates = self._aggregate(trials)
        artifacts = self._artifact_collection(name, study_id, trials, manifest)
        result = {
            "schema_version": 1,
            "study_id": study_id,
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_id": model_id,
            "manifest": dict(manifest),
            "seeds": [int(seed) for seed in seeds],
            "metadata": dict(metadata or {}),
            "stages": [
                {
                    "name": "dataset",
                    "kind": "dataset_manifest",
                    "result": {
                        **dict(manifest),
                        "example_ids": [str(example["id"]) for example in examples],
                    },
                },
                {
                    "name": "scan",
                    "kind": "artifact_scan",
                    "depends_on": ["dataset"],
                    "result": {"candidates": candidates},
                },
            ],
            "trials": [trial.to_dict() for trial in trials],
            "artifacts": artifacts.to_dict(),
            "candidates": candidates,
        }

        await self.context.message_bus.publish(
            client_token,
            {
                "type": "visualization",
                "module": "research_study",
                "subtype": "table",
                "data": {
                    "type": "table",
                    "headers": ["Layer", "Mean effect", "Std", "N"],
                    "rows": [
                        [
                            row["layer"],
                            round(row["mean_effect"], 6),
                            round(row["std_effect"], 6),
                            row["observations"],
                        ]
                        for row in candidates
                    ],
                    "study_id": result["study_id"],
                },
                "timestamp": result["created_at"],
            },
        )
        return result
