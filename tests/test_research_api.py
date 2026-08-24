from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import ltts
from ltts_engine import (
    ResearchStudyBuilder,
    StudyArtifactCollection,
    StudyArtifactRecord,
    archive_research_study,
)
from scripts.services.research_study import CapturedTrial, ResearchStudyService
from scripts.services.causal_scan import CausalScanService
from scripts.messages.message_bus import MessageBus


def test_research_route_is_registered():
    routes = {
        (route.path, method)
        for route in ltts.app.routes
        if hasattr(route, "path")
        for method in getattr(route, "methods", set())
    }
    for path in (
        "/api/research/studies/run",
        "/api/research/studies/archive",
        "/api/research/studies/restore",
        "/api/research/studies/compare",
        "/api/research/causal-scan",
    ):
        assert (path, "POST") in routes


def test_causal_scan_pairs_seeds_moves_only_the_intervention_and_emits_one_heatmap():
    async def scenario():
        published = []

        class Bus:
            async def publish(self, token, message):
                published.append((token, message))

        service = CausalScanService(SimpleNamespace(message_bus=Bus()))
        calls = []

        async def fake_trial(**kwargs):
            calls.append(kwargs)
            modules = kwargs["modules"] or {}
            intervention_layer = next(
                (
                    path
                    for path, specs in modules.items()
                    if any(
                        (spec if isinstance(spec, str) else spec.get("name")) == "neuron_ablation"
                        for spec in specs
                    )
                ),
                None,
            )
            value = {None: 1.0, "layers.0": 1.5, "layers.1": 0.5}[intervention_layer]
            return CapturedTrial(
                example_id="current-prompt",
                seed=kwargs["seed"],
                response={"status": "success"},
                artifacts=(
                    {
                        "type": "visualization",
                        "module": "activation_anisotropy",
                        "subtype": "scalar",
                        "layer_path": "layers.0",
                        "data": {"value": value},
                    },
                ),
            )

        service.study._run_trial = fake_trial
        original_modules = {
            "layers.0": [
                {"name": "activation_anisotropy", "parameters": {}},
                {
                    "name": "neuron_ablation",
                    "parameters": {"enabled": True, "neuron_indices": "0"},
                },
            ]
        }
        result = await service.run(
            model_id="debug/dummy",
            prompt="hello",
            layers=["layers.0", "layers.1"],
            seeds=[3, 7],
            parameters={},
            modules=original_modules,
            intervention={
                "name": "neuron_ablation",
                "parameters": {"enabled": True, "neuron_indices": "0"},
            },
            measurement_module="activation_anisotropy",
            metric="value",
            client_token="browser",
        )

        assert [point["delta_from_baseline"] for point in result["points"]] == [0.5, -0.5]
        assert len(calls) == 6
        assert all(
            all(
                (spec if isinstance(spec, str) else spec.get("name")) != "neuron_ablation"
                for specs in calls[index]["modules"].values()
                for spec in specs
            )
            for index in (0, 1)
        )
        assert original_modules["layers.0"][1]["parameters"].get("layer_path") is None
        assert len(published) == 1
        assert published[0][1]["module"] == "causal_scan"
        assert published[0][1]["data"]["values"] == [[0.5, 0.5], [-0.5, -0.5]]

    asyncio.run(scenario())


def test_research_contract_rejects_duplicate_examples_and_seeds():
    payload = {
        "model_id": "debug/dummy",
        "manifest": {"name": "smoke", "split": "validation"},
        "examples": [
            {"id": "same", "prompt": "one"},
            {"id": "same", "prompt": "two"},
        ],
        "seeds": [1, 1],
    }
    response = TestClient(ltts.app).post("/api/research/studies/run", json=payload)
    assert response.status_code == 422


def test_artifact_aggregation_keeps_provenance_and_ignores_non_numeric_outputs():
    trials = (
        CapturedTrial(
            example_id="a",
            seed=1,
            response={"status": "success"},
            artifacts=(
                {
                    "type": "visualization",
                    "module": "causal",
                    "subtype": "scalar",
                    "layer_path": "transformer.h.3",
                    "data": {"effect": 0.4},
                },
                {
                    "type": "visualization",
                    "layer_path": "transformer.h.4",
                    "data": {"effect": "not numeric"},
                },
            ),
        ),
        CapturedTrial(
            example_id="b",
            seed=2,
            response={"status": "success"},
            artifacts=(
                {
                    "type": "visualization",
                    "module": "causal",
                    "subtype": "scalar",
                    "module_path": "transformer.h.3",
                    "data": {"score": -0.2},
                },
            ),
        ),
    )

    rows = ResearchStudyService._aggregate(trials)
    assert len(rows) == 1
    assert rows[0]["layer"] == "transformer.h.3"
    assert rows[0]["mean_effect"] == pytest.approx(0.1)
    assert rows[0]["observations"] == 2
    assert {item["seed"] for item in rows[0]["provenance"]} == {1, 2}

    collection = ResearchStudyService._artifact_collection(
        "web-study",
        "study-1",
        trials,
        {"fingerprint": "sha256:dataset"},
    )
    restored = StudyArtifactCollection.from_dict(collection.to_dict())
    assert restored.manifest_fingerprint == "sha256:dataset"
    assert len(restored.records) == 2
    assert restored.records[0].stage == "scan"
    assert {record.metric for record in restored.records} == {"effect", "score"}
    assert {record.example_id for record in restored.records} == {"a", "b"}


def test_trial_capture_waits_for_real_message_bus_artifacts():
    async def scenario():
        bus = MessageBus()
        context = SimpleNamespace(message_bus=bus)
        service = ResearchStudyService(context)

        class FakeInference:
            async def run_inference(self, input):
                await bus.publish(
                    input.client_token,
                    {
                        "type": "visualization",
                        "module": "probe",
                        "subtype": "scalar",
                        "layer_path": "block.2",
                        "data": {"effect": 0.75},
                    },
                )
                await bus.publish(input.client_token, {"type": "inference_complete"})
                return {"status": "success", "result": "ok"}

        service.inference = FakeInference()
        processor = asyncio.create_task(bus.start_processing())
        await asyncio.sleep(0)
        try:
            trial = await service._run_trial(
                model_id="debug/dummy",
                example_id="one",
                prompt="hello",
                seed=3,
                parameters={},
                modules=None,
                parent_token="client",
            )
        finally:
            bus.stop_processing()
            await processor

        assert trial.seed == 3
        assert len(trial.artifacts) == 1
        assert trial.artifacts[0]["data"]["effect"] == 0.75

    asyncio.run(scenario())


def test_research_api_runs_reproducible_trials_on_debug_model():
    payload = {
        "name": "API smoke",
        "model_id": "debug/dummy",
        "manifest": {"name": "smoke", "split": "validation", "revision": "local"},
        "examples": [{"id": "prompt-1", "prompt": "hello research"}],
        "seeds": [7],
        "parameters": {"max_tokens": 2},
    }

    with TestClient(ltts.app) as client:
        first = client.post("/api/research/studies/run", json=payload)
        second = client.post("/api/research/studies/run", json=payload)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_result = first.json()
    second_result = second.json()
    assert first_result["schema_version"] == 1
    assert first_result["manifest"]["revision"] == "local"
    assert first_result["stages"][0]["kind"] == "dataset_manifest"
    assert first_result["stages"][1]["kind"] == "artifact_scan"
    assert len(first_result["trials"]) == 1
    assert first_result["trials"][0]["seed"] == 7
    assert first_result["artifacts"]["schema_version"] == 1
    assert first_result["artifacts"]["name"] == "API smoke"
    assert (
        first_result["trials"][0]["result"]
        == second_result["trials"][0]["result"]
    )


def test_research_archive_restore_and_compare_api_roundtrip():
    study = ResearchStudyBuilder("api-archive").add_stage(
        "scan", "artifact_scan", {"candidate_count": 1}
    ).build()
    baseline = StudyArtifactCollection(
        "baseline",
        (
            StudyArtifactRecord(
                "baseline-0",
                "effect",
                1.0,
                "probe",
                "scalar",
                layer_path="layers.0",
                example_id="example",
                seed=0,
            ),
        ),
        manifest_fingerprint="sha256:dataset",
    )
    candidate = StudyArtifactCollection(
        "candidate",
        (
            StudyArtifactRecord(
                "candidate-0",
                "effect",
                1.5,
                "probe",
                "scalar",
                layer_path="layers.0",
                example_id="example",
                seed=0,
            ),
        ),
        manifest_fingerprint="sha256:dataset",
    )
    archive = archive_research_study(
        study,
        baseline,
        manifest={"fingerprint": "sha256:dataset"},
        trials=({"example_id": "example", "seed": 0},),
    )

    with TestClient(ltts.app) as client:
        packed = client.post(
            "/api/research/studies/archive", json={"archive": archive.to_dict()}
        )
        restored = client.post(
            "/api/research/studies/restore", json={"bundle": packed.json()}
        )
        compared = client.post(
            "/api/research/studies/compare",
            json={
                "baseline": baseline.to_dict(),
                "candidate": candidate.to_dict(),
                "metadata": {"reason": "checkpoint"},
            },
        )

    assert packed.status_code == 200, packed.text
    assert packed.json()["kind"] == "research_study_archive"
    assert packed.json()["checksum"].startswith("sha256:")
    assert restored.status_code == 200, restored.text
    assert restored.json()["study"]["name"] == "api-archive"
    assert compared.status_code == 200, compared.text
    assert compared.json()["manifest_match"] is True
    assert compared.json()["points"][0]["delta"] == 0.5
    assert compared.json()["points"][0]["matched_trials"] == 1


def test_research_restore_rejects_tampered_bundle():
    with TestClient(ltts.app) as client:
        response = client.post(
            "/api/research/studies/restore",
            json={
                "bundle": {
                    "schema_version": 1,
                    "kind": "research_study_archive",
                    "created_at": "now",
                    "metadata": {},
                    "payload": {},
                    "checksum": "sha256:not-valid",
                }
            },
        )
    assert response.status_code == 422
    assert "checksum" in response.json()["detail"]


def test_inference_rejects_invalid_seed_without_changing_the_legacy_shape():
    with TestClient(ltts.app) as client:
        response = client.post(
            "/api/inference",
            json={
                "model_id": "debug/dummy",
                "prompt": "hello",
                "parameters": {"seed": -1},
            },
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "seed must be a uint32 value"
