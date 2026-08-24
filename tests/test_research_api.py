from __future__ import annotations

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import ltts
from scripts.services.research_study import CapturedTrial, ResearchStudyService
from scripts.messages.message_bus import MessageBus


def test_research_route_is_registered():
    routes = {
        (route.path, method)
        for route in ltts.app.routes
        if hasattr(route, "path")
        for method in getattr(route, "methods", set())
    }
    assert ("/api/research/studies/run", "POST") in routes


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
    assert (
        first_result["trials"][0]["result"]
        == second_result["trials"][0]["result"]
    )


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
