from __future__ import annotations

from dataclasses import dataclass, asdict

from fastapi.testclient import TestClient

import ltts


def test_core_routes_return_success():
    client = TestClient(ltts.app)

    expectations = {
        "/api/health": "status",
        "/api/readiness": "status",
        "/api/models": "models",
        "/api/modules": "modules",
        "/api/settings": "settings",
        "/api/settings/defaults": "settings",
        "/api/ltts/modules": "modules",
        "/api/ltts/health": "status",
    }

    for path, key in expectations.items():
        response = client.get(path)
        assert response.status_code == 200, path
        payload = response.json()
        assert key in payload, f"{path} missing key '{key}'"


def test_preflight_route_is_registered():
    routes = {
        (route.path, method)
        for route in ltts.app.routes
        if hasattr(route, "path")
        for method in getattr(route, "methods", set())
    }
    assert ("/api/models/preflight", "POST") in routes


def test_settings_route_exposes_registry_defaults():
    client = TestClient(ltts.app)

    response = client.get("/api/settings/defaults")
    assert response.status_code == 200

    settings = response.json()["settings"]
    # The key must exist so the settings UI can bind to it, but it ships empty
    # on purpose: a pinned value would be copied into every personal config and
    # would then outrank a local LTTS-modules checkout.
    assert "registry_url" in settings["modules"]
    assert isinstance(settings["modules"]["registry_url"], str)
    assert "blueprint" in settings


def test_readiness_route_reports_loaded_modules():
    client = TestClient(ltts.app)

    response = client.get("/api/readiness")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] in {"ready", "degraded"}
    assert payload["checks"]["config_exists"] is True
    assert payload["checks"]["models_registry_exists"] is True
    assert payload["details"]["modules_loaded"] >= 1


def test_registry_modules_route_uses_requested_registry_url(monkeypatch):
    @dataclass
    class FakeModule:
        name: str
        version: str
        description: str
        author: str
        event_types: list[str]
        dependencies: list[str]
        tags: list[str]
        repository_url: str
        module_path: str
        size_bytes: int
        created_at: str
        updated_at: str
        installed: bool = False
        local_path: str | None = None

    class FakeRegistry:
        def __init__(self):
            self.remote_registry_url = "https://example.invalid/default.json"
            self.debug_mode = True

        async def fetch_remote_modules(self):
            return [
                FakeModule(
                    name="remote_mod",
                    version="1.0.0",
                    description="Remote test module",
                    author="test",
                    event_types=["layer_after"],
                    dependencies=[],
                    tags=["remote"],
                    repository_url="https://example.invalid/repo",
                    module_path="ltts_modules/remote_mod",
                    size_bytes=1,
                    created_at="2026-01-01T00:00:00",
                    updated_at="2026-01-01T00:00:00",
                )
            ]

    fake_registry = FakeRegistry()

    import scripts.api_integration as api_integration

    monkeypatch.setattr(api_integration, "get_module_registry", lambda: fake_registry)

    client = TestClient(ltts.app)
    url = "https://registry.example.com/modules.json"
    response = client.get("/api/ltts/registry/modules", params={"registry_url": url})
    assert response.status_code == 200

    payload = response.json()
    assert payload["registry_url"] == url
    assert fake_registry.remote_registry_url == url
    assert fake_registry.debug_mode is False
    assert payload["modules"][0]["name"] == "remote_mod"


def test_lifespan_starts_and_stops_the_message_bus():
    """The FastAPI lifespan must start and stop the message bus."""
    from scripts.router import get_context

    context = get_context()
    assert getattr(context.message_bus, "running", False) is False

    with TestClient(ltts.app) as client:
        assert client.get("/api/health").json()["message_bus_running"] is True

    assert getattr(context.message_bus, "running", False) is False


def test_javascript_assets_are_served_with_an_executable_mime_type():
    """The built SPA must not be served as text/plain.

    Python reads MIME types from the Windows registry, where .js is commonly
    registered as text/plain. A browser refuses to execute a module script
    served that way, so the whole UI failed to start - a blank page whose only
    symptom was a console error - on exactly the platform the README says the
    project is developed on.
    """
    import mimetypes

    assert mimetypes.guess_type("bundle.js")[0] in {
        "text/javascript",
        "application/javascript",
    }
    assert mimetypes.guess_type("bundle.mjs")[0] in {
        "text/javascript",
        "application/javascript",
    }
    assert mimetypes.guess_type("style.css")[0] == "text/css"


def test_shipped_debug_model_runs_inference_without_a_hub_download():
    """The built-in smoke-test model must be usable on a fresh offline clone."""
    with TestClient(ltts.app) as client:
        response = client.post(
            "/api/inference",
            json={
                "model_id": "debug/dummy",
                "prompt": "hello",
                "max_tokens": 2,
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["model_id"] == "debug/dummy"


def test_shipped_debug_blueprint_exposes_real_research_capabilities():
    import json

    with TestClient(ltts.app) as client:
        response = client.post(
            "/api/blueprint_from_model",
            json={"id": "debug/dummy", "engine": "transformers"},
        )

    assert response.status_code == 200, response.text
    model = json.loads(response.json()["model"])
    categorization = model["categorization"]
    assert categorization["architectureTopology"] == "decoder_only"
    assert categorization["capabilities"]["hidden_states"] is True
    block = next(layer for layer in model["layers"] if layer["name"] == "layers.0")
    assert block["moduleRole"] == "block"
    assert block["architectureStage"] == "decoder"
