"""
Production readiness tests for LTTS.

Tests infrastructure, validation, and resilience — not model inference.
"""

import asyncio
import types
import pytest
import torch


# ---------------------------------------------------------------------------
# Message Bus Tests
# ---------------------------------------------------------------------------

class TestMessageBus:
    """Verify message bus robustness."""

    def _make_bus(self):
        from scripts.messages.message_bus import MessageBus
        return MessageBus()

    def test_queue_is_bounded(self):
        bus = self._make_bus()
        assert bus.message_queue.maxsize > 0, "Queue must have a max size"

    def test_publish_sync_drops_when_full(self):
        bus = self._make_bus()
        # Fill the queue
        for i in range(bus.message_queue.maxsize):
            bus.publish_sync("token", {"i": i})
        # This should not raise — it drops the message
        bus.publish_sync("token", {"overflow": True})
        assert bus._dropped_messages >= 1

    def test_stop_processing_flag(self):
        bus = self._make_bus()
        assert bus.running is False
        bus.stop_processing()
        assert bus.running is False


# ---------------------------------------------------------------------------
# Input Validation Tests
# ---------------------------------------------------------------------------

class TestInferenceInputValidation:
    """Verify request validation on the inference endpoint."""

    def _make_input(self, **kwargs):
        from scripts.routers.inference_router import InferenceInput
        defaults = {"model_id": "test/model", "prompt": "hello"}
        defaults.update(kwargs)
        return InferenceInput(**defaults)

    def test_valid_input(self):
        inp = self._make_input()
        assert inp.model_id == "test/model"
        assert inp.prompt == "hello"

    def test_rejects_empty_model_id(self):
        with pytest.raises(Exception):
            self._make_input(model_id="")

    def test_rejects_empty_prompt(self):
        with pytest.raises(Exception):
            self._make_input(prompt="")

    def test_rejects_oversized_prompt(self):
        with pytest.raises(Exception):
            self._make_input(prompt="x" * 200_000)

    def test_rejects_negative_max_tokens(self):
        with pytest.raises(Exception):
            self._make_input(max_tokens=-1)

    def test_rejects_huge_max_tokens(self):
        with pytest.raises(Exception):
            self._make_input(max_tokens=999_999)

    def test_accepts_valid_max_tokens(self):
        inp = self._make_input(max_tokens=256)
        assert inp.max_tokens == 256

    def test_rejects_too_many_modules(self):
        # 51 modules across layers should be rejected
        modules = {"layer.0": [f"mod_{i}" for i in range(51)]}
        with pytest.raises(Exception):
            self._make_input(modules=modules)

    def test_accepts_valid_modules(self):
        modules = {"layer.0": ["mod_a", "mod_b"]}
        inp = self._make_input(modules=modules)
        assert len(inp.modules["layer.0"]) == 2


# ---------------------------------------------------------------------------
# Layer Binding Tests
# ---------------------------------------------------------------------------

class TestLayerBinding:
    """Verify layer wrapping and path resolution."""

    def test_fallback_requires_two_segments(self):
        """Fallback layer matching must require 2+ significant segments.

        One shared segment is not evidence: "some.layers" overlaps every
        decoder path in the model and must not silently bind a module to an
        arbitrary one of them.
        """
        from scripts.core.module_registry import ModuleRegistry

        wrappers = {
            "model.layers.0.self_attn": object(),
            "model.layers.0.mlp": object(),
        }
        registry = ModuleRegistry(wrappers)

        class Mod:
            pass

        assert registry.attach_to_layer("some.layers", Mod()) is None
        assert registry.attach_to_layer("unrelated.foo.bar", Mod()) is None
        # Two significant segments are enough to identify a layer.
        assert (
            registry.attach_to_layer("layers.0.self_attn", Mod())
            == "model.layers.0.self_attn"
        )

    def test_exact_layer_match(self):
        from scripts.core.module_registry import ModuleRegistry

        wrappers = {"model.layers.0.self_attn": object()}
        registry = ModuleRegistry(wrappers)

        class Mod:
            pass

        result = registry.attach_to_layer("model.layers.0.self_attn", Mod())
        assert result == "model.layers.0.self_attn"


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------

class TestUserConfig:
    """Verify config defaults and validation."""

    def test_defaults_applied(self):
        from scripts.configs.user_config import UserConfig
        config = UserConfig()
        assert config.llm_engine == "transformers"
        assert config.autoload_blueprint is False

    def test_custom_values(self):
        from scripts.configs.user_config import UserConfig
        config = UserConfig(llm_engine="vllm", autoload_blueprint=True)
        assert config.llm_engine == "vllm"
        assert config.autoload_blueprint is True

    def test_extra_fields_ignored(self):
        from scripts.configs.user_config import UserConfig
        config = UserConfig(llm_engine="x", autoload_blueprint=False, unknown_field="abc")
        assert not hasattr(config, "unknown_field")


# ---------------------------------------------------------------------------
# Health Endpoint Test
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """Verify health check is registered."""

    def test_health_route_exists(self):
        from scripts.router import api
        routes = [r.path for r in api.routes if hasattr(r, "path")]
        assert "/api/health" in routes, f"Health route missing. Routes: {routes}"


# ---------------------------------------------------------------------------
# Emission Controller Tests
# ---------------------------------------------------------------------------

class TestEmissionController:
    """Verify emission state management."""

    def test_reset_clears_state(self):
        from scripts.core.emission_controller import (
            should_emit,
            reset_state,
            _STATE,
        )
        should_emit("test_mod", "layer.0", {"emit_mode": "all"})
        assert len(_STATE) > 0
        reset_state()
        assert len(_STATE) == 0

    def test_mode_first_emits_once(self):
        from scripts.core.emission_controller import should_emit, reset_state
        reset_state()
        r1 = should_emit("mod", "layer", {"emit_mode": "first"})
        r2 = should_emit("mod", "layer", {"emit_mode": "first"})
        assert r1["emit"] is True
        assert r2["emit"] is False

    def test_mode_every_n(self):
        from scripts.core.emission_controller import should_emit, reset_state
        reset_state()
        results = []
        for _ in range(6):
            r = should_emit("mod", "layer", {"emit_mode": "every_n", "emit_every_n": 3})
            results.append(r["emit"])
        # Passes 1-6: emit on 3, 6
        assert results == [False, False, True, False, False, True]


# ---------------------------------------------------------------------------
# Module Contract Smoke Test
# ---------------------------------------------------------------------------

class TestModuleContracts:
    """Quick smoke test that all modules load."""

    def test_modules_load(self):
        from scripts.router import context
        modules = context.module_loader.get_all_modules()
        assert len(modules) >= 5, f"Expected at least 5 modules, got {len(modules)}"

    def test_all_modules_have_metadata(self):
        from scripts.router import context
        for name, info in context.module_loader.get_all_modules().items():
            meta = info.get("metadata", {})
            assert meta.get("name"), f"{name} missing metadata.name"
            assert meta.get("version"), f"{name} missing metadata.version"
            assert meta.get("event_types"), f"{name} missing metadata.event_types"
