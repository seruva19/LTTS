import types
import asyncio
import inspect

import torch

from scripts.router import context


# The two emission channels carry different envelopes, and the UI treats
# `event_type` as optional (src/state/socket-store.js). Assert what each channel
# actually guarantees rather than one union that neither satisfies.
REQUIRED_REACTOR_VIS_KEYS = ("module", "subtype", "event_type", "data")
REQUIRED_SENDER_VIS_KEYS = ("module", "subtype", "data")

# Capture the message bus so sender-channel assertions exercise actual emissions.


class CapturingMessageBus:
    """Message bus stand-in that records what ModuleDataSender publishes."""

    def __init__(self):
        self.messages = []

    def publish_sync(self, client_token, message):
        self.messages.append(message)
        return True

    async def publish(self, client_token, message):
        self.messages.append(message)
        return True


def _build_synthetic_event(event_type: str):
    if event_type.startswith("model_"):
        module_path = "model"
        layer_type = "model"
    else:
        module_path = "model.layers.0.self_attn"
        layer_type = "attention"

    fake_backing_model = torch.nn.Sequential(
        torch.nn.Linear(16, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 16),
    )

    fake_context = types.SimpleNamespace(
        module_role="",
        architecture_stage="",
        layer_index=0,
        architecture={},
        batch_info={},
        device_info={},
        previous_outputs=[],
        analysis_results={},
        inputs={"input_ids": torch.tensor([[1, 2, 3, 4]])},
        outputs=torch.randn(1, 4, 16, requires_grad=True),
        module_path=module_path,
        module_class="SyntheticLayer",
        layer_type=layer_type,
        model_state={},
        layer_state={},
        message_bus=CapturingMessageBus(),
        attention_weights=torch.rand(1, 2, 4, 4),
        ltts_model=types.SimpleNamespace(
            original_model=fake_backing_model,
            layer_wrappers={},
            _last_input_tokens=["tok1", "tok2", "tok3", "tok4"],
            _tokenizer=None,
        ),
    )

    from scripts.core.ltts_event import LTTSReactor

    class FakeEvent:
        def __init__(self):
            self.event_type = event_type
            self.context = fake_context
            self.module_state = {}
            # Documented in docs/MODULE_AUTHORING.md as part of the contract.
            self.environment = {"module_config": {}}
            self.reactor = LTTSReactor(message_bus=None)

        def should_emit(self, *_args, **_kwargs):
            return {
                "emit": True,
                "mode": "all",
                "forward_pass": 1,
                "stride": None,
                "expected_passes": None,
                "deferred": False,
            }

        def mark_emission_finalized(self, *_args, **_kwargs):
            return None

        def __contains__(self, key):
            return key in {"attention", "embeddings", "ffn", "moe", "normalization"}

        def __getitem__(self, key):
            base_payload = {
                "layer_info": {"path": fake_context.module_path},
                "layer_name": fake_context.module_path,
                "layer_path": fake_context.module_path,
            }
            if key == "attention":
                return {
                    **base_payload,
                    "attention_weights": fake_context.attention_weights,
                    "token_data": {
                        "tokens": fake_context.ltts_model._last_input_tokens,
                    },
                }
            if key in {"embeddings", "ffn", "normalization"}:
                return {
                    **base_payload,
                    "hidden_states": fake_context.outputs,
                    "embeddings": fake_context.outputs,
                    "activations": fake_context.outputs,
                }
            if key == "moe":
                return {
                    **base_payload,
                    "router_logits": torch.randn(1, 4, 4),
                    "expert_indices": torch.randint(0, 4, (1, 4)),
                }
            raise KeyError(key)

    return FakeEvent()


def _collect_visualizations(event):
    """Every visualization payload the module emitted, with its channel."""
    payloads = []
    for signal in event.reactor.signals:
        if signal.get("type") == "visualization" and isinstance(signal.get("data"), dict):
            payloads.append(("reactor", signal["data"]))
    for message in event.context.message_bus.messages:
        if message.get("type") == "visualization":
            payloads.append(("sender", message))
    return payloads


async def _invoke_module_entrypoint(module_obj, event):
    interceptor = getattr(module_obj, "interceptor", None)
    process_event = getattr(module_obj, "process_event", None)

    if callable(interceptor):
        result = interceptor(event)
    elif callable(process_event):
        result = process_event(event)
    else:
        raise AssertionError(
            f"{module_obj} has no callable interceptor/process_event method."
        )

    if inspect.isawaitable(result):
        await result

    await asyncio.sleep(0)


def test_all_modules_load_and_emit_contract():
    modules = context.module_loader.get_all_modules()
    assert modules, "No modules loaded by module loader."

    for module_name, module_info in modules.items():
        metadata = module_info.get("metadata") or {}
        assert isinstance(metadata.get("name"), str) and metadata.get("name")
        assert isinstance(metadata.get("version"), str) and metadata.get("version")
        event_types = metadata.get("event_types") or []
        assert isinstance(event_types, list) and event_types

        module_obj = module_info.get("module_object")
        assert module_obj is not None, f"{module_name} has no module object."

        ui_schema = module_info.get("ui_schema")
        assert isinstance(
            ui_schema, dict
        ), f"{module_name} returned non-dict UI schema."

        synthetic_event = _build_synthetic_event(str(event_types[0]))
        asyncio.run(_invoke_module_entrypoint(module_obj, synthetic_event))

        emitted = _collect_visualizations(synthetic_event)
        for channel, payload in emitted:
            assert isinstance(
                payload, dict
            ), f"{module_name} emitted visualization payload that is not a dict."
            required = (
                REQUIRED_REACTOR_VIS_KEYS
                if channel == "reactor"
                else REQUIRED_SENDER_VIS_KEYS
            )
            for key in required:
                assert key in payload, (
                    f"{module_name} missing '{key}' in {channel}-channel "
                    f"visualization payload."
                )
            assert (
                isinstance(payload.get("module"), str) and payload.get("module")
            ), f"{module_name} emitted empty module name in visualization payload."


# Modules that produce a visualization from the synthetic event alone. The rest
# legitimately need something the synthetic fixture cannot provide (a real
# unembedding, gradients through a real graph, a loaded SAE, two or more layers
# seen in sequence); those are covered end-to-end by tests/manual_e2e_all_modules.py
# against a real checkpoint. This list is a regression guard: it exists so that
# the payload assertions above can never silently apply to nothing.
MODULES_EMITTING_ON_SYNTHETIC_EVENT = {
    "attn_visualizer",
    "layer_inspector",
    "layer_statistics",
    "neuron_tracker",
    "model_control_vector",
}


def test_synthetic_event_still_exercises_the_payload_assertions():
    """Guard against the contract assertions above becoming vacuous."""
    modules = context.module_loader.get_all_modules()
    assert modules, "No modules loaded by module loader."

    emitting = set()
    for module_name, module_info in modules.items():
        metadata = module_info.get("metadata") or {}
        event_types = metadata.get("event_types") or []
        module_obj = module_info.get("module_object")
        if module_obj is None or not event_types:
            continue
        synthetic_event = _build_synthetic_event(str(event_types[0]))
        asyncio.run(_invoke_module_entrypoint(module_obj, synthetic_event))
        if _collect_visualizations(synthetic_event):
            emitting.add(module_name)

    installed = set(modules)
    expected = MODULES_EMITTING_ON_SYNTHETIC_EVENT & installed
    regressed = expected - emitting
    assert not regressed, (
        f"Modules that used to emit on the synthetic event no longer do: "
        f"{sorted(regressed)}"
    )
    assert emitting, "No module emitted anything - the contract assertions are vacuous."


# Smallest thing the loader accepts as a module, for path-resolution tests.
STUB_MODULE = "METADATA = {}\n"

VALID_VIZ_DATA_TYPES = {"scalar", "chart", "table", "heatmap", "text", "image"}


def test_visualization_data_types():
    """Verify that modules emit structured visualization data with valid types."""
    modules = context.module_loader.get_all_modules()
    assert modules, "No modules loaded by module loader."

    for module_name, module_info in modules.items():
        metadata = module_info.get("metadata") or {}
        event_types = metadata.get("event_types") or []
        module_obj = module_info.get("module_object")
        if module_obj is None or not event_types:
            continue

        synthetic_event = _build_synthetic_event(str(event_types[0]))
        asyncio.run(_invoke_module_entrypoint(module_obj, synthetic_event))

        for _channel, payload in _collect_visualizations(synthetic_event):
            inner_data = payload.get("data")
            if not isinstance(inner_data, dict):
                continue

            data_type = inner_data.get("type")
            if data_type is None:
                continue

            assert data_type in VALID_VIZ_DATA_TYPES, (
                f"{module_name} emitted unknown data type '{data_type}'. "
                f"Expected one of {VALID_VIZ_DATA_TYPES}."
            )

            # Validate structure based on type
            if data_type == "scalar":
                assert "value" in inner_data, (
                    f"{module_name} scalar visualization missing 'value'"
                )
            elif data_type == "table":
                assert "headers" in inner_data and "rows" in inner_data, (
                    f"{module_name} table visualization missing 'headers' or 'rows'"
                )
                assert isinstance(inner_data["headers"], list), (
                    f"{module_name} table 'headers' must be a list"
                )
                assert isinstance(inner_data["rows"], list), (
                    f"{module_name} table 'rows' must be a list"
                )
            elif data_type == "chart":
                assert "chart_type" in inner_data, (
                    f"{module_name} chart visualization missing 'chart_type'"
                )
                assert "data" in inner_data, (
                    f"{module_name} chart visualization missing 'data'"
                )
            elif data_type == "heatmap":
                assert "values" in inner_data, (
                    f"{module_name} heatmap visualization missing 'values'"
                )
                assert isinstance(inner_data["values"], list), (
                    f"{module_name} heatmap 'values' must be a list"
                )


def test_layer_binding_resolution():
    """Verify that layer path resolution doesn't attach to wrong layers."""
    from scripts.core.module_registry import ModuleRegistry

    # Simulate a model with specific layers
    mock_wrappers = {
        "model.layers.0.self_attn": object(),
        "model.layers.0.mlp": object(),
        "model.layers.1.self_attn": object(),
        "model.layers.1.mlp": object(),
        "model.embed_tokens": object(),
    }
    registry = ModuleRegistry(mock_wrappers)

    class FakeModule:
        pass

    # Exact match should work
    mod1 = FakeModule()
    result = registry.attach_to_layer("model.layers.0.self_attn", mod1)
    assert result == "model.layers.0.self_attn", f"Expected exact match, got {result}"

    # Partial match should work
    mod2 = FakeModule()
    result = registry.attach_to_layer("layers.1.mlp", mod2)
    assert result is not None, "Partial match should resolve to a layer"
    assert "mlp" in result, f"Expected mlp layer, got {result}"

    # Completely unrelated path should NOT match
    mod3 = FakeModule()
    result = registry.attach_to_layer("nonexistent.foo.bar", mod3)
    assert result is None, (
        f"Unrelated layer path should not match, but got {result}"
    )


def test_is_transformer_layer_excludes_containers():
    """Verify that top-level model containers are not wrapped as layers."""
    from scripts.core.ltts_model import LTTSModel
    import torch.nn as nn

    # Create a minimal model instance (without wrapping a real model)
    dummy = nn.Linear(4, 4)
    ltts = LTTSModel.__new__(LTTSModel)

    # Simulate container class names
    class LlamaForCausalLM(nn.Module):
        pass

    class GemmaModel(nn.Module):
        pass

    class LlamaDecoderLayer(nn.Module):
        pass

    class LlamaMLP(nn.Module):
        pass

    class LlamaAttention(nn.Module):
        pass

    # Containers should be excluded
    assert not ltts._is_transformer_layer(LlamaForCausalLM()), (
        "LlamaForCausalLM should not be identified as a transformer layer"
    )
    assert not ltts._is_transformer_layer(GemmaModel()), (
        "GemmaModel should not be identified as a transformer layer"
    )

    # Actual sub-layers should be included
    assert ltts._is_transformer_layer(LlamaDecoderLayer()), (
        "LlamaDecoderLayer should be identified as a transformer layer"
    )
    assert ltts._is_transformer_layer(LlamaMLP()), (
        "LlamaMLP should be identified as a transformer layer"
    )
    assert ltts._is_transformer_layer(LlamaAttention()), (
        "LlamaAttention should be identified as a transformer layer"
    )


def test_legacy_statistics_modules_initialize_sender_context():
    """Module entrypoints must initialize the sender used by legacy visualizations."""
    from ltts_modules.layer_statistics import init as layer_statistics
    from ltts_modules.neuron_tracker import init as neuron_tracker
    from ltts_modules.token_contributor import init as token_contributor
    from scripts.core import module_data_sender

    fake_context = types.SimpleNamespace(
        message_bus=object(), module_data_sender=module_data_sender
    )

    layer_instance = layer_statistics.initialize(fake_context)
    neuron_instance = neuron_tracker.initialize(fake_context)
    token_instance = token_contributor.initialize(fake_context)

    assert layer_instance.data_sender.context is fake_context
    assert neuron_instance.data_sender.context is fake_context
    assert token_instance.data_sender.context is fake_context


def test_neuron_tracker_emits_computed_sparsity():
    """Sparsity lives at the analysis root, not inside basic statistics."""
    from ltts_modules.neuron_tracker.init import NeuronTracker

    class CapturingSender:
        def __init__(self):
            self.scalars = []

        def send_scalar(self, value, label=None, **_kwargs):
            self.scalars.append((value, label))
            return True

        def send_chart(self, *_args, **_kwargs):
            return True

        def send_table(self, *_args, **_kwargs):
            return True

    tracker = NeuronTracker()
    tracker.data_sender = CapturingSender()
    activations = torch.tensor([[[-1.0, 0.0, 1.0, 2.0]]])

    tracker.intercept_layer(
        activations,
        {"layer_index": 0, "layer_name": "model.layers.0"},
        types.SimpleNamespace(),
    )

    sparsity_values = [
        value for value, label in tracker.data_sender.scalars if "Sparsity" in label
    ]
    assert sparsity_values == [0.5]


def test_module_loader_replaces_empty_namespace_parent(tmp_path):
    """Import order must not turn a valid directory module into a false failure."""
    import sys

    from scripts.core.ltts_module import load_module_from_file

    module_name = "ltts_modules.synthetic_namespace_case"
    namespace_parent = types.ModuleType(module_name)
    namespace_parent.__path__ = []
    sys.modules[module_name] = namespace_parent

    module_file = tmp_path / "init.py"
    module_file.write_text(
        "METADATA = {"
        "'name': 'synthetic_namespace_case', "
        "'version': '1.0.0', "
        "'event_types': ['layer_after'], "
        "'methods': {}"
        "}\n",
        encoding="utf-8",
    )
    try:
        loaded = load_module_from_file(str(module_file), module_name)
        assert loaded is not None
        assert loaded["metadata"]["name"] == "synthetic_namespace_case"
    finally:
        sys.modules.pop(module_name, None)


def test_registry_never_installs_into_a_source_checkout(tmp_path, monkeypatch):
    """Registry installs must write only to the install directory."""
    from ltts_platform.module_paths import (
        resolve_install_dir,
        resolve_module_search_paths,
    )

    # LTTS_MODULES_DIR outranks a sibling checkout by design; this test is about
    # the sibling, so take the override out of the picture.
    monkeypatch.delenv("LTTS_MODULES_DIR", raising=False)

    project_root = tmp_path / "LTTS"
    (project_root / "ltts_modules").mkdir(parents=True)

    # A sibling checkout holding module source, exactly like a real setup.
    sibling = tmp_path / "LTTS-modules" / "some_module"
    sibling.mkdir(parents=True)
    (sibling / "init.py").write_text(STUB_MODULE, encoding="utf-8")

    install_dir = resolve_install_dir(project_root)
    search_paths = resolve_module_search_paths(project_root)

    assert install_dir == project_root / "ltts_modules"
    assert install_dir.resolve() != sibling.parent.resolve()
    # The checkout is still readable, and takes precedence over installed copies.
    assert search_paths[0].resolve() == sibling.parent.resolve()
    assert install_dir.resolve() in [p.resolve() for p in search_paths]


def test_loader_reads_from_every_search_path(tmp_path):
    """A module installed from the registry must load next to checkout modules."""
    from scripts.core.module_loader import ModuleLoader

    source_dir = tmp_path / "source"
    install_dir = tmp_path / "installed"
    for directory, name in ((source_dir, "from_source"), (install_dir, "from_registry")):
        module_dir = directory / name
        module_dir.mkdir(parents=True)
        (module_dir / "init.py").write_text(STUB_MODULE, encoding="utf-8")

    loader = ModuleLoader([str(source_dir), str(install_dir)])
    discovered = loader.discover_modules()

    assert set(discovered) == {"from_source", "from_registry"}
    assert loader._get_module_path("from_registry") == str(
        install_dir / "from_registry" / "init.py"
    )


def test_source_checkout_shadows_an_installed_copy(tmp_path):
    """The same module in both places resolves to the checkout, once."""
    from scripts.core.module_loader import ModuleLoader

    source_dir = tmp_path / "source"
    install_dir = tmp_path / "installed"
    for directory in (source_dir, install_dir):
        module_dir = directory / "shared_name"
        module_dir.mkdir(parents=True)
        (module_dir / "init.py").write_text(STUB_MODULE, encoding="utf-8")

    loader = ModuleLoader([str(source_dir), str(install_dir)])

    assert loader.discover_modules() == ["shared_name"]
    assert loader._get_module_path("shared_name") == str(
        source_dir / "shared_name" / "init.py"
    )


def test_categorized_layers_carry_the_role_the_ui_filters_on():
    """Categorized layers include the semantic role used by UI filters."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from scripts.core.ltts_model import LTTSModel
    from scripts.models.model_struct import traverse_model, model_to_json

    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=2, n_positions=32)
    ).eval()
    wrapped = LTTSModel(model, message_bus=None)

    with torch.no_grad():
        struct = traverse_model(wrapped)
    # model_to_json returns the serialized form the API hands the UI.
    payload = model_to_json(struct)
    if isinstance(payload, str):
        import json

        payload = json.loads(payload)

    categories = (payload.get("categorization") or {}).get("categories") or {}
    assert categories, "model produced no categorization for the UI to render"

    entries = [
        entry
        for category in categories.values()
        for entry in (category.get("layers") or [])
    ]
    assert entries, "categorization contains no layers"

    roles = {entry.get("moduleRole") for entry in entries if entry.get("moduleRole")}
    assert roles, "no categorized layer carries moduleRole - the role filter is blind"

    attention = [
        entry for entry in entries
        if str(entry.get("moduleRole", "")).endswith("attention")
    ]
    assert attention, f"no attention layer among categorized roles: {sorted(roles)}"
