"""Small public API for embedding LTTS instrumentation in other programs.

This module intentionally has no FastAPI, Socket.IO, filesystem registry, or
Vue dependencies.  A consumer supplies a ``torch.nn.Module`` and module files;
the engine returns ordinary Python artifacts and model outputs.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional

import torch.nn as nn

from scripts.core.ltts_model import LTTSModel


@dataclass(frozen=True)
class Artifact:
    """One renderable module output captured without a UI transport."""

    module: str
    subtype: str
    data: Any
    layer_path: Optional[str]
    raw: Dict[str, Any]


class _ArtifactBus:
    def __init__(self) -> None:
        self.messages: List[Dict[str, Any]] = []

    def _record(self, message: Any) -> None:
        if isinstance(message, dict):
            self.messages.append(message)

    async def publish(self, _client_token: str, message: Any) -> None:
        self._record(message)

    def publish_sync(self, _client_token: str, message: Any) -> None:
        self._record(message)

    def publish_from_thread(self, _client_token: str, message: Any) -> None:
        self._record(message)


class _ModuleContext:
    def __init__(self, message_bus: _ArtifactBus) -> None:
        self.message_bus = message_bus
        self.ltts_integration = None
        self.module_data_sender = None


def load_module(path: "str | Path", name: Optional[str] = None) -> ModuleType:
    """Load an LTTS module from an ``init.py`` file without the web registry."""

    source = Path(path).resolve()
    if source.is_dir():
        source = source / "init.py"
    if not source.is_file():
        raise FileNotFoundError(source)
    module_name = name or f"ltts_headless_{source.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load LTTS module from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HeadlessSession:
    """Instrument one model and execute modules without starting LTTS web app."""

    def __init__(self, model: nn.Module) -> None:
        self._bus = _ArtifactBus()
        self.model = LTTSModel(model, message_bus=self._bus)
        self._initialized: List[ModuleType] = []

    @property
    def architecture(self) -> Dict[str, Any]:
        return dict(self.model.architecture)

    @property
    def layer_descriptors(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.model.module_descriptors)

    def layers(self, role: Optional[str] = None) -> List[str]:
        items = self.model.module_descriptors.items()
        if role is not None:
            items = ((path, item) for path, item in items if item.get("role") == role)
        return [path for path, _item in items]

    def attach(
        self,
        module: ModuleType,
        *,
        layer: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        event_type: Optional[str] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Initialize and attach a module to a layer or model event."""

        if module not in self._initialized:
            initializer = getattr(module, "initialize_module", None) or getattr(
                module, "initialize", None
            )
            if callable(initializer):
                initializer(_ModuleContext(self._bus), **dict(config or {}))
            self._initialized.append(module)

        if layer is not None:
            resolved = self.model.attach_module_to_layer(
                layer, module, config, schedule
            )
            if resolved is None:
                raise ValueError(f"Layer could not be resolved: {layer}")
            return resolved

        metadata = getattr(module, "METADATA", {}) or {}
        events = [event_type] if event_type else list(metadata.get("event_types") or [])
        model_events = [item for item in events if str(item).startswith("model_")]
        if not model_events:
            raise ValueError("A layer or model-level event_type is required")
        for item in model_events:
            self.model.attach_module(str(item), module, config, schedule=schedule)
        return None

    def run(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.generate(*args, **kwargs)

    @property
    def artifacts(self) -> List[Artifact]:
        artifacts: List[Artifact] = []
        for message in self._bus.messages:
            if message.get("type") != "visualization":
                continue
            nested = message.get("data")
            if isinstance(nested, dict) and "module" in nested and "data" in nested:
                envelope = nested
            else:
                envelope = message
            artifacts.append(
                Artifact(
                    module=str(envelope.get("module") or "unknown"),
                    subtype=str(envelope.get("subtype") or "generic"),
                    data=envelope.get("data"),
                    layer_path=envelope.get("layer_path")
                    or envelope.get("module_path"),
                    raw=dict(message),
                )
            )
        return artifacts

    def clear_artifacts(self) -> None:
        self._bus.messages.clear()

    def reset_modules(self, *, clear_artifacts: bool = True) -> None:
        """Detach and clean up modules so the session can run another trial."""

        self.model.module_registry.detach_all_modules()
        for module in reversed(self._initialized):
            cleanup = getattr(module, "cleanup_module", None) or getattr(
                module, "cleanup", None
            )
            if callable(cleanup):
                cleanup()
        self._initialized.clear()
        if clear_artifacts:
            self.clear_artifacts()

    def close(self) -> None:
        self.reset_modules()
        self.model.close()

    def __enter__(self) -> "HeadlessSession":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


class HeadlessEngine:
    """Factory kept intentionally stateless for reuse by services and notebooks."""

    def instrument(self, model: nn.Module) -> HeadlessSession:
        return HeadlessSession(model)
