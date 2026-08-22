"""
LTTSEvent and LTTSContext Implementation

Core event container that provides lazy access to miners and carries complete context data.
"""

from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timezone
import torch
from .miners_registry import MinersRegistry, get_global_registry
from .emission_controller import (
    should_emit as emission_should_emit,
    mark_finalized as emission_mark_finalized,
)

# Keep strong references to fire-and-forget publish tasks so they are not
# garbage-collected mid-flight (same pattern as event_dispatcher).
_background_tasks = set()


@dataclass
class LTTSContext:
    """Complete context data available to miners and modules."""

    # Raw data from model execution
    inputs: Any = None
    outputs: Any = None
    module_path: str = ""
    module_class: str = ""
    layer_type: str = ""
    module_role: str = ""
    architecture_stage: str = ""
    layer_index: Optional[int] = None
    attention_kind: Optional[str] = None
    architecture: Dict[str, Any] = field(default_factory=dict)

    # Specialized data for different layer types
    attention_weights: Any = None  # Captured attention weights for attention layers

    # Model state information
    model_state: Dict[str, Any] = field(default_factory=dict)
    layer_state: Dict[str, Any] = field(default_factory=dict)

    # Runtime context
    batch_info: Dict[str, Any] = field(default_factory=dict)
    device_info: Dict[str, Any] = field(default_factory=dict)
    message_bus: Any = None
    ltts_model: Any = (
        None  # Reference to LTTSModel instance for accessing model-level data
    )

    # Historical and analysis data
    previous_outputs: List[Any] = field(default_factory=list)
    analysis_results: Dict[str, Any] = field(default_factory=dict)

    def add_analysis_result(self, key: str, result: Any):
        """Add analysis result for sharing between modules."""
        self.analysis_results[key] = result


class LTTSReactor:
    """Interface for client communication from modules."""

    def __init__(self, message_bus=None, default_client_token: str = "default"):
        self.message_bus = message_bus
        self.default_client_token = default_client_token
        self.signals = []

    @staticmethod
    def _normalize_visualization_payload(data: Any) -> Dict[str, Any]:
        if isinstance(data, dict):
            payload = dict(data)
        else:
            payload = {"data": data}

        module_name = payload.get("module") or payload.get("module_name") or "unknown"
        payload["module"] = str(module_name)

        if "data" not in payload:
            payload["data"] = payload.get("value")

        layer_path = (
            payload.get("layer_path")
            or payload.get("module_path")
            or payload.get("modulePath")
        )
        payload["layer_path"] = layer_path
        if layer_path is not None:
            payload.setdefault("module_path", layer_path)

        payload["event_type"] = payload.get("event_type") or payload.get("eventType") or "module_event"
        payload["subtype"] = payload.get("subtype") or payload.get("type") or "generic"

        return payload

    def signal(self, signal_type: str, data: Any, client_token: Optional[str] = None):
        """Send signal to client UI."""
        import logging
        logger = logging.getLogger(__name__)

        normalized_data = data
        if signal_type == "visualization":
            normalized_data = self._normalize_visualization_payload(data)

        signal_data = {
            "type": signal_type,
            "data": normalized_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.signals.append(signal_data)

        # Log when signals are sent to client as requested
        logger.info("REACTOR: Sending signal '%s' to client", signal_type)

        if self.message_bus:
            # Use asyncio to publish to message bus
            import asyncio

            target_token = client_token or self.default_client_token
            try:
                # Resolve the loop before constructing the coroutine.  Calling
                # create_task(self.message_bus.publish(...)) in a worker thread
                # constructs a coroutine first and then raises because there is
                # no running loop, leaking an un-awaited coroutine warning.
                loop = asyncio.get_running_loop()
                task = loop.create_task(
                    self.message_bus.publish(target_token, signal_data)
                )
                _background_tasks.add(task)
                task.add_done_callback(_background_tasks.discard)
            except RuntimeError:
                # Worker thread (inference runs in asyncio.to_thread): publish
                # through the bus's main loop instead of dropping the signal.
                try:
                    self.message_bus.publish_from_thread(target_token, signal_data)
                except Exception:
                    logger.warning(
                        "REACTOR: failed to deliver signal '%s' from worker thread",
                        signal_type,
                        exc_info=True,
                    )


class LTTSEvent:
    """
    Core event container with lazy miners access and complete context.

    Provides dictionary-style access to miners: event["attention"]
    Hard-coded properties for common miners: event.attention
    """

    def __init__(
        self,
        event_type: str,
        context: LTTSContext,
        reactor: Optional[LTTSReactor] = None,
        module_state: Optional[Dict[str, Any]] = None,
        parent_events: Optional[List["LTTSEvent"]] = None,
        environment: Optional[Dict[str, Any]] = None,
        miners_registry: Optional[MinersRegistry] = None,
    ):
        self.event_type = event_type
        self.context = context
        self.reactor = reactor or LTTSReactor()
        self.module_state = module_state or {}
        self.parent_events = parent_events or []
        self.environment = environment or {}

        # Miners system
        self.miners_registry = miners_registry or get_global_registry()
        self._cached_extractions: Dict[str, Any] = {}

    def __getitem__(self, miner_key: str) -> Any:
        """Dictionary-style access to miners with caching."""
        if miner_key not in self._cached_extractions:
            if not self.miners_registry.has_miner(miner_key):
                raise KeyError(f"Miner '{miner_key}' not registered")

            miner = self.miners_registry.get_miner(miner_key)

            # Check if miner is applicable to current context
            if not miner.is_applicable(
                self.context.module_path, self.context.module_class
            ):
                return None

            # Extract data and cache result
            raw_context = {
                "inputs": self.context.inputs,
                "outputs": self.context.outputs,
                "module_path": self.context.module_path,
                "module_class": self.context.module_class,
                "layer_type": self.context.layer_type,
                "module_role": self.context.module_role,
                "architecture_stage": self.context.architecture_stage,
                "layer_index": self.context.layer_index,
                "attention_kind": self.context.attention_kind,
                "architecture": self.context.architecture,
                "model_state": self.context.model_state,
                "layer_state": self.context.layer_state,
                "attention_weights": self.context.attention_weights,
                "ltts_model": self.context.ltts_model,
            }

            self._cached_extractions[miner_key] = miner.extract(raw_context)

        return self._cached_extractions[miner_key]

    def __contains__(self, miner_key: str) -> bool:
        """Check if miner is available."""
        return self.miners_registry.has_miner(miner_key)

    # Hard-coded properties for common miners
    @property
    def attention(self):
        """Access attention miner data."""
        return self["attention"] if "attention" in self else None

    @property
    def moe(self):
        """Access MoE miner data."""
        return self["moe"] if "moe" in self else None

    @property
    def embeddings(self):
        """Access embeddings miner data."""
        return self["embeddings"] if "embeddings" in self else None

    @property
    def ffn(self):
        """Access FFN miner data."""
        return self["ffn"] if "ffn" in self else None

    @property
    def normalization(self):
        """Access normalization miner data."""
        return self["normalization"] if "normalization" in self else None

    def get_available_miners(self) -> List[str]:
        """Get list of available miners."""
        return self.miners_registry.list_miners()

    def clear_cache(self):
        """Clear cached miner extractions."""
        self._cached_extractions.clear()

    # ------------------------------------------------------------------
    # Emission helpers

    def should_emit(
        self,
        module_name: str,
        layer_path: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate sampling strategy for the given module/layer."""

        effective_layer = layer_path or getattr(self.context, "module_path", None)
        effective_params = params or self.module_state or {}
        return emission_should_emit(module_name, effective_layer, effective_params)

    def mark_emission_finalized(
        self, module_name: str, layer_path: Optional[str] = None
    ) -> None:
        """Mark a module/layer as finalized (for emit_mode='final')."""

        effective_layer = layer_path or getattr(self.context, "module_path", None)
        emission_mark_finalized(module_name, effective_layer)
