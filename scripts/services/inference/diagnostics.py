from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List

from scripts.services.inference.attachments import AttachmentSpec

logger = logging.getLogger(__name__)


class ModuleDiagnosticsPublisher:
    """Publish diagnostics for modules that did not emit during inference."""

    def __init__(self, message_bus):
        self.message_bus = message_bus

    async def publish_uninvoked_module_logs(
        self,
        ltts_model,
        attached_specs: List[AttachmentSpec],
        token: str,
    ) -> None:
        dispatcher = getattr(ltts_model, "event_dispatcher", None)
        serialize_metrics = None
        if dispatcher and hasattr(dispatcher, "_serialize_metrics"):
            serialize_metrics = dispatcher._serialize_metrics

        for spec in attached_specs:
            event_type = spec.event_type
            module_instance = spec.module
            if not event_type or module_instance is None:
                continue

            try:
                entry = ltts_model.module_registry.get_attachment_entry(
                    event_type, module_instance
                )
            except Exception as diag_error:  # pragma: no cover - defensive logging
                logger.debug(
                    "? INFERENCE: Failed to fetch attachment entry for diagnostics: %s",
                    diag_error,
                )
                continue

            metrics: Dict[str, any] = {}
            target_layer = None
            if isinstance(entry, dict):
                metrics = entry.get("metrics") or {}
                target_layer = entry.get("target_layer") or spec.layer_name

            if metrics.get("invocations", 0) != 0:
                continue

            if serialize_metrics:
                sanitized_metrics = serialize_metrics(metrics)
            else:
                sanitized_metrics = {
                    key: (
                        value
                        if isinstance(value, (int, float, str, bool, type(None)))
                        else repr(value)
                    )
                    for key, value in metrics.items()
                }

            payload = {
                "type": "module_event_log",
                "event_type": "module_uninvoked",
                "layer_path": target_layer,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "modules": [
                    {
                        "module_repr": repr(module_instance),
                        "module_name": spec.module_name,
                        "target_layer": target_layer,
                        "requested_layer": spec.requested_layer_name,
                        "event_type": event_type,
                        "emitted": False,
                        "metrics": sanitized_metrics,
                    }
                ],
            }

            try:
                await self.message_bus.publish(token, payload)
            except Exception as publish_error:  # pragma: no cover - defensive logging
                logger.debug(
                    "? INFERENCE: Failed to publish uninvoked module summary: %s",
                    publish_error,
                )
