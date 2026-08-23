from typing import Any, Dict, List
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from .ltts_event import LTTSEvent, LTTSContext, LTTSReactor

logger = logging.getLogger(__name__)


class EventDispatcher:
    def __init__(self, message_bus: Any, miners_registry: Any):
        self.message_bus = message_bus
        self.miners_registry = miners_registry

    def dispatch(
        self,
        event_type: str,
        context: LTTSContext,
        attached_modules: Dict[str, List[Any]],
    ) -> None:
        if event_type not in attached_modules:
            return

        if getattr(context, "message_bus", None) is None:
            context.message_bus = self.message_bus

        modules_count = len(attached_modules[event_type])

        # Only log when there are actual modules to process
        if modules_count > 0:
            module_desc = [
                str(entry.get("module")) if isinstance(entry, dict) else str(entry)
                for entry in attached_modules[event_type]
            ]
            logger.debug(
                "📡 EVENT_DISPATCHER: Found %s modules for event '%s': %s",
                modules_count,
                event_type,
                module_desc,
            )

        reactor = LTTSReactor(message_bus=self.message_bus)
        attachments = attached_modules[event_type]
        event_summaries: List[Dict[str, Any]] = []

        for i, attachment in enumerate(attachments):
            # Pre-bind so the except block below can always reference them.
            module = None
            target_layer = None
            requested_layer = None
            try:
                if isinstance(attachment, dict):
                    module = attachment.get("module")
                    module_config = attachment.get("module_config") or {}
                    target_layer = attachment.get("target_layer")
                    requested_layer = attachment.get("requested_layer")
                    schedule = attachment.get("schedule") or {}
                    attachment_state = attachment.get("state") or {}
                    metrics = self._ensure_metrics_container(attachment)
                else:
                    module = attachment
                    module_config = {}
                    target_layer = None
                    requested_layer = None
                    schedule = {}
                    attachment_state = {}
                    metrics = None

                if not self._schedule_matches(schedule, context):
                    continue

                # Only log module calls at debug level
                logger.debug(
                    "🔧 EVENT_DISPATCHER: Calling module %s/%s: %s",
                    i + 1,
                    modules_count,
                    module,
                )

                module_event = LTTSEvent(
                    event_type=event_type,
                    context=context,
                    reactor=reactor,
                    module_state=dict(attachment_state),
                    miners_registry=self.miners_registry,
                )

                if target_layer:
                    module_event.environment["resolved_layer_path"] = target_layer
                if requested_layer:
                    module_event.environment["requested_layer_path"] = requested_layer
                module_event.environment["module_config"] = dict(module_config)
                module_event.environment["generation_active"] = bool(
                    context.generation_active
                )
                module_event.environment["generation_step"] = context.generation_step
                module_event.environment["token_positions"] = list(
                    context.token_positions
                )
                module_event.environment["output_token_position"] = (
                    context.output_token_position
                )
                module_event.environment["attachment_schedule"] = dict(schedule)

                before_signal_count = len(reactor.signals)
                call_started = time.time()
                result = self._call_module_interceptor(module, module_event)
                duration = max(time.time() - call_started, 0.0)
                after_signal_count = len(reactor.signals)

                emitted = self._did_module_emit(
                    result, before_signal_count, after_signal_count
                )

                if metrics is not None:
                    metrics["invocations"] = metrics.get("invocations", 0) + 1
                    metrics["emissions"] = (
                        metrics.get("emissions", 0) + 1
                        if emitted
                        else metrics.get("emissions", 0)
                    )
                    metrics["last_event_type"] = event_type
                    metrics["last_invoked_at"] = time.time()
                    metrics["last_duration"] = duration
                    metrics["last_status"] = (
                        result.get("status")
                        if isinstance(result, dict)
                        else ("ok" if result is not None else None)
                    )
                    metrics["last_result"] = result
                    metrics["last_emitted"] = emitted
                    metrics["last_error"] = None
                    metrics["last_signal_count"] = (
                        after_signal_count - before_signal_count
                    )

                event_summaries.append(
                    self._build_module_summary(
                        module,
                        metrics,
                        event_type=event_type,
                        target_layer=target_layer,
                        requested_layer=requested_layer,
                        emitted=emitted,
                    )
                )

                if isinstance(attachment, dict):
                    attachment["state"] = dict(module_event.module_state or {})

                # Only log successful completion at debug level
                logger.debug(
                    "✅ EVENT_DISPATCHER: Successfully processed module %s/%s",
                    i + 1,
                    modules_count,
                )

            except Exception as e:
                # Keep error logging at info level as requested
                logger.error("❌ EVENT_DISPATCHER: Error in module %s: %s", module, e)
                import traceback

                logger.error("Traceback: %s", traceback.format_exc())

                if isinstance(attachment, dict):
                    metrics = self._ensure_metrics_container(attachment)
                    metrics["errors"] = metrics.get("errors", 0) + 1
                    metrics["last_error"] = str(e)
                    metrics["last_event_type"] = event_type
                    metrics["last_invoked_at"] = time.time()
                    metrics["last_status"] = "error"
                    metrics["last_emitted"] = False

                    event_summaries.append(
                        self._build_module_summary(
                            module,
                            metrics,
                            event_type=event_type,
                            target_layer=target_layer,
                            requested_layer=requested_layer,
                            emitted=False,
                        )
                    )

                reactor.signal(
                    "error",
                    {
                        "module": str(module),
                        "error": str(e),
                        "event_type": event_type,
                        "target_layer": target_layer,
                        "requested_layer": requested_layer,
                    },
                )

        # Only log completion at debug level
        logger.debug(
            "🏁 EVENT_DISPATCHER: Completed dispatching event '%s' to %s modules",
            event_type,
            modules_count,
        )
        self._publish_event_log(event_type, context, event_summaries)

    def _call_module_interceptor(self, module: Any, event: LTTSEvent) -> Any:
        interceptor_func = getattr(module, "interceptor", None)

        if interceptor_func and callable(interceptor_func):
            # Only log interceptor calls at debug level
            logger.debug(
                "✅ EVENT_DISPATCHER: Calling interceptor on %s",
                module,
            )
            return interceptor_func(event)
        else:
            # Keep warnings for missing interceptors as they indicate potential issues
            logger.warning(
                "❌ EVENT_DISPATCHER: Module %s has no callable interceptor method",
                module,
            )

    def _ensure_metrics_container(self, attachment: Dict[str, Any]) -> Dict[str, Any]:
        metrics = attachment.get("metrics")
        if not isinstance(metrics, dict):
            metrics = {
                "invocations": 0,
                "emissions": 0,
                "errors": 0,
                "last_result": None,
                "last_error": None,
                "last_status": None,
                "last_emitted": False,
                "last_event_type": None,
                "last_invoked_at": None,
                "last_duration": None,
                "last_signal_count": 0,
            }
            attachment["metrics"] = metrics
        return metrics

    @staticmethod
    def _did_module_emit(result: Any, before_signals: int, after_signals: int) -> bool:
        if after_signals > before_signals:
            return True

        if isinstance(result, dict):
            emitted = result.get("emitted")
            status = str(result.get("status", "")).lower()
            if isinstance(emitted, (list, tuple)):
                return len(emitted) > 0
            if isinstance(emitted, bool):
                return emitted
            if emitted is not None:
                return bool(emitted)
            if status in {"ok", "success", "emitted"}:
                return True
            return False
        # Avoid bool(result) on arbitrary return values: a module that returns a
        # tensor (or any multi-element array) would raise "Boolean value of
        # Tensor with more than one value is ambiguous". Presence is enough here.
        return result is not None

    def _build_module_summary(
        self,
        module: Any,
        metrics: Dict[str, Any],
        *,
        event_type: str,
        target_layer: str,
        requested_layer: str,
        emitted: bool,
    ) -> Dict[str, Any]:
        module_name = None
        metadata = getattr(module, "METADATA", None)
        if isinstance(metadata, dict):
            module_name = metadata.get("name")
        if not module_name:
            module_name = getattr(module, "__name__", str(module))

        return {
            "module_repr": repr(module),
            "module_name": module_name,
            "target_layer": target_layer,
            "requested_layer": requested_layer,
            "event_type": event_type,
            "emitted": emitted,
            "metrics": self._serialize_metrics(metrics) if metrics else {},
        }

    @staticmethod
    def _serialize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        if not metrics:
            return {}
        serialized = dict(metrics)
        timestamp = serialized.get("last_invoked_at")
        if isinstance(timestamp, (int, float)):
            serialized["last_invoked_at"] = datetime.fromtimestamp(
                timestamp
            ).isoformat()
        last_result = serialized.get("last_result")
        if last_result is not None:
            try:
                json.dumps(last_result)
            except TypeError:
                serialized["last_result"] = repr(last_result)
        last_error = serialized.get("last_error")
        if last_error is not None and not isinstance(last_error, str):
            serialized["last_error"] = str(last_error)
        return serialized

    def _publish_event_log(
        self, event_type: str, context: LTTSContext, summaries: List[Dict[str, Any]]
    ) -> None:
        if not summaries:
            return
        if not self.message_bus:
            return

        layer_path = getattr(context, "module_path", None)
        client_token = "default"
        ltts_model = getattr(context, "ltts_model", None)
        if ltts_model is not None:
            client_token = getattr(ltts_model, "_client_token", client_token)

        payload = {
            "type": "module_event_log",
            "event_type": event_type,
            "layer_path": layer_path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "modules": summaries,
        }

        try:
            if hasattr(self.message_bus, "publish_sync"):
                self.message_bus.publish_sync(client_token, payload)
            elif hasattr(self.message_bus, "publish"):
                asyncio.create_task(self.message_bus.publish(client_token, payload))
        except Exception as publish_error:
            logger.debug(
                "EVENT_DISPATCHER: Failed to publish event log: %s", publish_error
            )

    @staticmethod
    def _schedule_matches(schedule: Dict[str, Any], context: LTTSContext) -> bool:
        if not schedule:
            return True
        step = context.generation_step
        generation_only = bool(
            schedule.get(
                "generation_only",
                "generation_steps" in schedule
                or "output_token_positions" in schedule,
            )
        )
        if generation_only and not context.generation_active:
            return False
        configured = schedule.get("generation_steps")
        if configured is not None:
            values = (
                configured
                if isinstance(configured, (list, tuple, set))
                else [configured]
            )
            try:
                if step is None or int(step) not in {int(value) for value in values}:
                    return False
            except (TypeError, ValueError):
                raise ValueError("generation_steps must contain integers")
        configured_positions = schedule.get("token_positions")
        if configured_positions is not None:
            values = (
                configured_positions
                if isinstance(configured_positions, (list, tuple, set))
                else [configured_positions]
            )
            try:
                selected_positions = {int(value) for value in values}
            except (TypeError, ValueError):
                raise ValueError("token_positions must contain integers")
            if not selected_positions.intersection(context.token_positions):
                return False
        configured_outputs = schedule.get("output_token_positions")
        if configured_outputs is not None:
            values = (
                configured_outputs
                if isinstance(configured_outputs, (list, tuple, set))
                else [configured_outputs]
            )
            try:
                output_positions = {int(value) for value in values}
            except (TypeError, ValueError):
                raise ValueError("output_token_positions must contain integers")
            if context.output_token_position not in output_positions:
                return False
        if step is not None:
            start = int(schedule.get("start", 0))
            stop = schedule.get("stop")
            every = int(schedule.get("every", 1))
            if every < 1:
                raise ValueError("schedule every must be at least one")
            if step < start or (stop is not None and step >= int(stop)):
                return False
            if (step - start) % every:
                return False
        return True
