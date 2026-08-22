import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Client token of the inference run currently in flight. Inference runs are
# serialized, so a single current token is safe; modules created by
# module_loader at startup cannot know it per-request.
_current_client_token = "default"

_background_tasks = set()


def set_current_client_token(token: Optional[str]) -> None:
    global _current_client_token
    _current_client_token = token or "default"


def get_current_client_token() -> str:
    return _current_client_token


class ModuleDataSender:
    """Simple helper for modules to send data via WebSocket."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.context = None

    def set_context(self, context):
        self.context = context

    def send_visualization(self, data: Any, subtype: str = "data", **kwargs):
        logger.debug(
            "[MODULE_DATA_SENDER] send_visualization called for %s", self.module_name
        )

        if not self.context:
            logger.warning(
                "[MODULE_DATA_SENDER] No context set for %s", self.module_name
            )
            return False

        if not hasattr(self.context, "message_bus"):
            logger.warning(
                "[MODULE_DATA_SENDER] Context has no message_bus attribute for %s",
                self.module_name,
            )
            return False

        if not self.context.message_bus:
            logger.warning(
                "[MODULE_DATA_SENDER] Context message_bus is None for %s",
                self.module_name,
            )
            return False

        try:
            message = {
                "type": "visualization",
                "module": self.module_name,
                "subtype": subtype,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }

            # Normalize layer identifiers so the UI can filter by layer
            layer_path = (
                kwargs.get("layer_path")
                or kwargs.get("module_path")
                or kwargs.get("layer_name")
            )
            if layer_path:
                message["layer_path"] = layer_path
                message["module_path"] = layer_path
                # Provide nested layer_info where consumers expect it
                if isinstance(message.get("data"), dict):
                    message["data"].setdefault("layer_info", {"path": layer_path})
                else:
                    message["layer_info"] = {"path": layer_path}
                # Keep one canonical artifact-type field.
                if "layer_name" in message:
                    del message["layer_name"]

            logger.debug("[MODULE_DATA_SENDER] Prepared message: %s", message)

            # Send via message bus
            client_token = kwargs.pop("client_token", None) or get_current_client_token()
            if hasattr(self.context.message_bus, "publish"):
                logger.debug("[MODULE_DATA_SENDER] Message bus has publish method")
                # Use async publish if available
                import asyncio

                if hasattr(self.context.message_bus, "publish_from_thread"):
                    logger.debug("[MODULE_DATA_SENDER] Using publish_from_thread method")
                    self.context.message_bus.publish_from_thread(client_token, message)
                elif hasattr(self.context.message_bus, "publish_sync"):
                    logger.debug("[MODULE_DATA_SENDER] Using publish_sync method")
                    self.context.message_bus.publish_sync(client_token, message)
                else:
                    logger.debug(
                        "[MODULE_DATA_SENDER] Using async publish with create_task"
                    )
                    # Fallback - create task
                    try:
                        task = asyncio.create_task(
                            self.context.message_bus.publish(client_token, message)
                        )
                        _background_tasks.add(task)
                        task.add_done_callback(_background_tasks.discard)
                    except RuntimeError:
                        logger.debug(
                            "[MODULE_DATA_SENDER] No running event loop; dropping message"
                        )
                        return False
            else:
                logger.warning("[MODULE_DATA_SENDER] Message bus has no publish method")
                return False

            logger.debug(
                "[MODULE_DATA_SENDER] Successfully sent visualization data for %s",
                self.module_name,
            )
            return True

        except Exception as e:
            logger.error(
                "[MODULE_DATA_SENDER] Error sending visualization data for %s: %s",
                self.module_name,
                e,
            )
            import traceback

            logger.debug("[MODULE_DATA_SENDER] Traceback: %s", traceback.format_exc())
            return False

    def send_scalar(self, value: float, label: str = None, **kwargs):
        """Send scalar value."""
        data = {
            "type": "scalar",
            "value": value,
            "label": label or f"{self.module_name} value",
        }
        return self.send_visualization(data, "scalar", **kwargs)

    def send_text(self, content: str, preformatted: bool = False, **kwargs):
        """Send text data."""
        data = {"type": "text", "content": content, "preformatted": preformatted}
        return self.send_visualization(data, "text", **kwargs)

    def send_image(self, image_data: str, image_type: str = "base64", **kwargs):
        """Send image data."""
        data = {"type": "image", image_type: image_data}
        return self.send_visualization(data, "image", **kwargs)

    def send_chart(self, chart_data: Any, chart_type: str = "line", **kwargs):
        """Send chart data."""
        data = {"type": "chart", "chart_type": chart_type, "data": chart_data}
        return self.send_visualization(data, "chart", **kwargs)

    def send_heatmap(self, values: list, dimensions: dict = None, **kwargs):
        """Send heatmap data."""
        data = {
            "type": "heatmap",
            "values": values,
            "dimensions": dimensions
            or {"rows": len(values), "cols": len(values[0]) if values else 0},
        }
        return self.send_visualization(data, "heatmap", **kwargs)

    def send_table(self, headers: list, rows: list, **kwargs):
        """Send table data."""
        data = {"type": "table", "headers": headers, "rows": rows}
        return self.send_visualization(data, "table", **kwargs)


# Global instance for modules to use
_senders = {}


def get_data_sender(module_name: str) -> ModuleDataSender:
    """Get or create a data sender for a module."""
    if module_name not in _senders:
        _senders[module_name] = ModuleDataSender(module_name)
    return _senders[module_name]
