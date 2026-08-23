"""
Module Registry

Manages module attachment, detachment, and layer path resolution
for the LTTS framework.
"""

from typing import Any, Dict, List, Optional
import logging

from scripts.core.layer_path_resolver import LayerPathResolver

logger = logging.getLogger(__name__)


class ModuleRegistry:
    """Manages module attachments to model events and layers."""

    def __init__(self, layer_wrappers: Dict[str, Any]):
        """
        Initialize module registry.

        Args:
            layer_wrappers: Dictionary mapping layer paths to LTTSLayer wrappers
        """
        self.layer_wrappers = layer_wrappers
        self.attached_modules: Dict[str, List[Any]] = {}
        self._resolver = LayerPathResolver(layer_wrappers)

    def attach_module(
        self,
        event_type: str,
        module: Any,
        module_config: Optional[Dict[str, Any]] = None,
        target_layer: Optional[str] = None,
        requested_layer: Optional[str] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Attach module to specific event type.

        Args:
            event_type: Event type to attach to
            module: Module instance to attach
            module_config: Optional module configuration
            target_layer: Resolved target layer path
            requested_layer: Originally requested layer path
        """
        if event_type not in self.attached_modules:
            self.attached_modules[event_type] = []

        attachments = self.attached_modules[event_type]

        duplicates = []
        for existing in list(attachments):
            existing_module = (
                existing.get("module") if isinstance(existing, dict) else existing
            )
            if existing_module is module:
                duplicates.append(existing)

        if duplicates:
            primary = duplicates[0]
            if isinstance(primary, dict):
                primary["module_config"] = module_config or {}
                primary["target_layer"] = target_layer
                primary["requested_layer"] = requested_layer
                primary["state"] = dict(module_config or {})
                primary["schedule"] = dict(schedule or {})
            else:
                idx = attachments.index(primary)
                primary = {
                    "module": module,
                    "module_config": module_config or {},
                    "target_layer": target_layer,
                    "requested_layer": requested_layer,
                    "state": dict(module_config or {}),
                    "schedule": dict(schedule or {}),
                }
                attachments[idx] = primary

            for duplicate in duplicates[1:]:
                if duplicate in attachments:
                    attachments.remove(duplicate)

            logger.debug(
                "Updated existing attachment for module %s on event %s",
                module,
                event_type,
            )
            return

        entry = {
            "module": module,
            "module_config": module_config or {},
            "target_layer": target_layer,
            "requested_layer": requested_layer,
            # Initialize per-attachment persistent state from module_config for compatibility
            "state": dict(module_config or {}),
            "schedule": dict(schedule or {}),
            "metrics": {
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
            },
        }

        self.attached_modules[event_type].append(entry)
        logger.info(
            "MODULE_REGISTRY: Attached %s to %s (target=%s requested=%s)",
            module,
            event_type,
            target_layer,
            requested_layer,
        )

    def attach_to_layer(
        self,
        layer_name: str,
        module: Any,
        module_config: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Attach module to specific layer with automatic path resolution.

        Args:
            layer_name: Layer name/path to attach to
            module: Module instance to attach
            module_config: Optional module configuration

        Returns:
            Resolved layer path if successful, None otherwise
        """
        logger.debug(
            "🔧 MODULE_REGISTRY: Attaching module %s to layer %s", module, layer_name
        )

        normalized_layer_name = self._resolver.normalize(layer_name, module)
        if normalized_layer_name != layer_name:
            logger.debug(
                "🔁 MODULE_REGISTRY: Remapped requested layer %s -> %s for module %s",
                layer_name,
                normalized_layer_name,
                getattr(module, "__name__", str(module)),
            )
            layer_name = normalized_layer_name

        # Try multiple layer path formats automatically
        layer_candidates = self._resolver.generate_candidates(layer_name)

        # Try exact matches first
        for candidate in layer_candidates:
            if candidate in self.layer_wrappers:
                layer_event_type = f"layer_after_{candidate.replace('.', '_')}"

                # Always refresh: a module re-attached to a different layer
                # must not keep the stale target.
                module._target_layer = candidate

                self.attach_module(
                    layer_event_type,
                    module,
                    module_config=module_config,
                    target_layer=candidate,
                    requested_layer=layer_name,
                    schedule=schedule,
                )
                logger.info(
                    "✅ MODULE_REGISTRY: Attached %s to %s (resolved from %s)",
                    module,
                    layer_event_type,
                    layer_name,
                )
                return candidate

        # Try partial matching on whole path segments: substring matching would
        # wrongly pair e.g. model.layers.1 with model.layers.10.
        for candidate in layer_candidates:
            candidate_parts = candidate.split(".")
            matching_layers = [
                name
                for name in self.layer_wrappers.keys()
                if all(
                    any(c_part == n_part for n_part in name.split("."))
                    for c_part in candidate_parts
                )
            ]
            if matching_layers:
                layer_event_type = f"layer_after_{matching_layers[0].replace('.', '_')}"

                if getattr(module, "_target_layer", None) != matching_layers[0]:
                    module._target_layer = matching_layers[0]

                self.attach_module(
                    layer_event_type,
                    module,
                    module_config=module_config,
                    target_layer=matching_layers[0],
                    requested_layer=layer_name,
                    schedule=schedule,
                )
                logger.info(
                    "✅ MODULE_REGISTRY: Attached %s to %s (partial from %s)",
                    module,
                    layer_event_type,
                    layer_name,
                )
                return matching_layers[0]

        # Final fallback - attach to a similar layer, but require at least
        # 2 significant path segments to match to avoid wrong-layer attachment.
        requested_parts = [part for part in layer_name.split(".") if len(part) > 2]
        if len(requested_parts) >= 2:
            for available_layer in self.layer_wrappers.keys():
                available_parts = available_layer.split(".")
                matching_parts = sum(
                    1 for part in requested_parts if part in available_parts
                )
                if matching_parts >= 2:
                    layer_event_type = (
                        f"layer_after_{available_layer.replace('.', '_')}"
                    )

                    if getattr(module, "_target_layer", None) != available_layer:
                        module._target_layer = available_layer

                    self.attach_module(
                        layer_event_type,
                        module,
                        module_config=module_config,
                        target_layer=available_layer,
                        requested_layer=layer_name,
                        schedule=schedule,
                    )
                    logger.info(
                        "✅ MODULE_REGISTRY: Attached %s to %s (fallback from %s, matched %d parts)",
                        module,
                        layer_event_type,
                        layer_name,
                        matching_parts,
                    )
                    return available_layer

        logger.warning(
            "❌ MODULE_REGISTRY: No suitable layer for '%s' (available: %s)",
            layer_name,
            list(self.layer_wrappers.keys()),
        )
        return None

    def detach_module(self, event_type: str, module: Any) -> None:
        """
        Detach module from event type.

        Args:
            event_type: Event type to detach from
            module: Module instance to detach
        """
        if event_type in self.attached_modules:
            removed = False
            entries = self.attached_modules[event_type]
            for entry in list(entries):
                candidate = entry.get("module") if isinstance(entry, dict) else entry
                if candidate is module:
                    entries.remove(entry)
                    removed = True
                    logger.debug("Detached module %s from event %s", module, event_type)
            if not removed:
                logger.warning("Module %s not found in %s", module, event_type)

    def has_modules_attached(self, event_type: str) -> bool:
        """
        Check if any modules are attached to event type.

        Args:
            event_type: Event type to check

        Returns:
            True if modules are attached, False otherwise
        """
        return (
            event_type in self.attached_modules
            and len(self.attached_modules[event_type]) > 0
        )

    def get_attachment_entry(
        self, event_type: str, module: Any
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve the attachment entry for a module on a given event.

        Args:
            event_type: Event type to inspect
            module: Module instance being queried

        Returns:
            Attachment dictionary if found, otherwise None.
        """
        entries = self.attached_modules.get(event_type) or []
        for entry in entries:
            candidate = entry.get("module") if isinstance(entry, dict) else entry
            if candidate is module:
                return entry if isinstance(entry, dict) else {"module": candidate}
        return None

    def detach_all_modules(self) -> None:
        """
        Detach all modules across all event types.

        Useful for ensuring a clean registry state before attaching a fresh set
        of modules (for example, at the start of an inference request).
        """
        if not self.attached_modules:
            return

        logger.info(
            "MODULE_REGISTRY: Detaching %d event(s) prior to reset",
            len(self.attached_modules),
        )
        for event_type, entries in list(self.attached_modules.items()):
            for entry in list(entries):
                module_obj = entry.get("module") if isinstance(entry, dict) else entry
                if module_obj is None:
                    continue
                self.detach_module(event_type, module_obj)
        self.attached_modules.clear()
