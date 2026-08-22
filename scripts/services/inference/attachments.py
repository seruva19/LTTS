from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from scripts.core.emission_controller import update_expected_passes
from scripts.services.model_compatibility import module_required_capabilities

logger = logging.getLogger(__name__)


@dataclass
class AttachmentSpec:
    layer_name: str
    requested_layer_name: str
    module_name: str
    module: Any
    parameters: Dict[str, Any]
    event_type: str


class ModuleAttachmentManager:
    """Handle module attachment lifecycle for an inference run."""

    def __init__(self, context, ltts_model):
        self.context = context
        self.ltts_model = ltts_model
        self.logger = logger

    def reset_attachments(self) -> None:
        try:
            self.ltts_model.module_registry.detach_all_modules()
        except Exception as reset_error:  # pragma: no cover - defensive logging
            self.logger.warning(
                "⚠ INFERENCE: Failed to reset module attachments before run: %s",
                reset_error,
            )

    def attach_modules(
        self, modules: Optional[Dict[str, List[Any]]]
    ) -> Tuple[List[AttachmentSpec], Set[str]]:
        attached_module_specs: List[AttachmentSpec] = []
        required_capabilities: Set[str] = set()

        attached_signatures = set()
        resolved_signatures = set()

        if not modules:
            self.logger.debug("?? INFERENCE: No modules specified for attachment")
            self.logger.info("ℹ INFERENCE: No modules were attached for this run")
            return attached_module_specs, required_capabilities

        self.logger.debug("🔧 INFERENCE: Attaching modules to layers: %s", modules)
        self.logger.debug("🔧 INFERENCE: Total layers to attach: %s", len(modules))

        for layer_name, module_entries in modules.items():
            if not module_entries:
                self.logger.debug(
                    "🔧 INFERENCE: Layer '%s' has no module entries", layer_name
                )
                continue

            if not isinstance(module_entries, list):
                module_entries = [module_entries]

            self.logger.debug(
                "🔧 INFERENCE: Processing layer '%s' with modules: %s",
                layer_name,
                module_entries,
            )

            for raw_entry in module_entries:
                module_name = None
                module_params: Dict[str, Any] = {}

                if isinstance(raw_entry, dict):
                    module_name = (
                        raw_entry.get("name")
                        or raw_entry.get("module")
                        or raw_entry.get("id")
                    )
                    module_params = dict(raw_entry.get("parameters") or {})
                    module_params.setdefault("layer_path", layer_name)
                else:
                    module_name = raw_entry

                if not module_name:
                    self.logger.warning(
                        "⚠ INFERENCE: Skipping module entry without name for layer %s: %s",
                        layer_name,
                        raw_entry,
                    )
                    continue

                requested_signature = (str(module_name).lower(), str(layer_name))
                if requested_signature in attached_signatures:
                    self.logger.debug(
                        "🔧 INFERENCE: Skipping duplicate module request %s on %s",
                        module_name,
                        layer_name,
                    )
                    continue

                try:
                    module_info = self.context.modules_store.get_module(module_name)
                    attached_signatures.add(requested_signature)

                    if not module_info or "module" not in module_info:
                        self.logger.warning(
                            "⚠ INFERENCE: Module %s not found or invalid", module_name
                        )
                        continue

                    module_instance = module_info["module"]
                    metadata = module_info.get("metadata", {}) or {}
                    meta_name = str(metadata.get("name", module_name)).lower()
                    meta_tags = [
                        str(tag).lower() for tag in (metadata.get("tags") or [])
                    ]
                    self.logger.debug(
                        "🔧 INFERENCE: Checking module %s: meta_name='%s', meta_tags=%s",
                        module_name,
                        meta_name,
                        meta_tags,
                    )

                    required_capabilities.update(
                        module_required_capabilities(str(module_name), metadata)
                    )
                    module_params.setdefault("expected_generation_passes", 1)

                    event_types = metadata.get("event_types", []) or []
                    is_model_level = metadata.get("module_type") == "model_level"
                    if not is_model_level:
                        is_model_level = any(
                            str(event_type).startswith("model_")
                            for event_type in event_types
                        )
                    if is_model_level:
                        if not event_types:
                            event_types = ["model_after"]
                        for event_type in event_types:
                            if not str(event_type).startswith("model_"):
                                continue
                            self.ltts_model.attach_module(
                                event_type,
                                module_instance,
                                module_config=module_params,
                                target_layer="model",
                                requested_layer=layer_name,
                            )
                            attached_module_specs.append(
                                AttachmentSpec(
                                    layer_name="model",
                                    requested_layer_name=layer_name,
                                    module_name=module_name,
                                    module=module_instance,
                                    parameters=module_params,
                                    event_type=event_type,
                                )
                            )
                        continue

                    resolved_layer_path = self.ltts_model.attach_module_to_layer(
                        layer_name,
                        module_instance,
                        module_config=module_params,
                    )

                    if not resolved_layer_path:
                        self.logger.warning(
                            "⚠ INFERENCE: Unable to resolve layer '%s' for module %s",
                            layer_name,
                            module_name,
                        )
                        continue

                    resolved_key = (
                        str(module_name).lower(),
                        str(resolved_layer_path),
                    )
                    if resolved_key in resolved_signatures:
                        continue

                    layer_event_type = (
                        f"layer_after_{resolved_layer_path.replace('.', '_')}"
                    )

                    self.logger.debug(
                        "✅ INFERENCE: Attached %s to %s (requested %s)",
                        module_name,
                        resolved_layer_path,
                        layer_name,
                    )

                    attached_signatures.add((str(module_name).lower(), resolved_layer_path))
                    resolved_signatures.add(resolved_key)

                    if module_params and hasattr(module_instance, "set_runtime_config"):
                        try:
                            module_instance.set_runtime_config(
                                resolved_layer_path, module_params
                            )
                            self.logger.debug(
                                "🔧 INFERENCE: Applied runtime config for %s on %s: %s",
                                module_name,
                                resolved_layer_path,
                                module_params,
                            )
                            update_expected_passes(
                                module_name,
                                resolved_layer_path,
                                module_params.get("expected_generation_passes"),
                            )
                        except Exception as config_error:
                            self.logger.warning(
                                "⚠ INFERENCE: Failed to apply runtime config for %s on %s: %s",
                                module_name,
                                resolved_layer_path,
                                config_error,
                            )

                    attached_module_specs.append(
                        AttachmentSpec(
                            layer_name=resolved_layer_path,
                            requested_layer_name=layer_name,
                            module_name=module_name,
                            module=module_instance,
                            parameters=module_params,
                            event_type=layer_event_type,
                        )
                    )

                except Exception as attach_error:  # pragma: no cover - defensive
                    self.logger.error(
                        "❌ INFERENCE: Failed to attach %s to %s: %s",
                        module_name,
                        layer_name,
                        attach_error,
                    )

        if attached_module_specs:
            attached_layers = {spec.layer_name for spec in attached_module_specs}
            self.logger.info(
                "✅ INFERENCE: Attached %d module(s) across %d layer(s)",
                len(attached_module_specs),
                len(attached_layers),
            )
        else:
            self.logger.info("ℹ INFERENCE: No modules were attached for this run")

        return attached_module_specs, required_capabilities
