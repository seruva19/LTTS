"""
Blueprint State Management - Complete system state serialization/deserialization.

Captures current model, attached modules, and all their configurations for save/load functionality.
"""

import json
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Handle pydantic import gracefully
try:
    from pydantic import BaseModel, Field  # type: ignore
except ImportError:
    # Fallback BaseModel for when pydantic is not available
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def dict(self):
            return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def Field(default_factory=None, **kwargs):  # type: ignore
        if default_factory:
            return default_factory()
        return None


class ModuleState(BaseModel):
    """State of an attached module."""

    name: str = ""
    event_type: str = ""
    parameters: Dict[str, Any] = {}
    ui_schema: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    attached_at: str = ""


class ModelState(BaseModel):
    """State of a loaded model."""

    model_id: str = ""
    engine: str = "transformers"
    loaded_at: str = ""
    model_class: Optional[str] = None
    layer_structure: Dict[str, Any] = {}
    attached_modules: List[ModuleState] = []


class BlueprintState(BaseModel):
    """Complete blueprint state capturing entire system configuration."""

    id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0.0"
    created_at: str = ""
    updated_at: str = ""
    thumbnail: Optional[str] = None  # Base64 encoded thumbnail image
    model_state: Optional[ModelState] = None
    global_modules: List[Any] = []
    client_config: Dict[str, Any] = {}
    execution_history: List[Any] = []
    tags: List[str] = []
    author: str = "LTTS User"


class BlueprintManager:
    """Manages blueprint state serialization and persistence."""

    def __init__(self, blueprints_dir: str = "blueprints"):
        self.blueprints_dir = Path(blueprints_dir)
        self.blueprints_dir.mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Internal sanitizers to keep module listings deduplicated

    def _clone_client_state(self, client_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if client_state is None:
            return None
        try:
            # Deep copy via JSON round-trip to avoid mutating caller state
            return json.loads(json.dumps(client_state))
        except Exception:
            # Fallback to shallow copy
            return client_state.copy() if isinstance(client_state, dict) else client_state

    def _sanitize_modules_list(self, modules: Optional[List[Any]]) -> List[Any]:
        if not isinstance(modules, list):
            return []

        sanitized: List[Any] = []
        seen: Dict[Any, Any] = {}

        for entry in modules:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("module", {}).get("name")
                layer_path = (
                    entry.get("layerPath")
                    or entry.get("layer_path")
                    or (entry.get("parameters") or {}).get("layer_path")
                )
                identifier = (name, layer_path)
            else:
                identifier = (str(entry), None)

            if identifier in seen:
                # Merge parameters if available
                existing = seen[identifier]
                if isinstance(entry, dict) and isinstance(existing, dict):
                    params = entry.get("parameters")
                    if isinstance(params, dict):
                        existing_params = existing.setdefault("parameters", {})
                        existing_params.update(params)
                continue

            seen[identifier] = entry
            sanitized.append(entry)

        return sanitized

    def _sanitize_layers(self, layers: Optional[List[Any]]) -> List[Any]:
        if not isinstance(layers, list):
            return []

        sanitized_layers: List[Any] = []
        for layer in layers:
            if isinstance(layer, dict):
                sanitized_layers.append(self._sanitize_layer(layer))
            else:
                sanitized_layers.append(layer)
        return sanitized_layers

    def _sanitize_layer(self, layer: Dict[str, Any]) -> Dict[str, Any]:
        for key in ("preModules", "postModules"):
            attachments = layer.get(key)
            if not isinstance(attachments, list):
                continue

            deduped: List[Any] = []
            seen: Dict[Any, Dict[str, Any]] = {}

            for attachment in attachments:
                if not isinstance(attachment, dict):
                    identifier = (str(attachment), None)
                    if identifier in seen:
                        continue
                    seen[identifier] = {"value": attachment}
                    deduped.append(attachment)
                    continue

                name = attachment.get("name") or attachment.get("module", {}).get("name")
                layer_path = (
                    attachment.get("layerPath")
                    or attachment.get("layer_path")
                    or layer.get("layerPath")
                    or layer.get("fullPath")
                    or layer.get("path")
                    or layer.get("name")
                )

                identifier = (name, layer_path)
                if identifier in seen:
                    existing = seen[identifier]
                    if isinstance(existing, dict):
                        params = attachment.get("parameters")
                        if isinstance(params, dict):
                            existing_params = existing.setdefault("parameters", {})
                            existing_params.update(params)
                    continue

                seen[identifier] = attachment
                deduped.append(attachment)

            layer[key] = deduped

        # Recursively sanitize nested layer collections if present
        nested_layers = layer.get("layers")
        if isinstance(nested_layers, list):
            layer["layers"] = self._sanitize_layers(nested_layers)

        return layer

    def _sanitize_blueprint_data(self, blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(blueprint_data, dict):
            return blueprint_data

        layers = blueprint_data.get("layers")
        if layers:
            blueprint_data["layers"] = self._sanitize_layers(layers)

        modules = blueprint_data.get("modules")
        if modules:
            blueprint_data["modules"] = self._sanitize_modules_list(modules)

        categorization = blueprint_data.get("categorization", {})
        if isinstance(categorization, dict):
            categories = categorization.get("categories")
            if isinstance(categories, dict):
                for category in categories.values():
                    if isinstance(category, dict):
                        category_layers = category.get("layers")
                        if isinstance(category_layers, list):
                            category["layers"] = self._sanitize_layers(category_layers)

        return blueprint_data

    def _sanitize_client_state(self, client_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(client_state, dict):
            return client_state

        # Sanitize top-level modules collection if present
        if "modules" in client_state:
            client_state["modules"] = self._sanitize_modules_list(client_state.get("modules"))

        blueprint_data = client_state.get("blueprint")
        if isinstance(blueprint_data, dict):
            client_state["blueprint"] = self._sanitize_blueprint_data(blueprint_data)

        model_state = client_state.get("model_state")
        if isinstance(model_state, dict):
            selected_model = model_state.get("selected_model")
            if isinstance(selected_model, dict):
                # No sanitization needed yet, but keep structure consistent
                model_state["selected_model"] = selected_model

        return client_state

    def capture_current_state(
        self,
        ltts_integration,
        name: str,
        description: str = "",
        client_state: Optional[Dict[str, Any]] = None,
    ) -> BlueprintState:
        """Capture current system state as a blueprint."""

        blueprint = BlueprintState(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        # Store client state if provided
        if client_state:
            sanitized_state = self._sanitize_client_state(self._clone_client_state(client_state))
            blueprint.client_config = sanitized_state or {}

            # Extract thumbnail if provided
            if (
                sanitized_state
                and "blueprint" in sanitized_state
                and isinstance(sanitized_state["blueprint"], dict)
                and "thumbnail" in sanitized_state["blueprint"]
            ):
                blueprint.thumbnail = sanitized_state["blueprint"]["thumbnail"]

        # Capture model state and blueprint data
        if client_state:
            # Handle blueprint data structure from frontend
            blueprint_data = (blueprint.client_config or {}).get("blueprint", {}) if blueprint.client_config else client_state.get("blueprint", {})
            model_state_data = (blueprint.client_config or {}).get("model_state", {}) if blueprint.client_config else client_state.get("model_state", {})

            # Extract layers and modules from blueprint data
            layers = blueprint_data.get("layers", []) if blueprint_data else []
            modules = blueprint_data.get("modules", []) if blueprint_data else []

            if layers:
                layers = self._sanitize_layers(layers)
            if modules:
                modules = self._sanitize_modules_list(modules)

            # Create model state if we have model information
            if model_state_data and "selected_model" in model_state_data:
                selected_model = model_state_data.get("selected_model", {})
                if selected_model and selected_model.get("loaded"):
                    model_id = selected_model.get("id")

                    # Create model state with layers and modules as dictionaries
                    blueprint.model_state = ModelState(
                        model_id=model_id,
                        engine=selected_model.get("engine", "transformers"),
                        loaded_at=selected_model.get(
                            "loadedAt", datetime.now().isoformat()
                        ),
                        model_class=(
                            blueprint_data.get("architecture", "Transformer")
                            if blueprint_data
                            else "Transformer"
                        ),
                        layer_structure={"layers": layers} if layers else {},
                        attached_modules=modules,  # Keep as raw dictionaries
                    )

        return blueprint

    def _capture_attached_modules(
        self,
        ltts_integration,
        model_id: str,
        client_state: Optional[Dict[str, Any]] = None,
    ) -> List[ModuleState]:
        """
        DEPRECATED: Modules are now client-side only.
        This method is kept for compatibility but returns empty list.
        Module state is stored in blueprint.client_config.modules instead.
        """
        return []

    def _get_module_ui_schema(self, module_obj) -> Dict[str, Any]:
        """Extract UI schema from module."""
        try:
            if hasattr(module_obj, "get_ui_schema"):
                return module_obj.get_ui_schema()
            elif hasattr(module_obj, "METADATA") and isinstance(
                module_obj.METADATA, dict
            ):
                editor_name = module_obj.METADATA.get("methods", {}).get("editor")
                if editor_name and hasattr(module_obj, editor_name):
                    editor_method = getattr(module_obj, editor_name)
                    return editor_method()
        except Exception as e:
            # Log and continue with empty UI schema
            logger.warning("Failed to get module UI schema: %s", e)
        return {}

    def _get_module_current_parameters(self, module_obj) -> Dict[str, Any]:
        """Return persisted runtime parameters, if supported."""
        # Module runtime parameters are not persisted.
        return {}

    def save_blueprint(self, blueprint: BlueprintState) -> str:
        """Save blueprint to file."""
        try:
            # A display name is untrusted input and must never become a path.
            # Use a deterministic digest of the id instead; the human-readable
            # name remains inside the JSON document.  This also works for old
            # or imported blueprints whose ids are not UUIDs.
            safe_id = hashlib.sha256(str(blueprint.id).encode("utf-8")).hexdigest()[:24]
            filename = f"blueprint_{safe_id}.json"
            filepath = self.blueprints_dir / filename

            # Update timestamp
            blueprint.updated_at = datetime.now().isoformat()

            # Save to file
            with open(filepath, "w", encoding="utf-8") as f:
                payload = (
                    blueprint.model_dump()
                    if hasattr(blueprint, "model_dump")
                    else blueprint.dict()
                )
                json.dump(payload, f, indent=2, ensure_ascii=False)

            return str(filepath)
        except Exception as e:
            raise Exception(f"Failed to save blueprint: {str(e)}")

    def load_blueprint(self, blueprint_id: str) -> Optional[BlueprintState]:
        """Load blueprint from file by ID."""
        if not blueprint_id:
            raise ValueError("Blueprint ID is required")

        for filepath in self.blueprints_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("id") == blueprint_id:
                        return BlueprintState(**data)
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in %s: %s", filepath, e)
            except Exception as e:
                logger.error("Error loading blueprint from %s: %s", filepath, e)
        return None

    def load_blueprint_by_name(self, name: str) -> Optional[BlueprintState]:
        """Load blueprint from file by name."""
        for filepath in self.blueprints_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("name") == name:
                        return BlueprintState(**data)
            except Exception as e:
                logger.error("Error loading blueprint by name from %s: %s", filepath, e)
        return None

    def list_blueprints(self) -> List[BlueprintState]:
        """List all available blueprints."""
        blueprints = []

        for filepath in self.blueprints_dir.glob("*.json"):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    blueprints.append(BlueprintState(**data))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in %s: %s", filepath, e)
            except Exception as e:
                logger.error("Error listing blueprint from %s: %s", filepath, e)
        return sorted(
            blueprints,
            key=lambda b: b.updated_at if isinstance(b.updated_at, str) else "",
            reverse=True,
        )

    def delete_blueprint(self, blueprint_id: str) -> bool:
        """Delete blueprint by ID."""
        if not blueprint_id:
            raise ValueError("Blueprint ID is required")

        for filepath in self.blueprints_dir.glob("*.json"):
            try:
                found_match = False
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("id") == blueprint_id:
                        found_match = True

                if found_match:
                    filepath.unlink()
                    return True
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON in %s: %s", filepath, e)
            except Exception as e:
                logger.error("Error deleting blueprint from %s: %s", filepath, e)

        return False

    def restore_blueprint(self, blueprint: BlueprintState, ltts_integration) -> bool:
        """
        Restore system state from blueprint.

        Since modules are now client-side, this method primarily validates
        that the blueprint can be loaded. The actual module restoration
        happens on the frontend when the client_config is applied.
        """
        try:
            if not blueprint:
                raise ValueError("Blueprint is required")

            # Validate that we have client config with modules
            if not blueprint.client_config:
                return True

            # Check if blueprint has module configuration in client_config
            modules = blueprint.client_config.get("modules", [])
            if modules:
                for module in modules:
                    if isinstance(module, dict):
                        module_name = module.get("name", "unknown")
                        # Placeholder for any server-side validation/logging
                        _ = module_name

            # The actual restoration happens client-side when the frontend
            # receives the blueprint.client_config and applies it
            return True

        except Exception as e:
            logger.error("Error restoring blueprint: %s", e)
            return False


# Global blueprint manager instance
_global_blueprint_manager = None


def get_blueprint_manager() -> BlueprintManager:
    """Get the global blueprint manager instance."""
    global _global_blueprint_manager

    if _global_blueprint_manager is None:
        _global_blueprint_manager = BlueprintManager()

    return _global_blueprint_manager
