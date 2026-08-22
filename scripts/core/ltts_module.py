"""
LTTS Module system - metadata-driven drop-in modules.

Modules are simple Python files with metadata that define callable methods.
No base class required - just drop a .py file into ltts_modules/ and it works.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import importlib.util
import inspect
import logging
import os
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModuleEventRequest(BaseModel):
    """Request model for module stateful events."""

    module_name: str
    event_data: Dict[str, Any] = {}
    module_state: Dict[str, Any] = {}
    client_token: Optional[str] = None


class ModuleControlButton(BaseModel):
    """Button control definition for modules."""

    id: str
    label: str
    route: str
    method: str = "POST"
    icon: Optional[str] = None
    variant: str = "primary"  # primary, secondary, danger, etc.
    disabled: bool = False
    tooltip: Optional[str] = None


class LTTSModuleBase:
    """Base class for LTTS modules with stateful event support."""

    def __init__(self, fastapi_app: Optional[APIRouter] = None, module_name: str = ""):
        self.fastapi_app = fastapi_app
        self.module_name = module_name
        self.module_state = {}
        self.registered_routes = []
        self.control_buttons = []

        # Auto-register routes if FastAPI app is provided
        if self.fastapi_app:
            self._register_module_routes()

    def register_route(self, route: str, handler_method: str, method: str = "POST"):
        """Register a custom route for this module."""
        if not self.fastapi_app:
            logger.warning("Cannot register route %s - no FastAPI app provided", route)
            return

        # Create the full route path
        full_route = f"/api/modules/{self.module_name}/events{route}"

        # Get the handler method from the module
        if not hasattr(self, handler_method):
            logger.error(
                "Handler method %s not found in module %s",
                handler_method,
                self.module_name,
            )
            return

        handler = getattr(self, handler_method)

        # Create wrapper function that handles the request
        async def route_wrapper(request: ModuleEventRequest):
            try:
                # Update module state with incoming state
                self.module_state.update(request.module_state)

                # Call the handler method with event data and current state
                result = await self._call_handler_safely(
                    handler, request.event_data, self.module_state
                )

                return {
                    "status": "success",
                    "result": result,
                    "module_state": self.module_state,
                    "module_name": self.module_name,
                }
            except Exception as e:
                logger.error(
                    "Error in module %s route %s: %s", self.module_name, route, e
                )
                raise HTTPException(status_code=500, detail="Module route execution failed")

        # Register the route with FastAPI
        if method.upper() == "POST":
            self.fastapi_app.post(full_route)(route_wrapper)
        elif method.upper() == "GET":
            self.fastapi_app.get(full_route)(route_wrapper)
        elif method.upper() == "PUT":
            self.fastapi_app.put(full_route)(route_wrapper)
        elif method.upper() == "DELETE":
            self.fastapi_app.delete(full_route)(route_wrapper)

        self.registered_routes.append(
            {"route": full_route, "method": method.upper(), "handler": handler_method}
        )

        logger.info(
            "Registered route %s (%s) for module %s",
            full_route,
            method.upper(),
            self.module_name,
        )

    def add_stateful_button(self, button_config: Dict[str, Any], handler_method: str):
        """Add a stateful button that will be included in UI schema and bound to a route."""
        route = button_config.get("route", f"/{button_config['id']}")

        # Register the route for this button
        self.register_route(route, handler_method)

        # Store button config for UI schema
        button_config["route"] = f"/api/modules/{self.module_name}/events{route}"
        button_config["handler_method"] = handler_method
        self.control_buttons.append(button_config)

        logger.info(
            "Added stateful button '%s' for module %s",
            button_config.get("label", button_config["id"]),
            self.module_name,
        )

    def get_control_buttons(self) -> List[Dict[str, Any]]:
        """Get all control buttons for this module."""
        return self.control_buttons.copy()

    def get_registered_routes(self) -> List[Dict[str, Any]]:
        """Get all registered routes for this module."""
        return self.registered_routes.copy()

    def update_state(self, new_state: Dict[str, Any]):
        """Update module state."""
        self.module_state.update(new_state)

    def get_state(self) -> Dict[str, Any]:
        """Get current module state."""
        return self.module_state.copy()

    async def _call_handler_safely(
        self,
        handler: Callable,
        event_data: Dict[str, Any],
        module_state: Dict[str, Any],
    ) -> Any:
        """Safely call a handler method, supporting both sync and async."""
        try:
            if inspect.iscoroutinefunction(handler):
                return await handler(event_data, module_state)
            else:
                return handler(event_data, module_state)
        except Exception as e:
            logger.error("Error calling handler %s: %s", handler.__name__, e)
            raise

    def _register_module_routes(self):
        """Override this method to register custom routes during initialization."""
        pass

    def get_ui_schema_with_buttons(self) -> Dict[str, Any]:
        """Get UI schema including stateful buttons. Override this in your module."""
        return {"controls": [], "stateful_buttons": self.control_buttons}


def load_module_from_file(
    module_path: str, module_name: str, fastapi_app: Optional[APIRouter] = None
) -> Optional[Dict[str, Any]]:
    """
    Load a module from a Python file and extract its metadata and callables.

    Args:
        module_path: Path to the .py file
        module_name: Name to assign to the module

    Returns:
        Dictionary containing module info, metadata, and callable methods
    """
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            return None

        # Ensure package modules support relative imports
        if os.path.basename(module_path) in {"__init__.py", "init.py"}:
            package_dir = os.path.dirname(module_path)
            if spec.submodule_search_locations is None:
                spec.submodule_search_locations = [package_dir]
            elif package_dir not in spec.submodule_search_locations:
                spec.submodule_search_locations.append(package_dir)

        # Reuse only a module that was actually loaded from this file. Importing
        # ``ltts_modules.<name>.init`` first can create an empty namespace parent
        # under ``ltts_modules.<name>``; treating that namespace as the analysis
        # module makes the loader report a false "missing METADATA" failure.
        existing = sys.modules.get(module_name)
        existing_file = getattr(existing, "__file__", None) if existing else None
        same_source = False
        if existing_file:
            try:
                same_source = os.path.samefile(existing_file, module_path)
            except (OSError, ValueError):
                same_source = os.path.abspath(existing_file) == os.path.abspath(
                    module_path
                )
        if existing is not None and same_source and hasattr(existing, "METADATA"):
            module = existing
            logger.info(
                "Module %s already loaded, reusing existing module", module_name
            )
        else:
            module = importlib.util.module_from_spec(spec)
            # Insert into sys.modules BEFORE exec_module to prevent recursive import issues
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

        # Extract metadata (required)
        if not hasattr(module, "METADATA"):
            logger.warning("Module %s missing METADATA - skipping", module_name)
            return None

        metadata = module.METADATA

        # Validate required metadata fields
        required_fields = ["name", "version", "event_types"]
        for field in required_fields:
            if field not in metadata:
                logger.warning(
                    "Module %s missing required metadata field '%s' - skipping",
                    module_name,
                    field,
                )
                return None

        # Check if module has a class that inherits from LTTSModuleBase
        module_instance = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                inspect.isclass(attr)
                and issubclass(attr, LTTSModuleBase)
                and attr != LTTSModuleBase
            ):
                # Instantiate the module class with FastAPI app
                try:
                    module_instance = attr(
                        fastapi_app=fastapi_app, module_name=module_name
                    )
                    logger.info(
                        "Instantiated stateful module class %s for %s",
                        attr_name,
                        module_name,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        "Failed to instantiate module class %s: %s", attr_name, e
                    )

        # Extract callable methods based on metadata
        callables = {}
        method_names = metadata.get("methods", {})

        for method_type, method_name in method_names.items():
            if hasattr(module, method_name) and callable(getattr(module, method_name)):
                callables[method_type] = getattr(module, method_name)
            else:
                logger.warning(
                    "Module %s declares method '%s' for '%s' but it's not found or not callable",
                    module_name,
                    method_name,
                    method_type,
                )

        # Extract UI schema via utor method
        ui_schema = {}
        if "utor" in callables:
            try:
                ui_schema = callables["utor"]()

                # If module has stateful instance, merge its buttons into UI schema
                if module_instance:
                    stateful_buttons = module_instance.get_control_buttons()
                    if stateful_buttons:
                        ui_schema.setdefault("stateful_buttons", []).extend(
                            stateful_buttons
                        )
                        logger.info(
                            "Added %s stateful buttons to UI schema for %s",
                            len(stateful_buttons),
                            module_name,
                        )

            except Exception as e:
                logger.warning(
                    "Error calling utor method for UI schema in %s: %s", module_name, e
                )

        result = {
            "name": metadata["name"],
            "metadata": metadata,
            "ui_schema": ui_schema,
            "callables": callables,
            "module_object": module,
            "file_path": module_path,
            "module_instance": module_instance,
            "has_stateful_events": module_instance is not None,
        }

        # Add control buttons and routes if module instance exists
        if module_instance:
            result["control_buttons"] = module_instance.get_control_buttons()
            result["registered_routes"] = module_instance.get_registered_routes()

            # Ensure UI schema includes stateful buttons
            if (
                not ui_schema.get("stateful_buttons")
                and module_instance.get_control_buttons()
            ):
                ui_schema["stateful_buttons"] = module_instance.get_control_buttons()
                result["ui_schema"] = ui_schema

        return result

    except Exception as e:
        logger.error("Error loading module %s from %s: %s", module_name, module_path, e)
        return None


def call_module_method(
    module_info: Dict[str, Any], method_type: str, *args, **kwargs
) -> Any:
    """
    Call a specific method on a loaded module.

    Args:
        module_info: Module info dict from load_module_from_file
        method_type: Type of method to call (e.g., 'interceptor', 'ctor', 'dtor')
        *args, **kwargs: Arguments to pass to the method

    Returns:
        Result of the method call, or None if method not found
    """
    if method_type not in module_info["callables"]:
        return None

    try:
        method = module_info["callables"][method_type]
        return method(*args, **kwargs)
    except Exception as e:
        logger.error(
            "Error calling %s on module %s: %s", method_type, module_info["name"], e
        )
        return None


def validate_module_metadata(metadata: Dict[str, Any]) -> List[str]:
    """
    Validate module metadata and return list of errors.

    Args:
        metadata: Module metadata dictionary

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required fields
    required_fields = {"name": str, "version": str, "event_types": list}

    for field, expected_type in required_fields.items():
        if field not in metadata:
            errors.append(f"Missing required field: {field}")
        elif not isinstance(metadata[field], expected_type):
            errors.append(f"Field {field} must be of type {expected_type.__name__}")

    # Optional but validated fields
    if "methods" in metadata:
        if not isinstance(metadata["methods"], dict):
            errors.append("Field 'methods' must be a dictionary")
        else:
            valid_method_types = ["ctor", "dtor", "ntor", "utor"]
            for method_type in metadata["methods"]:
                if method_type not in valid_method_types:
                    errors.append(f"Unknown method type: {method_type}")

    if "dependencies" in metadata and not isinstance(metadata["dependencies"], list):
        errors.append("Field 'dependencies' must be a list")

    return errors
