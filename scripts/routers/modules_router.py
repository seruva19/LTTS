import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field

from scripts.core.ltts_module import ModuleEventRequest
from scripts.services.model_compatibility import module_attachment_contract

logger = logging.getLogger(__name__)


class ModulesConfig(BaseModel):
    modules: Dict[str, Any]


class ExecuteModuleInput(BaseModel):
    module: str
    model_id: str
    engine: str
    params: Dict[str, Any] = Field(default_factory=dict)
    module_path: Optional[str] = None
    layer_index: Optional[int] = None
    client_token: Optional[str] = None


def create_modules_router(context):
    """Create and configure the modules router with context dependency injection."""
    router = APIRouter()

    @router.get("/api/modules", response_model=ModulesConfig)
    async def get_modules():
        # Use the unified module loader as the primary source
        serializable_modules = {}

        # Get all modules from the module loader (primary source)
        for name, module_info in context.module_loader.get_all_modules().items():
            metadata = module_info.get("metadata", {})
            serializable_modules[name] = {
                "metadata": metadata,
                "attachment_contract": module_attachment_contract(name, metadata),
                "ui_schema": module_info.get("ui_schema", {}),
                "has_stateful_events": module_info.get("has_stateful_events", False),
                "control_buttons": module_info.get("control_buttons", []),
                "registered_routes": module_info.get("registered_routes", []),
            }

        return ModulesConfig(modules=serializable_modules)

    @router.post("/api/execute_module")
    async def execute_module(input: ExecuteModuleInput):
        """
        Deprecated: module attachment is handled client-side.
        This endpoint is kept for backward compatibility but does nothing.
        Modules are attached via client-side state and executed during inference.
        """
        logger.warning(
            f"DEPRECATED: /api/execute_module called for module '{input.module}' - "
            "attachment is now handled client-side"
        )

        # Return success to avoid breaking existing client code
        token = input.client_token or "default"
        return {
            "status": "ok",
            "result": f"Module '{input.module}' registered (client-side attachment)",
            "client_token": token,
            "deprecated": True,
            "message": "Module attachment is now handled client-side",
        }

    @router.get("/api/modules/{module_name}/buttons")
    async def get_module_buttons(module_name: str):
        """Get control buttons for a specific module."""
        try:
            buttons = context.module_loader.get_module_control_buttons(module_name)
            return {"status": "success", "module_name": module_name, "buttons": buttons}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get buttons for module {module_name}: {str(e)}",
            )

    @router.get("/api/modules/buttons")
    async def get_all_module_buttons():
        """Get control buttons for all modules."""
        try:
            all_buttons = context.module_loader.get_all_control_buttons()
            return {"status": "success", "modules": all_buttons}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to get module buttons: {str(e)}"
            )

    @router.get("/api/modules/{module_name}/routes")
    async def get_module_routes(module_name: str):
        """Get registered routes for a specific module."""
        try:
            routes = context.module_loader.get_module_routes(module_name)
            return {"status": "success", "module_name": module_name, "routes": routes}
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to get routes for module {module_name}: {str(e)}",
            )

    @router.get("/api/modules/{module_name}/state")
    async def get_module_state(module_name: str):
        """Get current state of a stateful module."""
        try:
            module_instance = context.module_loader.call_module_event(
                module_name, {}, {}
            )
            return {
                "status": "success",
                "module_name": module_name,
                "state": module_instance.get_state(),
            }
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found or not stateful")
        except Exception as e:
            logger.error("Failed to get state for module %s: %s", module_name, e)
            raise HTTPException(status_code=500, detail="Failed to get module state")

    @router.post("/api/modules/{module_name}/state")
    async def update_module_state(
        module_name: str, new_state: Dict[str, Any] = Body(...)
    ):
        """Update state of a stateful module."""
        try:
            module_instance = context.module_loader.call_module_event(
                module_name, {}, new_state
            )
            return {
                "status": "success",
                "module_name": module_name,
                "state": module_instance.get_state(),
            }
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Module '{module_name}' not found or not stateful")
        except Exception as e:
            logger.error("Failed to update state for module %s: %s", module_name, e)
            raise HTTPException(status_code=500, detail="Failed to update module state")

    return router
