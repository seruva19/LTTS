import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from scripts.blueprints.blueprint_loader import BlueprintsConfig

logger = logging.getLogger(__name__)


class BlueprintSaveRequest(BaseModel):
    name: str
    description: str = ""
    model_id: Optional[str] = None
    client_state: Optional[Dict[str, Any]] = None


class BlueprintLoadRequest(BaseModel):
    blueprint_id: str


def create_blueprints_router(context):
    router = APIRouter()

    @router.get("/api/blueprints", response_model=BlueprintsConfig)
    async def get_blueprints():
        return context.blueprints

    @router.post("/api/blueprint/save")
    async def save_blueprint(request: BlueprintSaveRequest):
        try:
            from scripts.blueprints.blueprint_state import get_blueprint_manager

            blueprint_manager = get_blueprint_manager()
            ltts_integration = context.ltts_integration
            if ltts_integration is None:
                raise RuntimeError("LTTS integration is not initialized")

            blueprint = blueprint_manager.capture_current_state(
                ltts_integration,
                request.name,
                request.description,
                request.client_state,
            )

            if (
                request.model_id
                and hasattr(ltts_integration, "wrapped_models")
                and request.model_id in ltts_integration.wrapped_models
            ):
                try:
                    ltts_model = ltts_integration.wrapped_models[request.model_id]
                    if blueprint.model_state:
                        blueprint.model_state.model_id = request.model_id
                        blueprint.model_state.model_class = (
                            type(ltts_model.original_model).__name__
                            if hasattr(ltts_model, "original_model")
                            else "Unknown"
                        )
                except Exception as e:
                    logger.warning("Failed to attach model info to blueprint: %s", e)

            filepath = blueprint_manager.save_blueprint(blueprint)

            return {
                "status": "success",
                "blueprint_id": blueprint.id,
                "name": blueprint.name,
                "filepath": filepath,
                "created_at": blueprint.created_at,
                "has_model": blueprint.model_state is not None,
                "module_count": (
                    len(blueprint.client_config.get("modules", []))
                    if blueprint.client_config
                    else 0
                ),
            }

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to save blueprint: {str(e)}"
            )

    @router.post("/api/blueprint/load")
    async def load_blueprint(request: BlueprintLoadRequest):
        try:
            from scripts.blueprints.blueprint_state import get_blueprint_manager

            blueprint_manager = get_blueprint_manager()

            blueprint = blueprint_manager.load_blueprint(request.blueprint_id)
            if not blueprint:
                raise HTTPException(status_code=404, detail="Blueprint not found")

            return {
                "status": "success",
                "blueprint": blueprint.dict(),
                "message": f"Blueprint '{blueprint.name}' loaded successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to load blueprint: {str(e)}"
            )

    @router.get("/api/blueprints/list")
    async def list_blueprints():
        try:
            from scripts.blueprints.blueprint_state import get_blueprint_manager

            blueprint_manager = get_blueprint_manager()
            blueprints = blueprint_manager.list_blueprints()

            return {
                "status": "success",
                "blueprints": [
                    {
                        "id": bp.id,
                        "name": bp.name,
                        "description": bp.description,
                        "created_at": bp.created_at,
                        "updated_at": bp.updated_at,
                        "has_model": bp.model_state is not None,
                        "module_count": (
                            len(bp.model_state.attached_modules)
                            if bp.model_state
                            else 0
                        ),
                        "tags": bp.tags,
                        "thumbnail": bp.thumbnail if hasattr(bp, "thumbnail") else None,
                    }
                    for bp in blueprints
                ],
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Error listing blueprints: {str(e)}",
                "blueprints": [],
            }

    @router.delete("/api/blueprint/{blueprint_id}")
    async def delete_blueprint(blueprint_id: str):
        try:
            from scripts.blueprints.blueprint_state import get_blueprint_manager

            blueprint_manager = get_blueprint_manager()
            success = blueprint_manager.delete_blueprint(blueprint_id)

            if not success:
                raise HTTPException(status_code=404, detail="Blueprint not found")

            return {
                "status": "success",
                "message": f"Blueprint {blueprint_id} deleted successfully",
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to delete blueprint: {str(e)}"
            )

    return router
