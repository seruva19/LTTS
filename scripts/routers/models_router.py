from datetime import datetime
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from scripts.models.model_loader import ModelConfigItem
from scripts.models.model_struct import ModelInput, model_to_json

logger = logging.getLogger(__name__)


class ModelsResponse(BaseModel):
    models: Dict[str, ModelConfigItem]


class LoadModelInput(BaseModel):
    model_id: str
    engine: str
    params: Dict[str, Any] = Field(default_factory=dict)


class AddModelInput(BaseModel):
    model_id: str
    task: Optional[str] = None
    # Fetches config.json (a few KB) to confirm the repository exists and to
    # infer the task. Turn off for a model that is not reachable right now.
    # Not called `validate`: that name shadows a BaseModel attribute.
    check_remote: bool = True


def create_models_router(context):
    """Create and configure the models router with context dependency injection."""
    router = APIRouter()

    @router.get("/api/models", response_model=ModelsResponse)
    async def get_models():
        return ModelsResponse(models=context.models.config_list)

    @router.post("/api/models")
    async def add_model(input: AddModelInput):
        """Register a Hugging Face model id in the personal registry."""
        try:
            return {
                "status": "success",
                **context.models.add_model(
                    input.model_id, task=input.task, validate=input.check_remote
                ),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            # A bad repository id, a gated repo or no network all land here, and
            # the message is the only thing that tells the user which it was.
            logger.error("Failed to register model '%s': %s", input.model_id, exc)
            raise HTTPException(
                status_code=400,
                detail=f"Could not register '{input.model_id}': {exc}",
            ) from exc

    @router.delete("/api/models/{model_id:path}")
    async def remove_model(model_id: str):
        try:
            context.models.remove_model(model_id)
            return {"status": "success", "model_id": model_id}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Failed to remove model '%s': %s", model_id, exc)
            raise HTTPException(
                status_code=500, detail=f"Could not remove '{model_id}': {exc}"
            ) from exc

    @router.post("/api/blueprint_from_model")
    async def get_model_structure(model_input: ModelInput):
        model_id = model_input.id
        engine = model_input.engine

        try:
            if not engine:
                engine = "transformers"  # Default engine if not specified

            logger.info(
                "Loading model structure for '%s' with engine '%s'", model_id, engine
            )
            _, struct = context.models.load_model(model_id, engine)
            model_struct = model_to_json(struct)
            return {"model": model_struct}
        except Exception as e:
            logger.error("Error loading model structure: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to load model '{model_id}': {str(e)}"
            )

    @router.post("/api/models/load")
    async def load_model(input: LoadModelInput):
        try:
            # Check if model is already loaded
            existing_model = context.models.get_model(input.model_id)

            if (
                existing_model
                and existing_model.loaded
                and existing_model.model is not None
            ):
                logger.info(
                    "Model '%s' is already loaded, reusing existing instance",
                    input.model_id,
                )

                # Ensure the model is also wrapped in LTTS integration
                if input.model_id not in context.ltts_integration.wrapped_models:
                    context.ltts_integration.wrap_model(
                        existing_model.model, input.model_id
                    )
                    logger.info(
                        "Wrapped existing model '%s' with LTTS integration",
                        input.model_id,
                    )

                return {
                    "status": "ok",
                    "model_id": input.model_id,
                    "struct": model_to_json(existing_model.struct),
                    "reused": True,
                }

            # If model exists but is not loaded, unload it first to clear memory
            if existing_model and not existing_model.loaded:
                logger.info("Clearing unloaded model instance for '%s'", input.model_id)
                context.models.unload_model(input.model_id)

            # Load the model fresh
            logger.info(
                "Loading model '%s' with engine '%s'", input.model_id, input.engine
            )
            model, struct = context.models.load_model(
                input.model_id, input.engine, **input.params
            )

            # Wrap model with unified LTTS system
            if model is not None:
                context.ltts_integration.wrap_model(model, input.model_id)
                logger.info(
                    "Successfully loaded and wrapped model '%s'", input.model_id
                )

            return {
                "status": "ok",
                "model_id": input.model_id,
                "struct": model_to_json(struct),
                "reused": False,
            }

        except Exception as e:
            logger.error("Failed to load model '%s': %s", input.model_id, e)

            # Send error to clients via WebSocket
            await context.message_bus.publish(
                "default",
                {
                    "type": "model_load_error",
                    "model_id": input.model_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            raise HTTPException(
                status_code=500, detail=f"Failed to load model: {str(e)}"
            )

    return router
