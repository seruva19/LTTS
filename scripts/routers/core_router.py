import logging
from typing import Any, Dict
from fastapi import APIRouter
import yaml
from pydantic import BaseModel
from scripts.configs import user_files


from scripts.configs.user_config import UserConfig

logger = logging.getLogger(__name__)


def create_core_router(context):
    """Create and configure the core router with context dependency injection."""
    router = APIRouter()

    @router.get("/api/config")
    async def load_user_config() -> UserConfig:
        return context.user_config

    @router.post("/api/config/reload")
    async def reload_config() -> UserConfig:
        config_path = user_files.USER_CONFIG
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        context.user_config = UserConfig(**config)
        return context.user_config

    @router.get("/api/system/status")
    async def get_system_status():
        """Get current system status for debugging."""
        try:
            status = {"models": {}, "integration": {}, "modules": {}}

            # Check LTTS integration
            try:
                ltts_integration = context.ltts_integration
                if ltts_integration is None:
                    raise RuntimeError("LTTS integration is not initialized")

                status["integration"]["available"] = True
                status["integration"]["wrapped_models"] = list(
                    ltts_integration.wrapped_models.keys()
                )

                # Get model details
                for model_id, ltts_model in ltts_integration.wrapped_models.items():
                    status["models"][model_id] = {
                        "class": type(ltts_model.original_model).__name__,
                        "has_layer_structure_method": hasattr(
                            ltts_model, "_get_layer_structure"
                        ),
                        "layer_wrappers": (
                            list(ltts_model.layer_wrappers.keys())
                            if hasattr(ltts_model, "layer_wrappers")
                            else []
                        ),
                    }

            except Exception as e:
                status["integration"]["available"] = False
                status["integration"]["error"] = str(e)

            return {"status": "success", "system_status": status}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    return router
