import logging
import os
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from scripts.core.integration import get_integration

logger = logging.getLogger(__name__)

try:
    from ltts_platform.module_registry import get_module_registry
except ImportError as exc:  # pragma: no cover - depends on optional deps
    _registry_error = str(exc)

    def get_module_registry():
        raise RuntimeError(f"Module registry not available: {_registry_error}")


def create_ltts_api_router(message_bus=None, integration=None) -> APIRouter:
    router = APIRouter(prefix="/api/ltts")
    integration = integration or get_integration(message_bus)

    def _resolve_registry(registry_url: str | None = None):
        registry = get_module_registry()
        if registry_url:
            registry.remote_registry_url = registry_url
            if hasattr(registry, "debug_mode"):
                registry.debug_mode = False
        return registry

    @router.get("/modules")
    async def get_available_modules():
        try:
            modules = integration.get_available_modules()
            return {"modules": modules}
        except Exception as e:
            logger.error("Error getting modules: %s", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve modules")

    @router.get("/models")
    async def get_wrapped_models():
        try:
            models = {}
            for model_id in integration.wrapped_models.keys():
                models[model_id] = integration.get_model_info(model_id)
            return {"models": models}
        except Exception as e:
            logger.error("Error getting models: %s", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve models")

    @router.get("/models/{model_id}")
    async def get_model_info(model_id: str):
        try:
            model_info = integration.get_model_info(model_id)
            if model_info is None:
                raise HTTPException(
                    status_code=404, detail=f"Model '{model_id}' not found"
                )
            return model_info
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Error getting model info for %s: %s", model_id, e)
            raise HTTPException(status_code=500, detail="Failed to retrieve model info")

    @router.get("/miners")
    async def get_available_miners():
        try:
            registry = integration.miners_registry
            miners = registry.list_miners()
            return {"miners": miners}
        except Exception as e:
            logger.error("Error getting miners: %s", e)
            raise HTTPException(status_code=500, detail="Failed to retrieve miners")

    @router.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "loaded_modules": len(integration.module_loader.loaded_modules),
            "wrapped_models": len(integration.wrapped_models),
            "available_miners": len(integration.miners_registry.list_miners()),
        }

    @router.get("/registry/status")
    async def get_registry_status(registry_url: str | None = Query(default=None)):
        """Registry summary for the status bar: total / installed / updates."""
        try:
            registry = _resolve_registry(registry_url)
            return await registry.get_registry_status()
        except Exception as e:
            logger.error("Error getting registry status: %s", e)
            raise HTTPException(status_code=500, detail="Failed to get registry status")

    @router.get("/registry/modules")
    async def get_registry_modules(registry_url: str | None = Query(default=None)):
        try:
            registry = _resolve_registry(registry_url)
            modules = await registry.fetch_remote_modules()
            return {
                "modules": [asdict(module) for module in modules],
                "registry_url": getattr(registry, "remote_registry_url", None),
            }
        except RuntimeError as e:
            logger.error("Module registry unavailable: %s", e)
            raise HTTPException(status_code=503, detail="Module registry is not available")
        except Exception as e:
            logger.error("Error fetching registry modules: %s", e)
            raise HTTPException(status_code=500, detail="Failed to fetch registry modules")

    @router.get("/registry/modules/installed")
    async def get_installed_modules():
        try:
            registry = get_module_registry()
            modules = registry.get_installed_modules()
            return {"modules": [asdict(module) for module in modules]}
        except RuntimeError as e:
            logger.error("Module registry unavailable: %s", e)
            raise HTTPException(status_code=503, detail="Module registry is not available")
        except Exception as e:
            logger.error("Error getting installed modules: %s", e)
            raise HTTPException(status_code=500, detail="Failed to get installed modules")

    @router.get("/registry/modules/remote")
    async def get_remote_modules(registry_url: str | None = Query(default=None)):
        try:
            registry = _resolve_registry(registry_url)
            modules = registry.get_remote_modules()
            return {
                "modules": [asdict(module) for module in modules],
                "registry_url": getattr(registry, "remote_registry_url", None),
            }
        except RuntimeError as e:
            logger.error("Module registry unavailable: %s", e)
            raise HTTPException(status_code=503, detail="Module registry is not available")
        except Exception as e:
            logger.error("Error getting remote modules: %s", e)
            raise HTTPException(status_code=500, detail="Failed to get remote modules")

    def _check_install_allowed():
        if os.environ.get("LTTS_ALLOW_MODULE_INSTALL", "").lower() not in ("1", "true", "yes"):
            raise HTTPException(
                status_code=403,
                detail="Module installation is disabled. Set LTTS_ALLOW_MODULE_INSTALL=1 to enable it (executes third-party code).",
            )

    @router.post("/registry/modules/{module_name}/install")
    async def install_module(module_name: str):
        _check_install_allowed()
        try:
            registry = get_module_registry()
            success = await registry.install_module(module_name)
            if success:
                return {
                    "success": True,
                    "message": f"Module '{module_name}' installed successfully",
                }
            else:
                raise HTTPException(
                    status_code=400, detail=f"Failed to install module '{module_name}'"
                )
        except HTTPException:
            raise
        except RuntimeError as e:
            logger.error("Module registry unavailable: %s", e)
            raise HTTPException(status_code=503, detail="Module registry is not available")
        except Exception as e:
            logger.error("Error installing module %s: %s", module_name, e)
            raise HTTPException(status_code=500, detail="Module installation failed")

    @router.delete("/registry/modules/{module_name}/uninstall")
    async def uninstall_module(module_name: str):
        _check_install_allowed()
        try:
            registry = get_module_registry()
            success = registry.uninstall_module(module_name)
            if success:
                return {
                    "success": True,
                    "message": f"Module '{module_name}' uninstalled successfully",
                }
            else:
                raise HTTPException(
                    status_code=400, detail=f"Failed to uninstall module '{module_name}'"
                )
        except HTTPException:
            raise
        except RuntimeError as e:
            logger.error("Module registry unavailable: %s", e)
            raise HTTPException(status_code=503, detail="Module registry is not available")
        except Exception as e:
            logger.error("Error uninstalling module %s: %s", module_name, e)
            raise HTTPException(status_code=500, detail="Module uninstallation failed")

    return router
