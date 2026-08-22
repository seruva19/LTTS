import os
import logging
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import asyncio
from contextlib import asynccontextmanager
from threading import Lock

from scripts.blueprints.blueprint_loader import load_blueprints
from scripts.context import LTTSContext
from scripts.messages.message_bus import MessageBus
from scripts.configs import user_files
from scripts.configs.user_config import UserConfig
from scripts.models.model_loader import ModelsRegistry
from scripts.modules.modules_store import ModulesStore
from scripts.api_integration import create_ltts_api_router
from scripts.core.integration import LTTSIntegration
import scripts.core.integration as integration_module

from scripts.routers.core_router import create_core_router
from scripts.routers.models_router import create_models_router
from scripts.routers.modules_router import create_modules_router
from scripts.routers.blueprints_router import create_blueprints_router
from scripts.routers.settings_router import create_settings_router
from scripts.routers.inference_router import create_inference_router

logger = logging.getLogger(__name__)
_runtime_lock = Lock()
_runtime_context = None
_api_router = None


def _ensure_user_files() -> None:
    user_files.ensure_user_files()


def _load_user_config() -> UserConfig:
    config = user_files.read_user_config()
    return UserConfig(**config)


def _resolve_modules_dirs() -> list[str]:
    """Directories to load analysis modules from.

    Source normally lives in the sibling LTTS-modules repo; modules installed
    through the registry land in the local install directory. Both are loaded.
    """
    from ltts_platform.module_paths import resolve_module_search_paths

    paths = [str(path) for path in resolve_module_search_paths(Path.cwd())]
    logger.info("Loading analysis modules from %s", ", ".join(paths))
    return paths


def _build_runtime_context(api: APIRouter) -> LTTSContext:
    _ensure_user_files()
    modules_dirs = _resolve_modules_dirs()

    context = LTTSContext()
    context.router = api
    context.message_bus = MessageBus()
    context.user_config = _load_user_config()
    context.blueprints = load_blueprints()

    context.models = ModelsRegistry(context.user_config)
    context.models.load_models()

    from scripts.core.module_loader import ModuleLoader

    context.ltts_integration = LTTSIntegration(context.message_bus)
    context.ltts_integration._initialize_module_venvs()
    context.ltts_integration._load_miners()
    integration_module._global_integration = context.ltts_integration

    context.module_loader = ModuleLoader(
        modules_dirs, fastapi_app=api, message_bus=context.message_bus
    )
    context.module_loader.load_all_modules()
    context.ltts_integration.module_loader = context.module_loader

    context.modules_store = ModulesStore(modules_dirs)
    context.modules_store.discover_modules(context)
    return context


def create_api_router() -> APIRouter:
    api = APIRouter()
    context = _build_runtime_context(api)

    ltts_api_router = create_ltts_api_router(
        context.message_bus, integration=context.ltts_integration
    )
    api.include_router(ltts_api_router)
    api.include_router(create_core_router(context))
    api.include_router(create_models_router(context))
    api.include_router(create_modules_router(context))
    api.include_router(create_blueprints_router(context))
    api.include_router(create_settings_router(context))
    api.include_router(create_inference_router(context))

    async def startup_event():
        asyncio.create_task(context.message_bus.start_processing())
        logger.info("LTTS startup complete — message bus processing started.")
        asyncio.create_task(_scan_module_registry())

    async def _scan_module_registry():
        """Scan the modules registry and log a summary (registry/installed/updates)."""
        try:
            from ltts_platform.module_registry import get_module_registry

            registry = get_module_registry()
            status = await registry.get_registry_status()
            logger.info(
                "Module registry: %d modules in registry, %d installed, %d updates available (%s)",
                status["registry_count"],
                status["installed_count"],
                status["updates_available"],
                status["registry_url"],
            )
        except Exception as e:
            logger.warning("Module registry scan failed: %s", e)

    async def shutdown_event():
        """Gracefully shut down: stop message bus, notify clients, release models."""
        logger.info("LTTS shutting down...")
        context.message_bus.stop_processing()
        for ws in list(context.message_bus.websocket_clients):
            try:
                await ws.send_json(
                    {
                        "type": "server_shutdown",
                        "message": "Server is shutting down",
                    }
                )
                await ws.close()
            except Exception:
                pass
        context.message_bus.websocket_clients.clear()
        if hasattr(context, "models") and context.models:
            for model_id in list(context.models.ltts_models.keys()):
                try:
                    context.models.unload_model(model_id)
                except Exception as e:
                    logger.warning(
                        "Error unloading model %s during shutdown: %s", model_id, e
                    )
        logger.info("LTTS shutdown complete.")

    # Handed to the FastAPI app's lifespan (see lifespan() below and ltts.py)
    # rather than registered through the deprecated router event API.
    context.startup_event = startup_event
    context.shutdown_event = shutdown_event

    @api.get("/api/health")
    async def health_check():
        """Health check endpoint for container orchestration."""
        return {
            "status": "healthy",
            "websocket_clients": len(context.message_bus.websocket_clients),
            "message_bus_running": getattr(context.message_bus, "running", False),
        }

    @api.get("/api/readiness")
    async def readiness_check():
        """Readiness probe with basic runtime dependency checks."""
        config_exists = os.path.exists(user_files.USER_CONFIG)
        models_registry_exists = os.path.exists(user_files.USER_MODELS)
        module_count = len(context.module_loader.get_all_modules()) if context.module_loader else 0
        blueprint_count = len(getattr(context.blueprints, "blueprints", []) or [])
        ready = config_exists and models_registry_exists and module_count > 0

        return {
            "status": "ready" if ready else "degraded",
            "checks": {
                "config_exists": config_exists,
                "models_registry_exists": models_registry_exists,
                "modules_loaded": module_count > 0,
                "message_bus_initialized": context.message_bus is not None,
            },
            "details": {
                "modules_loaded": module_count,
                "configured_models": len(getattr(context.models, "config_list", {}) or {}),
                "available_blueprints": blueprint_count,
            },
        }

    _ws_max_message_size = 1 * 1024 * 1024

    @api.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        context.message_bus.add_websocket(websocket)
        try:
            while True:
                raw = await websocket.receive_text()
                if len(raw) > _ws_max_message_size:
                    logger.warning(
                        "WebSocket message too large (%d bytes), dropping", len(raw)
                    )
                    continue
                import json

                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError) as parse_err:
                    logger.warning(
                        "WebSocket received invalid JSON (%s), dropping", parse_err
                    )
                    continue
                token = data.get("token", "default")
                msg_type = data.get("type")
                if msg_type == "ping":
                    try:
                        await websocket.send_json(
                            {"type": "pong", "timestamp": data.get("timestamp")}
                        )
                    except Exception:
                        pass
                    continue
                if msg_type == "stop_inference":
                    from scripts.services.inference.cancellation import cancel_run

                    cancelled = cancel_run(token)
                    logger.info(
                        "Client requested inference stop (token=%s, active_run=%s)",
                        token,
                        cancelled,
                    )
                    await context.message_bus.publish(
                        token,
                        {
                            "type": "inference_cancelling" if cancelled else "info",
                            "message": (
                                "Stop request received. Inference will be aborted "
                                "at the next layer boundary."
                                if cancelled
                                else "Stop request received, but no active "
                                "inference run for this client."
                            ),
                        },
                    )
                    continue
                await context.message_bus.publish(token, data)
        except WebSocketDisconnect:
            logger.debug("WebSocket client disconnected normally")
        except Exception as e:
            logger.warning("WebSocket connection error: %s", e)
        finally:
            context.message_bus.remove_websocket(websocket)

    global _runtime_context
    _runtime_context = context
    return api


@asynccontextmanager
async def lifespan(_app):
    """Application lifespan: start the message bus, then release everything.

    Building the runtime is what registers the handlers, so the router is
    resolved first. Shutdown runs even if startup raised part-way through.
    """
    get_api_router()
    runtime = get_context()

    startup = getattr(runtime, "startup_event", None)
    if startup is not None:
        await startup()
    try:
        yield
    finally:
        shutdown = getattr(runtime, "shutdown_event", None)
        if shutdown is not None:
            await shutdown()


def get_api_router() -> APIRouter:
    global _api_router
    with _runtime_lock:
        if _api_router is None:
            _api_router = create_api_router()
        return _api_router


def get_context() -> LTTSContext:
    global _runtime_context
    if _runtime_context is None:
        get_api_router()
    return _runtime_context


def reset_runtime_state() -> None:
    global _api_router, _runtime_context
    with _runtime_lock:
        _api_router = None
        _runtime_context = None


def __getattr__(name: str):
    if name in {"api", "app"}:
        return get_api_router()
    if name == "context":
        return get_context()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "create_api_router",
    "get_api_router",
    "get_context",
    "reset_runtime_state",
    "api",
    "app",
    "context",
]
