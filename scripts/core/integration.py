import logging
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from .ltts_model import LTTSModel
from .module_loader import ModuleLoader
from .miners_registry import get_global_registry

try:
    from ltts_platform.module_venv_manager import ModuleVenvManager

    VENV_MANAGER_AVAILABLE = True
except ImportError:  # pragma: no cover - optional at runtime
    ModuleVenvManager = None
    VENV_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)


class LTTSIntegration:
    def __init__(self, message_bus=None):
        self.message_bus = message_bus
        self.module_loader = ModuleLoader(
            message_bus=message_bus, ltts_integration=self
        )
        self.miners_registry = get_global_registry()
        self.wrapped_models: Dict[str, LTTSModel] = {}

    def initialize(self):
        try:
            self._initialize_module_venvs()

            # Load miners
            self._load_miners()

            # Load modules
            self.module_loader.load_all_modules()

            logger.info("LTTS system initialized successfully")

        except Exception as e:
            logger.error("Error initializing LTTS system: %s", e)
            raise

    def wrap_model(self, model, model_id: str) -> LTTSModel:
        """Wrap a model with LTTS capabilities.

        Idempotent: ``ModelsRegistry.load_model`` already returns an
        ``LTTSModel``, and the inference path may call this again. Wrapping an
        already-wrapped model a second time would register the layer hooks
        twice (doubling every module emission) and bury the real HuggingFace
        model two levels deep, so an already-wrapped model is registered and
        returned as-is.
        """
        try:
            if isinstance(model, LTTSModel):
                ltts_model = model
                # ModelsRegistry constructs the LTTSModel with only a load-progress
                # callback as its "message_bus", which has no .publish — point it
                # (and its event dispatcher) at the integration's real bus so module
                # emissions and reactor signals reach subscribed clients.
                if self.message_bus is not None and hasattr(self.message_bus, "publish"):
                    ltts_model.message_bus = self.message_bus
                    dispatcher = getattr(ltts_model, "event_dispatcher", None)
                    if dispatcher is not None:
                        dispatcher.message_bus = self.message_bus
            else:
                ltts_model = LTTSModel(model, self.message_bus)
            self.wrapped_models[model_id] = ltts_model

            logger.info("Wrapped model: %s", model_id)
            return ltts_model

        except Exception as e:
            logger.error("Error wrapping model %s: %s", model_id, e)
            raise

    def get_available_modules(self) -> Dict[str, Any]:
        """Get all available modules with their metadata."""
        modules_info = {}

        for module_name, module in self.module_loader.get_all_modules().items():
            metadata = self.module_loader.get_module_metadata(module_name)
            modules_info[module_name] = {
                "metadata": metadata if metadata else {},
                "ui_definition": module.get("ui_schema", {}),
                "loaded": True,
            }

        return modules_info

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a wrapped model."""
        if model_id not in self.wrapped_models:
            return None

        ltts_model = self.wrapped_models[model_id]

        return {
            "model_id": model_id,
            "layer_wrappers": list(ltts_model.layer_wrappers.keys()),
            "model_class": type(ltts_model.original_model).__name__,
        }

    def _initialize_module_venvs(self):
        """Initialize virtual environments for all modules with dependencies."""
        if os.getenv("LTTS_SKIP_MODULE_VENVS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info(
                "Skipping per-module venv setup (LTTS_SKIP_MODULE_VENVS set); "
                "modules will import from the main environment."
            )
            return
        if not VENV_MANAGER_AVAILABLE or ModuleVenvManager is None:
            return

        try:
            logger.info("=" * 60)
            logger.info("Module Virtual Environment Setup")
            logger.info("=" * 60)

            project_root = Path.cwd()
            try:
                from ltts_platform.module_paths import resolve_modules_dir

                modules_dir = resolve_modules_dir(project_root)
            except ImportError:
                modules_dir = project_root / "ltts_modules"
            shared_venv_path = project_root / "venv"

            if not shared_venv_path.exists():
                logger.error(
                    "Shared LTTS venv not found at %s. Create it with "
                    "`python -m venv venv` and install project requirements.",
                    shared_venv_path,
                )
                return

            if not modules_dir.exists():
                logger.warning("ltts_modules directory not found")
                return

            # Initialize venv manager
            try:
                venv_manager = ModuleVenvManager(str(project_root))
            except ValueError as e:
                logger.error("Failed to initialize venv manager: %s", e)
                return

            # Scan for modules with requirements.txt
            modules_with_reqs = []
            for item in os.listdir(modules_dir):
                if item.startswith(".") or item == "__pycache__":
                    continue

                module_path = modules_dir / item
                if module_path.is_dir():
                    req_file = module_path / "requirements.txt"
                    if req_file.exists():
                        # Read requirements
                        with open(req_file, "r", encoding="utf-8") as f:
                            requirements = [
                                line.strip()
                                for line in f
                                if line.strip() and not line.startswith("#")
                            ]
                        if requirements:
                            modules_with_reqs.append((item, requirements))

            if not modules_with_reqs:
                logger.info("No modules require additional dependencies")
                return

            success_count = 0
            failed_count = 0

            for module_name, requirements in modules_with_reqs:
                try:
                    venv_manager.create_module_venv(module_name, requirements)
                    logger.info(
                        "[module] %-25s (%d pkg) OK", module_name, len(requirements)
                    )
                    success_count += 1
                except Exception as e:
                    logger.error(
                        "[module] %-25s (%d pkg) FAILED: %s",
                        module_name, len(requirements), str(e)[:80],
                    )
                    failed_count += 1

            # Summary
            logger.info("=" * 60)
            if success_count > 0:
                logger.info("OK: %d module(s) ready", success_count)
            if failed_count > 0:
                logger.error("FAILED: %d module(s)", failed_count)
            logger.info("=" * 60)

        except Exception as e:
            logger.error("Error initializing module venvs: %s", e)
            # Don't fail the whole initialization if venv setup fails

    def _load_miners(self):
        """Load all miners into the registry."""
        try:
            # Import miners to trigger registration
            import miners

            logger.info("Loaded miners successfully")

        except ImportError as e:
            logger.warning("Could not load miners: %s", e)


# Global integration instance
_global_integration = None


def get_integration(message_bus=None) -> LTTSIntegration:
    """Get the global LTTS integration instance."""
    global _global_integration

    if _global_integration is None:
        _global_integration = LTTSIntegration(message_bus)
        _global_integration.initialize()

    return _global_integration
