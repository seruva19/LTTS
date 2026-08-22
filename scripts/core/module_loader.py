import os
import sys
import contextlib
import importlib.util
import logging
from typing import Dict, Iterator, List, Any, Optional
from fastapi import APIRouter
from .ltts_module import (
    load_module_from_file,
    call_module_method,
    validate_module_metadata,
)
from . import module_data_sender

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _isolated_module_path(extra_paths: List[str]) -> Iterator[None]:
    """Path-scoped import isolation for module venvs.

    Temporarily appends the module's site-packages to sys.path and, on exit,
    restores sys.path and evicts modules that were loaded from the module's
    own site-packages during the import (plus keeps ``ltts_modules.*`` entries
    — the loaded module may register itself there).

    Only venv-sourced modules are evicted: third-party modules resolved from
    the main environment (e.g. lazily imported ``transformers`` submodules)
    are left in sys.modules, because evicting them would trigger a re-import
    later in the process and native extension DLLs cannot be safely loaded
    twice.

    NOTE: this is not full isolation. Imports performed later at runtime
    (e.g. inside module event calls) still resolve against the main process
    sys.path; this only prevents permanent pollution of the host process by
    module-venv dependencies ("shared dependency hijack").
    """
    if not extra_paths:
        yield
        return

    saved_path = list(sys.path)
    saved_modules = set(sys.modules.keys())
    normalized_paths = [
        os.path.normcase(os.path.abspath(path)) for path in extra_paths
    ]
    for path in extra_paths:
        if path not in sys.path:
            # Append instead of prepend: module venv packages must not
            # shadow the core process dependencies.
            sys.path.append(path)

    def _is_venv_module(module: Any) -> bool:
        module_file = getattr(module, "__file__", None)
        if not module_file:
            return False
        module_file = os.path.normcase(os.path.abspath(module_file))
        return any(module_file.startswith(p + os.sep) for p in normalized_paths)

    try:
        yield
    finally:
        sys.path[:] = saved_path
        for key in list(set(sys.modules.keys()) - saved_modules):
            if key.startswith("ltts_modules."):
                continue
            module = sys.modules.get(key)
            if module is not None and _is_venv_module(module):
                del sys.modules[key]


class ModuleContext:
    def __init__(
        self, message_bus=None, ltts_integration=None, module_data_sender=None
    ):
        self.message_bus = message_bus
        self.ltts_integration = ltts_integration
        self.module_data_sender = module_data_sender


class ModuleLoader:
    def __init__(
        self,
        modules_directory: "str | List[str]" = "ltts_modules",
        fastapi_app: Optional[APIRouter] = None,
        message_bus=None,
        ltts_integration=None,
    ):
        # Modules are read from more than one place: a source checkout (the
        # LTTS-modules repo) and the local install directory the registry
        # writes into. Accept either a single path or an ordered list.
        self.modules_directories = (
            [modules_directory]
            if isinstance(modules_directory, (str, os.PathLike))
            else [str(path) for path in modules_directory]
        )
        # Kept for callers that still read a single directory.
        self.modules_directory = self.modules_directories[0]
        self.fastapi_app = fastapi_app
        self.message_bus = message_bus
        self.ltts_integration = ltts_integration
        self.loaded_modules: Dict[str, Dict[str, Any]] = {}
        self.module_metadata: Dict[str, Dict[str, Any]] = {}

    def discover_modules(self) -> List[str]:
        existing = [d for d in self.modules_directories if os.path.exists(d)]
        if not existing:
            logger.warning(
                "No modules directory exists (looked in %s)",
                ", ".join(self.modules_directories),
            )
            return []

        module_names = []
        for directory in existing:
            logger.info("Scanning modules directory: %s", directory)
            module_names.extend(self._discover_in(directory, module_names))
        return module_names

    def _discover_in(self, directory: str, already_found: List[str]) -> List[str]:
        module_names = []
        for item in os.listdir(directory):
            if item.startswith(".") or item == "__pycache__":
                continue

            module_path = os.path.join(directory, item)

            if item.endswith(".py") and item != "__init__.py":
                name = item[:-3]
                if name in already_found:
                    logger.debug("  %s already provided by an earlier directory", name)
                    continue
                module_names.append(name)
                logger.debug("  Found module file: %s", item)
            elif os.path.isdir(module_path):
                init_file = os.path.join(module_path, "init.py")
                if os.path.exists(init_file):
                    if item in already_found:
                        logger.info(
                            "Module '%s' in %s is shadowed by the copy found earlier",
                            item,
                            directory,
                        )
                        continue
                    module_names.append(item)
                    venv_path = os.path.join(module_path, "venv")
                    has_venv = os.path.exists(venv_path)
                    venv_status = (
                        "with isolated venv" if has_venv else "using main venv"
                    )
                    logger.debug("  Found module directory: %s (%s)", item, venv_status)

        logger.info("✓ Discovered %s module(s)", len(module_names))
        for name in module_names:
            logger.debug("    - %s", name)
        return module_names

    def load_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        if module_name in self.loaded_modules:
            logger.debug(
                "Module %s already loaded, returning cached version", module_name
            )
            return self.loaded_modules[module_name]

        logger.info("📦 Loading module: %s...", module_name)

        module_path = self._get_module_path(module_name)
        if not module_path:
            logger.error("  ✗ Module %s not found", module_name)
            return None

        module_dir = os.path.dirname(module_path)
        venv_path = os.path.join(module_dir, "venv")
        has_venv = os.path.exists(venv_path)
        module_site_packages: List[str] = []
        if has_venv:
            logger.debug("  → Using isolated venv at %s", venv_path)

            if os.name == "nt":
                site_packages = os.path.join(venv_path, "Lib", "site-packages")
            else:
                site_packages = os.path.join(
                    venv_path,
                    "lib",
                    f"python{sys.version_info.major}.{sys.version_info.minor}",
                    "site-packages",
                )

            if os.path.exists(site_packages):
                module_site_packages.append(site_packages)
                logger.debug(
                    "  → Will import module with isolated site-packages: %s",
                    site_packages,
                )
            else:
                logger.warning(
                    "  ⚠ Module venv site-packages not found at %s", site_packages
                )
        else:
            logger.debug("  → Using main project venv")

        full_module_name = f"ltts_modules.{module_name}"

        # Import (and construct) the module with its venv site-packages visible
        # only for the duration of the import; see _isolated_module_path.
        with _isolated_module_path(module_site_packages):
            module_info = load_module_from_file(
                module_path, full_module_name, self.fastapi_app
            )
            if module_info is None:
                logger.error("  ✗ Failed to load module %s", module_name)
                return None

            errors = validate_module_metadata(module_info["metadata"])
            if errors:
                logger.error(
                    "  ✗ Module %s has invalid metadata: %s", module_name, errors
                )
                return None

            metadata = module_info["metadata"]
            logger.debug("  → Name: %s", metadata.get("name"))
            logger.debug("  → Version: %s", metadata.get("version"))
            logger.debug(
                "  → Event types: %s", ", ".join(metadata.get("event_types", []))
            )

            dependencies = metadata.get("dependencies", [])
            if dependencies:
                logger.debug("  → Dependencies: %s", ", ".join(dependencies))

            self.loaded_modules[module_name] = module_info
            self.module_metadata[module_name] = module_info["metadata"]

            context = ModuleContext(
                message_bus=self.message_bus,
                ltts_integration=self.ltts_integration,
                module_data_sender=self._get_module_data_sender(),
            )
            call_module_method(module_info, "ctor", context)

        if module_info.get("has_stateful_events"):
            buttons = len(module_info.get("control_buttons", []))
            routes = len(module_info.get("registered_routes", []))
            logger.info(
                "  ✓ Loaded stateful module: %s (%s buttons, %s routes)",
                module_name,
                buttons,
                routes,
            )
        else:
            logger.info("  ✓ Loaded module: %s", module_name)

        return module_info

    def load_all_modules(self) -> Dict[str, Dict[str, Any]]:
        logger.info("=" * 60)
        logger.info("🚀 LTTS Module Loading System")
        logger.info("=" * 60)

        module_names = self.discover_modules()

        if not module_names:
            logger.warning("⚠ No modules found to load")
            return {}

        logger.info("\n📚 Loading %s module(s)...", len(module_names))

        loaded_count = 0
        failed_count = 0

        for module_name in module_names:
            try:
                result = self.load_module(module_name)
                if result:
                    loaded_count += 1
                else:
                    failed_count += 1
            except Exception as e:
                logger.error("  ✗ Exception loading %s: %s", module_name, e)
                failed_count += 1

        logger.info("\n" + "=" * 60)
        logger.info(
            "✓ Module loading complete: %s loaded, %s failed",
            loaded_count,
            failed_count,
        )
        logger.info("=" * 60)

        return self.loaded_modules

    def get_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        return self.loaded_modules.get(module_name)

    def get_all_modules(self) -> Dict[str, Dict[str, Any]]:
        return self.loaded_modules.copy()

    def get_module_control_buttons(self, module_name: str) -> List[Dict[str, Any]]:
        module_info = self.loaded_modules.get(module_name)
        if module_info and module_info.get("has_stateful_events"):
            return module_info.get("control_buttons", [])
        return []

    def get_all_control_buttons(self) -> Dict[str, List[Dict[str, Any]]]:
        buttons = {}
        for module_name, module_info in self.loaded_modules.items():
            if module_info.get("has_stateful_events"):
                buttons[module_name] = module_info.get("control_buttons", [])
        return buttons

    def get_module_routes(self, module_name: str) -> List[Dict[str, Any]]:
        module_info = self.loaded_modules.get(module_name)
        if module_info and module_info.get("has_stateful_events"):
            return module_info.get("registered_routes", [])
        return []

    def call_module_event(
        self, module_name: str, event_data: Dict[str, Any], module_state: Dict[str, Any]
    ) -> Any:
        module_info = self.loaded_modules.get(module_name)
        if not module_info or not module_info.get("has_stateful_events"):
            raise ValueError(f"Module {module_name} does not support stateful events")

        module_instance = module_info.get("module_instance")
        if not module_instance:
            raise ValueError(f"Module {module_name} has no instance")

        module_instance.update_state(module_state)
        return module_instance

    def get_module_metadata(self, module_name: str) -> Optional[Dict[str, Any]]:
        return self.module_metadata.get(module_name)

    def unload_module(self, module_name: str) -> bool:
        if module_name in self.loaded_modules:
            module_info = self.loaded_modules[module_name]

            call_module_method(module_info, "dtor")

            if module_info.get("has_stateful_events"):
                module_instance = module_info.get("module_instance")
                if module_instance:
                    routes = len(module_instance.get_registered_routes())
                    logger.info(
                        "Cleaning up stateful module %s (%s routes)",
                        module_name,
                        routes,
                    )

            del self.loaded_modules[module_name]
            if module_name in self.module_metadata:
                del self.module_metadata[module_name]
            logger.info("Unloaded module: %s", module_name)
            return True
        return False

    def _get_module_path(self, module_name: str) -> Optional[str]:
        for directory in self.modules_directories:
            py_path = os.path.join(directory, f"{module_name}.py")
            if os.path.exists(py_path):
                return py_path

            init_path = os.path.join(directory, module_name, "init.py")
            if os.path.exists(init_path):
                return init_path

        return None

    def _get_module_data_sender(self):
        try:
            return module_data_sender
        except ImportError:
            logger.warning("module_data_sender not available")
            return None
