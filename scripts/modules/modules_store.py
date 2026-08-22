"""Compatibility wrapper for ModuleLoader-backed discovery and loading."""

import logging
from typing import Dict, Any, Optional
from scripts.core.module_loader import ModuleLoader
from scripts.core.ltts_module import load_module_from_file, call_module_method

logger = logging.getLogger(__name__)


class ModulesStore:
    """Expose ModuleLoader results through the module-store interface."""

    def __init__(self, modules_dir: "str | list[str]" = "ltts_modules"):
        self.modules_dir = modules_dir
        self.loaded = False
        self.modules: Dict[str, Dict[str, Any]] = {}
        self._module_loader = ModuleLoader(modules_dir)

    def discover_modules(self, context) -> Dict[str, Dict[str, Any]]:
        """Discover and load modules."""
        if not self.loaded:
            try:
                # Check if modules are already loaded by the module loader
                if hasattr(context, "module_loader") and context.module_loader:
                    loaded_modules = context.module_loader.get_all_modules()

                    # Convert to the module-store format.
                    for module_name, module_info in loaded_modules.items():
                        # ModuleLoader has already called the constructor.
                        self.modules[module_name] = {
                            "metadata": module_info["metadata"],
                            "ui_schema": module_info["ui_schema"],
                            "module": module_info["module_object"],
                        }

                    logger.info(
                        "Loaded %s modules", len(self.modules)
                    )
                    self.loaded = True
                else:
                    logger.warning("No module_loader found in context, modules_store will be empty")
                    self.modules = {}
                    self.loaded = True

            except Exception as e:
                logger.error("Failed to load modules: %s", e)
                self.modules = {}
                self.loaded = True

        return self.modules

    def get_module(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a module by name."""
        return self.modules.get(name)

    def reload_module(self, name: str, context) -> Optional[Dict[str, Any]]:
        """Reload a specific module."""
        try:
            # Unload the module
            self._module_loader.unload_module(name)
            if name in self.modules:
                del self.modules[name]

            # Reload it
            module_info = self._module_loader.load_module(name)
            if module_info and context is not None:
                call_module_method(module_info, "ctor", context)

            if module_info:
                self.modules[name] = {
                    "metadata": module_info["metadata"],
                    "ui_schema": module_info["ui_schema"],
                    "module": module_info["module_object"],
                }

            return self.modules.get(name)

        except Exception as e:
            logger.error("Failed to reload module %s: %s", name, e)
            return None
