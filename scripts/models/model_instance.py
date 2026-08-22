import time
from typing import Dict, List, Any, Union
from scripts.models.model_struct import ModelStruct
from scripts.core.ltts_model import LTTSModel


class ModelInstance:
    model: LTTSModel
    struct: ModelStruct
    loaded: bool
    modules: Dict[str, List[Dict[str, Any]]]
    loaded_at: float  # monotonic timestamp for LRU eviction

    def __init__(self, model, loaded, struct):
        self.model = model
        self.loaded = loaded
        self.struct = struct
        self.modules = {}
        self.loaded_at = time.monotonic()

    def register_module(self, model_id, module_info):
        if model_id not in self.modules:
            self.modules[model_id] = []
        self.modules[model_id].append(module_info)

    def unregister_module(self, model_id, module_info):
        if model_id not in self.modules:
            module_name = module_info.get("name", "unknown")
        else:
            module_list = self.modules[model_id]
            if module_info in module_list:
                module_list.remove(module_info)
                module_name = module_info.get("name", "unknown")
            else:
                module_name = module_info.get("name", "unknown")

    def get_model_modules(self, key) -> List[Dict[str, Any]]:
        if key in self.modules:
            return self.modules[key]
        else:
            raise KeyError(f"No modules found for model '{key}'.")

    def is_ltts_wrapped(self) -> bool:
        return hasattr(self.model, "original_model") and hasattr(
            self.model, "layer_wrappers"
        )

    def flush(self):
        self.modules.clear()

        if hasattr(self.model, "flush") and callable(getattr(self.model, "flush")):
            self.model.flush()

        self.model = None
        self.loaded = False
