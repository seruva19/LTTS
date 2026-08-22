"""
Normalization Miner - Extracts normalization layer information

Handles LayerNorm, RMSNorm, GroupNorm, and other normalization techniques.
"""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import logging
from scripts.core.miners_registry import BaseMiner
from .utils import safe_get_attr, collect_tensor_stats

logger = logging.getLogger(__name__)


class NormalizationMiner(BaseMiner):
    """Extracts normalization layer weights, statistics, and configuration."""

    def is_applicable(self, module_path: str, module_class: str) -> bool:
        """Check if this miner applies to normalization modules."""
        text_to_check = f"{module_path} {module_class}".lower()
        norm_patterns = [
            "norm", "layernorm", "rmsnorm", "groupnorm",
            "batchnorm", "instancenorm", "ln", "rms"
        ]
        return any(pattern in text_to_check for pattern in norm_patterns)

    def extract(self, raw_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract normalization data from raw context."""
        if not self.is_applicable(raw_context.get("module_path", ""),
                                  raw_context.get("module_class", "")):
            return None

        try:
            outputs = raw_context.get("outputs")
            inputs = raw_context.get("inputs")
            layer_state = raw_context.get("layer_state", {})
            module_class = raw_context.get("module_class", "")

            norm_data = {}

            # Extract normalization parameters
            params_info = self._extract_norm_parameters(layer_state)
            if params_info:
                norm_data["parameters"] = params_info

            # Extract normalization statistics
            if inputs and outputs:
                stats = self._extract_norm_statistics(inputs, outputs)
                if stats:
                    norm_data["statistics"] = stats

            # Extract configuration
            config = self._extract_norm_config(layer_state, module_class)
            if config:
                norm_data["config"] = config

            return norm_data if norm_data else None

        except Exception as e:
            logger.error(f"Error in NormalizationMiner.extract: {e}")
            return {"error": str(e)}

    def _extract_norm_parameters(self, layer_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract normalization parameters (weight, bias)."""
        try:
            parameters = layer_state.get("parameters", {})
            param_data = {}

            for param_name, param_info in parameters.items():
                if isinstance(param_info, list):
                    if "weight" in param_name.lower():
                        param_data["weight"] = {
                            "shape": param_info,
                            "dimension": param_info[0] if param_info else None
                        }
                    elif "bias" in param_name.lower():
                        param_data["bias"] = {
                            "shape": param_info,
                            "dimension": param_info[0] if param_info else None
                        }

            return param_data if param_data else None

        except Exception as e:
            logger.warning(f"Error extracting norm parameters: {e}")
            return None

    def _extract_norm_statistics(self, inputs: Any, outputs: Any) -> Optional[Dict[str, Any]]:
        """Extract normalization statistics from inputs and outputs."""
        try:
            stats = {}

            # Input statistics
            if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                first_input = inputs[0]
                if isinstance(first_input, torch.Tensor):
                    with torch.no_grad():
                        stats["input"] = {
                            "basic_stats": collect_tensor_stats(first_input),
                            "mean": float(first_input.mean()),
                            "std": float(first_input.std())
                        }

            # Output statistics
            if isinstance(outputs, torch.Tensor):
                with torch.no_grad():
                    stats["output"] = {
                        "basic_stats": collect_tensor_stats(outputs),
                        "mean": float(outputs.mean()),
                        "std": float(outputs.std())
                    }

                    # Basic effectiveness metrics
                    if "input" in stats:
                        input_tensor = inputs[0]
                        stats["effectiveness"] = {
                            "std_reduction": float(input_tensor.std() - outputs.std()),
                            "mean_shift": float(abs(outputs.mean() - input_tensor.mean()))
                        }

            return stats if stats else None

        except Exception as e:
            logger.warning(f"Error extracting norm statistics: {e}")
            return None

    def _extract_norm_config(self, layer_state: Dict[str, Any], module_class: str) -> Dict[str, Any]:
        """Extract normalization configuration."""
        config = {
            "normalization_type": self._identify_norm_type(module_class)
        }

        # Extract dimension information from parameters
        parameters = layer_state.get("parameters", {})
        for param_name, param_info in parameters.items():
            if "weight" in param_name.lower() and isinstance(param_info, list):
                config["normalized_dimension"] = param_info[0] if param_info else None
                break

        # Determine if bias is used
        config["has_bias"] = any("bias" in name.lower() for name in parameters.keys())

        return config

    def _identify_norm_type(self, module_class: str) -> str:
        """Identify the type of normalization from module class."""
        class_lower = module_class.lower()

        if "layernorm" in class_lower or "ln" in class_lower:
            return "layer_norm"
        elif "rmsnorm" in class_lower or "rms" in class_lower:
            return "rms_norm"
        elif "groupnorm" in class_lower:
            return "group_norm"
        elif "batchnorm" in class_lower:
            return "batch_norm"
        elif "instancenorm" in class_lower:
            return "instance_norm"
        elif "norm" in class_lower:
            return "generic_norm"
        else:
            return "unknown"
