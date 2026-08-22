"""
FFN Miner - Refined from FFNCollector

Extracts feed-forward network weights, activations, and configuration data.
"""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import logging
from scripts.core.miners_registry import BaseMiner
from .utils import safe_get_attr, collect_tensor_stats, extract_layer_weights

logger = logging.getLogger(__name__)


class FFNMiner(BaseMiner):
    """Extracts FFN/MLP weights, activations, and configuration."""

    def is_applicable(self, module_path: str, module_class: str) -> bool:
        """Check if this miner applies to FFN/MLP modules."""
        text_to_check = f"{module_path} {module_class}".lower()
        ffn_patterns = [
            "mlp", "ffn", "feedforward", "feed_forward",
            "swiglu", "gelu", "fc", "linear"
        ]
        return any(pattern in text_to_check for pattern in ffn_patterns)

    def extract(self, raw_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract FFN data from raw context."""
        if not self.is_applicable(raw_context.get("module_path", ""),
                                  raw_context.get("module_class", "")):
            return None

        try:
            outputs = raw_context.get("outputs")
            inputs = raw_context.get("inputs")
            layer_state = raw_context.get("layer_state", {})

            ffn_data = {}

            # Extract layer weights information
            weights_info = self._extract_ffn_weights(layer_state)
            if weights_info:
                ffn_data["weights"] = weights_info

            # Extract activation statistics
            activation_stats = self._extract_activation_stats(inputs, outputs)
            if activation_stats:
                ffn_data["activations"] = activation_stats

            # Extract FFN configuration
            config = self._extract_ffn_config(layer_state)
            if config:
                ffn_data["config"] = config

            return ffn_data if ffn_data else None

        except Exception as e:
            logger.error(f"Error in FFNMiner.extract: {e}")
            return {"error": str(e)}

    def _extract_ffn_weights(self, layer_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract FFN weight information from layer state."""
        try:
            parameters = layer_state.get("parameters", {})
            weights_data = {}

            # Common FFN layer patterns
            ffn_patterns = [
                "fc1", "fc2", "c_fc", "c_proj",  # Standard MLP
                "w1", "w2", "w3",                # Some transformer variants
                "gate_proj", "up_proj", "down_proj",  # SwiGLU
                "wi_0", "wi_1", "wo"             # T5-style
            ]

            for param_name, param_info in parameters.items():
                for pattern in ffn_patterns:
                    if pattern in param_name.lower():
                        if isinstance(param_info, list):  # Shape information
                            weights_data[param_name] = {
                                "shape": param_info,
                                "layer_type": self._identify_layer_type(param_name),
                                "parameter_count": self._compute_param_count(param_info)
                            }

            return weights_data if weights_data else None

        except Exception as e:
            logger.warning(f"Error extracting FFN weights: {e}")
            return None

    def _extract_activation_stats(self, inputs: Any, outputs: Any) -> Optional[Dict[str, Any]]:
        """Extract activation statistics from inputs and outputs."""
        try:
            activation_data = {}

            # Input activation stats
            if inputs and isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                first_input = inputs[0]
                if isinstance(first_input, torch.Tensor):
                    activation_data["input_stats"] = collect_tensor_stats(first_input)

            # Output activation stats
            if isinstance(outputs, torch.Tensor):
                activation_data["output_stats"] = collect_tensor_stats(outputs)

                # Basic activation patterns
                with torch.no_grad():
                    sparsity = float((outputs == 0).float().mean())
                    positive_ratio = float((outputs > 0).float().mean())

                    activation_data["patterns"] = {
                        "sparsity": sparsity,
                        "positive_ratio": positive_ratio
                    }

            return activation_data if activation_data else None

        except Exception as e:
            logger.warning(f"Error extracting activation stats: {e}")
            return None

    def _extract_ffn_config(self, layer_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract FFN configuration from layer state."""
        config = {}
        parameters = layer_state.get("parameters", {})

        # Infer dimensions from parameter shapes
        for param_name, param_info in parameters.items():
            if isinstance(param_info, list) and len(param_info) >= 2:
                if "fc1" in param_name or "w1" in param_name or "gate_proj" in param_name:
                    config["input_dim"] = param_info[1]
                    config["intermediate_dim"] = param_info[0]
                elif "fc2" in param_name or "w2" in param_name or "down_proj" in param_name:
                    config["output_dim"] = param_info[0]

        # Identify FFN architecture type
        param_names = list(parameters.keys())
        if any("gate_proj" in name for name in param_names):
            config["architecture"] = "swiglu"
        elif any("wi_0" in name for name in param_names):
            config["architecture"] = "t5_style"
        elif any("fc1" in name for name in param_names):
            config["architecture"] = "standard_mlp"
        else:
            config["architecture"] = "unknown"

        # Compute expansion ratio if possible
        if "input_dim" in config and "intermediate_dim" in config:
            config["expansion_ratio"] = config["intermediate_dim"] / config["input_dim"]

        return config

    def _identify_layer_type(self, param_name: str) -> str:
        """Identify the type of FFN layer from parameter name."""
        param_lower = param_name.lower()

        if any(pattern in param_lower for pattern in ["fc1", "w1", "gate_proj", "wi_0"]):
            return "input_projection"
        elif any(pattern in param_lower for pattern in ["fc2", "w2", "down_proj", "wo"]):
            return "output_projection"
        elif any(pattern in param_lower for pattern in ["up_proj", "wi_1"]):
            return "up_projection"
        elif "bias" in param_lower:
            return "bias"
        else:
            return "unknown"

    def _compute_param_count(self, shape: List[int]) -> int:
        """Compute parameter count from shape."""
        try:
            count = 1
            for dim in shape:
                count *= dim
            return count
        except Exception:
            return 0
