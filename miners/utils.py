"""
Utility functions for miners - refined from original tensor_utils
"""

import torch
import numpy as np
from typing import Any, Dict, Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


def safe_get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get attribute with error handling."""
    try:
        return getattr(obj, attr, default)
    except Exception as e:
        logger.warning(f"Error getting attribute {attr}: {str(e)}")
        return default


def collect_tensor_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    """Collect comprehensive tensor statistics."""
    try:
        with torch.no_grad():
            return {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "requires_grad": tensor.requires_grad,
                "mean": float(tensor.mean().item()) if tensor.numel() > 0 else 0.0,
                "std": float(tensor.std().item()) if tensor.numel() > 0 else 0.0,
                "min": float(tensor.min().item()) if tensor.numel() > 0 else 0.0,
                "max": float(tensor.max().item()) if tensor.numel() > 0 else 0.0,
                "norm": float(tensor.norm().item()) if tensor.numel() > 0 else 0.0,
                "memory_mb": tensor.element_size() * tensor.nelement() / (1024 * 1024)
            }
    except Exception as e:
        logger.warning(f"Error collecting tensor stats: {str(e)}")
        return {"error": str(e)}


def extract_attention_weights(outputs: Any) -> Optional[torch.Tensor]:
    """Extract attention weights from various output formats."""
    try:
        # Handle transformers model outputs with attentions field
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attentions = outputs.attentions
            if isinstance(attentions, (list, tuple)) and len(attentions) > 0:
                # Return attention weights from the last layer
                last_layer_attention = attentions[-1]
                if isinstance(last_layer_attention, torch.Tensor):
                    return last_layer_attention
                elif isinstance(last_layer_attention, (list, tuple)) and len(last_layer_attention) > 0:
                    return last_layer_attention[-1]  # Last element of last layer
            elif isinstance(attentions, torch.Tensor):
                return attentions

        # Handle tuple outputs (most common for layer outputs)
        if isinstance(outputs, tuple):
            # Look for attention weights specifically, skip logits (large last dimension)
            attention_candidates = []
            for i, output in enumerate(outputs):
                if isinstance(output, torch.Tensor):
                    # Check if tensor shape suggests attention weights [batch, heads, seq, seq]
                    if output.dim() >= 3 and output.shape[-1] == output.shape[-2]:
                        # Additional validation: attention weights should be non-negative and roughly sum to 1
                        if output.min() >= -1e-6 and torch.allclose(output.sum(dim=-1), torch.ones_like(output.sum(dim=-1)), atol=1e-2):
                            attention_candidates.append(output)

            # Return the best attention candidate (prefer ones with more heads)
            if attention_candidates:
                # Sort by number of heads (descending) to get the most likely attention weights
                attention_candidates.sort(key=lambda x: x.shape[-3] if x.dim() >= 3 else 0, reverse=True)
                return attention_candidates[0]

        # Handle direct tensor input
        elif isinstance(outputs, torch.Tensor):
            # Check if tensor shape suggests attention weights [batch, heads, seq, seq]
            if outputs.dim() >= 3 and outputs.shape[-1] == outputs.shape[-2]:
                # Additional validation: attention weights should be non-negative and roughly sum to 1
                if outputs.min() >= -1e-6 and torch.allclose(outputs.sum(dim=-1), torch.ones_like(outputs.sum(dim=-1)), atol=1e-2):
                    return outputs

        # Handle named outputs (HuggingFace style)
        elif hasattr(outputs, 'attention_weights') and outputs.attention_weights is not None:
            return outputs.attention_weights

        # Handle dict outputs
        elif isinstance(outputs, dict):
            for key in ['attention_weights', 'attentions', 'attn_weights', 'weights']:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    return outputs[key]
                elif key in outputs and isinstance(outputs[key], (list, tuple)) and len(outputs[key]) > 0:
                    # Handle list of attention tensors
                    last_attention = outputs[key][-1]
                    if isinstance(last_attention, torch.Tensor):
                        return last_attention

        # Handle ModelOutput objects from transformers
        elif hasattr(outputs, '__dict__') and hasattr(outputs, 'to_tuple'):
            # Try to convert to tuple and process
            try:
                tuple_outputs = outputs.to_tuple()
                if isinstance(tuple_outputs, tuple):
                    return extract_attention_weights(tuple_outputs)
            except Exception as e:
                logger.debug("ModelOutput to_tuple conversion failed: %s", e)

        return None
    except Exception as e:
        logger.warning(f"Error extracting attention weights: {e}")
        return None


def compute_attention_entropy(weights: torch.Tensor) -> float:
    """Compute mean attention entropy per attention distribution."""
    try:
        # Entropy per distribution (last dim), then mean over positions,
        # heads and batch instead of summing everything together.
        clamped = weights.clamp_min(1e-8)  # Avoid log(0)
        entropies = -(clamped * torch.log(clamped)).sum(dim=-1)
        return float(entropies.mean())
    except Exception as e:
        logger.warning(f"Error computing attention entropy: {e}")
        return 0.0


def compute_diagonal_dominance(weights: torch.Tensor) -> float:
    """Measure how much attention focuses on diagonal (self-attention)."""
    try:
        if weights.shape[-1] != weights.shape[-2]:
            return 0.0
        diagonal_sum = torch.diagonal(weights, dim1=-2, dim2=-1).sum()
        total_sum = weights.sum()
        return float(diagonal_sum / (total_sum + 1e-8))
    except Exception as e:
        logger.warning(f"Error computing diagonal dominance: {e}")
        return 0.0


def extract_layer_weights(module: torch.nn.Module, layer_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Extract weights from specified layer names."""
    weights_data = {}

    for layer_name in layer_names:
        layer = safe_get_attr(module, layer_name)
        if layer is not None and hasattr(layer, "weight"):
            layer_data = {
                "weights": collect_tensor_stats(layer.weight)
            }

            # Add bias if available
            if hasattr(layer, "bias") and layer.bias is not None:
                layer_data["bias"] = collect_tensor_stats(layer.bias)

            weights_data[layer_name] = layer_data

    return weights_data
