"""
Attention Miner - Refined from AttentionCollector

Extracts attention-specific data from transformer layers with improved architecture.
"""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import numpy as np
import logging
import time
from scripts.core.miners_registry import BaseMiner
from .utils import (
    safe_get_attr,
    collect_tensor_stats,
    extract_attention_weights,
    compute_attention_entropy,
    compute_diagonal_dominance,
    extract_layer_weights
)

logger = logging.getLogger(__name__)


class AttentionMiner(BaseMiner):
    """Extracts attention weights, patterns, and analysis from attention modules."""

    def is_applicable(self, module_path: str, module_class: str) -> bool:
        """Check if this miner applies to the current module."""
        text_to_check = f"{module_path} {module_class}".lower()
        attention_patterns = [
            "attention", "attn", "self_attn", "cross_attn",
            "multihead", "multi_head"
        ]
        return any(pattern in text_to_check for pattern in attention_patterns)

    def extract(self, raw_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract comprehensive attention data from raw context with LTTS integration."""
        if not self.is_applicable(raw_context.get("module_path", ""),
                                  raw_context.get("module_class", "")):
            return None

        try:
            outputs = raw_context.get("outputs")
            inputs = raw_context.get("inputs")
            module_path = raw_context.get("module_path", "")

            attention_data = {}

            # Try multiple sources for attention weights
            attention_weights = None

            # 1. PRIORITY: Get from LTTS dynamic capture system
            ltts_model = raw_context.get("ltts_model")
            if ltts_model and hasattr(ltts_model, 'captured_attn'):
                # Try exact module path match first
                captured_weights = ltts_model.captured_attn.get(module_path)
                if captured_weights:
                    # Get the latest capture (highest decode step)
                    latest_step = max(captured_weights.keys()) if captured_weights else 0
                    attention_weights = captured_weights[latest_step]
                    attention_data["capture_source"] = "ltts_dynamic_capture"
                    attention_data["decode_step"] = latest_step
                    attention_data["capture_meta"] = self._get_capture_meta(
                        ltts_model, module_path, latest_step
                    )
                else:
                    # Try alternative path patterns
                    import re
                    layer_match = re.search(r'layers\.(\d+)\.', module_path)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))
                        for alt_pattern in [
                            f"model.layers.{layer_idx}.self_attn",
                            f"layers.{layer_idx}.self_attn",
                            f"transformer.h.{layer_idx}.attn"
                        ]:
                            captured_weights = ltts_model.captured_attn.get(alt_pattern)
                            if captured_weights:
                                latest_step = max(captured_weights.keys())
                                attention_weights = captured_weights[latest_step]
                                attention_data["capture_source"] = f"ltts_dynamic_capture_{alt_pattern}"
                                attention_data["decode_step"] = latest_step
                                attention_data["capture_meta"] = self._get_capture_meta(
                                    ltts_model, alt_pattern, latest_step
                                )
                                break

            # Fall back to layer_attention_weights.
            if attention_weights is None and ltts_model and hasattr(ltts_model, 'layer_attention_weights'):
                layer_attention = ltts_model.layer_attention_weights.get(module_path)
                if layer_attention is not None:
                    attention_weights = layer_attention
                    attention_data["capture_source"] = "ltts_layer_weights"

            # 3. Try to get cached model-level attention weights
            if attention_weights is None and ltts_model and hasattr(ltts_model, 'cached_attention_weights'):
                attention_weights = ltts_model.cached_attention_weights
                attention_data["capture_source"] = "ltts_cached"

            # 4. Try direct context injection
            if attention_weights is None:
                attention_weights = raw_context.get("attention_weights")
                if attention_weights is not None:
                    attention_data["capture_source"] = "context_direct"

            # 5. Fallback to extracting from outputs
            if attention_weights is None:
                attention_weights = extract_attention_weights(outputs)
                if attention_weights is not None:
                    attention_data["capture_source"] = "extracted_from_outputs"

            if attention_weights is not None:
                # COMPREHENSIVE ATTENTION ANALYSIS
                attention_data["weights"] = self._analyze_attention_weights(attention_weights)

                # Add raw tensor for visualization
                attention_data["raw_tensor"] = attention_weights.detach().cpu()

                attention_data["quality"] = self._get_quality_flags(
                    attention_data.get("capture_source"),
                    attention_data.get("capture_meta"),
                )

                # Extract multi-layer comparison data if available
                if ltts_model and hasattr(ltts_model, 'captured_attn'):
                    attention_data["multi_layer_analysis"] = self._extract_multi_layer_data(ltts_model, module_path)

                # Generate visualization-ready data
                attention_data["visualization"] = self._prepare_visualization_data(attention_weights, raw_context)

                # Extract token-level analysis if tokenizer available
                attention_data["token_analysis"] = self._extract_token_analysis(attention_weights, raw_context)

                # Pattern classification and insights
                attention_data["insights"] = self._generate_attention_insights(attention_weights, module_path)

            # Extract attention configuration from layer state
            layer_state = raw_context.get("layer_state", {})
            attention_data["config"] = self._extract_attention_config(layer_state)

            # Extract QKV projections info if available in module state
            model_state = raw_context.get("model_state", {})
            if "layer_structure" in model_state:
                attention_data["projections"] = self._extract_projection_info(module_path, model_state)

            # Extract attention masks from inputs
            if inputs:
                masks = self._extract_attention_masks(inputs)
                if masks:
                    attention_data["masks"] = masks

            # Add metadata
            attention_data["metadata"] = {
                "module_path": module_path,
                "extraction_timestamp": time.time(),
                "ltts_version": "dynamic_layer_aware",
            }

            return attention_data if attention_data else None

        except Exception as e:
            logger.error(f"Error in AttentionMiner.extract: {e}")
            return {"error": str(e)}

    def _analyze_attention_weights(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention weights and compute patterns."""
        try:
            analysis = {
                "basic_stats": collect_tensor_stats(attention_weights),
                "num_heads": self._get_num_heads(attention_weights),
                "sequence_length": attention_weights.shape[-1] if attention_weights.dim() >= 2 else 0
            }

            # Head-wise analysis if multi-head
            if len(attention_weights.shape) >= 3:
                analysis["head_analysis"] = self._analyze_attention_heads(attention_weights)

            # Pattern analysis
            analysis["patterns"] = {
                "entropy": compute_attention_entropy(attention_weights),
                "diagonal_dominance": compute_diagonal_dominance(attention_weights)
            }

            return analysis

        except Exception as e:
            logger.warning(f"Error analyzing attention weights: {e}")
            return {"error": str(e)}

    def _get_num_heads(self, attention_weights: torch.Tensor) -> int:
        """Extract number of attention heads from tensor shape."""
        if len(attention_weights.shape) >= 3:
            return attention_weights.shape[-3]  # Usually [batch, heads, seq, seq]
        return 1

    def _analyze_attention_heads(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention patterns across heads."""
        try:
            if len(attention_weights.shape) < 3:
                return {}

            head_stats = []
            num_heads = attention_weights.shape[-3]

            for head_idx in range(min(num_heads, 32)):  # Limit to first 32 heads for performance
                head_weights = attention_weights[..., head_idx, :, :]

                # Basic metrics only
                entropy = compute_attention_entropy(head_weights)
                max_attention = float(head_weights.max())
                diagonal_dominance = compute_diagonal_dominance(head_weights)
                sparsity = float((head_weights < 0.01).float().mean())

                head_stats.append({
                    "head_index": head_idx,
                    "entropy": entropy,
                    "max_attention": max_attention,
                    "diagonal_dominance": diagonal_dominance,
                    "sparsity": sparsity
                })

            return {
                "head_statistics": head_stats,
                "total_heads": num_heads
            }

        except Exception as e:
            logger.warning(f"Error analyzing attention heads: {e}")
            return {"error": str(e)}

    def _extract_attention_config(self, layer_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract attention configuration from layer state."""
        config = {}

        # Extract parameter info that might indicate attention config
        parameters = layer_state.get("parameters", {})

        # Try to infer config from parameter shapes
        for param_name, param_info in parameters.items():
            if isinstance(param_info, list):  # Shape info
                if "q_proj" in param_name or "query" in param_name:
                    config["query_dim"] = param_info[0] if param_info else None
                elif "k_proj" in param_name or "key" in param_name:
                    config["key_dim"] = param_info[0] if param_info else None
                elif "v_proj" in param_name or "value" in param_name:
                    config["value_dim"] = param_info[0] if param_info else None

        return config

    def _get_capture_meta(
        self, ltts_model, module_path: str, decode_step: int
    ) -> Optional[Dict[str, Any]]:
        try:
            manager = getattr(ltts_model, "attention_manager", None)
            if manager is None or not hasattr(manager, "capture_meta"):
                return None
            return manager.capture_meta.get(module_path, {}).get(decode_step)
        except Exception:
            return None

    def _get_quality_flags(
        self,
        capture_source: Optional[str],
        capture_meta: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(capture_meta, dict):
            return {
                "approximate": bool(capture_meta.get("approximate", True)),
                "source": capture_meta.get("source", capture_source),
            }
        approximate = True
        if capture_source in {"extracted_from_outputs", "context_direct"}:
            approximate = False
        return {"approximate": approximate, "source": capture_source}

    def _extract_projection_info(self, module_path: str, model_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract QKV projection information from model structure."""
        projections = {}
        layer_structure = model_state.get("layer_structure", {})

        # Look for QKV projections in the layer structure
        qkv_patterns = ["q_proj", "k_proj", "v_proj", "query", "key", "value"]

        for layer_name, layer_info in layer_structure.items():
            if module_path in layer_name:
                for pattern in qkv_patterns:
                    if pattern in layer_name:
                        projections[pattern] = {
                            "class": layer_info.get("class", "unknown"),
                            "wrapped": layer_info.get("wrapped", False)
                        }

        return projections

    def _extract_attention_masks(self, inputs) -> Optional[Dict[str, Any]]:
        """Extract attention masks from inputs."""
        try:
            masks = {}

            if isinstance(inputs, dict):
                attention_mask = inputs.get("attention_mask")
                if isinstance(attention_mask, torch.Tensor):
                    masks["attention_mask"] = collect_tensor_stats(attention_mask)

            if isinstance(inputs, (list, tuple)):
                for i, input_item in enumerate(inputs):
                    if isinstance(input_item, torch.Tensor):
                        # Attention masks are typically boolean or float tensors
                        if input_item.dtype in [torch.bool, torch.float16, torch.float32]:
                            if input_item.dim() >= 2:
                                masks[f"mask_{i}"] = collect_tensor_stats(input_item)

            return masks if masks else None

        except Exception as e:
            logger.warning(f"Error extracting attention masks: {e}")
            return None

    def _extract_multi_layer_data(self, ltts_model, current_module_path: str) -> Dict[str, Any]:
        """Extract attention data from multiple layers for comparison."""
        try:
            multi_layer_data = {}

            if not hasattr(ltts_model, 'captured_attn'):
                return {}

            # Get data from all captured layers
            for layer_path, captures in ltts_model.captured_attn.items():
                if captures:
                    latest_step = max(captures.keys())
                    layer_weights = captures[latest_step]

                    # Extract layer index
                    import re
                    layer_match = re.search(r'layers\.(\d+)\.', layer_path)
                    if layer_match:
                        layer_idx = int(layer_match.group(1))

                        # Basic stats for this layer
                        multi_layer_data[f"layer_{layer_idx}"] = {
                            "mean": layer_weights.mean().item(),
                            "std": layer_weights.std().item(),
                            "entropy": compute_attention_entropy(layer_weights),
                            "max_attention": layer_weights.max().item(),
                            "layer_path": layer_path
                        }

            return multi_layer_data

        except Exception as e:
            logger.warning(f"Error extracting multi-layer data: {e}")
            return {}

    def _prepare_visualization_data(self, attention_weights: torch.Tensor, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data specifically for attention visualization."""
        try:
            vis_data = {}

            # Get tensor dimensions
            shape = attention_weights.shape
            vis_data["tensor_shape"] = list(shape)

            if len(shape) >= 2:
                # Average across batch dimension if present
                if len(shape) == 4:  # [batch, heads, seq, seq]
                    batch_avg = attention_weights.mean(dim=0)  # [heads, seq, seq]
                    vis_data["batch_averaged"] = batch_avg.detach().cpu()

                    # Create head-wise matrices for individual visualization
                    vis_data["per_head_matrices"] = []
                    for head_idx in range(min(shape[1], 12)):  # Limit to 12 heads
                        head_matrix = batch_avg[head_idx].detach().cpu().numpy()
                        vis_data["per_head_matrices"].append({
                            "head_index": head_idx,
                            "matrix": head_matrix.tolist(),
                            "shape": list(head_matrix.shape)
                        })

                    # Create aggregated matrix (average across heads)
                    aggregated = batch_avg.mean(dim=0).detach().cpu().numpy()
                    vis_data["aggregated_matrix"] = {
                        "matrix": aggregated.tolist(),
                        "shape": list(aggregated.shape)
                    }

                elif len(shape) == 3:  # [heads, seq, seq]
                    vis_data["per_head_matrices"] = []
                    for head_idx in range(min(shape[0], 12)):
                        head_matrix = attention_weights[head_idx].detach().cpu().numpy()
                        vis_data["per_head_matrices"].append({
                            "head_index": head_idx,
                            "matrix": head_matrix.tolist(),
                            "shape": list(head_matrix.shape)
                        })

                    # Average across heads
                    aggregated = attention_weights.mean(dim=0).detach().cpu().numpy()
                    vis_data["aggregated_matrix"] = {
                        "matrix": aggregated.tolist(),
                        "shape": list(aggregated.shape)
                    }

                elif len(shape) == 2:  # [seq, seq]
                    matrix = attention_weights.detach().cpu().numpy()
                    vis_data["aggregated_matrix"] = {
                        "matrix": matrix.tolist(),
                        "shape": list(matrix.shape)
                    }

            # Add colormap suggestions
            vis_data["colormap_suggestions"] = ["viridis", "plasma", "Blues", "Reds"]

            # Add normalization info
            vis_data["normalization"] = {
                "min_value": float(attention_weights.min()),
                "max_value": float(attention_weights.max()),
                "is_normalized": bool((attention_weights.sum(dim=-1) - 1.0).abs().max() < 0.01)
            }

            return vis_data

        except Exception as e:
            logger.warning(f"Error preparing visualization data: {e}")
            return {}

    def _extract_token_analysis(self, attention_weights: torch.Tensor, raw_context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract token-level attention analysis."""
        try:
            token_data = {}

            # Try to get tokens from context
            tokens = raw_context.get("tokens", [])
            input_ids = raw_context.get("input_ids")

            # If no tokens but have input_ids, try to get tokenizer
            if not tokens and input_ids is not None:
                # Try to find tokenizer in context or reconstruct
                # This is a simplified approach - in practice you'd want tokenizer access
                seq_len = attention_weights.shape[-1]
                tokens = [f"token_{i}" for i in range(seq_len)]

            if tokens and len(attention_weights.shape) >= 2:
                # Ensure we have the right number of tokens
                seq_len = attention_weights.shape[-1]
                if len(tokens) > seq_len:
                    tokens = tokens[:seq_len]
                elif len(tokens) < seq_len:
                    tokens.extend([f"token_{i}" for i in range(len(tokens), seq_len)])

                token_data["tokens"] = tokens
                token_data["sequence_length"] = seq_len

                # Get aggregated attention matrix (average across heads and batch)
                if len(attention_weights.shape) == 4:  # [batch, heads, seq, seq]
                    agg_matrix = attention_weights.mean(dim=(0, 1))
                elif len(attention_weights.shape) == 3:  # [heads, seq, seq]
                    agg_matrix = attention_weights.mean(dim=0)
                else:  # [seq, seq]
                    agg_matrix = attention_weights

                # Token-to-token attention analysis
                token_attention = {}
                for i, source_token in enumerate(tokens):
                    if i < agg_matrix.shape[0]:
                        attention_dist = agg_matrix[i].detach().cpu().numpy()

                        # Find top attended tokens
                        top_indices = attention_dist.argsort()[-5:][::-1]  # Top 5
                        top_attentions = []

                        for idx in top_indices:
                            if idx < len(tokens):
                                top_attentions.append({
                                    "target_token": tokens[idx],
                                    "attention_weight": float(attention_dist[idx]),
                                    "target_index": int(idx)
                                })

                        token_attention[source_token] = {
                            "source_index": i,
                            "attention_distribution": attention_dist.tolist(),
                            "top_attended": top_attentions,
                            "self_attention": float(attention_dist[i]) if i < len(attention_dist) else 0.0
                        }

                token_data["token_attention"] = token_attention

                # Global token statistics
                self_attention_scores = [agg_matrix[i, i].item() for i in range(min(agg_matrix.shape[0], len(tokens)))]
                token_data["global_stats"] = {
                    "average_self_attention": sum(self_attention_scores) / len(self_attention_scores) if self_attention_scores else 0.0,
                    "max_self_attention": max(self_attention_scores) if self_attention_scores else 0.0,
                    "min_self_attention": min(self_attention_scores) if self_attention_scores else 0.0
                }

            return token_data

        except Exception as e:
            logger.warning(f"Error extracting token analysis: {e}")
            return {}

    def _generate_attention_insights(self, attention_weights: torch.Tensor, module_path: str) -> Dict[str, Any]:
        """Generate basic insights about attention patterns."""
        try:
            insights = {}

            # Extract layer information
            import re
            layer_match = re.search(r'layers\.(\d+)\.', module_path)
            layer_idx = int(layer_match.group(1)) if layer_match else None

            if layer_idx is not None:
                insights["layer_index"] = layer_idx

            # Basic pattern analysis
            if len(attention_weights.shape) >= 2:
                # Get aggregated matrix
                if len(attention_weights.shape) == 4:
                    agg_matrix = attention_weights.mean(dim=(0, 1))
                elif len(attention_weights.shape) == 3:
                    agg_matrix = attention_weights.mean(dim=0)
                else:
                    agg_matrix = attention_weights

                # Basic pattern detection
                diagonal_strength = torch.diag(agg_matrix).mean().item()

                insights["pattern_analysis"] = {
                    "diagonal_dominance": diagonal_strength,
                    "attention_concentration": float(attention_weights.max()),
                    "attention_entropy": compute_attention_entropy(attention_weights)
                }

                # Quality assessment
                insights["quality_metrics"] = {
                    "is_well_normalized": bool((attention_weights.sum(dim=-1) - 1.0).abs().max() < 0.01),
                    "has_nan_values": bool(torch.isnan(attention_weights).any()),
                    "has_negative_values": bool((attention_weights < 0).any()),
                    "sparsity_ratio": float((attention_weights < 0.01).float().mean())
                }

            return insights

        except Exception as e:
            logger.warning(f"Error generating attention insights: {e}")
            return {}
