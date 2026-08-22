"""
MoE Miner - Refined from ExpertCollector

Extracts Mixture-of-Experts routing information, expert utilization, and load balancing data.
"""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import logging
from scripts.core.miners_registry import BaseMiner
from .utils import safe_get_attr, collect_tensor_stats

logger = logging.getLogger(__name__)


class MoEMiner(BaseMiner):
    """Extracts MoE routing, expert utilization, and load balancing information."""

    def is_applicable(self, module_path: str, module_class: str) -> bool:
        """Check if this miner applies to MoE modules."""
        text_to_check = f"{module_path} {module_class}".lower()
        moe_patterns = [
            "moe", "expert", "switch", "router", "gate", "mixture"
        ]
        return any(pattern in text_to_check for pattern in moe_patterns)

    def extract(self, raw_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract MoE data from raw context."""
        if not self.is_applicable(raw_context.get("module_path", ""),
                                  raw_context.get("module_class", "")):
            return None

        try:
            outputs = raw_context.get("outputs")
            inputs = raw_context.get("inputs")
            layer_state = raw_context.get("layer_state", {})

            moe_data = {}

            # Extract routing information from outputs
            routing_info = self._extract_routing_info(outputs)
            if routing_info:
                moe_data["routing"] = routing_info

            # Extract expert utilization metrics
            utilization = self._extract_expert_utilization(outputs, layer_state)
            if utilization:
                moe_data["utilization"] = utilization

            # Extract load balancing information
            load_balancing = self._extract_load_balancing(outputs)
            if load_balancing:
                moe_data["load_balancing"] = load_balancing

            # Extract expert configuration
            config = self._extract_moe_config(layer_state)
            if config:
                moe_data["config"] = config

            return moe_data if moe_data else None

        except Exception as e:
            logger.error(f"Error in MoEMiner.extract: {e}")
            return {"error": str(e)}

    def _extract_routing_info(self, outputs: Any) -> Optional[Dict[str, Any]]:
        """Extract routing weights and decisions from outputs."""
        try:
            routing_data = {}

            # Check for routing weights in outputs
            if isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    if isinstance(output, torch.Tensor):
                        # Router outputs are typically probability distributions
                        if output.dim() >= 2 and output.shape[-1] > 1:
                            # Check if this looks like routing probabilities (sum to 1, non-negative)
                            if output.min() >= -1e-6 and torch.allclose(output.sum(dim=-1), torch.ones(output.shape[:-1]), atol=1e-2):
                                routing_data[f"routing_probs_{i}"] = {
                                    "stats": collect_tensor_stats(output),
                                    "top_experts": self._get_top_experts(output),
                                    "entropy": self._compute_routing_entropy(output)
                                }
                            # Also check for logits that could be routing scores
                            elif output.dtype in [torch.float16, torch.float32, torch.float64]:
                                # Apply softmax to see if it creates valid probabilities
                                try:
                                    probs = torch.softmax(output, dim=-1)
                                    routing_data[f"routing_logits_{i}"] = {
                                        "stats": collect_tensor_stats(output),
                                        "probs_stats": collect_tensor_stats(probs),
                                        "top_experts": self._get_top_experts(probs),
                                        "entropy": self._compute_routing_entropy(probs)
                                    }
                                except Exception:
                                    pass

            elif isinstance(outputs, dict):
                for key, value in outputs.items():
                    if "router" in key.lower() or "gate" in key.lower():
                        if isinstance(value, torch.Tensor):
                            routing_data[key] = {
                                "stats": collect_tensor_stats(value),
                                "top_experts": self._get_top_experts(value),
                                "entropy": self._compute_routing_entropy(value)
                            }

            return routing_data if routing_data else None

        except Exception as e:
            logger.warning(f"Error extracting routing info: {e}")
            return None

    def _extract_expert_utilization(self, outputs: Any, layer_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract expert utilization statistics."""
        try:
            utilization_data = {}

            # Look for expert-related parameters in layer state
            parameters = layer_state.get("parameters", {})
            expert_params = {}

            for param_name, param_info in parameters.items():
                if "expert" in param_name.lower():
                    expert_params[param_name] = param_info

            if expert_params:
                utilization_data["expert_parameters"] = expert_params
                utilization_data["num_experts"] = len([p for p in expert_params.keys() if "expert_" in p])

            return utilization_data if utilization_data else None

        except Exception as e:
            logger.warning(f"Error extracting expert utilization: {e}")
            return None

    def _extract_load_balancing(self, outputs: Any) -> Optional[Dict[str, Any]]:
        """Extract load balancing loss and metrics."""
        try:
            load_balance_data = {}

            if isinstance(outputs, tuple):
                for i, output in enumerate(outputs):
                    if isinstance(output, torch.Tensor) and output.numel() == 1:
                        # Single scalar values might be load balancing losses
                        load_balance_data[f"potential_loss_{i}"] = {
                            "value": float(output.item()),
                            "stats": collect_tensor_stats(output)
                        }

            elif isinstance(outputs, dict):
                for key, value in outputs.items():
                    if any(term in key.lower() for term in ["loss", "balance", "aux"]):
                        if isinstance(value, torch.Tensor):
                            load_balance_data[key] = {
                                "value": float(value.item()) if value.numel() == 1 else "multi_value",
                                "stats": collect_tensor_stats(value)
                            }

            return load_balance_data if load_balance_data else None

        except Exception as e:
            logger.warning(f"Error extracting load balancing: {e}")
            return None

    def _extract_moe_config(self, layer_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract MoE configuration from layer state."""
        config = {}

        # Extract MoE-related configuration from parameters
        parameters = layer_state.get("parameters", {})

        # Count experts
        expert_count = len([p for p in parameters.keys() if "expert" in p.lower()])
        if expert_count > 0:
            config["estimated_num_experts"] = expert_count

        # Look for router/gate parameters
        router_params = [p for p in parameters.keys() if any(term in p.lower() for term in ["router", "gate"])]
        if router_params:
            config["router_parameters"] = router_params

        return config

    def _get_top_experts(self, routing_probs: torch.Tensor, top_k: int = 3) -> Dict[str, Any]:
        """Get top-k expert indices and their probabilities."""
        try:
            with torch.no_grad():
                top_probs, top_indices = torch.topk(routing_probs, min(top_k, routing_probs.shape[-1]), dim=-1)

                return {
                    "top_indices": top_indices.cpu().tolist(),
                    "top_probabilities": top_probs.cpu().tolist(),
                    "expert_distribution": routing_probs.mean(dim=0).cpu().tolist() if routing_probs.dim() > 1 else routing_probs.cpu().tolist()
                }
        except Exception as e:
            logger.warning(f"Error getting top experts: {e}")
            return {}

    def _compute_routing_entropy(self, routing_probs: torch.Tensor) -> float:
        """Compute entropy of routing distribution."""
        try:
            with torch.no_grad():
                # Compute entropy across expert dimension
                log_probs = torch.log(routing_probs + 1e-8)
                entropy = -(routing_probs * log_probs).sum(dim=-1).mean()
                return float(entropy.item())
        except Exception as e:
            logger.warning(f"Error computing routing entropy: {e}")
            return 0.0
