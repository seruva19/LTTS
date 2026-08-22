"""
Embeddings Miner - Extracts embedding layer information

Handles token embeddings, positional embeddings, and embedding statistics.
"""

from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import logging
from scripts.core.miners_registry import BaseMiner
from .utils import safe_get_attr, collect_tensor_stats

logger = logging.getLogger(__name__)


class EmbeddingsMiner(BaseMiner):
    """Extracts token and positional embedding information."""

    def is_applicable(self, module_path: str, module_class: str) -> bool:
        """Check if this miner applies to embedding modules."""
        text_to_check = f"{module_path} {module_class}".lower()
        embedding_patterns = [
            "embedding", "embed", "token", "position", "pos", "wte", "wpe"
        ]
        return any(pattern in text_to_check for pattern in embedding_patterns)

    def extract(self, raw_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract embedding data from raw context."""
        if not self.is_applicable(raw_context.get("module_path", ""),
                                  raw_context.get("module_class", "")):
            return None

        try:
            outputs = raw_context.get("outputs")
            inputs = raw_context.get("inputs")
            layer_state = raw_context.get("layer_state", {})
            module_path = raw_context.get("module_path", "")

            embedding_data = {}

            # Extract embedding weights information
            weights_info = self._extract_embedding_weights(layer_state)
            if weights_info:
                embedding_data["weights"] = weights_info

            # Extract embedding outputs
            if isinstance(outputs, torch.Tensor):
                embedding_data["output_stats"] = collect_tensor_stats(outputs)
                embedding_data["embedding_analysis"] = self._analyze_embeddings(outputs)

            # Extract embedding configuration
            config = self._extract_embedding_config(layer_state, module_path)
            if config:
                embedding_data["config"] = config

            return embedding_data if embedding_data else None

        except Exception as e:
            logger.error(f"Error in EmbeddingsMiner.extract: {e}")
            return {"error": str(e)}

    def _extract_embedding_weights(self, layer_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract embedding weight information."""
        try:
            parameters = layer_state.get("parameters", {})
            weights_data = {}

            for param_name, param_info in parameters.items():
                if "weight" in param_name.lower():
                    if isinstance(param_info, list) and len(param_info) >= 2:
                        weights_data[param_name] = {
                            "shape": param_info,
                            "vocab_size": param_info[0] if len(param_info) > 0 else None,
                            "embedding_dim": param_info[1] if len(param_info) > 1 else None,
                            "parameter_count": param_info[0] * param_info[1] if len(param_info) >= 2 else 0
                        }

            return weights_data if weights_data else None

        except Exception as e:
            logger.warning(f"Error extracting embedding weights: {e}")
            return None

    def _analyze_embeddings(self, embeddings: torch.Tensor) -> Dict[str, Any]:
        """Analyze embedding tensor properties."""
        try:
            with torch.no_grad():
                analysis = {
                    "norm_distribution": {
                        "mean_norm": float(embeddings.norm(dim=-1).mean()),
                        "std_norm": float(embeddings.norm(dim=-1).std()),
                        "min_norm": float(embeddings.norm(dim=-1).min()),
                        "max_norm": float(embeddings.norm(dim=-1).max())
                    }
                }

                # Basic similarity analysis
                if embeddings.shape[0] > 1:
                    sample_size = min(100, embeddings.shape[0])
                    sample_embeddings = embeddings[:sample_size]

                    # Cosine similarity analysis
                    cosine_sim = torch.nn.functional.cosine_similarity(
                        sample_embeddings.unsqueeze(1),
                        sample_embeddings.unsqueeze(0),
                        dim=-1
                    )

                    # Remove diagonal (self-similarity)
                    mask = ~torch.eye(cosine_sim.shape[0], dtype=torch.bool)
                    off_diag_sim = cosine_sim[mask]

                    analysis["similarity"] = {
                        "mean_cosine_sim": float(cosine_sim.mean()),
                        "std_cosine_sim": float(cosine_sim.std()),
                        "mean_off_diagonal_sim": float(off_diag_sim.mean())
                    }

                return analysis

        except Exception as e:
            logger.warning(f"Error analyzing embeddings: {e}")
            return {"error": str(e)}

    def _extract_embedding_config(self, layer_state: Dict[str, Any], module_path: str) -> Dict[str, Any]:
        """Extract embedding configuration."""
        config = {
            "embedding_type": self._identify_embedding_type(module_path)
        }

        parameters = layer_state.get("parameters", {})
        for param_name, param_info in parameters.items():
            if isinstance(param_info, list) and len(param_info) >= 2:
                config["vocab_size"] = param_info[0]
                config["embedding_dimension"] = param_info[1]
                break

        return config

    def _identify_embedding_type(self, module_path: str) -> str:
        """Identify the type of embedding from module path."""
        path_lower = module_path.lower()

        if any(pattern in path_lower for pattern in ["position", "pos", "wpe"]):
            return "positional"
        elif any(pattern in path_lower for pattern in ["token", "wte", "word"]):
            return "token"
        elif "embed" in path_lower:
            return "general"
        else:
            return "unknown"
