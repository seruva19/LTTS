from __future__ import annotations

import logging
from typing import Any, Dict, List


class LayerPathResolver:
    """Generate candidate layer paths and normalize module requests."""

    def __init__(self, layer_wrappers: Dict[str, Any]):
        self.layer_wrappers = layer_wrappers
        self.logger = logging.getLogger(__name__)

    def normalize(self, layer_name: str, module: Any) -> str:
        """Adjust layer path for modules targeting attention projections."""
        if not layer_name:
            return layer_name

        metadata = getattr(module, "METADATA", None)
        tags = []
        if isinstance(metadata, dict):
            tags = metadata.get("tags", []) or []

        wants_attention_block = any(tag.lower() == "attention" for tag in tags)
        if wants_attention_block or any(
            suffix in layer_name for suffix in (".q_proj", ".k_proj", ".v_proj")
        ):
            return self._strip_attention_projection(layer_name)

        return layer_name

    def generate_candidates(self, layer_name: str) -> List[str]:
        """Produce possible layer path variants for automatic mapping."""
        candidates = [layer_name]

        transformations = [
            lambda name: f"original_model.{name}",
            lambda name: f"original_model.model.{name}",
            lambda name: name.replace("model.", "").replace("original_model.", ""),
            lambda name: name.replace("original_model.model.", ""),
            lambda name: name.replace(".k_proj", "")
            .replace(".v_proj", "")
            .replace(".q_proj", ""),
            lambda name: name.replace(".self_attn.k_proj", ".self_attn")
            .replace(".self_attn.v_proj", ".self_attn")
            .replace(".self_attn.q_proj", ".self_attn"),
            lambda name: (
                name.replace("layers.", "model.layers.")
                if not name.startswith("original_model")
                else name
            ),
            lambda name: (
                f"original_model.model.{name}"
                if not name.startswith("original_model")
                else name
            ),
        ]

        for transform in transformations:
            try:
                transformed = transform(layer_name)
                if transformed != layer_name and transformed not in candidates:
                    candidates.append(transformed)
            except Exception:
                continue

        self.logger.info(
            "🔧 MODULE_REGISTRY: Generated layer candidates for '%s': %s",
            layer_name,
            candidates,
        )
        return candidates

    @staticmethod
    def _strip_attention_projection(layer_name: str) -> str:
        for suffix in (".q_proj", ".k_proj", ".v_proj", ".o_proj"):
            if layer_name.endswith(suffix):
                return layer_name[: -len(suffix)]

        replacements = {
            ".self_attn.q_proj": ".self_attn",
            ".self_attn.k_proj": ".self_attn",
            ".self_attn.v_proj": ".self_attn",
            ".self_attn.o_proj": ".self_attn",
        }
        for old, new in replacements.items():
            if old in layer_name:
                return layer_name.replace(old, new)

        return layer_name
