"""Feature-space interventions for SAEs and other encoder/decoder dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Dict, Sequence

import torch

FeatureTransform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class FeatureInterventionSpec:
    feature_ids: Sequence[int]
    operation: str = "add"
    value: float = 1.0
    clamp_min: float = -1.0
    clamp_max: float = 1.0
    positions: str = "all"
    emit_summary: bool = True

    def __post_init__(self) -> None:
        if not self.feature_ids:
            raise ValueError("Feature intervention requires at least one feature id")
        if self.operation not in {"add", "multiply", "set", "clamp"}:
            raise ValueError(f"Unsupported feature operation: {self.operation}")
        if self.positions not in {"all", "last"}:
            raise ValueError("Feature positions must be 'all' or 'last'")


def _hidden(outputs: Any) -> torch.Tensor:
    candidate = (
        outputs[0] if isinstance(outputs, (tuple, list)) and outputs else outputs
    )
    if not isinstance(candidate, torch.Tensor) or candidate.ndim != 3:
        raise TypeError(
            "Feature intervention requires [batch, sequence, hidden] output"
        )
    return candidate


def _replace(outputs: Any, hidden: torch.Tensor) -> Any:
    if isinstance(outputs, tuple):
        return (hidden, *outputs[1:])
    if isinstance(outputs, list):
        return [hidden, *outputs[1:]]
    return hidden


def build_feature_intervention(
    *,
    encoder: FeatureTransform,
    decoder: FeatureTransform,
    spec: FeatureInterventionSpec,
    name: str = "feature_intervention",
) -> ModuleType:
    """Create an LTTS module that edits encoded features during inference.

    The decoded feature delta is added to the original hidden state. This keeps
    the encoder/decoder reconstruction error out of the intervention itself.
    """

    module = ModuleType(name)
    module.METADATA = {
        "name": name,
        "version": "1",
        "description": "Runtime feature-space intervention",
        "event_types": ["layer_after"],
        "requires": ["hidden_states"],
        "target_roles": ["block", "mlp", "activation"],
    }

    def interceptor(event):
        hidden = _hidden(event.context.outputs)
        encoded = encoder(hidden)
        if (
            not isinstance(encoded, torch.Tensor)
            or encoded.shape[:-1] != hidden.shape[:-1]
        ):
            raise ValueError(
                "Feature encoder must preserve batch and sequence dimensions"
            )
        ids = tuple(dict.fromkeys(int(item) for item in spec.feature_ids))
        invalid = [item for item in ids if not 0 <= item < encoded.shape[-1]]
        if invalid:
            raise IndexError(
                f"Feature ids outside encoded width {encoded.shape[-1]}: {invalid}"
            )

        changed_features = encoded.clone()
        index = torch.tensor(ids, device=encoded.device, dtype=torch.long)
        selected = (
            changed_features[:, :, index]
            if spec.positions == "all"
            else changed_features[:, -1:, index]
        )
        if spec.operation == "add":
            changed = selected + spec.value
        elif spec.operation == "multiply":
            changed = selected * spec.value
        elif spec.operation == "set":
            changed = torch.full_like(selected, spec.value)
        else:
            low, high = sorted((spec.clamp_min, spec.clamp_max))
            changed = selected.clamp(low, high)
        if spec.positions == "all":
            changed_features[:, :, index] = changed
        else:
            changed_features[:, -1:, index] = changed

        decoded_original = decoder(encoded)
        decoded_changed = decoder(changed_features)
        if (
            not isinstance(decoded_original, torch.Tensor)
            or not isinstance(decoded_changed, torch.Tensor)
            or decoded_original.shape != hidden.shape
            or decoded_changed.shape != hidden.shape
        ):
            raise ValueError("Feature decoder must reconstruct the hidden-state shape")
        modified = hidden + (decoded_changed - decoded_original)
        event.context.analysis_results["modified_outputs"] = _replace(
            event.context.outputs, modified
        )
        delta_norm = float((modified - hidden).detach().float().norm())
        if spec.emit_summary:
            event.reactor.signal(
                "visualization",
                {
                    "module": name,
                    "subtype": "table",
                    "layer_path": event.context.module_path,
                    "data": {
                        "type": "table",
                        "columns": ["Metric", "Value"],
                        "rows": [
                            ["Operation", spec.operation],
                            ["Features", len(ids)],
                            ["Positions", spec.positions],
                            ["Delta L2 norm", delta_norm],
                        ],
                    },
                },
            )
        return {
            "status": "ok",
            "operation": spec.operation,
            "features": len(ids),
            "delta_norm": delta_norm,
            "emitted": spec.emit_summary,
        }

    module.interceptor = interceptor
    return module
