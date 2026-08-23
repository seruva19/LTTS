"""Capability-gated research primitives for non-standard model topologies."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .graphs import (
    AttributionEdge,
    AttributionGraph,
    AttributionGraphBuilder,
    AttributionNode,
)


@dataclass(frozen=True)
class ResearchCapturePoint:
    path: str
    role: str
    stage: str = "model"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "stage": self.stage,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ResearchArchitectureAdapter:
    """Discovered tensor boundaries and the operations they can support."""

    capabilities: FrozenSet[str]
    capture_points: Tuple[ResearchCapturePoint, ...]
    schema_version: int = 1

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def require(self, *capabilities: str) -> None:
        missing = sorted(set(capabilities).difference(self.capabilities))
        if missing:
            raise ValueError(
                "Architecture does not expose required research capabilities: "
                + ", ".join(missing)
            )

    def points(self, role: Optional[str] = None) -> Tuple[ResearchCapturePoint, ...]:
        if role is None:
            return self.capture_points
        return tuple(item for item in self.capture_points if item.role == role)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capabilities": sorted(self.capabilities),
            "capture_points": [item.to_dict() for item in self.capture_points],
        }


@dataclass(frozen=True)
class ResearchWorkflowRequirement:
    name: str
    required_capabilities: FrozenSet[str]


@dataclass(frozen=True)
class ResearchWorkflowCompatibility:
    name: str
    supported: bool
    required_capabilities: Tuple[str, ...]
    missing_capabilities: Tuple[str, ...]
    capture_points: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "supported": self.supported,
            "required_capabilities": list(self.required_capabilities),
            "missing_capabilities": list(self.missing_capabilities),
            "capture_points": list(self.capture_points),
        }


DEFAULT_RESEARCH_WORKFLOWS = (
    ResearchWorkflowRequirement(
        "ssm_effective_influence", frozenset({"effective_influence"})
    ),
    ResearchWorkflowRequirement(
        "ssm_mixer_intervention", frozenset({"ssm_mixer_intervention"})
    ),
    ResearchWorkflowRequirement(
        "ssm_raw_state_intervention", frozenset({"ssm_state_intervention"})
    ),
    ResearchWorkflowRequirement(
        "vision_patch_attribution", frozenset({"vision_patch_attribution"})
    ),
    ResearchWorkflowRequirement(
        "cross_modal_attention_attribution",
        frozenset({"cross_modal_attention_attribution"}),
    ),
)


def research_workflow_matrix(
    adapter: ResearchArchitectureAdapter,
    workflows: Sequence[ResearchWorkflowRequirement] = DEFAULT_RESEARCH_WORKFLOWS,
) -> Tuple[ResearchWorkflowCompatibility, ...]:
    """Explain which workflows are safe for one discovered architecture."""

    rows = []
    for workflow in workflows:
        required = tuple(sorted(workflow.required_capabilities))
        missing = tuple(
            item for item in required if item not in adapter.capabilities
        )
        relevant_points = tuple(
            point.path
            for point in adapter.capture_points
            if (
                ("ssm" in workflow.name and point.role == "state_mixer")
                or ("vision_patch" in workflow.name and point.role == "patch_embedding")
                or ("cross_modal" in workflow.name and point.role == "cross_attention")
            )
        )
        rows.append(
            ResearchWorkflowCompatibility(
                name=workflow.name,
                supported=not missing,
                required_capabilities=required,
                missing_capabilities=missing,
                capture_points=relevant_points,
            )
        )
    return tuple(rows)


def discover_research_adapter(model: nn.Module) -> ResearchArchitectureAdapter:
    """Discover capabilities from tensor boundaries, not architecture names."""

    capabilities = set()
    points = []
    for path, module in model.named_modules():
        if not path:
            continue
        if hasattr(module, "A_log") and (
            any(hasattr(module, item) for item in ("x_proj", "in_proj"))
            or any(hasattr(module, item) for item in ("B", "C", "D"))
        ):
            capabilities.update(
                {
                    "ssm_mixer_capture",
                    "ssm_mixer_intervention",
                    "effective_influence",
                }
            )
            explicit_state = bool(
                callable(getattr(module, "get_state", None))
                or getattr(module, "exposes_recurrent_state", False)
            )
            if explicit_state:
                capabilities.update({"ssm_state_capture", "ssm_state_intervention"})
            points.append(
                ResearchCapturePoint(
                    path=path,
                    role="state_mixer",
                    metadata={"explicit_state_boundary": explicit_state},
                )
            )
        patch_size = getattr(module, "patch_size", None)
        if patch_size is not None:
            capabilities.update({"vision_patch_capture", "vision_patch_attribution"})
            points.append(
                ResearchCapturePoint(
                    path=path,
                    role="patch_embedding",
                    stage="vision",
                    metadata={"patch_size": patch_size},
                )
            )
        try:
            parameters = inspect.signature(module.forward).parameters
        except (TypeError, ValueError):
            parameters = {}
        explicit_cross_attention = bool(
            getattr(module, "is_cross_attention", False)
            or getattr(module, "is_cross_attn", False)
        )
        if explicit_cross_attention or "encoder_hidden_states" in parameters:
            capabilities.update(
                {"cross_modal_capture", "cross_modal_attention_attribution"}
            )
            points.append(
                ResearchCapturePoint(
                    path=path,
                    role="cross_attention",
                    stage="fusion",
                    metadata={"explicit_cross_attention": explicit_cross_attention},
                )
            )
    has_explicit_cross_attention = any(
        point.role == "cross_attention"
        and point.metadata.get("explicit_cross_attention")
        for point in points
    )
    candidate_points = tuple(
        point
        for point in points
        if not (
            point.role == "cross_attention"
            and has_explicit_cross_attention
            and not point.metadata.get("explicit_cross_attention")
        )
    )
    filtered_points = tuple(
        point
        for point in candidate_points
        if not any(
            other is not point
            and other.role == point.role
            and point.role in {"patch_embedding", "cross_attention"}
            and other.path.startswith(point.path + ".")
            for other in candidate_points
        )
    )
    return ResearchArchitectureAdapter(
        capabilities=frozenset(capabilities), capture_points=filtered_points
    )


def ssm_effective_influence(
    decay: torch.Tensor, input_gain: torch.Tensor
) -> torch.Tensor:
    """Compute diagonal-SSM source-to-target influence without calling it attention.

    ``decay`` is either ``[channels]`` or ``[sequence, channels]`` and
    ``input_gain`` is ``[sequence, channels]``. The result is ``[target, source]``
    mean absolute recurrence influence. This is an effective influence matrix,
    not transformer attention probability.
    """

    if input_gain.ndim != 2:
        raise ValueError("input_gain must have shape [sequence, channels]")
    if decay.ndim == 1:
        decay = decay.unsqueeze(0).expand(input_gain.shape[0], -1)
    if decay.shape != input_gain.shape:
        raise ValueError("decay and input_gain must share sequence and channel shape")
    sequence = input_gain.shape[0]
    result = torch.zeros(
        (sequence, sequence), device=input_gain.device, dtype=input_gain.dtype
    )
    for source in range(sequence):
        influence = input_gain[source]
        result[source, source] = influence.abs().mean()
        for target in range(source + 1, sequence):
            influence = influence * decay[target]
            result[target, source] = influence.abs().mean()
    return result


def intervene_ssm_state(
    state: torch.Tensor,
    *,
    channels: Sequence[int],
    positions: Optional[Sequence[int]] = None,
    operation: str = "set",
    value: float = 0.0,
) -> torch.Tensor:
    """Return an edited state tensor; callers attach it at a discovered state boundary."""

    if state.ndim < 2:
        raise ValueError("SSM state must have at least position and channel dimensions")
    selected_channels = tuple(dict.fromkeys(int(item) for item in channels))
    if not selected_channels:
        raise ValueError("At least one state channel is required")
    invalid_channels = [
        item for item in selected_channels if not 0 <= item < state.shape[-1]
    ]
    if invalid_channels:
        raise IndexError(f"State channels outside width {state.shape[-1]}: {invalid_channels}")
    selected_positions = tuple(
        dict.fromkeys(int(item) for item in (positions or range(state.shape[-2])))
    )
    invalid_positions = [
        item for item in selected_positions if not 0 <= item < state.shape[-2]
    ]
    if invalid_positions:
        raise IndexError(
            f"State positions outside length {state.shape[-2]}: {invalid_positions}"
        )
    changed = state.clone()
    position_index = torch.tensor(selected_positions, device=state.device)
    channel_index = torch.tensor(selected_channels, device=state.device)
    mesh = torch.meshgrid(position_index, channel_index, indexing="ij")
    prefix = (slice(None),) * (state.ndim - 2)
    index = (*prefix, mesh[0], mesh[1])
    if operation == "set":
        changed[index] = value
    elif operation == "add":
        changed[index] += value
    elif operation == "multiply":
        changed[index] *= value
    else:
        raise ValueError("SSM state operation must be set, add, or multiply")
    return changed


def graph_from_vision_patch_scores(
    scores: Sequence[float],
    *,
    grid_shape: Tuple[int, int],
    target_label: str,
    class_token_offset: int = 0,
) -> AttributionGraph:
    """Build vision-patch-to-target edges with stable grid coordinates."""

    rows, columns = (int(grid_shape[0]), int(grid_shape[1]))
    if rows < 1 or columns < 1 or len(scores) != rows * columns:
        raise ValueError("Patch score count must match grid_shape")
    builder = AttributionGraphBuilder(metadata={"source": "vision_patch_attribution"})
    builder.add_node(AttributionNode(id="target", kind="metric", label=target_label))
    for offset, score in enumerate(scores):
        row, column = divmod(offset, columns)
        position = offset + int(class_token_offset)
        node_id = f"patch:{row}:{column}"
        builder.add_node(
            AttributionNode(
                id=node_id,
                kind="input_token",
                label=f"patch ({row}, {column})",
                position=position,
                metadata={"modality": "vision", "row": row, "column": column},
            )
        )
        builder.add_edge(
            AttributionEdge(
                source=node_id,
                target="target",
                score=float(score),
                method="vision_patch_attribution",
                metadata={"row": row, "column": column},
            )
        )
    return builder.build()


def graph_from_cross_modal_attention(
    values: Sequence[Sequence[float]],
    *,
    text_labels: Sequence[str],
    patch_labels: Sequence[str],
    layer_path: str,
    head_index: int,
) -> AttributionGraph:
    """Build vision-patch-to-text edges from [text query, vision key] weights."""

    matrix = tuple(tuple(float(item) for item in row) for row in values)
    if len(matrix) != len(text_labels) or any(
        len(row) != len(patch_labels) for row in matrix
    ):
        raise ValueError("Cross-modal matrix shape must match text and patch labels")
    builder = AttributionGraphBuilder(
        metadata={
            "source": "cross_modal_attention",
            "layer_path": layer_path,
            "head_index": int(head_index),
        }
    )
    for index, label in enumerate(text_labels):
        builder.add_node(
            AttributionNode(
                id=f"text:{index}",
                kind="input_token",
                label=str(label),
                position=index,
                metadata={"modality": "text"},
            )
        )
    for index, label in enumerate(patch_labels):
        builder.add_node(
            AttributionNode(
                id=f"vision:{index}",
                kind="input_token",
                label=str(label),
                position=index,
                metadata={"modality": "vision"},
            )
        )
    for query, row in enumerate(matrix):
        for key, score in enumerate(row):
            builder.add_edge(
                AttributionEdge(
                    source=f"vision:{key}",
                    target=f"text:{query}",
                    score=score,
                    method="cross_modal_attention",
                    metadata={
                        "layer_path": layer_path,
                        "head_index": int(head_index),
                        "text_position": query,
                        "patch_position": key,
                    },
                )
            )
    return builder.build()
