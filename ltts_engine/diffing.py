"""Paired model/checkpoint comparisons with semantic layer alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .inspection import ComponentInspector
from .runtime import HeadlessSession


@dataclass(frozen=True)
class LayerAlignment:
    left: str
    right: str
    role: str
    stage: str
    layer_index: Optional[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "role": self.role,
            "stage": self.stage,
            "layer_index": self.layer_index,
        }


@dataclass(frozen=True)
class ActivationDifference:
    alignment: LayerAlignment
    shape: Tuple[int, ...]
    cosine_similarity: float
    mean_absolute_delta: float
    relative_l2_delta: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alignment": self.alignment.to_dict(),
            "shape": list(self.shape),
            "cosine_similarity": self.cosine_similarity,
            "mean_absolute_delta": self.mean_absolute_delta,
            "relative_l2_delta": self.relative_l2_delta,
        }


@dataclass(frozen=True)
class LogitDifference:
    shape: Tuple[int, ...]
    cosine_similarity: float
    mean_absolute_delta: float
    symmetric_kl: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "shape": list(self.shape),
            "cosine_similarity": self.cosine_similarity,
            "mean_absolute_delta": self.mean_absolute_delta,
            "symmetric_kl": self.symmetric_kl,
        }


@dataclass(frozen=True)
class ModelDiffResult:
    alignments: Tuple[LayerAlignment, ...]
    activation_differences: Tuple[ActivationDifference, ...]
    logit_difference: Optional[LogitDifference]
    unmatched_left: Tuple[str, ...]
    unmatched_right: Tuple[str, ...]
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alignments": [item.to_dict() for item in self.alignments],
            "activation_differences": [
                item.to_dict() for item in self.activation_differences
            ],
            "logit_difference": (
                self.logit_difference.to_dict() if self.logit_difference else None
            ),
            "unmatched_left": list(self.unmatched_left),
            "unmatched_right": list(self.unmatched_right),
        }


def _semantic_key(descriptor: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        descriptor.get("role"),
        descriptor.get("stage"),
        descriptor.get("layer_index"),
        descriptor.get("attention_kind"),
    )


def align_layers(
    left: HeadlessSession,
    right: HeadlessSession,
    *,
    roles: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[LayerAlignment, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Align exact paths first, then unique semantic descriptor keys."""

    allowed = set(roles or ())
    left_items = {
        path: descriptor
        for path, descriptor in left.layer_descriptors.items()
        if not allowed or descriptor.get("role") in allowed
    }
    right_items = {
        path: descriptor
        for path, descriptor in right.layer_descriptors.items()
        if not allowed or descriptor.get("role") in allowed
    }
    pairs = []
    used_right = set()
    for path, descriptor in left_items.items():
        other = right_items.get(path)
        if other is not None and _semantic_key(descriptor) == _semantic_key(other):
            pairs.append((path, path, descriptor))
            used_right.add(path)

    right_by_key: Dict[Tuple[Any, ...], list] = {}
    for path, descriptor in right_items.items():
        if path not in used_right:
            right_by_key.setdefault(_semantic_key(descriptor), []).append(path)
    matched_left = {left_path for left_path, _right_path, _descriptor in pairs}
    for path, descriptor in left_items.items():
        if path in matched_left:
            continue
        candidates = right_by_key.get(_semantic_key(descriptor), [])
        if len(candidates) == 1:
            right_path = candidates.pop()
            pairs.append((path, right_path, descriptor))
            used_right.add(right_path)

    alignments = tuple(
        LayerAlignment(
            left=left_path,
            right=right_path,
            role=str(descriptor.get("role") or "other"),
            stage=str(descriptor.get("stage") or "model"),
            layer_index=descriptor.get("layer_index"),
        )
        for left_path, right_path, descriptor in pairs
    )
    aligned_left = {item.left for item in alignments}
    aligned_right = {item.right for item in alignments}
    return (
        alignments,
        tuple(path for path in left_items if path not in aligned_left),
        tuple(path for path in right_items if path not in aligned_right),
    )


def _activation_difference(
    alignment: LayerAlignment, left: torch.Tensor, right: torch.Tensor
) -> ActivationDifference:
    if left.shape != right.shape:
        raise ValueError(
            f"Aligned layers {alignment.left!r} and {alignment.right!r} have "
            f"different output shapes: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    left_float = left.detach().float().reshape(-1)
    right_float = right.detach().float().reshape(-1)
    delta = left_float - right_float
    return ActivationDifference(
        alignment=alignment,
        shape=tuple(left.shape),
        cosine_similarity=float(
            F.cosine_similarity(left_float, right_float, dim=0, eps=1e-12)
        ),
        mean_absolute_delta=float(delta.abs().mean()),
        relative_l2_delta=float(delta.norm() / left_float.norm().clamp_min(1e-12)),
    )


def _logit_difference(left_output: Any, right_output: Any) -> Optional[LogitDifference]:
    left = getattr(left_output, "logits", None)
    right = getattr(right_output, "logits", None)
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        return None
    if left.shape != right.shape:
        raise ValueError(
            f"Model logits have different shapes: {tuple(left.shape)} vs {tuple(right.shape)}"
        )
    left_float = left.detach().float()
    right_float = right.detach().float()
    left_log_probs = F.log_softmax(left_float, dim=-1)
    right_log_probs = F.log_softmax(right_float, dim=-1)
    left_probs = left_log_probs.exp()
    right_probs = right_log_probs.exp()
    kl_lr = F.kl_div(right_log_probs, left_probs, reduction="batchmean")
    kl_rl = F.kl_div(left_log_probs, right_probs, reduction="batchmean")
    return LogitDifference(
        shape=tuple(left.shape),
        cosine_similarity=float(
            F.cosine_similarity(left_float.reshape(-1), right_float.reshape(-1), dim=0)
        ),
        mean_absolute_delta=float((left_float - right_float).abs().mean()),
        symmetric_kl=max(0.0, float((kl_lr + kl_rl) / 2)),
    )


class ModelDiffer:
    """Compare aligned internal representations and output distributions."""

    def __init__(self, left: HeadlessSession, right: HeadlessSession) -> None:
        if left is right:
            raise ValueError("Model diff requires two distinct sessions")
        self.left = left
        self.right = right

    def compare(
        self,
        *,
        left_kwargs: Mapping[str, Any],
        right_kwargs: Optional[Mapping[str, Any]] = None,
        roles: Sequence[str] = ("block",),
    ) -> ModelDiffResult:
        alignments, unmatched_left, unmatched_right = align_layers(
            self.left, self.right, roles=roles
        )
        if not alignments:
            raise ValueError("Models have no semantically alignable layers")
        left_output, left_captures = ComponentInspector(self.left).capture(
            [item.left for item in alignments], kwargs=left_kwargs
        )
        right_output, right_captures = ComponentInspector(self.right).capture(
            [item.right for item in alignments],
            kwargs=right_kwargs or left_kwargs,
        )
        differences = tuple(
            _activation_difference(
                item, left_captures[item.left], right_captures[item.right]
            )
            for item in alignments
        )
        return ModelDiffResult(
            alignments=alignments,
            activation_differences=differences,
            logit_difference=_logit_difference(left_output, right_output),
            unmatched_left=unmatched_left,
            unmatched_right=unmatched_right,
        )
