"""Component capture and vocabulary-space drill-down primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .runtime import HeadlessSession


@dataclass(frozen=True)
class TokenScore:
    token_id: int
    score: float
    token: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"token_id": self.token_id, "token": self.token, "score": self.score}


@dataclass(frozen=True)
class ComponentComparison:
    """Vocabulary effects of one component relative to another component."""

    component_layer: str
    reference_layer: str
    position: int
    component_shape: Tuple[int, ...]
    reference_shape: Tuple[int, ...]
    promoted: Tuple[TokenScore, ...]
    suppressed: Tuple[TokenScore, ...]
    top_component_logits: Tuple[TokenScore, ...]
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "component_layer": self.component_layer,
            "reference_layer": self.reference_layer,
            "position": self.position,
            "component_shape": list(self.component_shape),
            "reference_shape": list(self.reference_shape),
            "promoted": [item.to_dict() for item in self.promoted],
            "suppressed": [item.to_dict() for item in self.suppressed],
            "top_component_logits": [
                item.to_dict() for item in self.top_component_logits
            ],
        }


def _first_hidden(output: Any) -> torch.Tensor:
    candidate = output[0] if isinstance(output, (tuple, list)) and output else output
    if not isinstance(candidate, torch.Tensor) or candidate.ndim < 2:
        raise TypeError("Selected component did not produce hidden-state-like output")
    return candidate


def _decode(decoder: Optional[Callable[[int], str]], token_id: int) -> Optional[str]:
    if decoder is None:
        return None
    try:
        return str(decoder(token_id))
    except Exception:
        return None


def _rank(
    values: torch.Tensor,
    *,
    largest: bool,
    top_k: int,
    decoder: Optional[Callable[[int], str]],
) -> Tuple[TokenScore, ...]:
    count = min(max(1, int(top_k)), int(values.shape[-1]))
    scores, indices = torch.topk(values.float(), count, largest=largest)
    return tuple(
        TokenScore(
            token_id=int(token_id),
            score=float(score),
            token=_decode(decoder, int(token_id)),
        )
        for score, token_id in zip(scores.detach().cpu(), indices.detach().cpu())
    )


class ComponentInspector:
    """Capture internal outputs and compare their vocabulary projections."""

    def __init__(self, session: HeadlessSession) -> None:
        self.session = session

    def capture(
        self,
        layers: Sequence[str],
        *,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run once and capture detached outputs from requested exact paths."""

        if not layers:
            raise ValueError("At least one layer is required")
        available = dict(self.session.model.original_model.named_modules())
        missing = [layer for layer in layers if layer not in available]
        if missing:
            raise KeyError(f"Unknown component layer(s): {', '.join(missing)}")

        captures: Dict[str, torch.Tensor] = {}
        handles = []
        for layer in layers:

            def capture_output(_module, _inputs, output, *, path=layer):
                captures[path] = _first_hidden(output).detach().clone()

            handles.append(available[layer].register_forward_hook(capture_output))
        try:
            output = self.session.run(*tuple(args), **dict(kwargs or {}))
        finally:
            for handle in handles:
                handle.remove()
        absent = [layer for layer in layers if layer not in captures]
        if absent:
            raise RuntimeError(
                f"Component layer(s) did not execute: {', '.join(absent)}"
            )
        return output, captures

    def compare_vocabulary_effects(
        self,
        *,
        component_layer: str,
        reference_layer: str,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
        output_embedding: Optional[nn.Module] = None,
        normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        position: int = -1,
        top_k: int = 10,
        decoder: Optional[Callable[[int], str]] = None,
    ) -> ComponentComparison:
        """Project component and component-minus-reference into vocabulary space."""

        _output, captures = self.capture(
            (reference_layer, component_layer), args=args, kwargs=kwargs
        )
        component = captures[component_layer]
        reference = captures[reference_layer]
        if component.shape != reference.shape:
            raise ValueError(
                "Component and reference outputs must have identical shapes; "
                f"got {tuple(component.shape)} and {tuple(reference.shape)}"
            )
        if component.ndim != 3:
            raise ValueError(
                "Vocabulary drill-down requires [batch, sequence, hidden] outputs"
            )
        sequence_length = component.shape[1]
        resolved_position = position if position >= 0 else sequence_length + position
        if not 0 <= resolved_position < sequence_length:
            raise IndexError(
                f"Position {position} is outside sequence length {sequence_length}"
            )

        embedding = output_embedding or self._output_embedding()
        weight = getattr(embedding, "weight", None)
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError("Output embedding must expose a rank-2 weight tensor")
        component_vector = component[0, resolved_position]
        reference_vector = reference[0, resolved_position]
        if component_vector.shape[-1] != weight.shape[-1]:
            raise ValueError(
                f"Hidden width {component_vector.shape[-1]} does not match "
                f"output embedding width {weight.shape[-1]}"
            )
        if normalizer is not None:
            component_vector = normalizer(component_vector)
            reference_vector = normalizer(reference_vector)
        bias = getattr(embedding, "bias", None)
        component_logits = F.linear(component_vector.to(weight.device), weight, bias)
        reference_logits = F.linear(reference_vector.to(weight.device), weight, bias)
        delta = component_logits - reference_logits
        return ComponentComparison(
            component_layer=component_layer,
            reference_layer=reference_layer,
            position=resolved_position,
            component_shape=tuple(component.shape),
            reference_shape=tuple(reference.shape),
            promoted=_rank(delta, largest=True, top_k=top_k, decoder=decoder),
            suppressed=_rank(delta, largest=False, top_k=top_k, decoder=decoder),
            top_component_logits=_rank(
                component_logits, largest=True, top_k=top_k, decoder=decoder
            ),
        )

    def _output_embedding(self) -> nn.Module:
        model = self.session.model.original_model
        getter = getattr(model, "get_output_embeddings", None)
        embedding = getter() if callable(getter) else None
        if embedding is None:
            embedding = getattr(model, "lm_head", None)
        if not isinstance(embedding, nn.Module):
            raise ValueError("Model does not expose an output embedding")
        return embedding
