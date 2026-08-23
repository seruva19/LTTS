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
class ResearchSelectionContext:
    """Stable drill-down coordinates shared by artifacts and interventions."""

    layer_path: str
    position: int
    component_kind: str = "component"
    component_index: Optional[int] = None
    output_token_id: Optional[int] = None
    run_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.layer_path:
            raise ValueError("Selection context requires a layer path")
        if self.position < 0:
            raise ValueError("Selection context position cannot be negative")
        if self.component_index is not None and self.component_index < 0:
            raise ValueError("Selection component index cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "output_token_id": self.output_token_id,
            "layer_path": self.layer_path,
            "component_kind": self.component_kind,
            "component_index": self.component_index,
            "position": self.position,
        }


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
    selection: Optional[ResearchSelectionContext] = None

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
            "selection": self.selection.to_dict() if self.selection else None,
        }


@dataclass(frozen=True)
class ComponentContribution:
    """One head or neuron contribution projected into vocabulary space."""

    kind: str
    index: int
    position: Optional[int]
    vector_norm: float
    promoted: Tuple[TokenScore, ...]
    suppressed: Tuple[TokenScore, ...]
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "index": self.index,
            "position": self.position,
            "vector_norm": self.vector_norm,
            "promoted": [item.to_dict() for item in self.promoted],
            "suppressed": [item.to_dict() for item in self.suppressed],
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


def _position(tensor: torch.Tensor, position: int) -> Tuple[torch.Tensor, int]:
    if tensor.ndim != 3:
        raise ValueError("Component decomposition requires [batch, sequence, width]")
    resolved = position if position >= 0 else tensor.shape[1] + position
    if not 0 <= resolved < tensor.shape[1]:
        raise IndexError(
            f"Position {position} is outside sequence length {tensor.shape[1]}"
        )
    return tensor[0, resolved], int(resolved)


def _projection_weight(projection: nn.Module) -> torch.Tensor:
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise TypeError("Projection must expose a rank-2 weight tensor")
    return weight


def _canonical_projection_weight(
    projection: nn.Module, input_width: int, layout: str
) -> torch.Tensor:
    """Return a projection matrix in [output, input] layout."""

    weight = _projection_weight(projection)
    if layout not in {"auto", "out_in", "in_out"}:
        raise ValueError("weight_layout must be auto, out_in, or in_out")
    if layout == "out_in" or (layout == "auto" and isinstance(projection, nn.Linear)):
        if weight.shape[-1] != input_width:
            raise ValueError("Projection input width does not match component width")
        return weight
    if layout == "in_out":
        if weight.shape[0] != input_width:
            raise ValueError("Projection input width does not match component width")
        return weight.T
    out_in = weight.shape[1] == input_width
    in_out = weight.shape[0] == input_width
    if out_in == in_out:
        raise ValueError(
            "Projection weight layout is ambiguous; pass weight_layout explicitly"
        )
    return weight if out_in else weight.T


def _vocabulary_scores(
    vector: torch.Tensor,
    *,
    output_embedding: nn.Module,
    normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]],
    top_k: int,
    decoder: Optional[Callable[[int], str]],
) -> Tuple[Tuple[TokenScore, ...], Tuple[TokenScore, ...]]:
    weight = _projection_weight(output_embedding)
    if vector.shape[-1] != weight.shape[-1]:
        raise ValueError(
            f"Contribution width {vector.shape[-1]} does not match output "
            f"embedding width {weight.shape[-1]}"
        )
    projected = normalizer(vector) if normalizer is not None else vector
    logits = F.linear(projected.to(weight.device), weight, None)
    return (
        _rank(logits, largest=True, top_k=top_k, decoder=decoder),
        _rank(logits, largest=False, top_k=top_k, decoder=decoder),
    )


def decompose_attention_heads(
    concatenated_heads: torch.Tensor,
    *,
    output_projection: nn.Module,
    output_embedding: nn.Module,
    num_heads: int,
    weight_layout: str = "auto",
    position: int = -1,
    top_k: int = 10,
    normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    decoder: Optional[Callable[[int], str]] = None,
) -> Tuple[ComponentContribution, ...]:
    """Project each pre-output-projection attention head independently.

    ``concatenated_heads`` is the tensor immediately before the attention output
    projection. Supporting this explicit boundary avoids pretending that heads
    can be recovered exactly from an already projected residual contribution.
    """

    vector, resolved_position = _position(concatenated_heads, position)
    if num_heads < 1 or vector.shape[-1] % num_heads:
        raise ValueError("Hidden width must be divisible by num_heads")
    projection_weight = _canonical_projection_weight(
        output_projection, vector.shape[-1], weight_layout
    )
    head_width = vector.shape[-1] // num_heads
    results = []
    for index in range(num_heads):
        start = index * head_width
        stop = start + head_width
        contribution = F.linear(
            vector[start:stop].to(projection_weight.device),
            projection_weight[:, start:stop],
            None,
        )
        promoted, suppressed = _vocabulary_scores(
            contribution,
            output_embedding=output_embedding,
            normalizer=normalizer,
            top_k=top_k,
            decoder=decoder,
        )
        results.append(
            ComponentContribution(
                kind="attention_head",
                index=index,
                position=resolved_position,
                vector_norm=float(contribution.float().norm().detach().cpu()),
                promoted=promoted,
                suppressed=suppressed,
            )
        )
    return tuple(results)


def decompose_mlp_neurons(
    neuron_activations: torch.Tensor,
    *,
    output_projection: nn.Module,
    output_embedding: nn.Module,
    neuron_ids: Optional[Sequence[int]] = None,
    weight_layout: str = "auto",
    position: int = -1,
    top_k: int = 10,
    normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    decoder: Optional[Callable[[int], str]] = None,
) -> Tuple[ComponentContribution, ...]:
    """Project individual post-activation MLP neurons through the down projection."""

    vector, resolved_position = _position(neuron_activations, position)
    projection_weight = _canonical_projection_weight(
        output_projection, vector.shape[-1], weight_layout
    )
    selected = tuple(
        dict.fromkeys(int(item) for item in neuron_ids or range(vector.shape[-1]))
    )
    invalid = [item for item in selected if not 0 <= item < vector.shape[-1]]
    if invalid:
        raise IndexError(f"Neuron ids outside width {vector.shape[-1]}: {invalid}")
    results = []
    for index in selected:
        contribution = (
            vector[index].to(projection_weight.device) * projection_weight[:, index]
        )
        promoted, suppressed = _vocabulary_scores(
            contribution,
            output_embedding=output_embedding,
            normalizer=normalizer,
            top_k=top_k,
            decoder=decoder,
        )
        results.append(
            ComponentContribution(
                kind="mlp_neuron",
                index=index,
                position=resolved_position,
                vector_norm=float(contribution.float().norm().detach().cpu()),
                promoted=promoted,
                suppressed=suppressed,
            )
        )
    return tuple(results)


def decompose_sae_features(
    decoder: Callable[[torch.Tensor], torch.Tensor],
    *,
    feature_width: int,
    feature_ids: Sequence[int],
    output_embedding: nn.Module,
    position: Optional[int] = None,
    top_k: int = 10,
    normalizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    token_decoder: Optional[Callable[[int], str]] = None,
) -> Tuple[ComponentContribution, ...]:
    """Project bias-free SAE decoder directions into vocabulary space."""

    width = int(feature_width)
    if width < 1:
        raise ValueError("feature_width must be at least one")
    selected = tuple(dict.fromkeys(int(item) for item in feature_ids))
    if not selected:
        raise ValueError("At least one SAE feature id is required")
    invalid = [item for item in selected if not 0 <= item < width]
    if invalid:
        raise IndexError(f"SAE feature ids outside width {width}: {invalid}")
    embedding_weight = _projection_weight(output_embedding)
    device = embedding_weight.device
    dtype = embedding_weight.dtype
    owner = getattr(decoder, "parameters", None)
    if callable(owner):
        parameter = next(owner(), None)
        if parameter is not None:
            device, dtype = parameter.device, parameter.dtype
    basis = torch.zeros((len(selected), width), device=device, dtype=dtype)
    basis[
        torch.arange(len(selected), device=device),
        torch.tensor(selected, device=device),
    ] = 1
    zero = torch.zeros((1, width), device=device, dtype=dtype)
    decoded = _first_hidden(decoder(basis))
    decoded_zero = _first_hidden(decoder(zero))
    if decoded.ndim != 2 or decoded.shape[0] != len(selected):
        raise ValueError("SAE decoder must map [features, width] to [features, hidden]")
    if decoded_zero.shape != (1, decoded.shape[-1]):
        raise ValueError("SAE decoder output shape changed for the zero feature vector")
    directions = decoded - decoded_zero
    results = []
    for offset, feature_id in enumerate(selected):
        direction = directions[offset]
        promoted, suppressed = _vocabulary_scores(
            direction,
            output_embedding=output_embedding,
            normalizer=normalizer,
            top_k=top_k,
            decoder=token_decoder,
        )
        results.append(
            ComponentContribution(
                kind="sae_feature",
                index=feature_id,
                position=position,
                vector_norm=float(direction.float().norm().detach().cpu()),
                promoted=promoted,
                suppressed=suppressed,
            )
        )
    return tuple(results)


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

    def capture_inputs(
        self,
        layers: Sequence[str],
        *,
        args: Sequence[Any] = (),
        kwargs: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, torch.Tensor]]:
        """Run once and capture inputs at exact pre-projection boundaries."""

        if not layers:
            raise ValueError("At least one layer is required")
        available = dict(self.session.model.original_model.named_modules())
        missing = [layer for layer in layers if layer not in available]
        if missing:
            raise KeyError(f"Unknown component layer(s): {', '.join(missing)}")
        captures: Dict[str, torch.Tensor] = {}
        handles = []
        for layer in layers:

            def capture_input(_module, inputs, *, path=layer):
                captures[path] = _first_hidden(inputs).detach().clone()

            handles.append(available[layer].register_forward_pre_hook(capture_input))
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
        run_id: Optional[str] = None,
        output_token_id: Optional[int] = None,
        component_kind: str = "component",
        component_index: Optional[int] = None,
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
            selection=ResearchSelectionContext(
                run_id=run_id,
                output_token_id=output_token_id,
                layer_path=component_layer,
                component_kind=component_kind,
                component_index=component_index,
                position=resolved_position,
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
