"""Training-free residual-stream recirculation experiments.

Implements the method introduced by Mozer et al., "Recirculation" (2026):
https://arxiv.org/abs/2608.17981
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RecirculationSpec:
    """Coordinates and mixing rule for one recurrent residual connection."""

    source_layer: str
    destination_layer: str
    alpha: float = 0.07
    beta: Optional[float] = None
    normalization: str = "l2"
    ramp_steps: int = 0

    def __post_init__(self) -> None:
        if not self.source_layer or not self.destination_layer:
            raise ValueError("Recirculation requires source and destination layers")
        if self.source_layer == self.destination_layer:
            raise ValueError("Recirculation layers must be different")
        if not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("Recirculation alpha must be finite and non-negative")
        if self.beta is not None and not math.isfinite(self.beta):
            raise ValueError("Recirculation beta must be finite")
        if self.normalization not in {"l2", "none"}:
            raise ValueError("Recirculation normalization must be 'l2' or 'none'")
        if self.ramp_steps < 0:
            raise ValueError("Recirculation ramp_steps cannot be negative")

    @property
    def resolved_beta(self) -> float:
        return 1.0 - self.alpha if self.beta is None else self.beta

    def alpha_at(self, position: int) -> float:
        if position < 0:
            raise ValueError("Recirculation position cannot be negative")
        if self.ramp_steps == 0:
            return self.alpha
        return min(position / self.ramp_steps, 1.0) * self.alpha

    def beta_at(self, position: int) -> float:
        """Resolve convex beta against the ramped alpha unless beta is explicit."""

        return 1.0 - self.alpha_at(position) if self.beta is None else self.beta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_layer": self.source_layer,
            "destination_layer": self.destination_layer,
            "alpha": self.alpha,
            "beta": self.resolved_beta,
            "normalization": self.normalization,
            "ramp_steps": self.ramp_steps,
        }


@dataclass(frozen=True)
class RecirculationEvaluation:
    """Paired next-token likelihoods from ordinary and recirculated execution."""

    spec: RecirculationSpec
    baseline_mean_nll: float
    recirculation_mean_nll: float
    baseline_perplexity: float
    recirculation_perplexity: float
    perplexity_change_percent: float
    token_count: int
    baseline_token_nll: Tuple[float, ...]
    recirculation_token_nll: Tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "baseline_mean_nll": self.baseline_mean_nll,
            "recirculation_mean_nll": self.recirculation_mean_nll,
            "baseline_perplexity": self.baseline_perplexity,
            "recirculation_perplexity": self.recirculation_perplexity,
            "perplexity_change_percent": self.perplexity_change_percent,
            "token_count": self.token_count,
            "baseline_token_nll": list(self.baseline_token_nll),
            "recirculation_token_nll": list(self.recirculation_token_nll),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecirculationSweepResult:
    """Comparable evaluations for an explicit recirculation search space."""

    evaluations: Tuple[RecirculationEvaluation, ...]
    best_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.evaluations:
            raise ValueError("Recirculation sweep requires at least one evaluation")
        if not 0 <= self.best_index < len(self.evaluations):
            raise ValueError("Recirculation sweep best_index is out of range")

    @property
    def best(self) -> RecirculationEvaluation:
        return self.evaluations[self.best_index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "best_index": self.best_index,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecirculationGeneration:
    """Paired greedy trajectories for ordinary and recirculated decoding."""

    spec: RecirculationSpec
    baseline_sequence: Tuple[int, ...]
    recirculation_sequence: Tuple[int, ...]
    prompt_length: int
    changed_positions: Tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "baseline_sequence": list(self.baseline_sequence),
            "recirculation_sequence": list(self.recirculation_sequence),
            "prompt_length": self.prompt_length,
            "changed_positions": list(self.changed_positions),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecirculationCompatibility:
    """Discovered ordered residual boundaries for one decoder model."""

    supported: bool
    layer_paths: Tuple[str, ...]
    container_path: Optional[str]
    reason: Optional[str] = None
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supported": self.supported,
            "layer_paths": list(self.layer_paths),
            "container_path": self.container_path,
            "reason": self.reason,
        }


def discover_recirculation_layers(model: nn.Module) -> RecirculationCompatibility:
    """Find the longest ordered ModuleList suitable for residual recirculation."""

    candidates = []
    for path, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) >= 2:
            paths = tuple(f"{path}.{index}" if path else str(index) for index in range(len(module)))
            candidates.append((len(paths), path, paths))
    if not candidates:
        return RecirculationCompatibility(
            supported=False,
            layer_paths=(),
            container_path=None,
            reason="No ordered decoder block ModuleList was found",
        )
    _, path, paths = min(candidates, key=lambda item: (-item[0], item[1]))
    return RecirculationCompatibility(
        supported=True,
        layer_paths=paths,
        container_path=path,
    )


@dataclass(frozen=True)
class RecirculationDatasetEvaluation:
    """Token-weighted recirculation measurement over multiple examples."""

    spec: RecirculationSpec
    examples: Tuple[RecirculationEvaluation, ...]
    token_count: int
    baseline_mean_nll: float
    recirculation_mean_nll: float
    baseline_perplexity: float
    recirculation_perplexity: float
    perplexity_change_percent: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "spec": self.spec.to_dict(),
            "examples": [item.to_dict() for item in self.examples],
            "example_count": len(self.examples),
            "token_count": self.token_count,
            "baseline_mean_nll": self.baseline_mean_nll,
            "recirculation_mean_nll": self.recirculation_mean_nll,
            "baseline_perplexity": self.baseline_perplexity,
            "recirculation_perplexity": self.recirculation_perplexity,
            "perplexity_change_percent": self.perplexity_change_percent,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RecirculationDatasetSweepResult:
    evaluations: Tuple[RecirculationDatasetEvaluation, ...]
    best_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    @property
    def best(self) -> RecirculationDatasetEvaluation:
        return self.evaluations[self.best_index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluations": [item.to_dict() for item in self.evaluations],
            "best_index": self.best_index,
            "metadata": dict(self.metadata),
        }


def recirculation_sweep_heatmap(
    sweep: Any,
) -> Dict[str, Any]:
    """Build a renderer-neutral source-by-destination perplexity heatmap."""

    evaluations = tuple(sweep.evaluations)
    if not evaluations:
        raise ValueError("Recirculation heatmap requires evaluations")
    settings = {
        (
            item.spec.alpha,
            item.spec.resolved_beta,
            item.spec.normalization,
            item.spec.ramp_steps,
        )
        for item in evaluations
    }
    if len(settings) != 1:
        raise ValueError("Recirculation heatmap requires fixed mixing settings")
    sources = tuple(sorted({item.spec.source_layer for item in evaluations}))
    destinations = tuple(
        sorted({item.spec.destination_layer for item in evaluations})
    )
    points = {
        (item.spec.source_layer, item.spec.destination_layer): item
        for item in evaluations
    }
    if len(points) != len(evaluations):
        raise ValueError("Recirculation heatmap layer pairs must be unique")
    expected = {(source, destination) for source in sources for destination in destinations}
    if set(points) != expected:
        raise ValueError("Recirculation heatmap requires a complete rectangular grid")
    values = [
        [points[(source, destination)].perplexity_change_percent for destination in destinations]
        for source in sources
    ]
    alpha, beta, normalization, ramp_steps = next(iter(settings))
    return {
        "type": "heatmap",
        "values": values,
        "dimensions": {"rows": len(sources), "cols": len(destinations)},
        "row_labels": list(sources),
        "column_labels": list(destinations),
        "color_scale": "diverging_zero",
        "metric": "perplexity_change_percent",
        "alpha": alpha,
        "beta": beta,
        "normalization": normalization,
        "ramp_steps": ramp_steps,
    }


def mix_recirculated_residual(
    source: torch.Tensor,
    destination: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    normalization: str = "l2",
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Apply the paper's source-to-destination residual mixing equation."""

    if source.shape != destination.shape:
        raise ValueError("Recirculation source and destination shapes must match")
    if source.device != destination.device:
        raise ValueError("Recirculation tensors must share a device")
    if source.dtype != destination.dtype:
        raise ValueError("Recirculation tensors must share a dtype")
    if not math.isfinite(alpha) or not math.isfinite(beta):
        raise ValueError("Recirculation coefficients must be finite")
    if normalization == "l2":
        source_norm = torch.linalg.vector_norm(source, dim=-1, keepdim=True)
        destination_norm = torch.linalg.vector_norm(
            destination, dim=-1, keepdim=True
        )
        scale = destination_norm / source_norm.clamp_min(epsilon)
        source = source * scale
    elif normalization != "none":
        raise ValueError("Recirculation normalization must be 'l2' or 'none'")
    return beta * destination + alpha * source


class RecirculationRunner:
    """Reference sequential executor for Hugging Face-style causal LMs.

    The implementation deliberately disables KV caching and recomputes each
    prefix. This is slower than an optimized architecture-specific executor but
    gives a clear reference: a token is first processed normally, then its mixed
    destination state becomes visible while processing later tokens.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def evaluate(
        self,
        input_ids: torch.Tensor,
        spec: RecirculationSpec,
        *,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RecirculationEvaluation:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Reference recirculation expects input_ids shaped [1, seq]")
        if input_ids.shape[1] < 2:
            raise ValueError("Recirculation evaluation requires at least two tokens")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        source_module = self.model.get_submodule(spec.source_layer)
        destination_module = self.model.get_submodule(spec.destination_layer)
        self._validate_layer_order(source_module, destination_module)

        was_training = self.model.training
        self.model.eval()
        try:
            baseline = self._token_nll(input_ids, attention_mask)
            recirculated = self._recirculated_token_nll(
                input_ids,
                attention_mask,
                source_module,
                destination_module,
                spec,
            )
        finally:
            self.model.train(was_training)

        baseline_mean = sum(baseline) / len(baseline)
        recirculated_mean = sum(recirculated) / len(recirculated)
        baseline_ppl = math.exp(baseline_mean)
        recirculated_ppl = math.exp(recirculated_mean)
        return RecirculationEvaluation(
            spec=spec,
            baseline_mean_nll=baseline_mean,
            recirculation_mean_nll=recirculated_mean,
            baseline_perplexity=baseline_ppl,
            recirculation_perplexity=recirculated_ppl,
            perplexity_change_percent=(
                (recirculated_ppl - baseline_ppl) / baseline_ppl * 100.0
            ),
            token_count=len(baseline),
            baseline_token_nll=baseline,
            recirculation_token_nll=recirculated,
            metadata=dict(metadata or {}),
        )

    def sweep(
        self,
        input_ids: torch.Tensor,
        specs: Sequence[RecirculationSpec],
        *,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RecirculationSweepResult:
        """Evaluate an explicit grid and select the lowest perplexity result."""

        if not specs:
            raise ValueError("Recirculation sweep requires at least one spec")
        keys = [
            (
                item.source_layer,
                item.destination_layer,
                item.alpha,
                item.resolved_beta,
                item.normalization,
                item.ramp_steps,
            )
            for item in specs
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("Recirculation sweep specs must be unique")
        evaluations = tuple(
            self.evaluate(
                input_ids,
                spec,
                attention_mask=attention_mask,
                metadata=metadata,
            )
            for spec in specs
        )
        baseline = evaluations[0].baseline_token_nll
        if any(item.baseline_token_nll != baseline for item in evaluations[1:]):
            raise RuntimeError("Paired recirculation sweep produced inconsistent baselines")
        best_index = min(
            range(len(evaluations)),
            key=lambda index: (evaluations[index].recirculation_perplexity, index),
        )
        return RecirculationSweepResult(
            evaluations=evaluations,
            best_index=best_index,
            metadata=dict(metadata or {}),
        )

    def evaluate_dataset(
        self,
        examples: Sequence[torch.Tensor],
        spec: RecirculationSpec,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RecirculationDatasetEvaluation:
        """Aggregate examples by predicted-token count, not document count."""

        if not examples:
            raise ValueError("Recirculation dataset requires at least one example")
        results = tuple(
            self.evaluate(
                input_ids,
                spec,
                metadata={"example_index": index},
            )
            for index, input_ids in enumerate(examples)
        )
        token_count = sum(item.token_count for item in results)
        baseline_mean = sum(
            item.baseline_mean_nll * item.token_count for item in results
        ) / token_count
        recirculation_mean = sum(
            item.recirculation_mean_nll * item.token_count for item in results
        ) / token_count
        baseline_ppl = math.exp(baseline_mean)
        recirculation_ppl = math.exp(recirculation_mean)
        return RecirculationDatasetEvaluation(
            spec=spec,
            examples=results,
            token_count=token_count,
            baseline_mean_nll=baseline_mean,
            recirculation_mean_nll=recirculation_mean,
            baseline_perplexity=baseline_ppl,
            recirculation_perplexity=recirculation_ppl,
            perplexity_change_percent=(
                (recirculation_ppl - baseline_ppl) / baseline_ppl * 100.0
            ),
            metadata=dict(metadata or {}),
        )

    def sweep_dataset(
        self,
        examples: Sequence[torch.Tensor],
        specs: Sequence[RecirculationSpec],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RecirculationDatasetSweepResult:
        if not specs:
            raise ValueError("Recirculation dataset sweep requires at least one spec")
        keys = tuple(
            (
                item.source_layer,
                item.destination_layer,
                item.alpha,
                item.resolved_beta,
                item.normalization,
                item.ramp_steps,
            )
            for item in specs
        )
        if len(keys) != len(set(keys)):
            raise ValueError("Recirculation dataset sweep specs must be unique")
        evaluations = tuple(
            self.evaluate_dataset(examples, spec, metadata=metadata) for spec in specs
        )
        best_index = min(
            range(len(evaluations)),
            key=lambda index: (evaluations[index].recirculation_perplexity, index),
        )
        return RecirculationDatasetSweepResult(
            evaluations=evaluations,
            best_index=best_index,
            metadata=dict(metadata or {}),
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        spec: RecirculationSpec,
        *,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RecirculationGeneration:
        """Run paired deterministic greedy decoding from an identical prompt."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
            raise ValueError("Recirculation generation expects input_ids shaped [1, seq]")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        source_module = self.model.get_submodule(spec.source_layer)
        destination_module = self.model.get_submodule(spec.destination_layer)
        self._validate_layer_order(source_module, destination_module)
        was_training = self.model.training
        self.model.eval()
        try:
            baseline = self._greedy_baseline(
                input_ids.clone(), max_new_tokens, eos_token_id
            )
            recirculated = self._greedy_recirculation(
                input_ids.clone(),
                source_module,
                destination_module,
                spec,
                max_new_tokens,
                eos_token_id,
            )
        finally:
            self.model.train(was_training)

        baseline_ids = tuple(int(item) for item in baseline[0].detach().cpu())
        recirculated_ids = tuple(int(item) for item in recirculated[0].detach().cpu())
        overlap = min(len(baseline_ids), len(recirculated_ids))
        changed = tuple(
            index
            for index in range(input_ids.shape[1], overlap)
            if baseline_ids[index] != recirculated_ids[index]
        )
        changed += tuple(range(overlap, max(len(baseline_ids), len(recirculated_ids))))
        return RecirculationGeneration(
            spec=spec,
            baseline_sequence=baseline_ids,
            recirculation_sequence=recirculated_ids,
            prompt_length=input_ids.shape[1],
            changed_positions=changed,
            metadata=dict(metadata or {}),
        )

    def _token_nll(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> Tuple[float, ...]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = _logits(output)[:, :-1, :]
        targets = input_ids[:, 1:]
        values = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        )
        return tuple(float(item) for item in values.detach().cpu())

    def _recirculated_token_nll(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        source_module: nn.Module,
        destination_module: nn.Module,
        spec: RecirculationSpec,
    ) -> Tuple[float, ...]:
        stored = []
        losses = []
        for position in range(input_ids.shape[1] - 1):
            mask = attention_mask[:, : position + 1] if attention_mask is not None else None
            logits, state = self._recirculated_step(
                input_ids[:, : position + 1],
                mask,
                stored,
                source_module,
                destination_module,
                spec,
            )
            stored.append(state)
            target = input_ids[:, position + 1]
            loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
            losses.append(float(loss.item()))
        return tuple(losses)

    def _recirculated_step(
        self,
        prefix: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        stored: Sequence[torch.Tensor],
        source_module: nn.Module,
        destination_module: nn.Module,
        spec: RecirculationSpec,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position = prefix.shape[1] - 1
        captured: Dict[str, torch.Tensor] = {}

        def destination_hook(_module: nn.Module, _args: Any, output: Any) -> Any:
            tensor = _first_tensor(output)
            captured["destination"] = tensor[:, position : position + 1].detach()
            if stored:
                replacement = torch.cat(tuple(stored), dim=1).to(
                    device=tensor.device, dtype=tensor.dtype
                )
                modified = tensor.clone()
                modified[:, : len(stored), :] = replacement
                return _replace_first_tensor(output, modified)
            return output

        def source_hook(_module: nn.Module, _args: Any, output: Any) -> Any:
            tensor = _first_tensor(output)
            captured["source"] = tensor[:, position : position + 1].detach()
            return output

        handles = (
            destination_module.register_forward_hook(destination_hook),
            source_module.register_forward_hook(source_hook),
        )
        try:
            output = self.model(
                input_ids=prefix, attention_mask=attention_mask, use_cache=False
            )
        finally:
            for handle in handles:
                handle.remove()
        if set(captured) != {"source", "destination"}:
            raise RuntimeError("Recirculation layers did not expose tensor outputs")
        state = mix_recirculated_residual(
            captured["source"],
            captured["destination"],
            alpha=spec.alpha_at(position),
            beta=spec.beta_at(position),
            normalization=spec.normalization,
        ).detach()
        return _logits(output)[:, -1, :], state

    def _greedy_baseline(
        self, sequence: torch.Tensor, max_new_tokens: int, eos_token_id: Optional[int]
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits = _logits(self.model(input_ids=sequence, use_cache=False))[:, -1, :]
            token = logits.argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
            if eos_token_id is not None and int(token.item()) == eos_token_id:
                break
        return sequence

    def _greedy_recirculation(
        self,
        sequence: torch.Tensor,
        source_module: nn.Module,
        destination_module: nn.Module,
        spec: RecirculationSpec,
        max_new_tokens: int,
        eos_token_id: Optional[int],
    ) -> torch.Tensor:
        stored = []
        logits = None
        for end in range(1, sequence.shape[1] + 1):
            logits, state = self._recirculated_step(
                sequence[:, :end], None, stored, source_module, destination_module, spec
            )
            stored.append(state)
        for _ in range(max_new_tokens):
            token = logits.argmax(dim=-1, keepdim=True)
            sequence = torch.cat((sequence, token), dim=1)
            if eos_token_id is not None and int(token.item()) == eos_token_id:
                break
            logits, state = self._recirculated_step(
                sequence, None, stored, source_module, destination_module, spec
            )
            stored.append(state)
        return sequence

    def _validate_layer_order(
        self, source_module: nn.Module, destination_module: nn.Module
    ) -> None:
        module_order = {
            id(module): index for index, module in enumerate(self.model.modules())
        }
        if module_order[id(source_module)] <= module_order[id(destination_module)]:
            raise ValueError("Recirculation source must execute after destination")


def _logits(output: Any) -> torch.Tensor:
    value = getattr(output, "logits", None)
    if value is None and isinstance(output, (tuple, list)) and output:
        value = output[0]
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise TypeError("Causal LM output must expose rank-3 logits")
    return value


def _first_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        tensor = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        tensor = output[0]
    else:
        raise TypeError("Recirculation layer output must contain a tensor")
    if tensor.ndim != 3:
        raise TypeError("Recirculation residual must have shape [batch, sequence, width]")
    return tensor


def _replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return tensor
    if isinstance(output, tuple):
        return (tensor,) + output[1:]
    if isinstance(output, list):
        return [tensor] + output[1:]
    raise TypeError("Recirculation layer output must contain a tensor")
