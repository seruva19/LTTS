from __future__ import annotations

import inspect
import importlib.util
from typing import Any, Dict, Iterable, List, Optional, Set


ATTENTION_MODULES = {
    "attention_entropy_profiler",
    "attention_head_grid",
    "attention_rollout",
    "attn_visualizer",
    "induction_head_detector",
}

GRADIENT_MODULES = {"gradient_importance"}

INTERVENTION_MODULES = {"neuron_ablation", "activation_patching"}

TOKENIZER_MODULES = {
    "contrastive_logit_difference",
    "direct_logit_attribution",
    "embedding_projector",
    "logit_lens",
    "logit_lens_grid",
    "next_token_predictions",
}

UNEMBEDDING_MODULES = {
    "contrastive_logit_difference",
    "direct_logit_attribution",
    "logit_lens",
    "logit_lens_grid",
    "next_token_predictions",
}

NO_HIDDEN_STATE_MODULES = {
    "memory_monitor",
    "model_introspector",
    "sample_module",
}

ALL_ATTACHABLE_ROLES = {
    "block",
    "self_attention",
    "cross_attention",
    "mlp",
    "moe_router",
    "moe_expert",
    "normalization",
    "embedding",
    "positional_embedding",
    "activation",
}

MODEL_LEVEL_MODULES = {
    "memory_monitor",
    "model_control_vector",
    "model_introspector",
    "occlusion_attribution",
}

SELF_ATTENTION_ONLY_MODULES = {
    "attention_rollout",
    "induction_head_detector",
}

ATTENTION_TARGET_MODULES = ATTENTION_MODULES - SELF_ATTENTION_ONLY_MODULES

BLOCK_STREAM_MODULES = {
    "activation_trajectory",
    "contrastive_logit_difference",
    "direct_logit_attribution",
    "embedding_projector",
    "logit_lens",
    "logit_lens_grid",
    "next_token_predictions",
    "representation_similarity",
    "residual_stream_norm",
    "sae_features",
}

NEURON_MODULES = {
    "neuron_ablation",
    "neuron_activation_map",
    "neuron_tracker",
}

OPTIONAL_MODULE_PACKAGES = {
    "model_control_vector": ("repeng",),
    "sae_features": ("sae_lens",),
}


def module_target_roles(
    module_name: str, metadata: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """Return the architecture roles on which a module has a meaningful contract."""

    metadata = metadata or {}
    explicit = metadata.get("target_roles")
    if isinstance(explicit, (list, tuple, set)):
        return {str(item) for item in explicit if item}

    name = str(module_name or metadata.get("name") or "").lower()
    if metadata.get("module_type") == "model_level" or name in MODEL_LEVEL_MODULES:
        return {"model"}
    if name in SELF_ATTENTION_ONLY_MODULES:
        return {"self_attention"}
    if name in ATTENTION_TARGET_MODULES or name.startswith("attention_"):
        return {"self_attention", "cross_attention"}
    if name in BLOCK_STREAM_MODULES:
        return {"block"}
    if name in NEURON_MODULES:
        return {"block", "mlp", "activation"}
    if name == "token_contributor":
        return {"block"}
    return set(ALL_ATTACHABLE_ROLES)


def module_supported_topologies(
    module_name: str, metadata: Optional[Dict[str, Any]] = None
) -> Set[str]:
    metadata = metadata or {}
    explicit = metadata.get("supported_topologies")
    if isinstance(explicit, (list, tuple, set)):
        return {str(item) for item in explicit if item}

    name = str(module_name or metadata.get("name") or "").lower()
    if name in {"model_control_vector", "sae_features"}:
        return {"decoder_only"}
    return {
        "decoder_only",
        "encoder_only",
        "encoder_decoder",
        "multimodal",
        "unknown",
    }


def module_dependency_status(module_name: str) -> List[Dict[str, Any]]:
    """Describe optional features without importing heavy dependency packages."""

    statuses = []
    for package in OPTIONAL_MODULE_PACKAGES.get(str(module_name).lower(), ()):
        available = importlib.util.find_spec(package) is not None
        statuses.append(
            {
                "package": package,
                "available": available,
                "impact": (
                    "full feature set available"
                    if available
                    else "optional feature is disabled until the package is installed"
                ),
            }
        )
    return statuses


def module_attachment_contract(
    module_name: str, metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build the shared backend/UI attachment contract for a shipped module."""

    metadata = metadata or {}
    return {
        "target_roles": sorted(module_target_roles(module_name, metadata)),
        "supported_topologies": sorted(
            module_supported_topologies(module_name, metadata)
        ),
        "required_capabilities": sorted(
            module_required_capabilities(module_name, metadata)
        ),
        "optional_dependencies": module_dependency_status(module_name),
        "model_level": "model" in module_target_roles(module_name, metadata),
    }


def module_required_capabilities(
    module_name: str, metadata: Optional[Dict[str, Any]] = None
) -> Set[str]:
    """Return explicit or conservatively inferred runtime requirements."""
    metadata = metadata or {}
    explicit = metadata.get("requires")
    if isinstance(explicit, (list, tuple, set)):
        return {str(item) for item in explicit if item}

    name = str(module_name or metadata.get("name") or "").lower()
    required: Set[str] = set()
    if name not in NO_HIDDEN_STATE_MODULES:
        required.add("hidden_states")
    if name in ATTENTION_MODULES or name.startswith("attention_"):
        required.add("eager_attention")
    if name in GRADIENT_MODULES:
        required.add("gradients")
    if name in TOKENIZER_MODULES:
        required.add("tokenizer_decoding")
    if name in UNEMBEDDING_MODULES:
        required.add("unembedding")
    return required


def selected_module_requests(
    modules: Optional[Dict[str, List[Any]]]
) -> Dict[str, Set[str]]:
    """Normalize inference attachment input into module -> requested layers."""
    selected: Dict[str, Set[str]] = {}
    for layer_name, entries in (modules or {}).items():
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("module") or entry.get("id")
            else:
                name = entry
            if name:
                selected.setdefault(str(name), set()).add(str(layer_name))
    return selected


def _supports_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    if name in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _unwrap_model(ltts_model: Any) -> Any:
    return getattr(ltts_model, "original_model", None) or ltts_model


def _has_unembedding(model: Any) -> bool:
    try:
        getter = getattr(model, "get_output_embeddings", None)
        if callable(getter) and getter() is not None:
            return True
    except Exception:
        pass
    return getattr(model, "lm_head", None) is not None


def _has_trainable_floating_parameter(model: Any) -> bool:
    try:
        return any(
            parameter.requires_grad and parameter.is_floating_point()
            for parameter in model.parameters()
        )
    except Exception:
        return False


def inspect_model_capabilities(ltts_model: Any) -> Dict[str, Dict[str, Any]]:
    """Inspect capabilities without running inference or changing model state."""
    model = _unwrap_model(ltts_model)
    config = getattr(model, "config", None)
    tokenizer = getattr(ltts_model, "_tokenizer", None)
    forward = getattr(model, "forward", None)
    architecture = getattr(ltts_model, "architecture", None) or {}
    architecture_capabilities = architecture.get("capabilities") or {}

    hidden_states = bool(
        architecture_capabilities.get("hidden_states")
        or (config is not None and hasattr(config, "output_hidden_states"))
        or (callable(forward) and _supports_kwarg(forward, "output_hidden_states"))
    )

    current_attention = None
    if config is not None:
        current_attention = getattr(config, "_attn_implementation", None)
    eager_attention = bool(
        callable(getattr(model, "set_attn_implementation", None))
        or current_attention in {"eager", "eager_paged", "flex_attention"}
        or config is not None and hasattr(config, "_attn_implementation")
    )

    tokenizer_decoding = bool(
        tokenizer is not None and callable(getattr(tokenizer, "decode", None))
    )

    capabilities = {
        "hidden_states": {
            "supported": hidden_states,
            "detail": "Model forward accepts hidden-state output requests"
            if hidden_states
            else "Model does not expose hidden-state output controls",
        },
        "eager_attention": {
            "supported": eager_attention,
            "detail": f"Current attention implementation: {current_attention or 'unspecified'}",
        },
        "gradients": {
            "supported": _has_trainable_floating_parameter(model),
            "detail": "Floating-point trainable parameters are available",
        },
        "tokenizer_decoding": {
            "supported": tokenizer_decoding,
            "detail": "Tokenizer decode is available"
            if tokenizer_decoding
            else "Tokenizer has not been attached or cannot decode",
        },
        "unembedding": {
            "supported": _has_unembedding(model),
            "detail": "Output embedding / lm_head is available",
        },
    }
    for name in (
        "self_attention",
        "cross_attention",
        "encoder",
        "decoder",
        "causal_lm",
        "masked_lm",
        "seq2seq_lm",
        "moe",
        "multimodal",
    ):
        supported = bool(architecture_capabilities.get(name, False))
        capabilities[name] = {
            "supported": supported,
            "detail": (
                f"Architecture topology provides {name.replace('_', ' ')}"
                if supported
                else f"Architecture topology does not provide {name.replace('_', ' ')}"
            ),
        }
    return capabilities


def build_compatibility_report(
    context: Any,
    ltts_model: Any,
    model_id: str,
    modules: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    capabilities = inspect_model_capabilities(ltts_model)
    requests = selected_module_requests(modules)
    all_modules = context.module_loader.get_all_modules()
    model_layers = set(getattr(ltts_model, "layer_wrappers", {}).keys())
    module_descriptors = getattr(ltts_model, "module_descriptors", {}) or {}
    architecture = getattr(ltts_model, "architecture", None) or {}
    topology = str(architecture.get("topology") or "unknown")

    module_reports = []
    # Gradient attribution and activation interventions mutate the forward
    # pass in incompatible ways: patched/ablated activations would make the
    # measured gradients describe the intervention, not the model.
    conflict_reasons: Dict[str, List[str]] = {}
    intervention_selected = sorted(requests.keys() & INTERVENTION_MODULES)
    if intervention_selected:
        for gradient_name in requests.keys() & GRADIENT_MODULES:
            conflict_reasons[gradient_name] = [
                "incompatible with real-time intervention module(s): "
                + ", ".join(intervention_selected)
            ]
    for name, requested_layers in sorted(requests.items()):
        info = all_modules.get(name) or {}
        metadata = info.get("metadata") or {}
        contract = module_attachment_contract(name, metadata)
        required = sorted(module_required_capabilities(name, metadata))
        missing = [
            capability
            for capability in required
            if not capabilities.get(capability, {}).get("supported", False)
        ]
        unresolved_layers = []
        role_mismatches = []
        for layer in sorted(requested_layers):
            if contract["model_level"]:
                continue
            if layer == "__model__":
                role_mismatches.append(
                    {
                        "layer": layer,
                        "role": "model",
                        "allowed_roles": contract["target_roles"],
                    }
                )
                continue
            resolved_layer = None
            if layer in model_layers:
                resolved_layer = layer
            else:
                matches = sorted(
                    candidate
                    for candidate in model_layers
                    if candidate.endswith(layer) or layer.endswith(candidate)
                )
                if matches:
                    resolved_layer = matches[0]
            if resolved_layer is None:
                unresolved_layers.append(layer)
                continue
            role = (module_descriptors.get(resolved_layer) or {}).get("role")
            if role and role not in contract["target_roles"]:
                role_mismatches.append(
                    {
                        "layer": layer,
                        "resolved_layer": resolved_layer,
                        "role": role,
                        "allowed_roles": contract["target_roles"],
                    }
                )

        topology_supported = topology in contract["supported_topologies"]
        missing_optional_dependencies = [
            dependency["package"]
            for dependency in contract["optional_dependencies"]
            if not dependency.get("available", False)
        ]

        installed = bool(info)
        module_reports.append(
            {
                "name": name,
                "installed": installed,
                "requested_layers": sorted(requested_layers),
                "required_capabilities": required,
                "missing_capabilities": missing,
                "target_roles": contract["target_roles"],
                "role_mismatches": role_mismatches,
                "supported_topologies": contract["supported_topologies"],
                "topology_supported": topology_supported,
                "optional_dependencies": contract["optional_dependencies"],
                "missing_optional_dependencies": missing_optional_dependencies,
                "unresolved_layers": unresolved_layers,
                "conflict_reasons": conflict_reasons.get(name, []),
                "compatible": bool(
                    installed
                    and not missing
                    and not missing_optional_dependencies
                    and not unresolved_layers
                    and not role_mismatches
                    and not conflict_reasons.get(name)
                    and topology_supported
                ),
            }
        )

    incompatible = [report["name"] for report in module_reports if not report["compatible"]]
    model = _unwrap_model(ltts_model)
    config = getattr(model, "config", None)
    return {
        "model_id": model_id,
        "model_class": type(model).__name__,
        "model_revision": getattr(config, "_commit_hash", None),
        "architecture": architecture,
        "capabilities": capabilities,
        "modules": module_reports,
        "compatible": not incompatible,
        "incompatible_modules": incompatible,
    }


def apply_capability_override(
    report: Dict[str, Any], capability: str, status: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply an active capability probe and refresh per-module verdicts."""
    report.setdefault("capabilities", {})[capability] = dict(status)
    for module_report in report.get("modules", []):
        required = module_report.get("required_capabilities", [])
        module_report["missing_capabilities"] = [
            name
            for name in required
            if not report["capabilities"].get(name, {}).get("supported", False)
        ]
        module_report["compatible"] = bool(
            module_report.get("installed")
            and not module_report["missing_capabilities"]
            and not module_report.get("missing_optional_dependencies")
            and not module_report.get("unresolved_layers")
            and not module_report.get("role_mismatches")
            and module_report.get("topology_supported", True)
        )

    incompatible = [
        item["name"]
        for item in report.get("modules", [])
        if not item.get("compatible", False)
    ]
    report["incompatible_modules"] = incompatible
    report["compatible"] = not incompatible
    return report
