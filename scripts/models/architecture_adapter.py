from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Optional

import torch.nn as nn


ENCODER_MODEL_TYPES = {
    "albert",
    "bert",
    "camembert",
    "deberta",
    "deberta-v2",
    "distilbert",
    "electra",
    "ernie",
    "flaubert",
    "longformer",
    "mpnet",
    "rembert",
    "roberta",
    "xlm-roberta",
}

ENCODER_DECODER_MODEL_TYPES = {
    "bart",
    "blenderbot",
    "fsmt",
    "led",
    "longt5",
    "m2m-100",
    "marian",
    "mbart",
    "mt5",
    "pegasus",
    "switch_transformers",
    "t5",
    "umt5",
}

MULTIMODAL_MODEL_TYPES = {
    "blip",
    "blip-2",
    "chameleon",
    "flamingo",
    "idefics",
    "idefics2",
    "idefics3",
    "llava",
    "llava_next",
    "mllama",
    "paligemma",
    "qwen2_vl",
}

DECODER_MODEL_TYPES = {
    "baichuan",
    "bloom",
    "cohere",
    "deepseek_v2",
    "deepseek_v3",
    "falcon",
    "gemma",
    "gemma2",
    "gpt2",
    "gpt_bigcode",
    "gpt_neo",
    "gpt_neox",
    "internlm",
    "llama",
    "mistral",
    "mixtral",
    "mpt",
    "olmo",
    "opt",
    "phi",
    "phi3",
    "qwen2",
    "qwen2_moe",
    "stablelm",
}


@dataclass(frozen=True)
class ArchitectureCapabilities:
    hidden_states: bool = True
    attention_outputs: bool = False
    self_attention: bool = False
    cross_attention: bool = False
    encoder: bool = False
    decoder: bool = False
    causal_lm: bool = False
    masked_lm: bool = False
    seq2seq_lm: bool = False
    unembedding: bool = False
    gradients: bool = False
    moe: bool = False
    multimodal: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


@dataclass(frozen=True)
class ArchitectureDescriptor:
    family: str
    topology: str
    model_type: str
    model_class: str
    display_name: str
    capabilities: ArchitectureCapabilities

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["capabilities"] = self.capabilities.to_dict()
        return payload


@dataclass(frozen=True)
class ModuleDescriptor:
    path: str
    class_name: str
    role: str
    stage: str
    layer_index: Optional[int]
    attention_kind: Optional[str]
    is_block: bool
    is_leaf: bool
    attachable: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _config_architectures(config: Any) -> list[str]:
    values = getattr(config, "architectures", None) or []
    return [str(value) for value in values]


def _has_output_embedding(model: nn.Module) -> bool:
    try:
        getter = getattr(model, "get_output_embeddings", None)
        if callable(getter) and getter() is not None:
            return True
    except Exception:
        pass
    return bool(getattr(model, "lm_head", None) is not None)


def _config_count(config: Any, *keys: str) -> int:
    """Read an integer-like architecture count without trusting config types."""

    for key in keys:
        value = getattr(config, key, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return 0


class TransformerArchitectureAdapter:
    """Describe transformer topology and module roles without path-specific APIs."""

    INTERCEPTABLE_ROLES = {
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

    def __init__(self, model: nn.Module):
        self.model = model
        self.config = getattr(model, "config", None)
        self.model_type = _lower(getattr(self.config, "model_type", None))
        self.model_class = type(model).__name__
        self._named_modules = dict(model.named_modules())
        self.descriptor = self._describe_architecture()
        self._module_descriptors: Dict[str, ModuleDescriptor] = {}

    def _describe_architecture(self) -> ArchitectureDescriptor:
        architectures = _config_architectures(self.config)
        architecture_text = " ".join(architectures).lower()
        model_class_lower = self.model_class.lower()

        has_multimodal_config = bool(
            self.config is not None
            and getattr(self.config, "vision_config", None) is not None
            and (
                getattr(self.config, "text_config", None) is not None
                or getattr(self.config, "language_config", None) is not None
            )
        )
        is_multimodal = (
            has_multimodal_config
            or self.model_type in MULTIMODAL_MODEL_TYPES
            or any(token in model_class_lower for token in ("vision2seq", "llava", "idefics"))
        )

        is_encoder_decoder = bool(
            getattr(self.config, "is_encoder_decoder", False)
            or self.model_type in ENCODER_DECODER_MODEL_TYPES
        )
        is_encoder_only = bool(
            not is_encoder_decoder
            and (
                self.model_type in ENCODER_MODEL_TYPES
                or any(
                    token in architecture_text
                    for token in ("maskedlm", "sequenceclassification", "tokenclassification")
                )
            )
        )
        is_decoder_only = bool(
            not is_encoder_decoder
            and not is_encoder_only
            and (
                self.model_type in DECODER_MODEL_TYPES
                or "causallm" in architecture_text
                or model_class_lower.endswith(("causallm", "lmheadmodel"))
            )
        )

        if is_multimodal:
            topology = "multimodal"
        elif is_encoder_decoder:
            topology = "encoder_decoder"
        elif is_encoder_only:
            topology = "encoder_only"
        elif is_decoder_only:
            topology = "decoder_only"
        else:
            topology = "unknown"

        module_text = " ".join(
            f"{name} {type(module).__name__}" for name, module in self._named_modules.items()
        ).lower()
        is_moe = bool(
            _config_count(
                self.config,
                "num_experts",
                "num_local_experts",
                "n_routed_experts",
            )
            > 1
            or re.search(
                r"(^|[.\s_])(moe|experts?|router|sparse_moe)([.\s_]|$)",
                module_text,
            )
        )

        has_attention = "attention" in module_text or "attn" in module_text
        has_cross_attention = bool(
            is_encoder_decoder
            or "cross_attn" in module_text
            or "crossattention" in module_text
            or "encoder_attn" in module_text
        )
        has_gradients = any(
            parameter.requires_grad and parameter.is_floating_point()
            for parameter in self.model.parameters()
        )
        has_unembedding = _has_output_embedding(self.model)

        capabilities = ArchitectureCapabilities(
            hidden_states=callable(getattr(self.model, "forward", None)),
            attention_outputs=has_attention,
            self_attention=has_attention,
            cross_attention=has_cross_attention,
            encoder=is_encoder_decoder or is_encoder_only,
            decoder=is_encoder_decoder or is_decoder_only or is_multimodal,
            causal_lm=is_decoder_only,
            masked_lm=is_encoder_only and has_unembedding,
            seq2seq_lm=is_encoder_decoder,
            unembedding=has_unembedding,
            gradients=has_gradients,
            moe=is_moe,
            multimodal=is_multimodal,
        )

        family = self.model_type or self._infer_family_from_class()
        display_family = family.replace("_", " ").replace("-", " ").title()
        display_topology = topology.replace("_", " ").title()
        display_name = (
            f"{display_family} · {display_topology}"
            if display_family and display_family != "Unknown"
            else display_topology
        )
        if is_moe:
            display_name = f"{display_name} · MoE"

        return ArchitectureDescriptor(
            family=family or "unknown",
            topology=topology,
            model_type=self.model_type or "unknown",
            model_class=self.model_class,
            display_name=display_name,
            capabilities=capabilities,
        )

    def _infer_family_from_class(self) -> str:
        value = self.model_class
        value = re.sub(
            r"(ForCausalLM|ForMaskedLM|ForConditionalGeneration|LMHeadModel|Model)$",
            "",
            value,
        )
        value = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower().strip("_")
        return value or "unknown"

    def describe_module(self, path: str, module: nn.Module) -> ModuleDescriptor:
        cached = self._module_descriptors.get(path)
        if cached is not None:
            return cached

        class_name = type(module).__name__
        class_lower = class_name.lower()
        path_lower = _lower(path)
        tokens = [token for token in path_lower.split(".") if token]
        leaf = tokens[-1] if tokens else ""
        is_leaf = not any(True for _ in module.children())
        stage = self._detect_stage(path_lower)
        layer_index = self._detect_layer_index(tokens)

        is_cross_attention = bool(
            any(
                token in path_lower or token in class_lower
                for token in ("cross_attn", "crossattention", "encoder_attn", "encdecattention")
            )
        )
        is_attention = bool(
            is_cross_attention
            or "attention" in class_lower
            or "attn" in class_lower
            or leaf in {"attn", "attention", "self_attn", "selfattention"}
        )
        is_attention_projection = bool(
            leaf
            in {
                "q",
                "k",
                "v",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "out_proj",
                "c_attn",
                "c_proj",
                "query",
                "key",
                "value",
            }
            and any(token in path_lower for token in ("attn", "attention"))
        )

        is_block = self._is_block(path_lower, class_lower)
        role = "other"
        attention_kind: Optional[str] = None

        if is_block:
            role = "block"
        elif is_attention_projection:
            role = "attention_projection"
        elif is_attention:
            role = "cross_attention" if is_cross_attention else "self_attention"
            attention_kind = "cross" if is_cross_attention else "self"
        elif self._is_router(path_lower, class_lower, leaf):
            role = "moe_router"
        elif self._is_expert(path_lower, class_lower):
            role = "moe_expert"
        elif self._is_embedding(path_lower, class_lower):
            role = (
                "positional_embedding"
                if any(token in path_lower for token in ("position", "pos_embed", ".wpe"))
                else "embedding"
            )
        elif self._is_normalization(path_lower, class_lower):
            role = "normalization"
        elif self._is_mlp(path_lower, class_lower):
            role = "mlp"
        elif any(token in class_lower for token in ("gelu", "silu", "relu", "activation")):
            role = "activation"
        elif any(
            token == leaf
            for token in ("lm_head", "classifier", "classification_head", "score")
        ):
            role = "output"
        elif "dropout" in class_lower:
            role = "dropout"

        attachable = bool(path and role in self.INTERCEPTABLE_ROLES)
        descriptor = ModuleDescriptor(
            path=path,
            class_name=class_name,
            role=role,
            stage=stage,
            layer_index=layer_index,
            attention_kind=attention_kind,
            is_block=is_block,
            is_leaf=is_leaf,
            attachable=attachable,
        )
        self._module_descriptors[path] = descriptor
        return descriptor

    def should_wrap_module(self, path: str, module: nn.Module) -> bool:
        return self.describe_module(path, module).attachable

    def module_descriptors(self) -> Dict[str, ModuleDescriptor]:
        for path, module in self._named_modules.items():
            if path:
                self.describe_module(path, module)
        return dict(self._module_descriptors)

    def _detect_stage(self, path: str) -> str:
        if any(token in path for token in ("vision_model", "vision_tower", "visual.")):
            return "vision"
        if re.search(r"(^|\.)(encoder)(\.|$)", path):
            return "encoder"
        if re.search(r"(^|\.)(decoder)(\.|$)", path):
            return "decoder"
        if any(token in path for token in ("language_model", "text_model")):
            return (
                "decoder"
                if self.descriptor.topology in {"decoder_only", "multimodal"}
                else "text"
            )
        if self.descriptor.topology == "encoder_only":
            return "encoder"
        if self.descriptor.topology == "decoder_only":
            return "decoder"
        if path.startswith("shared"):
            return "shared"
        return "model"

    @staticmethod
    def _detect_layer_index(tokens: Iterable[str]) -> Optional[int]:
        values = list(tokens)
        for index, token in enumerate(values):
            if token.isdigit():
                previous = values[index - 1] if index > 0 else ""
                if previous in {"h", "layer", "layers", "block", "blocks", "expert", "experts"}:
                    return int(token)
        for token in values:
            if token.isdigit():
                return int(token)
        return None

    @staticmethod
    def _is_block(path: str, class_name: str) -> bool:
        class_patterns = (
            "decoderlayer",
            "encoderlayer",
            "transformerblock",
            "gpt2block",
            "gptneoxlayer",
            "bloomblock",
            "t5block",
            "bartdecoderlayer",
            "bartencoderlayer",
            "bertlayer",
            "robertalayer",
        )
        if any(pattern in class_name for pattern in class_patterns):
            return True
        return bool(
            re.search(r"(^|\.)(layers?|h|blocks?)\.\d+$", path)
            and not any(
                token in path
                for token in (".experts.", ".expert.", ".attention.", ".attn.")
            )
        )

    @staticmethod
    def _is_router(path: str, class_name: str, leaf: str) -> bool:
        return bool(
            leaf in {"router", "gate", "router_gate"}
            or "router" in class_name
            or "router" in path
        )

    @staticmethod
    def _is_expert(path: str, class_name: str) -> bool:
        return bool(
            "expert" in class_name
            or re.search(r"(^|\.)(experts?|expert)\.\d+($|\.)", path)
        )

    @staticmethod
    def _is_embedding(path: str, class_name: str) -> bool:
        return bool(
            "embedding" in class_name
            or any(
                token in path
                for token in (
                    "embed_tokens",
                    "embeddings",
                    "word_embeddings",
                    "position_embeddings",
                    "wte",
                    "wpe",
                    "shared",
                )
            )
        )

    @staticmethod
    def _is_normalization(path: str, class_name: str) -> bool:
        return bool(
            "norm" in class_name
            or re.search(r"(^|\.)(ln(_\d+|_f)?|norm|layer_norm|final_layer_norm)($|\.)", path)
        )

    @staticmethod
    def _is_mlp(path: str, class_name: str) -> bool:
        return bool(
            any(token in class_name for token in ("mlp", "feedforward", "ffn", "intermediate"))
            or any(
                token in path
                for token in (
                    ".mlp",
                    ".ffn",
                    ".feed_forward",
                    ".intermediate",
                    ".dense_relu_dense",
                )
            )
        )


def get_architecture_adapter(model: nn.Module) -> TransformerArchitectureAdapter:
    return TransformerArchitectureAdapter(model)


def describe_standalone_module(module: nn.Module) -> ModuleDescriptor:
    """Classify an isolated module for compatibility with legacy callers/tests."""

    class _StandaloneModel(nn.Module):
        def __init__(self, child: nn.Module):
            super().__init__()
            self.layer = child

    adapter = TransformerArchitectureAdapter(_StandaloneModel(module))
    return adapter.describe_module("layer", module)
