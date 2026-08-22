from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
)

from scripts.blueprints.blueprint_loader import load_blueprints
from scripts.models.architecture_adapter import get_architecture_adapter
from scripts.models.model_loader import ModelConfigItem, ModelsRegistry
from scripts.models.model_struct import traverse_model


class LlamaAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)


class LlamaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(8, 16)
        self.down_proj = nn.Linear(16, 8)


class LlamaDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = LlamaAttention()
        self.mlp = LlamaMLP()
        self.input_layernorm = nn.LayerNorm(8)


class LlamaForCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="llama",
            architectures=["LlamaForCausalLM"],
            is_encoder_decoder=False,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([LlamaDecoderLayer(), LlamaDecoderLayer()])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


class BertAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.self = nn.Linear(8, 8)


class BertIntermediate(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(8, 16)


class BertLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = BertAttention()
        self.intermediate = BertIntermediate()
        self.LayerNorm = nn.LayerNorm(8)


class BertForMaskedLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="bert",
            architectures=["BertForMaskedLM"],
            is_encoder_decoder=False,
        )
        self.bert = nn.Module()
        self.bert.encoder = nn.Module()
        self.bert.encoder.layer = nn.ModuleList([BertLayer()])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


class T5LayerSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.SelfAttention = nn.Linear(8, 8)


class T5LayerCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.EncDecAttention = nn.Linear(8, 8)


class T5Block(nn.Module):
    def __init__(self, cross_attention: bool):
        super().__init__()
        self.layer = nn.ModuleList([T5LayerSelfAttention()])
        if cross_attention:
            self.layer.append(T5LayerCrossAttention())
        self.layer.append(nn.LayerNorm(8))


class T5ForConditionalGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            model_type="t5",
            architectures=["T5ForConditionalGeneration"],
            is_encoder_decoder=True,
        )
        self.shared = nn.Embedding(32, 8)
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([T5Block(cross_attention=False)])
        self.decoder = nn.Module()
        self.decoder.block = nn.ModuleList([T5Block(cross_attention=True)])
        self.lm_head = nn.Linear(8, 32, bias=False)

    def forward(self, hidden_states):
        return self.lm_head(hidden_states)


class MixtralSparseMoeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(8, 2)
        self.experts = nn.ModuleList([LlamaMLP(), LlamaMLP()])


class MixtralDecoderLayer(LlamaDecoderLayer):
    def __init__(self):
        super().__init__()
        self.block_sparse_moe = MixtralSparseMoeBlock()


class MixtralForCausalLM(LlamaForCausalLM):
    def __init__(self):
        super().__init__()
        self.config.model_type = "mixtral"
        self.config.architectures = ["MixtralForCausalLM"]
        self.config.num_local_experts = 2
        self.model.layers = nn.ModuleList([MixtralDecoderLayer()])


def test_decoder_only_adapter_describes_blocks_attention_and_mlp():
    adapter = get_architecture_adapter(LlamaForCausalLM())

    assert adapter.descriptor.topology == "decoder_only"
    assert adapter.descriptor.family == "llama"
    assert adapter.descriptor.capabilities.causal_lm is True
    assert adapter.describe_module(
        "model.layers.0", adapter.model.model.layers[0]
    ).role == "block"
    attention = adapter.describe_module(
        "model.layers.0.self_attn", adapter.model.model.layers[0].self_attn
    )
    assert attention.role == "self_attention"
    assert attention.stage == "decoder"
    assert attention.layer_index == 0
    assert attention.attachable is True
    assert adapter.describe_module(
        "model.layers.0.mlp", adapter.model.model.layers[0].mlp
    ).role == "mlp"


def test_encoder_only_adapter_and_model_struct_expose_ui_metadata():
    model = BertForMaskedLM()
    adapter = get_architecture_adapter(model)
    struct = traverse_model(model)
    payload = struct.to_dict()

    assert adapter.descriptor.topology == "encoder_only"
    assert adapter.descriptor.capabilities.masked_lm is True
    assert payload["architectureFamily"] == "bert"
    assert payload["architectureTopology"] == "encoder_only"
    assert payload["capabilities"]["encoder"] is True
    layer = next(
        item for item in payload["layers"] if item["name"] == "bert.encoder.layer.0"
    )
    assert layer["moduleRole"] == "block"
    assert layer["architectureStage"] == "encoder"
    assert layer["attachable"] is True


def test_encoder_decoder_adapter_distinguishes_self_and_cross_attention():
    model = T5ForConditionalGeneration()
    adapter = get_architecture_adapter(model)

    assert adapter.descriptor.topology == "encoder_decoder"
    assert adapter.descriptor.capabilities.seq2seq_lm is True
    assert adapter.descriptor.capabilities.cross_attention is True
    descriptors = adapter.module_descriptors()
    cross = [
        descriptor
        for descriptor in descriptors.values()
        if descriptor.role == "cross_attention"
    ]
    assert cross
    assert all(item.stage == "decoder" for item in cross)
    encoder_block = descriptors["encoder.block.0"]
    decoder_block = descriptors["decoder.block.0"]
    assert encoder_block.stage == "encoder"
    assert decoder_block.stage == "decoder"


def test_moe_adapter_marks_router_and_experts():
    model = MixtralForCausalLM()
    adapter = get_architecture_adapter(model)
    descriptors = adapter.module_descriptors()

    assert adapter.descriptor.capabilities.moe is True
    assert descriptors["model.layers.0.block_sparse_moe.gate"].role == "moe_router"
    assert descriptors["model.layers.0.block_sparse_moe.experts.0"].role == "moe_expert"
    assert descriptors["model.layers.0.block_sparse_moe.experts.0"].attachable is True


def test_model_task_resolution_supports_aliases_and_three_core_topologies():
    registry = ModelsRegistry.__new__(ModelsRegistry)
    registry.config_list = {
        "my-llama": ModelConfigItem(id="org/real-llama", data={}),
        "forced-base": ModelConfigItem(
            id="org/encoder", data={"task": "base_model"}
        ),
    }

    assert registry._resolve_model_source("my-llama") == "org/real-llama"
    assert registry._resolve_model_source("direct/model") == "direct/model"

    task, loader = registry._resolve_model_task(
        "my-llama",
        SimpleNamespace(
            model_type="llama", architectures=["LlamaForCausalLM"], is_encoder_decoder=False
        ),
    )
    assert (task, loader) == ("causal_lm", AutoModelForCausalLM)

    task, loader = registry._resolve_model_task(
        "encoder",
        SimpleNamespace(
            model_type="bert", architectures=["BertForMaskedLM"], is_encoder_decoder=False
        ),
    )
    assert (task, loader) == ("masked_lm", AutoModelForMaskedLM)

    task, loader = registry._resolve_model_task(
        "seq2seq",
        SimpleNamespace(
            model_type="t5",
            architectures=["T5ForConditionalGeneration"],
            is_encoder_decoder=True,
        ),
    )
    assert (task, loader) == ("seq2seq_lm", AutoModelForSeq2SeqLM)

    task, _ = registry._resolve_model_task(
        "forced-base",
        SimpleNamespace(
            model_type="bert", architectures=["BertForMaskedLM"], is_encoder_decoder=False
        ),
    )
    assert task == "base_model"


def test_blueprint_loader_bootstraps_missing_directory(tmp_path):
    folder = tmp_path / "blueprints"

    result = load_blueprints(str(folder))

    assert folder.is_dir()
    assert result.blueprints == []
