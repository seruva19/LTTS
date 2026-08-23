from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from ltts_engine import (
    discover_research_adapter,
    graph_from_cross_modal_attention,
    graph_from_vision_patch_scores,
    intervene_ssm_state,
    research_workflow_matrix,
    ssm_effective_influence,
)


class StateMixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(4))
        self.D = nn.Parameter(torch.ones(4))
        self.x_proj = nn.Linear(4, 4)

    def forward(self, hidden):
        return hidden


class PatchBoundary(nn.Module):
    patch_size = (16, 16)

    def forward(self, pixels):
        return pixels


class FusionBoundary(nn.Module):
    def forward(self, hidden_states, encoder_hidden_states=None):
        return hidden_states


class MixedTopology(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = StateMixer()
        self.patches = PatchBoundary()
        self.fusion = FusionBoundary()


def test_capabilities_are_discovered_from_boundaries_not_model_names():
    adapter = discover_research_adapter(MixedTopology())

    assert adapter.supports("ssm_mixer_intervention")
    assert not adapter.supports("ssm_state_intervention")
    assert adapter.supports("vision_patch_attribution")
    assert adapter.supports("cross_modal_attention_attribution")
    assert adapter.points("state_mixer")[0].path == "state"
    assert not adapter.points("state_mixer")[0].metadata["explicit_state_boundary"]
    assert adapter.points("patch_embedding")[0].metadata["patch_size"] == (16, 16)
    adapter.require("effective_influence", "cross_modal_capture")
    with pytest.raises(ValueError, match="missing_capability"):
        adapter.require("missing_capability")
    assert json.loads(json.dumps(adapter.to_dict()))["schema_version"] == 1


def test_ssm_influence_and_state_edit_have_explicit_recurrence_semantics():
    influence = ssm_effective_influence(
        decay=torch.full((1,), 0.5), input_gain=torch.ones(3, 1)
    )
    assert torch.allclose(
        influence,
        torch.tensor([[1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.25, 0.5, 1.0]]),
    )

    original = torch.ones(1, 3, 4)
    changed = intervene_ssm_state(
        original, channels=[1, 3], positions=[2], operation="set", value=0.0
    )
    assert torch.equal(original, torch.ones_like(original))
    assert changed[0, 2].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_vision_and_cross_modal_graphs_retain_modality_coordinates():
    patches = graph_from_vision_patch_scores(
        [0.1, -0.2, 0.3, 0.4], grid_shape=(2, 2), target_label="cat"
    )
    assert patches.node("patch:1:0").metadata == {
        "modality": "vision",
        "row": 1,
        "column": 0,
    }
    assert [edge.score for edge in patches.edges] == [0.1, -0.2, 0.3, 0.4]

    cross = graph_from_cross_modal_attention(
        [[0.7, 0.3]],
        text_labels=["cat"],
        patch_labels=["left", "right"],
        layer_path="fusion.0",
        head_index=2,
    )
    assert len(cross.nodes) == 3
    assert cross.node("vision:0").metadata["modality"] == "vision"
    assert all(edge.metadata["head_index"] == 2 for edge in cross.edges)


def test_real_transformers_mamba_and_vit_boundaries_are_discovered():
    from transformers import MambaConfig, MambaForCausalLM, ViTConfig, ViTModel

    mamba = MambaForCausalLM(
        MambaConfig(
            vocab_size=32,
            hidden_size=16,
            state_size=8,
            num_hidden_layers=1,
            expand=2,
            conv_kernel=4,
        )
    )
    mamba_adapter = discover_research_adapter(mamba)
    assert mamba_adapter.supports("effective_influence")
    assert mamba_adapter.supports("ssm_mixer_intervention")
    assert not mamba_adapter.supports("ssm_state_intervention")
    assert mamba_adapter.points("state_mixer")[0].path == "backbone.layers.0.mixer"

    vit = ViTModel(
        ViTConfig(
            image_size=16,
            patch_size=8,
            num_channels=3,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
        )
    )
    vit_adapter = discover_research_adapter(vit)
    assert vit_adapter.supports("vision_patch_attribution")
    assert [point.path for point in vit_adapter.points("patch_embedding")] == [
        "embeddings.patch_embeddings"
    ]

    mamba_matrix = {row.name: row for row in research_workflow_matrix(mamba_adapter)}
    assert mamba_matrix["ssm_effective_influence"].supported
    assert mamba_matrix["ssm_mixer_intervention"].capture_points == (
        "backbone.layers.0.mixer",
    )
    assert not mamba_matrix["ssm_raw_state_intervention"].supported
    assert mamba_matrix["ssm_raw_state_intervention"].missing_capabilities == (
        "ssm_state_intervention",
    )

    vit_matrix = {row.name: row for row in research_workflow_matrix(vit_adapter)}
    assert vit_matrix["vision_patch_attribution"].supported
    assert vit_matrix["vision_patch_attribution"].capture_points == (
        "embeddings.patch_embeddings",
    )


def test_real_vision_encoder_decoder_exposes_exact_cross_modal_boundary():
    from transformers import (
        GPT2Config,
        ViTConfig,
        VisionEncoderDecoderConfig,
        VisionEncoderDecoderModel,
    )

    encoder = ViTConfig(
        image_size=16,
        patch_size=8,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    decoder = GPT2Config(
        vocab_size=32,
        n_embd=16,
        n_layer=1,
        n_head=2,
        n_positions=16,
        add_cross_attention=True,
        is_decoder=True,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = VisionEncoderDecoderModel(
        VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder, decoder)
    )
    adapter = discover_research_adapter(model)
    matrix = {row.name: row for row in research_workflow_matrix(adapter)}

    assert [point.path for point in adapter.points("patch_embedding")] == [
        "encoder.embeddings.patch_embeddings"
    ]
    assert [point.path for point in adapter.points("cross_attention")] == [
        "decoder.transformer.h.0.crossattention"
    ]
    assert matrix["cross_modal_attention_attribution"].supported
    assert matrix["cross_modal_attention_attribution"].capture_points == (
        "decoder.transformer.h.0.crossattention",
    )

    output = model(
        pixel_values=torch.randn(1, 3, 16, 16),
        decoder_input_ids=torch.tensor([[1, 2]]),
    )
    assert output.logits.shape == (1, 2, 32)
