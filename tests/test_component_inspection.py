from __future__ import annotations

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import (
    ComponentInspector,
    HeadlessEngine,
    graph_from_component_comparison,
)


def _model():
    torch.manual_seed(23)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def test_component_drilldown_captures_once_and_projects_signed_logit_effects():
    model = _model()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    with HeadlessEngine().instrument(model) as session:
        layers = session.layers("block")
        comparison = ComponentInspector(session).compare_vocabulary_effects(
            reference_layer=layers[0],
            component_layer=layers[1],
            kwargs=inputs,
            normalizer=model.transformer.ln_f,
            top_k=4,
            decoder=lambda token_id: f"token-{token_id}",
        )

    assert comparison.position == 3
    assert comparison.component_shape == (1, 4, 16)
    assert len(comparison.promoted) == 4
    assert len(comparison.suppressed) == 4
    assert comparison.promoted[0].score >= comparison.promoted[-1].score
    assert comparison.suppressed[0].score <= comparison.suppressed[-1].score
    assert comparison.promoted[0].token.startswith("token-")
    assert json.loads(json.dumps(comparison.to_dict()))["schema_version"] == 1

    graph = graph_from_component_comparison(comparison)
    assert graph.node("component").layer_path == layers[1]
    assert {edge.method for edge in graph.edges} == {"vocabulary_projection_delta"}
    assert any(edge.score > 0 for edge in graph.edges)
    assert any(edge.score < 0 for edge in graph.edges)


def test_component_capture_removes_temporary_hooks():
    model = _model()
    session = HeadlessEngine().instrument(model)
    layer = session.layers("block")[0]
    block = dict(model.named_modules())[layer]
    before = len(block._forward_hooks)

    _output, captures = ComponentInspector(session).capture(
        [layer], kwargs={"input_ids": torch.tensor([[1, 2]])}
    )

    assert captures[layer].shape == (1, 2, 16)
    assert len(block._forward_hooks) == before
    session.close()
