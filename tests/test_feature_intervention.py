from __future__ import annotations

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import (
    FeatureInterventionSpec,
    HeadlessEngine,
    build_feature_intervention,
)


def _model():
    torch.manual_seed(43)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def test_feature_intervention_changes_logits_and_emits_causal_summary():
    model = _model()
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    module = build_feature_intervention(
        encoder=lambda hidden: hidden,
        decoder=lambda features: features,
        spec=FeatureInterventionSpec(
            feature_ids=[0, 1], operation="add", value=10.0, positions="last"
        ),
    )
    with HeadlessEngine().instrument(model) as session:
        with torch.no_grad():
            baseline = session.run(**inputs).logits.detach().clone()
        session.attach(module, layer=session.layers("block")[0])
        with torch.no_grad():
            changed = session.run(**inputs).logits.detach().clone()
        artifacts = session.artifacts

    assert not torch.allclose(baseline, changed)
    assert len(artifacts) == 1
    assert artifacts[0].module == "feature_intervention"
    assert artifacts[0].data["type"] == "table"
