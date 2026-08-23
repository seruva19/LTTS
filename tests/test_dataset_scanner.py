from __future__ import annotations

import json

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import (
    ActivationDatasetScanner,
    DatasetExample,
    HeadlessEngine,
)


def _model():
    torch.manual_seed(29)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def test_dataset_scanner_keeps_bounded_top_examples_and_running_statistics():
    model = _model()
    examples = [
        DatasetExample(
            id=f"example-{index}",
            inputs={"input_ids": torch.tensor([[1, 2, 3 + index]])},
            metadata={"index": index},
        )
        for index in range(5)
    ]
    with HeadlessEngine().instrument(model) as session:
        layer = session.layers("block")[0]
        result = ActivationDatasetScanner(session).scan(
            examples,
            layer=layer,
            feature_extractor=lambda hidden: hidden,
            feature_ids=[0, 3],
            top_k=2,
            activation_threshold=0.0,
            compute_correlations=True,
        )

    assert result.examples_seen == 5
    assert result.vectors_seen == 15
    assert [feature.feature_id for feature in result.features] == [0, 3]
    for feature in result.features:
        assert feature.count == 15
        assert feature.variance >= 0
        assert 0 <= feature.activation_frequency <= 1
        assert len(feature.top_activations) == 2
        assert feature.top_activations[0].value >= feature.top_activations[1].value
        assert feature.top_activations[0].metadata["index"] in range(5)
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1
    assert len(result.correlations) == 1
    assert result.correlations[0].left_feature_id == 0
    assert result.correlations[0].right_feature_id == 3
    assert -1 <= result.correlations[0].correlation <= 1


def test_dataset_scanner_honors_example_limit():
    model = _model()
    examples = (
        DatasetExample(id=str(index), inputs={"input_ids": torch.tensor([[1, 2]])})
        for index in range(100)
    )
    with HeadlessEngine().instrument(model) as session:
        result = ActivationDatasetScanner(session).scan(
            examples,
            layer=session.layers("block")[0],
            feature_extractor=lambda hidden: hidden,
            feature_ids=[0],
            max_examples=3,
        )

    assert result.examples_seen == 3
    assert result.vectors_seen == 6
