from __future__ import annotations

import copy
import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import HeadlessEngine, ModelDiffer


def _model():
    torch.manual_seed(37)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def test_model_diff_aligns_layers_and_detects_checkpoint_change():
    left_model = _model()
    right_model = copy.deepcopy(left_model)
    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}

    with (
        HeadlessEngine().instrument(left_model) as left,
        HeadlessEngine().instrument(right_model) as right,
    ):
        identical = ModelDiffer(left, right).compare(left_kwargs=inputs)
        assert len(identical.alignments) == 2
        assert not identical.unmatched_left
        assert not identical.unmatched_right
        assert all(
            item.mean_absolute_delta == pytest.approx(0.0)
            for item in identical.activation_differences
        )
        assert identical.logit_difference.mean_absolute_delta == pytest.approx(0.0)
        assert identical.token_alignment.exact_match

        with torch.no_grad():
            right_model.transformer.h[0].mlp.c_proj.weight.zero_()
            right_model.lm_head.weight[0].add_(1.0)
        changed = ModelDiffer(left, right).compare(left_kwargs=inputs)

    assert any(item.mean_absolute_delta > 0 for item in changed.activation_differences)
    assert changed.logit_difference.mean_absolute_delta > 0
    assert changed.logit_difference.symmetric_kl > 0
    assert json.loads(json.dumps(changed.to_dict()))["schema_version"] == 1


def test_model_diff_reports_token_mismatch_instead_of_hiding_it():
    left_model = _model()
    right_model = copy.deepcopy(left_model)
    with (
        HeadlessEngine().instrument(left_model) as left,
        HeadlessEngine().instrument(right_model) as right,
    ):
        result = ModelDiffer(left, right).compare(
            left_kwargs={"input_ids": torch.tensor([[1, 2, 3]])},
            right_kwargs={"input_ids": torch.tensor([[1, 9, 3]])},
        )

    assert not result.token_alignment.exact_match
    assert result.token_alignment.mismatch_positions == ((0, 1),)
    assert not result.token_alignment.truncated_mismatches
