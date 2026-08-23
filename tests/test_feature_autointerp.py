from __future__ import annotations

import json

import pytest

from ltts_engine import (
    DatasetExample,
    FeatureAutointerpreter,
    FeatureEvidence,
    InterpretationDraft,
    build_dataset_manifest,
    partition_feature_evidence,
    score_explanation,
)


class KeywordProvider:
    def explain(self, feature_id, evidence):
        assert feature_id == 12
        assert {item.example_id for item in evidence} == {"train-a", "train-b"}
        return InterpretationDraft(
            label="mentions sunlight",
            explanation="The feature responds to text about the sun.",
            notes=("Check metaphorical uses separately.",),
        )


def _predict(_draft, evidence):
    return float("sun" in evidence.text.lower())


def test_autointerp_keeps_generation_and_heldout_scoring_separate():
    result = FeatureAutointerpreter(KeywordProvider()).interpret(
        12,
        training=(
            FeatureEvidence("train-a", "sunrise", 2.0),
            FeatureEvidence("train-b", "night", 0.0),
        ),
        held_out=(
            FeatureEvidence("test-a", "warm sun", 3.0),
            FeatureEvidence("test-b", "rain cloud", 0.1),
            FeatureEvidence("test-c", "the sun", 2.4),
            FeatureEvidence("test-d", "cold rain", 0.0),
        ),
        predictor=_predict,
    )

    assert result.label == "mentions sunlight"
    assert result.training_examples == ("train-a", "train-b")
    assert result.held_out_score.examples == 4
    assert result.held_out_score.correlation > 0.9
    assert result.held_out_score.top_bottom_separation == 1.0
    assert json.loads(json.dumps(result.to_dict()))["schema_version"] == 1


def test_autointerp_rejects_data_leakage_and_nonfinite_predictions():
    interpreter = FeatureAutointerpreter(KeywordProvider())
    shared = FeatureEvidence("train-a", "sun", 1.0)
    with pytest.raises(ValueError, match="disjoint"):
        interpreter.interpret(
            12,
            training=(shared,),
            held_out=(shared, FeatureEvidence("other", "rain", 0.0)),
            predictor=_predict,
        )

    draft = InterpretationDraft("label", "explanation")
    with pytest.raises(ValueError, match="finite"):
        score_explanation(
            draft,
            (FeatureEvidence("a", "a", 1.0), FeatureEvidence("b", "b", 0.0)),
            predictor=lambda _draft, _evidence: float("nan"),
        )


def test_feature_evidence_uses_manifest_splits_without_leakage():
    examples = [DatasetExample(str(index), {}) for index in range(100)]
    manifest = build_dataset_manifest(
        examples,
        name="feature-prompts",
        revision="v1",
        seed=11,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    evidence = tuple(
        FeatureEvidence(item.id, f"text {item.id}", float(index))
        for index, item in enumerate(examples)
    )
    training, held_out = partition_feature_evidence(evidence, manifest)

    assert {item.example_id for item in training} == set(manifest.ids("train"))
    assert {item.example_id for item in held_out} == set(manifest.ids("test"))
    assert not set(item.example_id for item in training).intersection(
        item.example_id for item in held_out
    )

    omitted = manifest.ids("train")[0]
    with pytest.raises(ValueError, match="missing"):
        partition_feature_evidence(
            tuple(item for item in evidence if item.example_id != omitted), manifest
        )
