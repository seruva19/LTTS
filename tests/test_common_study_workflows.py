from __future__ import annotations

from types import ModuleType

import torch
import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from ltts_engine import (
    CausalStudyRunner,
    DatasetExample,
    FeatureEvidence,
    FeatureStudyRunner,
    HeadlessEngine,
    InterpretationDraft,
    build_dataset_manifest,
)


def _model():
    torch.manual_seed(61)
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_embd=16,
            n_layer=2,
            n_head=2,
            n_positions=16,
        )
    ).eval()


def _intervention():
    module = ModuleType("causal_study_intervention")
    module.METADATA = {
        "name": "causal_study_intervention",
        "event_types": ["layer_after"],
    }

    def interceptor(event):
        output = event.context.outputs
        hidden = output[0] if isinstance(output, tuple) else output
        changed = hidden.clone()
        changed[..., 0] += float(event.module_state.get("amount", 1.0))
        event.context.analysis_results["modified_outputs"] = (
            (changed, *output[1:]) if isinstance(output, tuple) else changed
        )
        return {"status": "ok"}

    module.interceptor = interceptor
    return module


class SunProvider:
    def explain(self, feature_id, evidence):
        assert feature_id == 0
        assert evidence
        return InterpretationDraft(
            label="sun-related text",
            explanation="The feature responds to examples mentioning the sun.",
        )


def test_causal_study_runner_executes_sweep_and_graph_without_web_app():
    model = _model()
    inputs = {"input_ids": torch.tensor([[1, 2, 3]])}
    with HeadlessEngine().instrument(model) as session:
        study = CausalStudyRunner(session).run(
            "causal-layer-study",
            module=_intervention(),
            layers=session.layers("block"),
            scorer=lambda output: float(output.logits[0, -1, 0].detach()),
            target_label="token 0 logit",
            config={"amount": 2.0},
            kwargs=inputs,
            seed=9,
            metadata={"model": "tiny-gpt2"},
        )

    assert [stage.name for stage in study.stages] == [
        "causal_sweep",
        "attribution_graph",
    ]
    graph = study.stage("attribution_graph").result
    assert len(graph["edges"]) == 2
    assert all(edge["method"] == "causal_intervention" for edge in graph["edges"])


def test_feature_study_runner_uses_manifest_train_and_test_splits():
    model = _model()
    examples = [
        DatasetExample(
            id=f"example-{index}",
            inputs={"input_ids": torch.tensor([[1, 2, 3 + index % 10]])},
        )
        for index in range(30)
    ]
    manifest = build_dataset_manifest(
        examples,
        name="feature-prompts",
        revision="v1",
        seed=5,
        validation_fraction=0.2,
        test_fraction=0.2,
    )
    evidence = tuple(
        FeatureEvidence(
            example.id,
            "sun" if index % 2 == 0 else "rain",
            1.0 if index % 2 == 0 else 0.0,
        )
        for index, example in enumerate(examples)
    )

    with HeadlessEngine().instrument(model) as session:
        study = FeatureStudyRunner(session).run(
            "feature-study",
            examples=examples,
            manifest=manifest,
            layer=session.layers("block")[0],
            feature_extractor=lambda hidden: hidden,
            feature_id=0,
            evidence=evidence,
            provider=SunProvider(),
            predictor=lambda _draft, item: float("sun" in item.text),
            top_k=3,
        )

    scan = study.stage("feature_scan")
    interpretation = study.stage("feature_interpretation")
    assert scan.metadata["split"] == "train"
    assert scan.result["examples_seen"] == len(manifest.ids("train"))
    assert interpretation.metadata["held_out_split"] == "test"
    assert interpretation.result["held_out_score"]["examples"] == len(
        manifest.ids("test")
    )
    assert interpretation.result["held_out_score"]["correlation"] == pytest.approx(
        1.0
    )
