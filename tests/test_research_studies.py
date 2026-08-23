from __future__ import annotations

import pytest

from ltts_engine import (
    AttributionGraphBuilder,
    AttributionNode,
    DatasetExample,
    ResearchStudyBuilder,
    ResearchStudyResult,
    ResearchStudyPlanStep,
    ResearchStudyRunner,
    ResearchStudyStage,
    build_dataset_manifest,
    bundle_result,
    load_research_bundle,
    save_research_bundle,
)


def test_research_study_composes_manifest_measurement_and_graph(tmp_path):
    manifest = build_dataset_manifest(
        [DatasetExample(str(index), {}) for index in range(10)],
        name="prompts",
        revision="v1",
        seed=4,
    )
    measurement = {"feature_id": 7, "mean_activation": 0.4}
    graph = (
        AttributionGraphBuilder()
        .add_node(AttributionNode("target", "metric", "feature activation"))
        .build()
    )
    study = (
        ResearchStudyBuilder(
            "feature-7-study", metadata={"model": "tiny", "revision": "abc"}
        )
        .add_stage("dataset", "dataset_manifest", manifest)
        .add_stage(
            "scan", "feature_scan", measurement, depends_on=("dataset",)
        )
        .add_stage(
            "graph", "attribution_graph", graph, depends_on=("scan",)
        )
        .build()
    )

    assert study.stage("graph").depends_on == ("scan",)
    assert study.stage("dataset").result["fingerprint"] == manifest.fingerprint
    path = save_research_bundle(
        bundle_result("research_study", study), tmp_path / "study.json"
    )
    restored = load_research_bundle(path)
    assert restored.payload["stages"][2]["name"] == "graph"
    assert restored.payload["metadata"]["revision"] == "abc"


def test_study_builder_rejects_forward_dependencies_and_tensor_results():
    builder = ResearchStudyBuilder("study")
    with pytest.raises(ValueError, match="after their dependencies"):
        builder.add_stage("graph", "graph", {}, depends_on=("scan",))

    with pytest.raises(TypeError, match="mapping"):
        builder.add_stage("tensor", "invalid", object())


def test_study_result_rejects_dependency_cycles():
    left = ResearchStudyStage("left", "test", {}, depends_on=("right",))
    right = ResearchStudyStage("right", "test", {}, depends_on=("left",))
    with pytest.raises(ValueError, match="cycle"):
        ResearchStudyResult("cycle", (left, right))


def test_study_runner_executes_dependencies_and_keeps_serializable_results():
    calls = []
    study = ResearchStudyRunner().run(
        "executed-study",
        (
            ResearchStudyPlanStep(
                "graph",
                "graph",
                lambda dependencies: {
                    "score": dependencies["scan"]["mean"] * 2
                },
                depends_on=("scan",),
            ),
            ResearchStudyPlanStep(
                "scan",
                "scan",
                lambda _dependencies: calls.append("scan") or {"mean": 0.5},
            ),
        ),
    )

    assert calls == ["scan"]
    assert [stage.name for stage in study.stages] == ["scan", "graph"]
    assert study.stage("graph").result == {"score": 1.0}


def test_study_runner_rejects_plan_cycles_before_execution():
    steps = (
        ResearchStudyPlanStep("left", "test", lambda _deps: {}, ("right",)),
        ResearchStudyPlanStep("right", "test", lambda _deps: {}, ("left",)),
    )
    with pytest.raises(ValueError, match="cycle"):
        ResearchStudyRunner().run("cycle", steps)
