import json

import pytest

from ltts_engine import (
    ResearchStudyBuilder,
    StudyArtifactCollection,
    StudyArtifactRecord,
    archive_research_study,
    bundle_study_archive,
    compare_study_artifacts,
    load_research_bundle,
    save_research_bundle,
    study_archive_from_bundle,
)


def _record(
    artifact_id,
    value,
    *,
    metric="effect",
    layer="layers.0",
    seed=0,
    example="example-a",
    module="activation_anisotropy",
):
    return StudyArtifactRecord(
        artifact_id=artifact_id,
        metric=metric,
        value=value,
        module=module,
        subtype="scalar",
        layer_path=layer,
        seed=seed,
        example_id=example,
        stage="scan",
        run_id="run-1",
        metadata={"source": "module_artifact"},
    )


def test_artifact_contract_roundtrips_and_rejects_ambiguous_values():
    record = _record("artifact-1", 0.25, seed=7)
    restored = StudyArtifactRecord.from_dict(record.to_dict())

    assert restored == record
    assert restored.comparison_key == record.comparison_key
    assert restored.trial_key == ("example-a", 7)
    assert json.loads(json.dumps(record.to_dict()))["value"] == 0.25

    with pytest.raises(ValueError, match="finite"):
        _record("nan", float("nan"))
    with pytest.raises(ValueError, match="uint32"):
        _record("bad-seed", 1.0, seed=-1)
    with pytest.raises(ValueError, match="unique"):
        StudyArtifactCollection("duplicate", (record, record))


def test_complete_study_archive_uses_existing_integrity_checked_bundle(tmp_path):
    study = (
        ResearchStudyBuilder("portable-study", metadata={"model": "debug/dummy"})
        .add_stage("dataset", "dataset_manifest", {"fingerprint": "sha256:data"})
        .add_stage(
            "scan",
            "module_scan",
            {"candidate_count": 1},
            depends_on=("dataset",),
        )
        .build()
    )
    artifacts = StudyArtifactCollection(
        "portable-study",
        (_record("artifact-1", 0.25),),
        manifest_fingerprint="sha256:data",
    )
    archive = archive_research_study(
        study,
        artifacts,
        manifest={"schema_version": 1, "fingerprint": "sha256:data"},
        trials=({"example_id": "example-a", "seed": 0, "score": 0.25},),
        graphs=({"schema_version": 1, "nodes": [], "edges": []},),
        metadata={"model_revision": "revision-a"},
    )

    destination = save_research_bundle(
        bundle_study_archive(archive), tmp_path / "study.json"
    )
    restored = study_archive_from_bundle(load_research_bundle(destination))

    assert restored.study["name"] == "portable-study"
    assert restored.manifest["fingerprint"] == "sha256:data"
    assert restored.trials[0]["seed"] == 0
    assert restored.graphs[0]["schema_version"] == 1
    assert restored.artifacts.records[0].artifact_id == "artifact-1"
    assert restored.metadata["model_revision"] == "revision-a"


def test_study_comparison_reports_effects_presence_and_paired_seed_stability():
    baseline = StudyArtifactCollection(
        "baseline",
        (
            _record("b0", 1.0, seed=0),
            _record("b1", 3.0, seed=1),
            _record("gone", 5.0, metric="removed", seed=0),
        ),
        manifest_fingerprint="sha256:same",
    )
    candidate = StudyArtifactCollection(
        "candidate",
        (
            _record("c0", 2.0, seed=0),
            _record("c1", 6.0, seed=1),
            _record("new", 9.0, metric="added", seed=0),
        ),
        manifest_fingerprint="sha256:same",
    )

    comparison = compare_study_artifacts(baseline, candidate)
    points = {point.metric: point for point in comparison.points}

    assert comparison.manifest_match is True
    assert points["effect"].status == "matched"
    assert points["effect"].baseline_mean == 2.0
    assert points["effect"].candidate_mean == 4.0
    assert points["effect"].delta == 2.0
    assert points["effect"].matched_trials == 2
    assert points["effect"].paired_delta_mean == 2.0
    assert points["effect"].paired_delta_std == 1.0
    assert points["added"].status == "appeared"
    assert points["removed"].status == "disappeared"
    assert json.loads(json.dumps(comparison.to_dict()))["schema_version"] == 1


def test_comparison_does_not_pair_different_examples_or_unknown_manifests():
    baseline = StudyArtifactCollection(
        "left", (_record("left", 1.0, example="left-example"),)
    )
    candidate = StudyArtifactCollection(
        "right", (_record("right", 2.0, example="right-example"),)
    )

    point = compare_study_artifacts(baseline, candidate).points[0]

    assert point.delta == 1.0
    assert point.matched_trials == 0
    assert point.paired_delta_mean is None
    assert compare_study_artifacts(baseline, candidate).manifest_match is None

    no_provenance_left = StudyArtifactCollection(
        "unknown-left",
        (
            StudyArtifactRecord(
                "unknown-left", "effect", 1.0, "probe", "scalar"
            ),
        ),
    )
    no_provenance_right = StudyArtifactCollection(
        "unknown-right",
        (
            StudyArtifactRecord(
                "unknown-right", "effect", 2.0, "probe", "scalar"
            ),
        ),
    )
    assert (
        compare_study_artifacts(no_provenance_left, no_provenance_right)
        .points[0]
        .matched_trials
        == 0
    )
