from __future__ import annotations

import json

import pytest

from ltts_engine import (
    DatasetExample,
    build_dataset_manifest,
    bundle_result,
    dataset_split_for_id,
    examples_for_split,
)


def _examples(count=100):
    return [DatasetExample(id=f"item-{index}", inputs={"value": index}) for index in range(count)]


def test_dataset_manifest_is_order_independent_and_bundle_ready():
    examples = _examples()
    manifest = build_dataset_manifest(
        examples,
        name="interpretability-prompts",
        revision="v1",
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        metadata={"license": "test-only"},
    )
    reordered = build_dataset_manifest(
        reversed(examples),
        name="interpretability-prompts",
        revision="v1",
        seed=17,
        validation_fraction=0.2,
        test_fraction=0.2,
        metadata={"license": "test-only"},
    )

    assert manifest.fingerprint == reordered.fingerprint
    assert manifest.entries == reordered.entries
    assert set(entry.split for entry in manifest.entries) == {
        "train",
        "validation",
        "test",
    }
    assert manifest.split_for("item-7") == dataset_split_for_id(
        "item-7", seed=17, validation_fraction=0.2, test_fraction=0.2
    )
    payload = bundle_result("dataset_manifest", manifest).to_dict()
    assert json.loads(json.dumps(payload))["payload"]["fingerprint"].startswith(
        "sha256:"
    )


def test_manifest_split_filter_rejects_dataset_drift():
    examples = _examples(30)
    manifest = build_dataset_manifest(
        examples, name="prompts", revision="commit-abc", seed=3
    )
    train = list(examples_for_split(reversed(examples), manifest, "train"))
    assert {item.id for item in train} == set(manifest.ids("train"))

    with pytest.raises(ValueError, match="absent"):
        list(
            examples_for_split(
                [DatasetExample(id="new-item", inputs={})], manifest, "test"
            )
        )

    selected_id = manifest.ids("train")[0]
    with pytest.raises(ValueError, match="missing"):
        list(
            examples_for_split(
                [item for item in examples if item.id != selected_id],
                manifest,
                "train",
            )
        )

    selected = next(item for item in examples if item.id == selected_id)
    with pytest.raises(ValueError, match="Duplicate"):
        list(
            examples_for_split(
                [*examples, selected], manifest, "train", require_complete=False
            )
        )


def test_manifest_rejects_duplicate_ids_and_invalid_fractions():
    duplicate = [
        DatasetExample(id="same", inputs={}),
        DatasetExample(id="same", inputs={}),
    ]
    with pytest.raises(ValueError, match="Duplicate"):
        build_dataset_manifest(duplicate, name="data", revision="v1")
    with pytest.raises(ValueError, match="sum below 1"):
        dataset_split_for_id("item", seed=0, validation_fraction=0.5, test_fraction=0.5)
