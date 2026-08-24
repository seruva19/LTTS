from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from ltts_engine import (
    AttributionGraphBuilder,
    AttributionNode,
    bundle_result,
    load_research_bundle,
    research_bundle_from_dict,
    save_research_bundle,
)


def _graph():
    return (
        AttributionGraphBuilder(metadata={"prompt": "The sun"})
        .add_node(AttributionNode(id="target", kind="metric", label="score"))
        .build()
    )


def test_research_bundle_roundtrip_is_versioned_and_integrity_checked(tmp_path):
    bundle = bundle_result(
        "attribution_graph",
        _graph(),
        metadata={"model": "tiny-test", "revision": "abc123"},
    )
    path = save_research_bundle(bundle, tmp_path / "nested" / "run.json")
    restored = load_research_bundle(path)

    assert restored.kind == "attribution_graph"
    assert restored.payload == _graph().to_dict()
    assert restored.metadata["revision"] == "abc123"
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["checksum"].startswith("sha256:")
    assert not list(path.parent.glob("*.tmp"))
    assert research_bundle_from_dict(document).payload == restored.payload


def test_research_bundle_rejects_tampering_and_nonfinite_values(tmp_path):
    path = save_research_bundle(bundle_result("graph", _graph()), tmp_path / "run.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["metadata"]["prompt"] = "changed"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_research_bundle(path)

    with pytest.raises(ValueError, match="finite"):
        bundle_result("invalid", {"value": float("nan")})


def test_bundle_checksum_survives_equivalent_json_number_serialization():
    document = bundle_result("metrics", {"whole": 1.0, "fraction": 1.25}).to_dict()
    document["payload"]["whole"] = 1

    assert research_bundle_from_dict(document).payload["whole"] == 1

    stable = bundle_result("metrics", {"whole": 1.0})
    stable_document = stable.to_dict()
    instant = datetime.fromisoformat(stable_document["created_at"])
    stable_document["created_at"] = instant.astimezone(
        timezone(timedelta(hours=3))
    ).isoformat()
    assert research_bundle_from_dict(stable_document).kind == "metrics"


def test_bundle_loader_accepts_pre_normalization_checksums():
    document = bundle_result("metrics", {"whole": 1.0}).to_dict()
    content = {key: value for key, value in document.items() if key != "checksum"}
    legacy = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    document["checksum"] = "sha256:" + hashlib.sha256(legacy).hexdigest()

    assert research_bundle_from_dict(document).payload["whole"] == 1.0
