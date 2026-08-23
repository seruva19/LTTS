from __future__ import annotations

import json

import pytest

from ltts_engine import (
    AttributionGraphBuilder,
    AttributionNode,
    bundle_result,
    load_research_bundle,
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


def test_research_bundle_rejects_tampering_and_nonfinite_values(tmp_path):
    path = save_research_bundle(bundle_result("graph", _graph()), tmp_path / "run.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["payload"]["metadata"]["prompt"] = "changed"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_research_bundle(path)

    with pytest.raises(ValueError, match="finite"):
        bundle_result("invalid", {"value": float("nan")})
