"""Report fields the UI reads that the API never sends, for real checkpoints.

    python tests/manual_audit_ui_fields.py

A class of defect in this codebase is invisible at runtime: the frontend reads
a property the payload does not carry, gets `undefined`, and a guard written as
`if (value && ...)` quietly becomes "no constraint" while a template written as
`v-if="layer.value"` quietly renders nothing. Nothing errors.

`tests/test_ui_field_flow.py` enforces the machine-checkable part of this on
models built from a config. This tool is the wider sweep: it runs against the
real checkpoints in the registry, which is where architecture-specific fields
show up, and prints three buckets:

  1. blueprint properties no payload carries - mostly saved-blueprint fields and
     settings objects that merely share the name `blueprint`; read the list, do
     not assume every entry is a defect;
  2. layer properties no payload carries - usually fallback chains
     (`layer.layerPath || layer.name`) that resolve elsewhere;
  3. layer properties present in `layers` but MISSING from `categorization` -
     this bucket must stay empty. The cards render from `categorization`, so
     anything here reads as undefined in the UI.

Models that are not in the registry, or cannot be downloaded, are skipped and
named.
"""
import io
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("LTTS_SKIP_MODULE_VENVS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient  # noqa: E402

import ltts  # noqa: E402

MODELS = [
    "HuggingFaceTB/SmolLM-135M",
    "gpt2",
    "google-bert/bert-base-uncased",
    "google-t5/t5-small",
]

# Property reads in the frontend, by the object they are read from.
PATTERNS = {
    "blueprint": re.compile(r"(?:loadedBlueprint|props\.blueprint|blueprint)\??\.([a-zA-Z_][a-zA-Z0-9_]*)"),
    "layer": re.compile(r"(?:props\.layer|layer)\??\.([a-zA-Z_][a-zA-Z0-9_]*)"),
}

# Names that are not payload fields: locals, methods the store adds, DOM.
IGNORE = {
    "blueprint": {
        "value", "layers", "modules", "modelModules", "addPreModule", "addPostModule",
        "addModelModule", "removeModelModule", "run", "stop", "source", "token", "type",
        "name", "id", "map", "filter", "forEach", "find", "length", "push", "some",
        "every", "reduce", "slice", "includes", "then", "catch", "toFixed",
    },
    "layer": {
        "value", "map", "filter", "forEach", "find", "length", "push", "some", "every",
        "reduce", "slice", "includes", "preModules", "postModules", "id", "name",
        "toFixed", "split", "replace", "replaceAll", "startsWith", "endsWith",
        "toLowerCase", "toUpperCase", "trim", "indexOf", "join", "concat", "sort",
    },
}


def frontend_reads():
    found = {"blueprint": set(), "layer": set()}
    src_root = Path(__file__).resolve().parent.parent / "src"
    for root, dirs, files in os.walk(src_root):
        dirs[:] = [d for d in dirs if d != "node_modules"]
        for name in files:
            if not name.endswith((".js", ".vue")) or name.endswith(".spec.js"):
                continue
            text = io.open(os.path.join(root, name), encoding="utf-8").read()
            for kind, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    prop = match.group(1)
                    if prop not in IGNORE[kind]:
                        found[kind].add(prop)
    return found


def payload_keys(client, model_id):
    response = client.post(
        "/api/blueprint_from_model", json={"id": model_id, "engine": "transformers"}
    )
    if response.status_code >= 400:
        return None
    struct = response.json()["model"]
    if isinstance(struct, str):
        struct = json.loads(struct)

    root = set(struct)
    layers = set()
    for layer in struct.get("layers") or []:
        layers |= set(layer)

    categorized = set()
    for category in (struct.get("categorization") or {}).get("categories", {}).values():
        for entry in category.get("layers") or []:
            categorized |= set(entry)

    return {"root": root, "layers": layers, "categorized": categorized}


def main():
    reads = frontend_reads()
    client = TestClient(ltts.app)

    per_model = {}
    for model_id in MODELS:
        keys = payload_keys(client, model_id)
        if keys is None:
            print(f"  (skipped {model_id}: could not build a blueprint)")
            continue
        per_model[model_id] = keys

    if not per_model:
        print("no model could be loaded - nothing to audit")
        return 1

    root_union = set().union(*(k["root"] for k in per_model.values()))
    layer_union = set().union(*(k["layers"] for k in per_model.values()))
    cat_union = set().union(*(k["categorized"] for k in per_model.values()))

    print(f"models audited: {', '.join(per_model)}")
    print()

    missing_bp = sorted(p for p in reads["blueprint"] if p not in root_union)
    print(f"BLUEPRINT properties the UI reads that no model payload carries ({len(missing_bp)}):")
    for prop in missing_bp:
        print("   ", prop)

    print()
    # Cards render from categorized entries, so a layer property must exist there.
    missing_layer = sorted(
        p for p in reads["layer"] if p not in cat_union and p not in layer_union
    )
    print(f"LAYER properties the UI reads that no payload carries ({len(missing_layer)}):")
    for prop in missing_layer:
        print("   ", prop)

    print()
    only_flat = sorted(p for p in reads["layer"] if p in layer_union and p not in cat_union)
    print(f"LAYER properties present in `layers` but MISSING from `categorization` ({len(only_flat)}):")
    print("  (cards are rendered from categorization, so these read as undefined)")
    for prop in only_flat:
        print("   ", prop)
    return 0


if __name__ == "__main__":
    sys.exit(main())
