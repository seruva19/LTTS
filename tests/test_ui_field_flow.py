"""Guard the bug class where the UI reads a field the payload never carries.

Two defects found in review had the same shape. The blueprint dropped
`architectureTopology` and `capabilities`, so every module that declares a
supported topology looked architecture-incompatible. The categorization entries
dropped `moduleRole`, so the compatibility filter had nothing to compare against
and offered every module on every layer.

Both are invisible at runtime: a missing property reads as `undefined`, a guard
written as `if (value && ...)` becomes "no constraint", and a template written
as `v-if="layer.value"` silently renders nothing. Nobody sees an error.

The mechanical invariant that catches the whole class: the UI renders its layer
cards from `categorization`, so any layer property the UI reads must be
reachable there. A property that exists on the flat `layers` array but not on
the categorized entries is exactly the failure both defects were.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel

from scripts.core.ltts_model import LTTSModel
from scripts.models.model_struct import model_to_json, traverse_model

SRC = Path(__file__).resolve().parent.parent / "src"

# `layer.foo` / `props.layer.foo` / `layer?.foo` in any component or store.
LAYER_READ = re.compile(r"(?:props\.)?layer\??\.([a-zA-Z_][a-zA-Z0-9_]*)")

# Reads that are not payload fields: array/string methods, and the keys the
# blueprint store attaches to each layer after the fact.
NOT_PAYLOAD_FIELDS = {
    "preModules", "postModules", "id", "value", "length",
    "map", "filter", "forEach", "find", "push", "some", "every", "reduce",
    "slice", "includes", "sort", "concat", "join", "indexOf",
    "split", "replace", "replaceAll", "startsWith", "endsWith", "trim",
    "toLowerCase", "toUpperCase", "toFixed",
    # Populated only after the layer_inspector module has run.
    "inspector_info",
}


def _ui_layer_reads() -> set[str]:
    reads: set[str] = set()
    for path in SRC.rglob("*"):
        if path.suffix not in {".js", ".vue"} or path.name.endswith(".spec.js"):
            continue
        for match in LAYER_READ.finditer(path.read_text(encoding="utf-8")):
            prop = match.group(1)
            if prop not in NOT_PAYLOAD_FIELDS:
                reads.add(prop)
    return reads


def _payload(model):
    wrapped = LTTSModel(model.eval(), message_bus=None)
    with torch.no_grad():
        struct = traverse_model(wrapped)
    payload = model_to_json(struct)
    return json.loads(payload) if isinstance(payload, str) else payload


def _key_sets(payload):
    flat: set[str] = set()
    for layer in payload.get("layers") or []:
        flat |= set(layer)

    categorized: set[str] = set()
    for category in (payload.get("categorization") or {}).get("categories", {}).values():
        for entry in category.get("layers") or []:
            categorized |= set(entry)

    return flat, categorized


MODELS = {
    "decoder_only": lambda: GPT2LMHeadModel(
        GPT2Config(vocab_size=64, n_embd=32, n_layer=2, n_head=2, n_positions=32)
    ),
    "encoder_only": lambda: BertForMaskedLM(
        BertConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=32,
        )
    ),
}


def test_every_layer_field_the_ui_reads_survives_categorization():
    reads = _ui_layer_reads()
    assert reads, "found no layer property reads in src/ - the scan is broken"

    for topology, build in MODELS.items():
        payload = _payload(build())
        flat, categorized = _key_sets(payload)
        assert flat and categorized, f"{topology}: payload carries no layers"

        stranded = sorted(
            prop for prop in reads if prop in flat and prop not in categorized
        )
        assert not stranded, (
            f"{topology}: the UI reads {stranded} from a layer, and those fields "
            f"exist on the flat `layers` array but not on the categorized entries "
            f"the cards are rendered from. They will read as undefined."
        )


def test_the_compatibility_fields_reach_the_cards():
    """The specific fields the attachment filter compares against.

    Spelled out separately from the scan above so the failure names the
    consequence rather than a property list.
    """
    required = {"moduleRole", "architectureStage"}

    for topology, build in MODELS.items():
        payload = _payload(build())
        _flat, categorized = _key_sets(payload)
        missing = sorted(required - categorized)
        assert not missing, (
            f"{topology}: categorized layers lack {missing}. Without moduleRole "
            f"the UI cannot tell which modules target a layer and offers all of "
            f"them; the backend then rejects the attachment at run time."
        )


def test_the_blueprint_root_describes_the_architecture():
    """Topology and capabilities decide module compatibility in the UI."""
    for topology, build in MODELS.items():
        payload = _payload(build())
        for field in ("architectureTopology", "capabilities"):
            assert payload.get(field), (
                f"{topology}: model payload has no {field}; the UI falls back to "
                f"topology 'unknown' and reports compatible modules as unavailable"
            )
