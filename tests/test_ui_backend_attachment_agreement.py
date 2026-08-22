"""The UI must offer exactly the attachments the backend accepts.

The backend decides in `build_compatibility_report`; the frontend decides
independently in `filterOptionsForLayer` (src/components/blocks/blueprint/
LayerCard.vue). Two implementations of one rule drift, and the drift is
invisible: offering too much means the user picks a module and the run dies
with a 422, offering too little means a usable module is missing from the list
with no explanation.

That is what the `moduleRole` defect did. The categorized layer entries carried
no role, so the UI's role check had nothing to compare against and offered 25
modules on a self-attention layer where the backend accepts 12.

This walks the whole matrix — every installed module against every layer of a
real model — and asserts the two verdicts agree.

The UI rule below is a transcription. Keep it in step with
`filterOptionsForLayer`; a divergence here is exactly the thing being tested.
"""

from __future__ import annotations

import json

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, GPT2Config, GPT2LMHeadModel

from scripts.core.ltts_model import LTTSModel
from scripts.models.model_struct import model_to_json, traverse_model
from scripts.router import context
from scripts.services.model_compatibility import (
    build_compatibility_report,
    module_attachment_contract,
)


def _ui_accepts(entry, contract, topology, capabilities):
    """Transcription of filterOptionsForLayer() in LayerCard.vue."""
    role = entry.get("moduleRole")
    target_roles = contract.get("target_roles") or []
    supported_topologies = contract.get("supported_topologies") or []
    required_capabilities = contract.get("required_capabilities") or []
    optional_dependencies = contract.get("optional_dependencies") or []

    role_compatible = (not role) or (not target_roles) or (role in target_roles)
    topology_compatible = (
        not supported_topologies or topology in supported_topologies
    )
    # Fail closed, matching LayerCard.vue: an unreported capability is absent.
    capabilities_compatible = all(
        capabilities.get(capability) is True
        for capability in required_capabilities
    )
    dependencies_available = all(
        dependency.get("available") is not False
        for dependency in optional_dependencies
    )
    return (
        role_compatible
        and topology_compatible
        and capabilities_compatible
        and dependencies_available
    )


def _model_payload(model):
    wrapped = LTTSModel(model.eval(), message_bus=None)
    with torch.no_grad():
        struct = traverse_model(wrapped)
    payload = model_to_json(struct)
    payload = json.loads(payload) if isinstance(payload, str) else payload
    return wrapped, payload


def _categorized_entries(payload):
    entries = []
    for category in (payload.get("categorization") or {}).get("categories", {}).values():
        entries.extend(category.get("layers") or [])
    return entries


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


@pytest.mark.parametrize("topology_name", sorted(MODELS))
def test_ui_and_backend_agree_on_every_module_and_layer(topology_name):
    wrapped, payload = _model_payload(MODELS[topology_name]())
    entries = _categorized_entries(payload)
    assert entries, f"{topology_name}: model produced no categorized layers"

    # What the UI has: the blueprint's architecture description.
    topology = payload.get("architectureTopology") or "unknown"
    capabilities = payload.get("capabilities") or {}

    modules = context.module_loader.get_all_modules()
    assert modules, "no modules installed"

    # One layer per distinct role is enough coverage and keeps the matrix small.
    by_role = {}
    for entry in entries:
        role = entry.get("moduleRole")
        if role and role not in by_role:
            by_role[role] = entry
    assert by_role, f"{topology_name}: no categorized layer carries a role"

    disagreements = []
    for module_name in sorted(modules):
        metadata = (modules.get(module_name) or {}).get("metadata") or {}
        if module_attachment_contract(module_name, metadata)["model_level"]:
            # Offered by the model-level panel, never by a layer card.
            continue
        for role, entry in sorted(by_role.items()):
            layer_name = entry["name"]
            report = build_compatibility_report(
                context, wrapped, topology_name, {layer_name: [module_name]}
            )
            module_reports = report.get("modules") or []
            if not module_reports:
                continue
            backend_ok = bool(module_reports[0].get("compatible"))

            contract = {
                "target_roles": module_reports[0].get("target_roles") or [],
                "supported_topologies": module_reports[0].get("supported_topologies") or [],
                "required_capabilities": module_reports[0].get("required_capabilities") or [],
                "optional_dependencies": module_reports[0].get("optional_dependencies") or [],
            }
            ui_ok = _ui_accepts(entry, contract, topology, capabilities)

            if ui_ok != backend_ok:
                disagreements.append(
                    f"{module_name} on {role} ({layer_name}): "
                    f"UI={'offers' if ui_ok else 'hides'} "
                    f"backend={'accepts' if backend_ok else 'rejects'}"
                )

    assert not disagreements, (
        f"{topology_name}: UI and backend disagree on "
        f"{len(disagreements)} module/layer pairs:\n  "
        + "\n  ".join(disagreements[:20])
    )
