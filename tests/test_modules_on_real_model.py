"""Run modules against config-built transformers without downloads.

Assertions cover tensor shapes, hooks, dispatch, and UI message delivery.
Random model outputs are not evaluated.
"""

from __future__ import annotations

import pytest
import torch
from transformers import (
    BertConfig,
    BertForMaskedLM,
    GPT2Config,
    GPT2LMHeadModel,
)

from scripts.core.ltts_model import LTTSModel
from scripts.messages.message_bus import MessageBus
from scripts.router import context
from scripts.services.inference.attachments import ModuleAttachmentManager

VALID_DATA_TYPES = {"scalar", "chart", "table", "heatmap", "text", "image"}


def _resolve(message):
    """Module name and payload, however the envelope nests them.

    Two shapes reach the socket: ModuleDataSender publishes a flat envelope,
    while a reactor signal arrives wrapped one level deeper. The UI resolves
    both (src/state/socket-store.js), so the contract is that the name and the
    payload are *reachable*, not that they sit at a fixed depth.
    """
    nested = message.get("data") if isinstance(message.get("data"), dict) else {}
    name = message.get("module") or nested.get("module")
    payload = nested.get("data") if isinstance(nested.get("data"), dict) else nested
    subtype = message.get("subtype") or nested.get("subtype")
    return name, subtype, payload


def _tiny_gpt2():
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=64,
            n_embd=32,
            n_layer=3,
            n_head=2,
            n_positions=32,
            attn_implementation="eager",
        )
    )
    return model.eval()


def _tiny_bert():
    model = BertForMaskedLM(
        BertConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=32,
            attn_implementation="eager",
        )
    )
    return model.eval()


def _paths_by_role(wrapped, role):
    descriptors = wrapped.module_descriptors
    return sorted(
        (path for path, d in descriptors.items() if d.get("role") == role),
        key=lambda path: descriptors[path].get("layer_index") or 0,
    )


def _run_with_modules(model, attachments_for):
    """Attach modules, run one forward, return everything published."""
    wrapped = LTTSModel(model, message_bus=context.message_bus)
    # Modules read the token strings when they have them; inference normally
    # sets this, and there is no tokenizer here.
    wrapped._last_input_tokens = ["a", "b", "c", "d"]

    attachments = attachments_for(wrapped)
    if attachments is None:
        pytest.skip("model exposes no layer of the required role")

    captured = []
    original = MessageBus.publish_from_thread
    MessageBus.publish_from_thread = (
        lambda self, token, message: captured.append(message)
    )
    try:
        manager = ModuleAttachmentManager(context, wrapped)
        manager.reset_attachments()
        specs, _capabilities = manager.attach_modules(attachments)
        assert specs, "no module attached - the layer paths did not resolve"

        with torch.no_grad():
            wrapped(input_ids=torch.tensor([[1, 2, 3, 4]]), output_attentions=True)
    finally:
        MessageBus.publish_from_thread = original
        try:
            wrapped.module_registry.detach_all_modules()
        except Exception:  # pragma: no cover - cleanup must not mask a failure
            pass

    return [m for m in captured if isinstance(m, dict) and m.get("type") == "visualization"]


def _emitting_modules(messages):
    names = {}
    for message in messages:
        name, _subtype, _payload = _resolve(message)
        names[name] = names.get(name, 0) + 1
    return names


def test_block_modules_emit_on_a_real_decoder():
    """Modules reading the residual stream produce output from a real forward."""
    expected = {
        "layer_statistics",
        "residual_stream_norm",
        "massive_activations",
        "layer_redundancy",
    }

    def attachments_for(wrapped):
        blocks = _paths_by_role(wrapped, "block")
        if not blocks:
            return None
        return {
            blocks[0]: [
                {"name": name, "parameters": {"emit_mode": "all"}}
                for name in sorted(expected)
            ]
        }

    messages = _run_with_modules(_tiny_gpt2(), attachments_for)
    emitted = _emitting_modules(messages)

    installed = set(context.module_loader.get_all_modules())
    for name in sorted(expected & installed):
        assert emitted.get(name), f"{name} produced nothing on a real forward pass"


def test_attention_module_emits_on_a_real_decoder():
    """Attention capture reaches a module through the real hook path."""

    def attachments_for(wrapped):
        attention = _paths_by_role(wrapped, "self_attention")
        if not attention:
            return None
        return {
            attention[0]: [
                {"name": "attn_visualizer", "parameters": {"emit_mode": "all"}}
            ]
        }

    if "attn_visualizer" not in context.module_loader.get_all_modules():
        pytest.skip("attn_visualizer is not installed")

    messages = _run_with_modules(_tiny_gpt2(), attachments_for)
    assert _emitting_modules(messages).get("attn_visualizer"), (
        "attn_visualizer produced nothing - attention weights did not reach it"
    )


def test_block_modules_emit_on_a_real_encoder():
    """The same modules work on an encoder-only model, not just decoders."""
    expected = {"layer_statistics", "residual_stream_norm"}

    def attachments_for(wrapped):
        blocks = _paths_by_role(wrapped, "block")
        if not blocks:
            return None
        return {
            blocks[0]: [
                {"name": name, "parameters": {"emit_mode": "all"}}
                for name in sorted(expected)
            ]
        }

    messages = _run_with_modules(_tiny_bert(), attachments_for)
    emitted = _emitting_modules(messages)

    installed = set(context.module_loader.get_all_modules())
    for name in sorted(expected & installed):
        assert emitted.get(name), f"{name} produced nothing on an encoder model"


def test_payloads_from_a_real_run_satisfy_the_contract():
    """Everything published during a real run is renderable by the UI."""

    def attachments_for(wrapped):
        blocks = _paths_by_role(wrapped, "block")
        if not blocks:
            return None
        return {
            blocks[0]: [
                {"name": "layer_statistics", "parameters": {"emit_mode": "all"}},
                {"name": "massive_activations", "parameters": {"emit_mode": "all"}},
            ]
        }

    messages = _run_with_modules(_tiny_gpt2(), attachments_for)
    assert messages, "a real forward pass published nothing at all"

    for message in messages:
        name, subtype, payload = _resolve(message)
        assert name, f"published payload names no module: {message!r}"
        assert subtype, f"{name} published a payload with no subtype: {message!r}"
        assert isinstance(payload, dict), f"{name} published a non-dict payload"

        data_type = payload.get("type")
        if data_type is None:
            continue
        assert data_type in VALID_DATA_TYPES, (
            f"{name} published unknown data type {data_type!r}"
        )
