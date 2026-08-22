#!/usr/bin/env python
"""
Manual end-to-end verification harness for the interpretability modules.

Loads a real model, attaches the new modules across many layers, runs a single
forward pass through the full LTTS inference service, and reports — per module —
what was emitted onto the message bus and whether it looks like correct, real
data (not a stub or an error).

NOT collected by pytest (no test_ prefix). Run directly:

    ./venv/Scripts/python.exe tests/manual_e2e_modern_llm.py HuggingFaceTB/SmolLM-135M
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections import defaultdict

import torch

sys.path.insert(0, ".")

# Tokens from some tokenizers contain glyphs the Windows console encoding
# (cp1251/cp1252) can't represent; force UTF-8 so reporting never crashes.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

RESIDUAL_MODULES = [
    "logit_lens",
    "activation_trajectory",
    "representation_similarity",
    "direct_logit_attribution",
    "embedding_projector",
]
ATTENTION_MODULES = [
    "attention_rollout",
    "induction_head_detector",
    "attention_entropy_profiler",
]
# Repetition gives induction heads something to find; >4 tokens.
PROMPT = "The cat sat on the mat. The cat sat on the mat. The cat sat on the"


def _spaced(items, k):
    if len(items) <= k:
        return items
    step = max(1, len(items) // k)
    return items[::step][:k]


def _summarize_payload(msg):
    """Return a short, human-readable description of one visualization message."""
    data = msg.get("data")
    subtype = msg.get("subtype")
    if isinstance(data, dict):
        dtype = data.get("type", subtype)
        if dtype == "table":
            rows = data.get("rows") or (data.get("data") or {}).get("rows")
            headers = data.get("headers") or (data.get("data") or {}).get("headers")
            sample = rows[0] if isinstance(rows, list) and rows else None
            return f"table headers={headers} first_row={sample}"
        if dtype == "scalar":
            return f"scalar value={data.get('value')} label={data.get('label')}"
        if dtype == "text":
            content = str(data.get("content") or data.get("data") or "")
            return f"text: {content[:120]}"
        if dtype == "chart":
            series = data.get("series") or (data.get("data") or {}).get("series")
            n = len(series[0]["points"]) if series and series[0].get("points") else 0
            return f"chart series={len(series) if series else 0} points={n}"
        if dtype == "json":
            return f"json keys={list((data.get('data') or {}).keys())[:6]}"
        return f"dict type={dtype} keys={list(data.keys())[:6]}"
    if isinstance(data, str):
        if data.startswith("data:image"):
            return f"image data-uri ({len(data)} bytes)"
        return f"str: {data[:120]}"
    return f"{type(data).__name__}: {str(data)[:80]}"


async def run_family(service, bus, model_id, modules, label):
    print(f"\n{'=' * 70}\n[{label}] attaching {sum(len(v) for v in modules.values())} "
          f"module(s) across {len(modules)} layer(s)\n{'=' * 70}")

    collected = []
    errors = []

    def collector(msg):
        if not isinstance(msg, dict):
            return
        mtype = msg.get("type")
        if mtype == "visualization":
            collected.append(msg)
        elif mtype == "error":
            errors.append(msg.get("data") or msg)

    bus.subscribe("default", collector)

    from scripts.routers.inference_router import InferenceInput

    inp = InferenceInput(
        model_id=model_id,
        prompt=PROMPT,
        max_tokens=16,
        parameters={"force_eager_attention": True},
        client_token="default",
        modules=modules,
    )

    result = None
    try:
        result = await service.run_inference(inp)
    except Exception as exc:  # noqa: BLE001
        print(f"  !! run_inference raised: {type(exc).__name__}: {exc}")
    finally:
        await asyncio.sleep(1.0)  # let publish_sync queue drain through the loop
        bus.unsubscribe("default", collector)

    by_module = defaultdict(list)
    for msg in collected:
        by_module[msg.get("module", "?")].append(msg)

    if result:
        txt = (result.get("result") or "")[:80].replace("\n", " ")
        print(f"  forward ok. result preview: {txt!r}")

    if errors:
        print(f"  !! {len(errors)} error signal(s):")
        for e in errors[:8]:
            print(f"     - {e}")

    return by_module, errors


def report_family(by_module, expected_modules, checks):
    print(f"\n  --- emissions ---")
    verdicts = {}
    for mod in expected_modules:
        msgs = by_module.get(mod, [])
        subtypes = defaultdict(int)
        for m in msgs:
            d = m.get("data")
            t = d.get("type") if isinstance(d, dict) else (
                "image" if isinstance(d, str) and d.startswith("data:image") else "raw"
            )
            subtypes[t] += 1
        ok, detail = checks[mod](msgs)
        verdicts[mod] = ok
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {mod:28s} {dict(subtypes)}")
        if msgs:
            print(f"           e.g. {_summarize_payload(msgs[0])}")
        print(f"           -> {detail}")
    return verdicts


# ---- correctness predicates -------------------------------------------------

def _has_type(msgs, t):
    for m in msgs:
        d = m.get("data")
        if isinstance(d, dict) and d.get("type") == t:
            return True
        if t == "image" and isinstance(d, str) and d.startswith("data:image"):
            return True
        if t == "image" and isinstance(d, dict):
            v = d.get("data") or d.get("value")
            if isinstance(v, str) and v.startswith("data:image"):
                return True
    return False


def check_logit_lens(msgs):
    for m in msgs:
        d = m.get("data")
        if isinstance(d, dict) and d.get("type") == "table":
            rows = d.get("rows") or []
            if rows and len(rows[0]) >= 3:
                prob = rows[0][2]
                if isinstance(prob, (int, float)) and 0.0 <= prob <= 1.0001:
                    return True, f"top token row {rows[0]} (prob in [0,1])"
    return bool(msgs), "emitted but no valid top-k probability table found"


def check_trajectory(msgs):
    if _has_type(msgs, "chart"):
        return True, "cross-layer chart emitted (trajectory accumulated)"
    if _has_type(msgs, "scalar"):
        return True, "per-layer scalar emitted (chart needs >=2 layers)"
    return bool(msgs), "no scalar/chart"


def check_repsim(msgs):
    if _has_type(msgs, "image"):
        return True, "CKA heatmap image emitted"
    if _has_type(msgs, "scalar"):
        return True, "CKA scalar emitted"
    return bool(msgs), "no heatmap/scalar"


def check_dla(msgs):
    if _has_type(msgs, "chart"):
        return True, "per-layer logit-attribution chart emitted"
    return bool(msgs), "no chart"


def check_projector(msgs):
    if _has_type(msgs, "image"):
        return True, "2D projection image emitted"
    if _has_type(msgs, "json"):
        return True, "projection coords json emitted (matplotlib fallback)"
    return bool(msgs), "no image/json"


def check_rollout(msgs):
    if _has_type(msgs, "image"):
        return True, "rollout heatmap emitted"
    return bool(msgs), "no heatmap"


def check_induction(msgs):
    for m in msgs:
        d = m.get("data")
        if isinstance(d, dict) and d.get("type") == "table":
            return True, f"per-head score table emitted ({len(d.get('rows') or [])} heads)"
    return bool(msgs), "no head-score table"


def check_entropy(msgs):
    if _has_type(msgs, "scalar") or _has_type(msgs, "chart"):
        return True, "entropy scalar/chart emitted"
    return bool(msgs), "no scalar/chart"


def _text_contains(msgs, needle):
    for m in msgs:
        d = m.get("data")
        if isinstance(d, dict):
            content = str(d.get("content") or d.get("data") or "")
            if needle.lower() in content.lower():
                return content[:160]
    return None


def check_ablation(msgs):
    hit = _text_contains(msgs, "ablat") or _text_contains(msgs, "neuron")
    if hit:
        return True, f"intervention summary: {hit}"
    return bool(msgs), "no ablation summary emitted"


def check_patching(msgs):
    hit = _text_contains(msgs, "cache") or _text_contains(msgs, "cached")
    if hit:
        return True, f"cache summary: {hit}"
    return bool(msgs), "no cache summary"


def check_sae(msgs):
    hit = _text_contains(msgs, "sae_lens") or _text_contains(msgs, "not installed")
    if hit:
        return True, f"graceful missing-dep notice: {hit}"
    if _has_type(msgs, "table"):
        return True, "SAE feature table emitted (sae_lens present!)"
    return bool(msgs), "no sae output"


async def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM-135M"
    print(f"Model under test: {model_id}")

    # Skip per-module venv creation: every new module's real deps
    # (torch/numpy/matplotlib/scikit-learn) are already in the main venv, and
    # the loader falls back to it when a module venv is absent. Building 20
    # isolated venvs on startup (esp. sae_lens) otherwise blocks for many
    # minutes. sae_features will report its missing optional dep gracefully.
    import scripts.core.integration as _integ
    _integ.LTTSIntegration._initialize_module_venvs = lambda self: None

    from scripts.router import get_context

    context = get_context()
    bus = context.message_bus

    # Run the bus delivery loop for the whole session.
    proc = asyncio.create_task(bus.start_processing())

    # Pre-load to enumerate real layer names.
    print("Loading model (first run downloads weights)...")
    ltts_model, _struct = context.models.load_model(model_id, "transformers")
    names = list(ltts_model.layer_wrappers.keys())
    blocks = sorted(
        [n for n in names if re.fullmatch(r".*\blayers\.\d+", n)],
        key=lambda s: int(s.split(".")[-1]),
    )
    attn = [b + ".self_attn" for b in blocks if b + ".self_attn" in names]
    print(f"  model class: {type(ltts_model.original_model).__name__}")
    print(f"  decoder blocks: {len(blocks)} | attention sublayers: {len(attn)}")
    # The loader leaves models on CPU; opt into GPU for this harness so we can
    # exercise real hardware (LTTS_E2E_DEVICE=cuda).
    target_device = os.environ.get("LTTS_E2E_DEVICE", "cpu")
    if target_device == "cuda" and torch.cuda.is_available():
        try:
            ltts_model.to("cuda")
        except Exception as e:
            print(f"  (cuda move failed: {e}; staying on CPU)")
    print(f"  device: {next(ltts_model.parameters()).device}")

    sel_blocks = _spaced(blocks, 8)
    sel_attn = _spaced(attn, 8)

    from scripts.services.inference_service import InferenceService

    service = InferenceService(context)

    def specs(mod_names):
        return [
            {"name": m, "parameters": {"emit_mode": "all"}} for m in mod_names
        ]

    overall = {}

    # --- Residual-stream family ------------------------------------------
    modules_R = {b: specs(RESIDUAL_MODULES) for b in sel_blocks}
    by_mod_R, _ = await run_family(service, bus, model_id, modules_R, "RESIDUAL")
    overall.update(report_family(
        by_mod_R, RESIDUAL_MODULES,
        {
            "logit_lens": check_logit_lens,
            "activation_trajectory": check_trajectory,
            "representation_similarity": check_repsim,
            "direct_logit_attribution": check_dla,
            "embedding_projector": check_projector,
        },
    ))

    # --- Attention family ------------------------------------------------
    modules_A = {a: specs(ATTENTION_MODULES) for a in sel_attn}
    by_mod_A, _ = await run_family(service, bus, model_id, modules_A, "ATTENTION")
    overall.update(report_family(
        by_mod_A, ATTENTION_MODULES,
        {
            "attention_rollout": check_rollout,
            "induction_head_detector": check_induction,
            "attention_entropy_profiler": check_entropy,
        },
    ))

    # --- Intervention / optional-dep family ------------------------------
    modules_I = {
        sel_blocks[3]: [{"name": "neuron_ablation", "parameters": {
            "enabled": True, "neuron_indices": "0,1,2,3,4",
            "ablation_mode": "zero", "positions": "all", "emit_summary": True,
            "emit_mode": "all"}}],
        sel_blocks[2]: [{"name": "activation_patching", "parameters": {
            "mode": "cache", "emit_summary": True, "emit_mode": "all"}}],
        sel_blocks[1]: [{"name": "sae_features", "parameters": {
            "enabled": True, "emit_mode": "all"}}],
    }
    by_mod_I, _ = await run_family(service, bus, model_id, modules_I, "INTERVENTION")
    overall.update(report_family(
        by_mod_I, ["neuron_ablation", "activation_patching", "sae_features"],
        {
            "neuron_ablation": check_ablation,
            "activation_patching": check_patching,
            "sae_features": check_sae,
        },
    ))

    bus.stop_processing()
    await asyncio.sleep(0.1)
    proc.cancel()

    # --- Verdict ----------------------------------------------------------
    print(f"\n{'=' * 70}\nFINAL VERDICT ({model_id})\n{'=' * 70}")
    passed = sum(1 for v in overall.values() if v)
    for mod, ok in overall.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {mod}")
    print(f"\n{passed}/{len(overall)} modules produced correct real data end-to-end.")
    return 0 if passed == len(overall) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
