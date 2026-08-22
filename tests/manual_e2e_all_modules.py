#!/usr/bin/env python
"""
Attach EVERY discovered module to representative layers, run one inference, and
report per-module health (emitted real data / skipped / errored).

  LTTS_E2E_DEVICE=cuda python tests/manual_e2e_all_modules.py [model_id]
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, ".")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch

import scripts.core.integration as _integ
_integ.LTTSIntegration._initialize_module_venvs = lambda self: None

PROMPT = "The cat sat on the mat. The cat sat on the"

# Modules that only emit once they have seen >= 2 layers (cross-layer charts).
CROSS_LAYER = {"activation_trajectory", "representation_similarity"}
SEPARATE_INTERVENTIONS = {"neuron_ablation", "activation_patching"}

# Per-module parameters needed to make a module actually emit.
PARAM_OVERRIDES = {
    "neuron_ablation": {"enabled": True, "neuron_indices": "0,1,2",
                        "ablation_mode": "zero", "emit_summary": True},
    "activation_patching": {"mode": "cache", "emit_summary": True},
    "token_contributor": {
        "occlusion_mode": "full_model_forward",
        "max_full_occlusion_tokens": 16,
    },
}


async def main():
    model_id = sys.argv[1] if len(sys.argv) > 1 else "HuggingFaceTB/SmolLM-135M"
    from scripts.router import get_context

    context = get_context()
    bus = context.message_bus

    ltts_model, _ = context.models.load_model(model_id, "transformers")
    if os.environ.get("LTTS_E2E_DEVICE") == "cuda" and torch.cuda.is_available():
        try:
            ltts_model.to("cuda")
        except Exception as e:
            print(f"(cuda move failed: {e})")

    names = list(ltts_model.layer_wrappers.keys())
    descriptors = ltts_model.module_descriptors
    paths_by_role = defaultdict(list)
    for path, descriptor in descriptors.items():
        paths_by_role[descriptor.get("role", "other")].append(path)
    for role in paths_by_role:
        paths_by_role[role].sort(
            key=lambda path: (
                descriptors[path].get("layer_index") is None,
                descriptors[path].get("layer_index") or 0,
                path,
            )
        )
    blocks = paths_by_role.get("block", [])

    all_mods = sorted(context.module_loader.get_all_modules().keys())
    requested_modules = {
        name.strip()
        for name in os.environ.get("LTTS_E2E_MODULES", "").split(",")
        if name.strip()
    }
    if requested_modules:
        all_mods = [name for name in all_mods if name in requested_modules]
    print(f"Model: {type(ltts_model.original_model).__name__}  "
          f"device: {next(ltts_model.parameters()).device}")
    print(f"Discovered {len(all_mods)} modules.")
    print("Architecture:", ltts_model.architecture)

    def _spec(m):
        params = {"emit_mode": "all", "enabled": True}
        params.update(PARAM_OVERRIDES.get(m, {}))
        return {"name": m, "parameters": dict(params)}

    from scripts.services.model_compatibility import module_attachment_contract

    attachments = defaultdict(list)
    intervention_attachments = defaultdict(list)
    optional_missing = {}
    target_priority = (
        "block",
        "self_attention",
        "cross_attention",
        "mlp",
        "normalization",
        "embedding",
        "activation",
        "moe_router",
        "moe_expert",
    )
    for module_name in all_mods:
        info = context.module_loader.get_all_modules()[module_name]
        contract = module_attachment_contract(
            module_name, info.get("metadata") or {}
        )
        missing = [
            item["package"]
            for item in contract["optional_dependencies"]
            if not item["available"]
        ]
        if missing:
            optional_missing[module_name] = missing
            continue

        if contract["model_level"]:
            target = "__model__"
        else:
            target = None
            for role in target_priority:
                if role in contract["target_roles"] and paths_by_role.get(role):
                    target = paths_by_role[role][len(paths_by_role[role]) // 2]
                    break
            if target is None:
                raise RuntimeError(
                    f"No compatible target for {module_name}; "
                    f"roles={contract['target_roles']}"
                )
        destination = (
            intervention_attachments
            if module_name in SEPARATE_INTERVENTIONS
            else attachments
        )
        destination[target].append(_spec(module_name))

        if module_name in CROSS_LAYER and len(blocks) > 1:
            second = blocks[-1] if target != blocks[-1] else blocks[0]
            destination[second].append(_spec(module_name))

    passes = [("analysis_modules", dict(attachments))]
    if intervention_attachments:
        passes.append(("intervention_modules", dict(intervention_attachments)))
    for pass_name, pass_attachments in passes:
        print(
            f"{pass_name} targets: "
            + ", ".join(
                f"{path}={len(items)}" for path, items in pass_attachments.items()
            )
        )

    viz = defaultdict(int)
    errors = []
    diagnostics = defaultdict(list)
    captured_messages = []

    def collector(msg):
        if not isinstance(msg, dict):
            return
        captured_messages.append(msg)
        t = msg.get("type")
        if t == "visualization":
            mod = msg.get("module")
            data = msg.get("data")
            if not mod and isinstance(data, dict):
                mod = data.get("module") or data.get("module_name")
            viz[mod or "?"] += 1
        elif t == "error":
            errors.append(str(msg.get("data") or msg))
        elif t == "module_event_log":
            for entry in msg.get("modules") or []:
                module_name = entry.get("module_name")
                if module_name:
                    diagnostics[module_name].append(entry.get("metrics") or {})

    bus.subscribe("default", collector)
    proc = asyncio.create_task(bus.start_processing())

    from scripts.routers.inference_router import InferenceInput
    from scripts.services.inference_service import InferenceService

    service = InferenceService(context)
    for name, mods in passes:
        inp = InferenceInput(
            model_id=model_id, prompt=PROMPT, max_tokens=8,
            parameters={"force_eager_attention": True}, client_token="default",
            modules=mods,
        )
        try:
            result = await service.run_inference(inp)
            print(f"[{name}] inference success | result: {result['result'][:50]!r}")
        except Exception as e:
            print(f"[{name}] inference RAISED: {type(e).__name__}: {e}")
        await asyncio.sleep(1.0)
    await asyncio.sleep(1.0)
    bus.stop_processing()
    await asyncio.sleep(0.1)
    proc.cancel()
    bus.unsubscribe("default", collector)

    def declared_skip(module_name):
        """The reason a module gave for emitting nothing, if it gave one."""
        for entry in reversed(diagnostics.get(module_name) or []):
            if entry.get("last_status") != "skipped":
                continue
            result = entry.get("last_result")
            reason = result.get("reason") if isinstance(result, dict) else None
            return f"declared: {reason}" if reason else "declared"
        return None

    # Per-module verdict.
    print(f"\n{'=' * 64}\nPER-MODULE RESULT ({model_id})\n{'=' * 64}")
    counts = {"PASS": 0, "SKIP": 0, "FAIL": 0}
    for m in all_mods:
        emitted = viz.get(m, 0)
        # an error references this module if its name appears in any error payload
        errored = sum(1 for e in errors if m in e)
        if errored:
            verdict, counts_key = "FAIL", "FAIL"
        elif emitted > 0:
            verdict, counts_key = "PASS", "PASS"
        elif m in optional_missing:
            verdict, counts_key = (
                "SKIP(optional dependency: "
                + ",".join(optional_missing[m])
                + ")",
                "SKIP",
            )
        elif declared_skip(m):
            # A module that declines and says why did its job. Only silence is
            # a failure - the point of the diagnostics channel is that "no
            # output" and "no output, because X" are different outcomes.
            verdict, counts_key = f"SKIP({declared_skip(m)})", "SKIP"
        else:
            verdict, counts_key = "FAIL(no emission)", "FAIL"
        counts[counts_key] += 1
        print(f"  [{verdict:14.14s}] {m:28s} emitted={emitted} errored={errored}")
        if counts_key == "FAIL" and diagnostics.get(m):
            last = diagnostics[m][-1]
            print(
                " " * 20
                + f"last_status={last.get('last_status')} "
                + f"last_result={last.get('last_result')} "
                + f"last_error={last.get('last_error')}"
            )

    print(f"\nTotals: PASS={counts['PASS']}  SKIP={counts['SKIP']}  FAIL={counts['FAIL']}")
    if errors:
        print(f"\n{len(errors)} error signal(s). Unique samples:")
        seen = set()
        for e in errors:
            key = e[:80]
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {e[:200]}")

    artifact_path = os.environ.get("LTTS_E2E_ARTIFACT_PATH")
    if artifact_path:
        parent = os.path.dirname(os.path.abspath(artifact_path))
        os.makedirs(parent, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as artifact_file:
            json.dump(
                {
                    "model_id": model_id,
                    "modules": all_mods,
                    "counts": counts,
                    "visualization_counts": dict(viz),
                    "diagnostics": dict(diagnostics),
                    "errors": errors,
                    "messages": captured_messages,
                },
                artifact_file,
                ensure_ascii=False,
                indent=2,
                default=repr,
            )
        print(f"\nRaw artifact evidence: {artifact_path}")
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
