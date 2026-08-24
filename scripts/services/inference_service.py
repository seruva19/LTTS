from __future__ import annotations

import asyncio
import logging
import random
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
import numpy as np
from fastapi import HTTPException
from scripts.core.emission_controller import reset_state as reset_emission_state
from scripts.core.module_data_sender import set_current_client_token
from scripts.services.inference import (
    AttachmentSpec,
    ModuleAttachmentManager,
    ModuleDiagnosticsPublisher,
    InputPreparer,
    ModelManager,
    OutputProcessor,
    detach_modules,
)
from scripts.services.inference.cancellation import (
    InferenceCancelledError,
    begin_run,
    end_run,
    is_cancelled,
)
from scripts.services.model_compatibility import (
    apply_capability_override,
    build_compatibility_report,
)
from scripts.services.run_export import build_run_export

if TYPE_CHECKING:  # pragma: no cover - import only for typing
    from scripts.services.inference.contracts import InferenceInput


logger = logging.getLogger(__name__)

# Emission-controller state is process-global and reset per request, and the
# shared wrapped models are not safe for concurrent forward passes, so whole
# inference runs are serialized instead of interleaved.
_global_inference_lock = asyncio.Lock()

_model_locks: Dict[str, threading.Lock] = {}
_model_locks_guard = threading.Lock()


def _get_model_lock(model_id: str) -> threading.Lock:
    with _model_locks_guard:
        lock = _model_locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            _model_locks[model_id] = lock
        return lock


@contextmanager
def _reproducible_rng(seed: int, model: Any):
    """Seed one serialized inference and restore every caller RNG afterward."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = sorted(
        {
            parameter.device.index
            for parameter in model.parameters()
            if parameter.device.type == "cuda" and parameter.device.index is not None
        }
    )
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        for device in devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


class InferenceService:
    """Run inference and preserve the public service contract."""

    def __init__(self, context):
        self.context = context
        self.model_manager = ModelManager(context)
        self.input_preparer = InputPreparer()
        self.output_processor = OutputProcessor()
        self.diagnostics_publisher = ModuleDiagnosticsPublisher(context.message_bus)

    async def run_inference(self, input: "InferenceInput") -> Dict[str, Any]:
        async with _global_inference_lock:
            return await self._run_inference_serialized(input)

    async def _run_inference_serialized(self, input: "InferenceInput") -> Dict[str, Any]:
        attached_module_specs: List[AttachmentSpec] = []
        ltts_model = None
        compatibility = None
        rng_context = None
        token = input.client_token or "default"
        begin_run(token)

        try:
            reset_emission_state()
            set_current_client_token(token)

            def _check_cancelled():
                return is_cancelled(token)

            def _raise_if_cancelled(stage: str):
                if is_cancelled(token):
                    raise InferenceCancelledError(f"Inference cancelled by client ({stage})")
            ltts_model = await asyncio.to_thread(
                self.model_manager.ensure_model_ready, input.model_id
            )
            try:
                ltts_model.attention_manager.reset_run()
            except Exception as e:
                logger.debug("attention_manager.reset_run() failed (non-critical): %s", e)

            attachment_manager = ModuleAttachmentManager(self.context, ltts_model)
            attachment_manager.reset_attachments()
            compatibility = build_compatibility_report(
                self.context,
                ltts_model,
                input.model_id,
                input.modules,
            )
            if not compatibility.get("compatible", False):
                failures = []
                for module_report in compatibility.get("modules", []):
                    if module_report.get("compatible"):
                        continue
                    reasons = []
                    if not module_report.get("installed"):
                        reasons.append("module is not installed")
                    if module_report.get("missing_capabilities"):
                        reasons.append(
                            "missing capabilities: "
                            + ", ".join(module_report["missing_capabilities"])
                        )
                    if module_report.get("missing_optional_dependencies"):
                        reasons.append(
                            "missing optional dependencies: "
                            + ", ".join(
                                module_report["missing_optional_dependencies"]
                            )
                        )
                    if module_report.get("unresolved_layers"):
                        reasons.append(
                            "unknown targets: "
                            + ", ".join(module_report["unresolved_layers"])
                        )
                    if module_report.get("conflict_reasons"):
                        reasons.append(
                            "; ".join(module_report["conflict_reasons"])
                        )
                    if module_report.get("role_mismatches"):
                        details = [
                            f"{item.get('layer')} is {item.get('role')}"
                            for item in module_report["role_mismatches"]
                        ]
                        reasons.append("invalid target roles: " + ", ".join(details))
                    if not module_report.get("topology_supported", True):
                        reasons.append(
                            "unsupported architecture topology "
                            + str(
                                compatibility.get("architecture", {}).get(
                                    "topology", "unknown"
                                )
                            )
                        )
                    failures.append(
                        f"{module_report.get('name')}: "
                        + ("; ".join(reasons) or "incompatible")
                    )
                raise HTTPException(
                    status_code=422,
                    detail="Module compatibility preflight failed. "
                    + " | ".join(failures),
                )

            attached_module_specs, required_capabilities = (
                attachment_manager.attach_modules(input.modules)
            )
            requires_attention_outputs = "eager_attention" in required_capabilities
            requires_gradients = "gradients" in required_capabilities

            if requires_attention_outputs:
                force_eager = True
                try:
                    params = input.parameters or {}
                    if isinstance(params, dict):
                        force_eager = bool(
                            params.get("force_eager_attention", True)
                        )
                except Exception as e:
                    logger.debug("Error reading force_eager_attention param: %s", e)
                    force_eager = True
                attention_status = self.model_manager.configure_attention_outputs(
                    ltts_model,
                    force_eager=force_eager,
                )
                apply_capability_override(
                    compatibility, "eager_attention", attention_status
                )
                if not attention_status.get("supported"):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Selected modules require eager attention outputs, but "
                            f"the model cannot provide them: {attention_status.get('detail')}"
                        ),
                    )

            try:
                setattr(ltts_model, "_client_token", token)
                setattr(ltts_model, "_model_id", input.model_id)
                setattr(ltts_model, "_cancellation_check", _check_cancelled)
            except Exception as e:
                logger.debug("Failed to set _client_token on ltts_model: %s", e)

            raw_seed = (input.parameters or {}).get("seed")
            if raw_seed is not None:
                try:
                    seed = int(raw_seed)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=422, detail="seed must be an integer") from exc
                if seed < 0 or seed > 2**32 - 1:
                    raise HTTPException(status_code=422, detail="seed must be a uint32 value")
                rng_context = _reproducible_rng(seed, ltts_model.original_model)
                rng_context.__enter__()

            inputs, tokenizer = await asyncio.to_thread(
                self.input_preparer.prepare_inputs,
                ltts_model,
                input.model_id,
                input.prompt,
                requires_attention_outputs,
            )

            await self.context.message_bus.publish(
                token,
                {
                    "type": "inference_start",
                    "model_id": input.model_id,
                    "prompt": input.prompt,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            gradient_context = torch.enable_grad() if requires_gradients else torch.no_grad()

            def _run_forward():
                _raise_if_cancelled("before forward")
                with _get_model_lock(input.model_id):
                    with gradient_context:
                        self.model_manager.force_attention_extraction(
                            ltts_model, requires_attention_outputs
                        )
                        return ltts_model(**inputs)

            outputs = await asyncio.to_thread(_run_forward)
            _raise_if_cancelled("after forward")

            if requires_gradients:
                target, target_info = self._select_gradient_target(
                    ltts_model,
                    inputs,
                    outputs,
                    tokenizer,
                    attached_module_specs,
                )
                try:
                    await asyncio.to_thread(target.backward, retain_graph=True)
                except Exception as backward_error:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Selected gradient modules require a differentiable "
                            f"model target, but backward failed: {backward_error}"
                        ),
                    ) from backward_error
                self._finalize_gradient_modules(
                    attached_module_specs, target_info
                )
                try:
                    ltts_model.zero_grad(set_to_none=True)
                except TypeError:  # older torch compatibility
                    ltts_model.zero_grad()

            result_text = self.output_processor.interpret_outputs(
                ltts_model,
                inputs,
                outputs,
                tokenizer,
                attached_module_specs,
                input,
                requires_attention_outputs,
            )

            self.output_processor.finalize_emit_mode_modules(attached_module_specs)

            await self.context.message_bus.publish(
                token,
                {
                    "type": "inference_complete",
                    "model_id": input.model_id,
                    "result": result_text,
                    "output_info": {
                        "type": type(outputs).__name__,
                        "shape": getattr(outputs, "shape", None),
                        "device": str(getattr(outputs, "device", "unknown")),
                    },
                    "timestamp": datetime.now().isoformat(),
                },
            )

            output_info = {
                "type": type(outputs).__name__,
                "shape": getattr(outputs, "shape", None),
            }
            run_export = build_run_export(
                model_id=input.model_id,
                prompt=input.prompt,
                parameters=input.parameters,
                modules=input.modules,
                result=result_text,
                output_info=output_info,
                compatibility=compatibility,
            )

            return {
                "status": "success",
                "result": result_text,
                "model_id": input.model_id,
                "client_token": token,
                "output_info": output_info,
                "compatibility": compatibility,
                "run_export": run_export,
            }

        except InferenceCancelledError as exc:
            logger.info("Inference cancelled by client for model %s: %s", input.model_id, exc)
            await self.context.message_bus.publish(
                token,
                {
                    "type": "inference_cancelled",
                    "model_id": input.model_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return {
                "status": "cancelled",
                "model_id": input.model_id,
                "client_token": token,
                "detail": str(exc),
                "compatibility": compatibility,
            }

        except HTTPException as exc:
            logger.error("Inference rejected for model %s: %s", input.model_id, exc.detail)
            await self.context.message_bus.publish(
                token,
                {
                    "type": "inference_error",
                    "model_id": input.model_id,
                    "error": str(exc.detail),
                    "timestamp": datetime.now().isoformat(),
                },
            )
            raise

        except Exception as exc:
            logger.error("Inference error for model %s: %s", input.model_id, exc)

            await self.context.message_bus.publish(
                token,
                {
                    "type": "inference_error",
                    "model_id": input.model_id,
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(),
                },
            )

            raise HTTPException(status_code=500, detail=f"Inference failed: {str(exc)}")

        finally:
            if rng_context is not None:
                rng_context.__exit__(None, None, None)
            set_current_client_token("default")
            end_run(token)
            if ltts_model:
                try:
                    ltts_model._cancellation_check = None
                except Exception:
                    pass
            if ltts_model:
                try:
                    self.model_manager.restore_attention_outputs(ltts_model)
                except Exception as e:
                    logger.debug("restore_attention_outputs failed: %s", e)
            if ltts_model and attached_module_specs:
                await self.diagnostics_publisher.publish_uninvoked_module_logs(
                    ltts_model, attached_module_specs, token
                )
                detach_modules(ltts_model, attached_module_specs)
                attached_module_specs.clear()

    @staticmethod
    def _select_gradient_target(
        ltts_model,
        inputs: Dict[str, Any],
        outputs: Any,
        tokenizer: Any,
        attached_specs: List[AttachmentSpec],
    ):
        """Choose a scalar model output for honest target-score attribution."""

        gradient_params: Dict[str, Any] = {}
        for spec in attached_specs:
            if spec.module_name == "gradient_importance":
                gradient_params = spec.parameters or {}
                break

        try:
            configured_position = int(gradient_params.get("target_position", -1))
        except (TypeError, ValueError):
            configured_position = -1
        try:
            configured_id = int(gradient_params.get("target_token_id", -1))
        except (TypeError, ValueError):
            configured_id = -1

        logits = getattr(outputs, "logits", None)
        if isinstance(logits, torch.Tensor):
            if logits.ndim == 3:
                sequence_length = int(logits.shape[1])
                position = configured_position
                if position < 0:
                    position = sequence_length - 1
                    if (
                        getattr(ltts_model, "_model_task", None) == "masked_lm"
                        and tokenizer is not None
                        and getattr(tokenizer, "mask_token_id", None) is not None
                    ):
                        mask_positions = (
                            inputs["input_ids"][0]
                            == int(tokenizer.mask_token_id)
                        ).nonzero(as_tuple=False)
                        if mask_positions.numel():
                            position = int(mask_positions[0].item())
                position = max(0, min(sequence_length - 1, position))
                row = logits[:, position, :]
                target_id = configured_id
                if target_id < 0:
                    target_id = int(torch.argmax(row[0].detach()).item())
                target_id = max(0, min(int(row.shape[-1]) - 1, target_id))
                target = row[:, target_id].sum()
                label = None
                if tokenizer is not None:
                    try:
                        label = tokenizer.decode(
                            [target_id], skip_special_tokens=False
                        )
                    except Exception:
                        label = None
                return target, {
                    "kind": "token_logit",
                    "position": position,
                    "target_id": target_id,
                    "target_label": label,
                    "score": float(target.detach()),
                }

            if logits.ndim == 2:
                target_id = configured_id
                if target_id < 0:
                    target_id = int(torch.argmax(logits[0].detach()).item())
                target_id = max(0, min(int(logits.shape[-1]) - 1, target_id))
                target = logits[:, target_id].sum()
                config = getattr(
                    getattr(ltts_model, "original_model", None), "config", None
                )
                id2label = getattr(config, "id2label", None) or {}
                label = id2label.get(
                    target_id, id2label.get(str(target_id), str(target_id))
                )
                return target, {
                    "kind": "class_logit",
                    "target_id": target_id,
                    "target_label": label,
                    "score": float(target.detach()),
                }

        start_logits = getattr(outputs, "start_logits", None)
        end_logits = getattr(outputs, "end_logits", None)
        if isinstance(start_logits, torch.Tensor) and isinstance(
            end_logits, torch.Tensor
        ):
            start = int(torch.argmax(start_logits[0].detach()).item())
            end = int(torch.argmax(end_logits[0].detach()).item())
            target = start_logits[:, start].sum() + end_logits[:, end].sum()
            return target, {
                "kind": "answer_span_logit",
                "start": start,
                "end": end,
                "score": float(target.detach()),
            }

        hidden = getattr(outputs, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor) and hidden.requires_grad:
            target = hidden[:, -1].square().mean()
            return target, {
                "kind": "hidden_state_energy",
                "position": int(hidden.shape[1]) - 1,
                "score": float(target.detach()),
                "note": "Base model has no task head; using last-position hidden-state energy",
            }

        raise HTTPException(
            status_code=422,
            detail=(
                "No differentiable logits, QA scores, or hidden state are "
                "available for gradient attribution"
            ),
        )

    @staticmethod
    def _finalize_gradient_modules(
        attached_specs: List[AttachmentSpec], target_info: Dict[str, Any]
    ) -> None:
        finalized = set()
        for spec in attached_specs:
            if spec.module_name != "gradient_importance":
                continue
            module = spec.module
            identity = id(module)
            if identity in finalized:
                continue
            finalized.add(identity)
            finalizer = getattr(module, "finalize_gradients", None)
            if callable(finalizer):
                finalizer(target_info=target_info)
