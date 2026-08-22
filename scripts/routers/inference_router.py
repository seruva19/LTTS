import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from scripts.services.inference_service import InferenceService
from scripts.services.model_compatibility import (
    apply_capability_override,
    build_compatibility_report,
)

logger = logging.getLogger(__name__)

# Limits to prevent resource exhaustion
_MAX_PROMPT_LENGTH = 100_000  # characters
_MAX_TOKENS_UPPER = 8192
_MAX_MODULES_PER_REQUEST = 50


class InferenceInput(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=_MAX_PROMPT_LENGTH)
    max_tokens: Optional[int] = Field(None, ge=1, le=_MAX_TOKENS_UPPER)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    client_token: Optional[str] = Field(None, max_length=128)
    modules: Optional[Dict[str, List[Union[str, Dict[str, Any]]]]] = (
        None  # layer_name -> module specs
    )

    @field_validator("modules")
    @classmethod
    def validate_modules(cls, v):
        if v is not None:
            total = sum(len(specs) for specs in v.values())
            if total > _MAX_MODULES_PER_REQUEST:
                raise ValueError(
                    f"Too many module attachments ({total}), max is {_MAX_MODULES_PER_REQUEST}"
                )
        return v


class PreflightInput(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=256)
    modules: Optional[Dict[str, List[Union[str, Dict[str, Any]]]]] = None


# Concurrency limiter — only N inferences at once
import asyncio

_inference_semaphore: Optional[asyncio.Semaphore] = None


def _get_semaphore() -> asyncio.Semaphore:
    global _inference_semaphore
    if _inference_semaphore is None:
        _inference_semaphore = asyncio.Semaphore(2)
    return _inference_semaphore


# Request timeout (seconds)
_INFERENCE_TIMEOUT = 300  # 5 minutes


def create_inference_router(context):
    """Create and configure the inference router with context dependency injection."""
    router = APIRouter()

    service = InferenceService(context)

    @router.post("/api/models/preflight")
    async def preflight(input: PreflightInput):
        """Load or reuse a model and report selected-module compatibility."""
        try:
            ltts_model = service.model_manager.ensure_model_ready(input.model_id)
            report = build_compatibility_report(
                context, ltts_model, input.model_id, input.modules
            )
            needs_attention = any(
                "eager_attention" in item.get("required_capabilities", [])
                for item in report.get("modules", [])
            )
            if needs_attention:
                attention_status = service.model_manager.configure_attention_outputs(
                    ltts_model, force_eager=True
                )
                apply_capability_override(
                    report, "eager_attention", attention_status
                )
            return report
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Compatibility preflight failed for %s: %s", input.model_id, exc)
            raise HTTPException(
                status_code=500,
                detail=f"Compatibility preflight failed: {exc}",
            ) from exc

    @router.post("/api/inference")
    async def run_inference(input: InferenceInput):
        """Run model inference with attached modules processing through event system."""
        sem = _get_semaphore()
        if sem.locked():
            logger.warning("Inference request queued — semaphore at capacity")

        async with sem:
            try:
                return await asyncio.wait_for(
                    service.run_inference(input),
                    timeout=_INFERENCE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error("Inference timed out after %ds for model %s", _INFERENCE_TIMEOUT, input.model_id)
                raise HTTPException(
                    status_code=504,
                    detail=f"Inference timed out after {_INFERENCE_TIMEOUT} seconds",
                )

    return router
