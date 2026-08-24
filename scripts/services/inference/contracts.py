from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


MAX_PROMPT_LENGTH = 100_000
MAX_TOKENS_UPPER = 8192
MAX_MODULES_PER_REQUEST = 50


class InferenceInput(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=MAX_PROMPT_LENGTH)
    max_tokens: Optional[int] = Field(None, ge=1, le=MAX_TOKENS_UPPER)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    client_token: Optional[str] = Field(None, max_length=128)
    modules: Optional[Dict[str, List[Union[str, Dict[str, Any]]]]] = None

    @field_validator("modules")
    @classmethod
    def validate_modules(cls, value):
        if value is not None:
            total = sum(len(specs) for specs in value.values())
            if total > MAX_MODULES_PER_REQUEST:
                raise ValueError(
                    f"Too many module attachments ({total}), "
                    f"max is {MAX_MODULES_PER_REQUEST}"
                )
        return value


class PreflightInput(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=256)
    modules: Optional[Dict[str, List[Union[str, Dict[str, Any]]]]] = None
