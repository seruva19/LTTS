from __future__ import annotations

import logging
from typing import List

from .attachments import AttachmentSpec

logger = logging.getLogger(__name__)


def detach_modules(ltts_model, attached_specs: List[AttachmentSpec]) -> None:
    """Detach all modules that were attached for the inference run."""
    for spec in attached_specs:
        event_type = spec.event_type
        module_instance = spec.module
        if not event_type or module_instance is None:
            continue
        try:
            ltts_model.detach_module(event_type, module_instance)
            logger.debug(
                "✅ INFERENCE: Detached %s from %s after inference run",
                spec.module_name,
                event_type,
            )
        except Exception as detach_error:
            logger.warning(
                "⚠ INFERENCE: Failed to detach %s from %s: %s",
                spec.module_name,
                event_type,
                detach_error,
            )
