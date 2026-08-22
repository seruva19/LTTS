from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

EmissionKey = Tuple[str, str]


def _normalize_layer_path(layer_path: Optional[str]) -> str:
    if not layer_path:
        return "global"
    return layer_path


class _EmissionState:
    __slots__ = ("forward_pass", "finalized", "expected_passes", "last_emit")

    def __init__(self) -> None:
        self.forward_pass: int = 0
        self.finalized: bool = False
        self.expected_passes: Optional[int] = None
        self.last_emit: Optional[int] = None


_STATE: Dict[EmissionKey, _EmissionState] = defaultdict(_EmissionState)


def reset_state() -> None:
    if _STATE:
        logger.debug("EmissionController: resetting %d tracked entries", len(_STATE))
    _STATE.clear()


def update_expected_passes(
    module_name: str, layer_path: Optional[str], expected: Optional[int]
) -> None:
    key = (module_name, _normalize_layer_path(layer_path))
    state = _STATE[key]
    if expected is None:
        state.expected_passes = None
        return
    try:
        value = int(expected)
        if value < 1:
            value = 1
    except (TypeError, ValueError):
        logger.debug(
            "EmissionController: invalid expected passes value %r; ignoring", expected
        )
        return
    state.expected_passes = value
    logger.debug(
        "EmissionController: set expected passes to %d for %s/%s",
        value,
        module_name,
        layer_path,
    )


def mark_finalized(module_name: str, layer_path: Optional[str]) -> None:
    key = (module_name, _normalize_layer_path(layer_path))
    state = _STATE[key]
    state.finalized = True
    logger.debug("EmissionController: marked %s/%s finalized", module_name, layer_path)


def reset_entry(module_name: str, layer_path: Optional[str]) -> None:
    key = (module_name, _normalize_layer_path(layer_path))
    if key in _STATE:
        del _STATE[key]
        logger.debug(
            "EmissionController: reset entry for %s/%s", module_name, layer_path
        )


def should_emit(
    module_name: str,
    layer_path: Optional[str],
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    params = params or {}
    key = (module_name, _normalize_layer_path(layer_path))
    state = _STATE[key]

    state.forward_pass += 1
    forward_pass = state.forward_pass

    mode = params.get("emit_mode") or "all"
    emit_every_n = params.get("emit_every_n") or 1
    expected = params.get("expected_generation_passes")

    if expected is not None:
        try:
            state.expected_passes = max(1, int(expected))
        except (TypeError, ValueError):
            logger.debug(
                "EmissionController: invalid expected_generation_passes=%r for %s/%s",
                expected,
                module_name,
                layer_path,
            )

    emit = False
    deferred = False
    stride: Optional[int] = None

    if mode == "all":
        emit = True
    elif mode == "first":
        emit = forward_pass == 1
    elif mode == "every_n":
        try:
            stride = max(1, int(emit_every_n))
        except (TypeError, ValueError):
            stride = 1
        emit = forward_pass % stride == 0
    elif mode == "final":
        if state.finalized:
            emit = False
        else:
            expected_passes = state.expected_passes
            if expected_passes is not None:
                if forward_pass >= expected_passes:
                    emit = True
                    state.finalized = True
                else:
                    deferred = True
            else:
                # When expected passes is unknown, emit on first pass only
                emit = forward_pass == 1
                if emit:
                    state.finalized = True
                else:
                    deferred = True
    elif mode == "manual":
        emit = False
        deferred = True
    else:
        emit = True

    if emit:
        state.last_emit = forward_pass
        logger.debug(
            "EmissionController: emit=True for %s/%s (mode=%s pass=%s)",
            module_name,
            layer_path,
            mode,
            forward_pass,
        )
    else:
        logger.debug(
            "EmissionController: emit=False for %s/%s (mode=%s pass=%s)",
            module_name,
            layer_path,
            mode,
            forward_pass,
        )

    return {
        "emit": emit,
        "mode": mode,
        "forward_pass": forward_pass,
        "stride": stride,
        "expected_passes": state.expected_passes,
        "deferred": deferred,
    }
