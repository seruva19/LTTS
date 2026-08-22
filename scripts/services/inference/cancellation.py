"""Cooperative mid-run inference cancellation.

A client sends ``{"type": "stop_inference", "token": ...}`` over WebSocket;
the flag is set here and the running inference pipeline checks it at safe
points (between layers, before forward/backward) and aborts with
``InferenceCancelledError``.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional


class InferenceCancelledError(Exception):
    """Raised inside the model pipeline when the client cancels the run."""


_lock = threading.Lock()
_active_runs: Dict[str, threading.Event] = {}


def begin_run(token: str) -> threading.Event:
    """Register a run for ``token`` and return its cancellation event."""
    event = threading.Event()
    with _lock:
        _active_runs[token] = event
    return event


def cancel_run(token: str) -> bool:
    """Request cancellation of the run bound to ``token``.

    Returns True when a matching active run was found.
    """
    with _lock:
        event = _active_runs.get(token)
    if event is None:
        return False
    event.set()
    return True


def end_run(token: str) -> None:
    """Unregister the run for ``token`` (called in the request's finally)."""
    with _lock:
        _active_runs.pop(token, None)


def is_cancelled(token: Optional[str]) -> bool:
    if not token:
        return False
    with _lock:
        event = _active_runs.get(token)
    return event is not None and event.is_set()
