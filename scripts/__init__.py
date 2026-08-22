"""LTTS package exports with lazy runtime initialization."""

from scripts.router import (
    create_api_router,
    get_api_router,
    get_context,
    reset_runtime_state,
)

__all__ = [
    "create_api_router",
    "get_api_router",
    "get_context",
    "reset_runtime_state",
    "api",
    "app",
    "context",
]


def __getattr__(name: str):
    if name in {"api", "app"}:
        return get_api_router()
    if name == "context":
        return get_context()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
