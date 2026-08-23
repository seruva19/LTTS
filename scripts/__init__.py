"""LTTS application exports.

Keep the web runtime lazy: the reusable :mod:`ltts_engine` package imports the
model instrumentation under ``scripts.core`` and must not pull FastAPI into a
headless process merely because Python initializes this package first.
"""

__all__ = [
    "create_api_router",
    "get_api_router",
    "get_context",
    "reset_runtime_state",
    "api",
    "app",
    "context",
]


def create_api_router():
    from scripts.router import create_api_router as factory

    return factory()


def get_api_router():
    from scripts.router import get_api_router as getter

    return getter()


def get_context():
    from scripts.router import get_context as getter

    return getter()


def reset_runtime_state():
    from scripts.router import reset_runtime_state as reset

    return reset()


def __getattr__(name: str):
    if name in {"api", "app"}:
        return get_api_router()
    if name == "context":
        return get_context()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
