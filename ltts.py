import contextvars
import mimetypes
import os
import uuid
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from scripts import api
from scripts.router import lifespan
from scripts.configs.user_config import UserConfig
from scripts.utils.hf_auth import setup_hf_auth

# Context variable for request ID — accessible from any logger in the call stack
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


def _install_request_id_record_factory() -> None:
    """Ensure every log record has request_id, including third-party logs."""
    current_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get("-")
        return record

    logging.setLogRecordFactory(record_factory)


# Configure logging — default to INFO in production, DEBUG only when explicitly set
log_level = os.getenv("LTTS_LOG_LEVEL", "INFO").upper()
_install_request_id_record_factory()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] %(message)s',
)

# Setup HuggingFace authentication on startup
setup_hf_auth()

app = FastAPI(
    title="LTTS",
    description="Transformer model interpretability toolkit",
    version="0.1.0",
    lifespan=lifespan,
)

allowed_origins_env = os.getenv("LTTS_ALLOWED_ORIGINS", "")
allowed_origins = (
    [origin.strip() for origin in allowed_origins_env.split(",") if origin.strip()]
    if allowed_origins_env
    else [
        "http://localhost:5173",
        "http://localhost:4173",
        "http://localhost:8000",
    ]
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID to every HTTP request for tracing."""
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        request.state.request_id = rid
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            request_id_var.reset(token)


app.add_middleware(RequestIDMiddleware)

# Compress responses — the model-structure payload is 1-2 MB of JSON per load,
# which is slow to ship (especially over an SSH tunnel) and to parse client-side.
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Client-Token", "X-Request-ID"],
)

app.include_router(api)

# Serve the built SPA (dist/) on the same origin as the API when present, so the
# whole app is reachable on one port (e.g. behind an SSH tunnel). Mounted after
# the API router so /api and /ws take precedence; harmless when dist/ is absent.
# Python reads MIME types from the Windows registry, where .js is commonly
# registered as text/plain. Browsers refuse to execute a module script served
# that way, so the built SPA silently fails to start - a blank page whose only
# symptom is a console error. Pin the types the bundle needs, regardless of
# whether dist/ has been built yet.
for _suffix, _mime_type in (
    (".js", "text/javascript"),
    (".mjs", "text/javascript"),
    (".css", "text/css"),
    (".json", "application/json"),
    (".svg", "image/svg+xml"),
    (".wasm", "application/wasm"),
):
    mimetypes.add_type(_mime_type, _suffix)

if os.path.isdir("dist"):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory="dist", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("LTTS_HOST", "127.0.0.1")
    port = int(os.getenv("LTTS_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
