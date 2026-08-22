from __future__ import annotations

import platform
import uuid
from datetime import datetime, timezone
from importlib import metadata
from typing import Any, Dict, Optional


def _package_version(name: str) -> Optional[str]:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_run_export(
    *,
    model_id: str,
    prompt: str,
    parameters: Dict[str, Any],
    modules: Optional[Dict[str, Any]],
    result: str,
    output_info: Dict[str, Any],
    compatibility: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a JSON-serializable, reproducible record of one inference run."""
    return {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "id": model_id,
            "revision": compatibility.get("model_revision"),
            "class": compatibility.get("model_class"),
        },
        "input": {
            "prompt": prompt,
            "parameters": dict(parameters or {}),
            "modules": modules or {},
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                package: _package_version(package)
                for package in (
                    "torch",
                    "transformers",
                    "fastapi",
                    "numpy",
                    "captum",
                )
            },
        },
        "compatibility": compatibility,
        "output": {
            "result": result,
            "info": output_info,
        },
    }
