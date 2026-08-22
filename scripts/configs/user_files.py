"""User config file scheme.

Defaults ship with the application (user/default_config.yaml,
user/default_models.yaml — tracked in git). On first start LTTS generates
personal copies with a ``.user`` marker: user/config.user.yaml and
user/models.user.yaml. Everything matching ``user/*.user.yaml`` is
gitignored, so personal settings (e.g. the local modules-registry path)
never reach the repository.

The personal model registry is reconciled with the shipped defaults on every
start: models added to user/default_models.yaml by an upgrade appear, and
nothing the user added or edited is touched. Removing a shipped model through
the API records it under ``removed_defaults`` so it stays removed.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "user/default_config.yaml"
USER_CONFIG = "user/config.user.yaml"

DEFAULT_MODELS = "user/default_models.yaml"
USER_MODELS = "user/models.user.yaml"


def _ensure_user_file(user_path: str, default_path: str) -> None:
    if os.path.exists(user_path):
        return

    os.makedirs(os.path.dirname(user_path), exist_ok=True)

    if os.path.exists(default_path):
        shutil.copy2(default_path, user_path)
        logger.info(
            "Generated %s from %s — customize it freely, it is gitignored.",
            user_path,
            default_path,
        )
        return

    with open(user_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump({}, handle)
    logger.warning(
        "Generated empty %s (no default file found at %s).", user_path, default_path
    )


def sync_default_models(
    user_path: str = USER_MODELS, default_path: str = DEFAULT_MODELS
) -> List[str]:
    """Add shipped models the personal registry does not have yet.

    Purely additive: an entry the user edited keeps its edits, and one they
    removed through the API stays removed via ``removed_defaults``. Without
    this a model added to the shipped defaults by an upgrade would only ever
    reach fresh installs.
    """
    defaults = read_yaml(default_path).get("models")
    if not isinstance(defaults, dict) or not defaults:
        return []

    registry = read_yaml(user_path)
    models = registry.get("models")
    if not isinstance(models, dict):
        models = {}

    removed = registry.get("removed_defaults")
    removed = set(removed) if isinstance(removed, list) else set()

    added = [
        model_id
        for model_id in defaults
        if model_id not in models and model_id not in removed
    ]
    if not added:
        return []

    for model_id in added:
        models[model_id] = defaults[model_id]
    registry["models"] = models
    write_yaml(user_path, registry)
    logger.info("Added %s from the shipped defaults to %s", ", ".join(added), user_path)
    return added


def forget_default_model(
    model_id: str, user_path: str = USER_MODELS, default_path: str = DEFAULT_MODELS
) -> None:
    """Record that a shipped model was removed, so the sync leaves it alone."""
    defaults = read_yaml(default_path).get("models")
    if not isinstance(defaults, dict) or model_id not in defaults:
        return

    registry = read_yaml(user_path)
    removed = registry.get("removed_defaults")
    removed = list(removed) if isinstance(removed, list) else []
    if model_id in removed:
        return

    removed.append(model_id)
    registry["removed_defaults"] = removed
    write_yaml(user_path, registry)


def ensure_user_files() -> None:
    _ensure_user_file(USER_CONFIG, DEFAULT_CONFIG)
    _ensure_user_file(USER_MODELS, DEFAULT_MODELS)
    sync_default_models()


def read_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return {}


def read_user_config() -> Dict[str, Any]:
    return read_yaml(USER_CONFIG)


def read_default_config() -> Dict[str, Any]:
    return read_yaml(DEFAULT_CONFIG)


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
