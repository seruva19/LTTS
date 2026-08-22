"""The personal model registry must track the shipped defaults.

It is generated once from user/default_models.yaml. Without reconciliation a
model added to the defaults by an upgrade reaches fresh installs only, and an
existing user never sees it — which is how the shipped encoder-only and
encoder-decoder checkpoints stayed invisible on this machine.

Reconciliation has to be additive: the personal file is the user's, so their
own entries and edits survive, and a shipped model they removed stays removed.
"""

from __future__ import annotations

import yaml

from scripts.configs import user_files

DEFAULTS = {
    "models": {
        "local/DEBUG": {"id": "debug/dummy", "data": {}},
        "shipped/one": {"id": "shipped/one", "data": {"task": "causal_lm"}},
        "shipped/two": {"id": "shipped/two", "data": {}},
    }
}


def _write(path, payload):
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read(path):
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def test_new_shipped_models_reach_an_existing_install(tmp_path):
    default_path = tmp_path / "default_models.yaml"
    user_path = tmp_path / "models.user.yaml"
    _write(default_path, DEFAULTS)
    # A registry generated before "shipped/two" existed, with the user's own
    # model added afterwards.
    _write(
        user_path,
        {
            "models": {
                "local/DEBUG": {"id": "debug/dummy", "data": {}},
                "shipped/one": {"id": "shipped/one", "data": {"task": "causal_lm"}},
                "mine/private": {"id": "mine/private", "data": {}},
            }
        },
    )

    added = user_files.sync_default_models(str(user_path), str(default_path))

    assert added == ["shipped/two"]
    models = _read(user_path)["models"]
    assert "shipped/two" in models
    assert "mine/private" in models, "the user's own entry was dropped"


def test_user_edits_to_a_shipped_entry_are_not_overwritten(tmp_path):
    default_path = tmp_path / "default_models.yaml"
    user_path = tmp_path / "models.user.yaml"
    _write(default_path, DEFAULTS)
    _write(
        user_path,
        {
            "models": {
                "shipped/one": {"id": "shipped/one", "data": {"task": "base_model"}},
            }
        },
    )

    user_files.sync_default_models(str(user_path), str(default_path))

    models = _read(user_path)["models"]
    assert models["shipped/one"]["data"]["task"] == "base_model"


def test_a_removed_shipped_model_stays_removed(tmp_path):
    default_path = tmp_path / "default_models.yaml"
    user_path = tmp_path / "models.user.yaml"
    _write(default_path, DEFAULTS)
    _write(user_path, {"models": dict(DEFAULTS["models"])})

    registry = _read(user_path)
    del registry["models"]["shipped/one"]
    _write(user_path, registry)
    user_files.forget_default_model(
        "shipped/one", user_path=str(user_path), default_path=str(default_path)
    )

    user_files.sync_default_models(str(user_path), str(default_path))

    models = _read(user_path)["models"]
    assert "shipped/one" not in models, "a model the user removed came back"
    assert _read(user_path)["removed_defaults"] == ["shipped/one"]


def test_the_sync_is_idempotent(tmp_path):
    default_path = tmp_path / "default_models.yaml"
    user_path = tmp_path / "models.user.yaml"
    _write(default_path, DEFAULTS)
    _write(user_path, {"models": {}})

    first = user_files.sync_default_models(str(user_path), str(default_path))
    second = user_files.sync_default_models(str(user_path), str(default_path))

    assert sorted(first) == sorted(DEFAULTS["models"])
    assert second == [], "a second start rewrote the registry with no changes"


def test_no_legacy_filenames_remain():
    """Pre-`.user` filenames cannot override shipped defaults."""
    assert not hasattr(user_files, "LEGACY_CONFIG")
    assert not hasattr(user_files, "LEGACY_MODELS")
