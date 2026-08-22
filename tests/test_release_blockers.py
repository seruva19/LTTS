from __future__ import annotations

import warnings
from pathlib import Path


def test_blueprint_display_name_cannot_escape_storage_directory(tmp_path):
    from scripts.blueprints.blueprint_state import BlueprintManager, BlueprintState

    storage = tmp_path / "blueprints"
    manager = BlueprintManager(str(storage))
    blueprint = BlueprintState(
        id="../../outside-id",
        name="../../outside-name",
        created_at="2026-01-01T00:00:00",
        updated_at="2026-01-01T00:00:00",
    )

    saved = Path(manager.save_blueprint(blueprint)).resolve()

    assert saved.parent == storage.resolve()
    assert saved.name.startswith("blueprint_")
    assert "outside" not in saved.name
    assert saved.is_file()
    assert not (tmp_path / "outside-name").exists()


def test_sync_reactor_signal_does_not_leak_an_unawaited_coroutine():
    from scripts.core.ltts_event import LTTSReactor
    from scripts.messages.message_bus import MessageBus

    reactor = LTTSReactor(message_bus=MessageBus())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reactor.signal("text", {"value": "hello"})

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
        and "was never awaited" in str(warning.message)
    ]
