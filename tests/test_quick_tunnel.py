from __future__ import annotations

import io
import types

import pytest

from scripts.utils import quick_tunnel


class _Process:
    def __init__(self, output):
        self.stdout = io.StringIO(output)
        self.terminated = False

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0

    def kill(self):
        self.terminated = True


def test_quick_tunnel_extracts_url_without_blocking(monkeypatch):
    process = _Process(
        "starting connector\nhttps://quiet-sun-42.trycloudflare.com is ready\n"
    )
    monkeypatch.setattr(quick_tunnel.subprocess, "Popen", lambda *a, **k: process)

    returned, url, log = quick_tunnel.start_quick_tunnel(8000, timeout=1)

    assert returned is process
    assert url == "https://quiet-sun-42.trycloudflare.com"
    assert "starting connector" in log


def test_quick_tunnel_reports_process_output_on_failure(monkeypatch):
    process = _Process("network unavailable\n")
    monkeypatch.setattr(quick_tunnel.subprocess, "Popen", lambda *a, **k: process)

    with pytest.raises(RuntimeError, match="network unavailable"):
        quick_tunnel.start_quick_tunnel(8000, timeout=0.1)

    assert process.terminated is True
