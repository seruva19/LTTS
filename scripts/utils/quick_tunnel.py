"""Start a temporary Cloudflare Quick Tunnel and return its public URL."""

from __future__ import annotations

import queue
import re
import subprocess
import threading
import time
from collections.abc import Sequence

TUNNEL_URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com", re.I)


def start_quick_tunnel(
    port: int,
    *,
    executable: str = "cloudflared",
    timeout: float = 60,
    extra_args: Sequence[str] = (),
):
    """Start cloudflared without an account and wait for its assigned URL.

    Returns ``(process, public_url, startup_log)``. Output is consumed on a
    background thread so a quiet process cannot block the timeout.
    """
    command = [
        executable,
        "tunnel",
        "--url",
        f"http://127.0.0.1:{int(port)}",
        "--no-autoupdate",
        "--protocol",
        "http2",
        *extra_args,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    log: list[str] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            line = lines.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
        except queue.Empty:
            if process.poll() is not None:
                break
            continue
        if line is None:
            break
        log.append(line.rstrip())
        match = TUNNEL_URL_RE.search(line)
        if match:
            return process, match.group(0), "\n".join(log)

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
    detail = "\n".join(log[-20:]) or "cloudflared produced no output"
    raise RuntimeError(f"Cloudflare Quick Tunnel did not start within {timeout}s:\n{detail}")
