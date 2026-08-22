"""Resolve the directories analysis modules are loaded from and installed into.

Reading and writing are deliberately different paths. Module *source* normally
lives in a sibling LTTS-modules checkout, which is a git working tree the user
may be editing. Registry installs must never touch it: they clone over the
target directory, so installing into a checkout would silently discard local
work. Installs therefore always land in ``<project root>/ltts_modules``, and
the loader scans both.


The single source of truth for module code is the separate LTTS-modules
repository. Modules are loaded directly from there — no copies. Resolution
order:

1. ``LTTS_MODULES_DIR`` env var (absolute path, or relative to cwd);
2. a sibling ``<project root>/../LTTS-modules`` checkout containing at least
   one ``*/init.py``;
3. ``<project root>/ltts_modules`` (registry install directory).
"""

from __future__ import annotations

import os
from pathlib import Path

_MODULES_REPO_DIRNAME = "LTTS-modules"


def _has_modules(directory: Path) -> bool:
    try:
        return any(
            child.is_dir() and (child / "init.py").exists()
            for child in directory.iterdir()
        )
    except OSError:
        return False


def resolve_modules_dir(project_root: str | Path | None = None) -> Path:
    env_dir = os.environ.get("LTTS_MODULES_DIR", "").strip()
    if env_dir:
        candidate = Path(env_dir)
        if not candidate.is_absolute():
            candidate = Path.cwd() / candidate
        if _has_modules(candidate):
            return candidate

    root = (Path(project_root) if project_root else Path.cwd()).resolve()
    sibling = root.parent / _MODULES_REPO_DIRNAME
    if _has_modules(sibling):
        return sibling

    return root / "ltts_modules"


def resolve_install_dir(project_root: str | Path | None = None) -> Path:
    """Where registry installs are written. Never a source checkout."""
    root = Path(project_root) if project_root else Path.cwd()
    return root / "ltts_modules"


def resolve_module_search_paths(project_root: str | Path | None = None) -> list[Path]:
    """Every directory to load modules from, in precedence order.

    The resolved source directory comes first: an explicit LTTS_MODULES_DIR or a
    sibling checkout takes precedence over the registry install directory.
    """
    root = Path(project_root) if project_root else Path.cwd()
    paths = [resolve_modules_dir(root), resolve_install_dir(root)]

    seen: set[str] = set()
    ordered: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(path)
    return ordered
