"""Test-session setup.

Importing ``scripts.router`` builds the whole runtime, which by default also
creates one virtualenv per installed module and pip-installs into each. That is
minutes of work and network traffic before the first assertion runs, so the
test session opts out. Set here rather than in each test module because the
environment has to be in place before ``scripts.router`` is first imported.
"""

import os
import sys
import types
from pathlib import Path

os.environ.setdefault("LTTS_SKIP_MODULE_VENVS", "1")

# Allow both `pytest` and `python -m pytest`: only the latter puts the repo root
# on sys.path by itself.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Modules now live in the separate LTTS-modules repository (loaded from there
# directly — see ltts_platform.module_paths). Tests import them under the
# legacy `ltts_modules` package name, so alias that name to the resolved
# directory as a namespace package.
if "ltts_modules" not in sys.modules:
    from ltts_platform.module_paths import resolve_modules_dir

    _modules_dir = resolve_modules_dir(ROOT)
    if _modules_dir.is_dir():
        _alias = types.ModuleType("ltts_modules")
        _alias.__path__ = [str(_modules_dir)]
        sys.modules["ltts_modules"] = _alias
