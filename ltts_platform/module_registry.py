"""
LTTS Module Registry - Remote module repository system for production

This system handles:
1. Remote module discovery and installation
2. Debug mode emulation for local development
3. Module metadata validation and caching
"""

import ast
import os
import json
import re
import requests

try:
    from .module_paths import resolve_modules_dir, resolve_install_dir, resolve_module_search_paths
except ImportError:  # pragma: no cover - direct script execution
    from module_paths import resolve_modules_dir, resolve_install_dir, resolve_module_search_paths
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from .module_venv_manager import ModuleVenvManager

logger = logging.getLogger(__name__)


def _version_tuple(version: str) -> tuple:
    """'1.2.0' -> (1, 2, 0); non-numeric segments are skipped."""
    parts = re.findall(r"\d+", str(version))
    return tuple(int(p) for p in parts)

@dataclass
class ModuleInfo:
    """Module metadata for registry"""
    name: str
    version: str
    description: str
    author: str
    event_types: List[str]
    dependencies: List[str]
    tags: List[str]
    repository_url: str  # Git repository URL
    module_path: str  # Path within repository (e.g., "ltts_modules/module_name")
    size_bytes: int
    created_at: str
    updated_at: str
    installed: bool = False
    local_path: Optional[str] = None

class ModuleRegistry:
    """Manages remote and local LTTS modules"""

    def __init__(self, project_root: str = ".", debug_mode: bool = False):
        self.project_root = Path(project_root)
        # Reads may come from a source checkout; writes must not. Cloning over
        # a directory inside the user's LTTS-modules working tree would discard
        # their local edits, so installs always target the install directory.
        self.modules_dir = resolve_install_dir(self.project_root)
        self.module_search_paths = resolve_module_search_paths(self.project_root)
        self.cache_dir = self.project_root / ".ltts_cache"
        self.registry_cache = self.cache_dir / "module_registry.json"
        self.debug_mode = debug_mode

        # Initialize virtual environment manager
        try:
            self.venv_manager = ModuleVenvManager(str(self.project_root))
        except ValueError as e:
            logger.warning(f"ModuleVenvManager initialization failed: {e}. Virtual environment features disabled.")
            self.venv_manager = None

        # Create directories if they don't exist
        self.modules_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)

        # Remote repository configuration. Precedence: LTTS_MODULES_REGISTRY
        # env var, then the personal user config (modules.registry_url in
        # user/config.user.yaml), then a sibling LTTS-modules checkout,
        # falling back to the GitHub raw URL. Accepts an http(s) URL or a
        # local path to a registry.json / modules repo checkout.
        default_registry = (
            "https://raw.githubusercontent.com/seruva19/"
            "LTTS-modules/main/registry.json"
        )
        sibling = (
            self.project_root.resolve().parent / "LTTS-modules" / "registry.json"
        )
        if sibling.exists():
            default_registry = str(sibling)
        self.remote_registry_url = (
            os.environ.get("LTTS_MODULES_REGISTRY")
            or self._registry_url_from_user_config()
            or default_registry
        )

        # Initialize cached modules and load from cache
        self._cached_modules: Dict[str, ModuleInfo] = {}
        self._load_cache()

    @staticmethod
    def _registry_url_from_user_config() -> Optional[str]:
        """Read modules.registry_url from the personal user config (if set)."""
        try:
            import yaml

            for path in (
                Path("user/config.user.yaml"),
                Path("user/config.yaml"),
            ):
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as handle:
                    config = yaml.safe_load(handle) or {}
                url = (config.get("modules") or {}).get("registry_url")
                if url:
                    return str(url)
        except Exception as e:
            logger.debug("Could not read registry_url from user config: %s", e)
        return None

    def _load_cache(self):
        """Load cached module registry"""
        if self.registry_cache.exists():
            try:
                with open(self.registry_cache, 'r') as f:
                    data = json.load(f)
                    self._cached_modules = {
                        name: ModuleInfo(**info)
                        for name, info in data.items()
                    }
            except Exception as e:
                logger.warning(f"Failed to load module registry cache: {e}")

    def _save_cache(self):
        """Save module registry to cache"""
        try:
            cache_data = {
                name: asdict(info)
                for name, info in self._cached_modules.items()
            }
            with open(self.registry_cache, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save module registry cache: {e}")

    async def fetch_remote_modules(self) -> List[ModuleInfo]:
        """Fetch available modules from remote registry"""
        if self.debug_mode:
            return self._get_debug_modules()

        try:
            url = self.remote_registry_url
            if url.startswith(("http://", "https://")):
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                modules_data = response.json()
            else:
                # Local path to a registry.json (e.g., a sibling repo checkout)
                path = Path(url)
                if path.is_dir():
                    path = path / "registry.json"
                with open(path, 'r', encoding='utf-8') as f:
                    modules_data = json.load(f)
            modules = [ModuleInfo(**module) for module in modules_data['modules']]

            # The registry file is the source of truth: replace the cache
            # wholesale so stale entries (renamed/removed modules) don't linger.
            self._cached_modules = {module.name: module for module in modules}
            self._save_cache()

            return modules

        except Exception as e:
            logger.error(f"Failed to fetch remote modules: {e}")
            # Fallback to cached modules
            return list(self._cached_modules.values())

    def _get_debug_modules(self) -> List[ModuleInfo]:
        """Debug mode: Generate module info from local ltts_modules directory"""
        debug_modules = []

        # Scan every directory modules can be loaded from
        seen = set()
        for directory in self.module_search_paths:
            if not directory.exists():
                continue
            for module_path in directory.iterdir():
                if not module_path.is_dir() or module_path.name.startswith('.'):
                    continue
                if module_path.name in seen:
                    continue
                seen.add(module_path.name)

                module_info = self._analyze_local_module(module_path)
                if module_info:
                    debug_modules.append(module_info)

        # Add some "remote" modules for testing
        debug_modules.extend(self._get_mock_remote_modules())

        # Update cache
        for module in debug_modules:
            self._cached_modules[module.name] = module
        self._save_cache()

        return debug_modules

    def _analyze_local_module(self, module_path: Path) -> Optional[ModuleInfo]:
        """Analyze local module and extract metadata"""
        init_file = module_path / "init.py"
        if not init_file.exists():
            return None

        try:
            # Read module metadata
            with open(init_file, 'r') as f:
                content = f.read()

            # Extract METADATA dict (simple parsing)
            metadata = self._extract_metadata_from_content(content)
            if not metadata:
                return None

            return ModuleInfo(
                name=metadata.get('name', module_path.name),
                version=metadata.get('version', '1.0.0'),
                description=metadata.get('description', 'Local LTTS module'),
                author=metadata.get('author', 'Local Developer'),
                event_types=metadata.get('event_types', []),
                dependencies=metadata.get('dependencies', []),
                tags=metadata.get('tags', ['local']),
                repository_url=f"file://{module_path}",
                module_path=module_path.name,
                size_bytes=self._module_size_bytes(module_path),
                created_at=datetime.fromtimestamp(module_path.stat().st_ctime).isoformat(),
                updated_at=datetime.fromtimestamp(module_path.stat().st_mtime).isoformat(),
                installed=True,
                local_path=str(module_path)
            )

        except Exception as e:
            logger.warning(f"Failed to analyze module {module_path.name}: {e}")
            return None

    @staticmethod
    def _module_size_bytes(module_path: Path) -> int:
        """Module size without walking per-module venvs and caches."""
        total = 0
        skip = {"venv", "__pycache__", ".git"}
        for dirpath, dirnames, filenames in os.walk(module_path):
            dirnames[:] = [d for d in dirnames if d not in skip]
            for filename in filenames:
                try:
                    total += (Path(dirpath) / filename).stat().st_size
                except OSError:
                    continue
        return total

    def _extract_metadata_from_content(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract METADATA dict from module content"""
        try:
            # AST-based extraction is robust to nested dicts and comments
            # (a regex stops at the first '}', dropping any nested mapping).
            tree = ast.parse(content)
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "METADATA":
                        # literal_eval only accepts plain literals, so scanning
                        # an untrusted module file cannot execute code
                        return ast.literal_eval(node.value)

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return None

    def _get_mock_remote_modules(self) -> List[ModuleInfo]:
        """Generate mock remote modules for debug mode"""
        return [
            ModuleInfo(
                name="gradient_flow_analyzer",
                version="1.2.0",
                description="Analyzes gradient flow patterns through transformer layers",
                author="LTTS Community",
                event_types=["layer_before", "layer_after"],
                dependencies=["gradients"],
                tags=["analysis", "gradients", "debugging"],
                repository_url="https://github.com/seruva19/LTTS-modules.git",
                module_path="gradient_flow_analyzer",
                size_bytes=45678,
                created_at="2024-01-15T10:30:00Z",
                updated_at="2024-02-20T14:45:00Z",
                installed=False
            ),
            ModuleInfo(
                name="token_importance_tracker",
                version="2.0.1",
                description="Tracks token importance scores across model layers",
                author="Research Team",
                event_types=["model_before", "model_after"],
                dependencies=["embeddings", "attention"],
                tags=["tokens", "importance", "analysis"],
                repository_url="https://github.com/seruva19/LTTS-modules.git",
                module_path="token_importance_tracker",
                size_bytes=32145,
                created_at="2024-02-01T09:15:00Z",
                updated_at="2024-03-10T16:20:00Z",
                installed=False
            ),
            ModuleInfo(
                name="layer_similarity_matrix",
                version="1.0.3",
                description="Generates similarity matrices between layer representations",
                author="LTTS Core Team",
                event_types=["layer_after"],
                dependencies=["embeddings"],
                tags=["similarity", "layers", "visualization"],
                repository_url="https://github.com/seruva19/LTTS-modules.git",
                module_path="layer_similarity_matrix",
                size_bytes=28934,
                created_at="2024-01-20T11:00:00Z",
                updated_at="2024-02-15T13:30:00Z",
                installed=False
            )
        ]

    def get_available_modules(self) -> List[ModuleInfo]:
        """Get all available modules (installed + remote)"""
        return list(self._cached_modules.values())

    def get_installed_modules(self) -> List[ModuleInfo]:
        """Get only installed modules"""
        return [module for module in self._cached_modules.values() if module.installed]

    def get_remote_modules(self) -> List[ModuleInfo]:
        """Get only remote (not installed) modules"""
        return [module for module in self._cached_modules.values() if not module.installed]

    async def get_registry_status(self) -> Dict[str, Any]:
        """Compare the registry with locally installed modules.

        Returns counts for the status bar: how many modules the registry
        offers, how many are installed locally, and for how many of those
        the registry advertises a newer version.
        """
        registry_modules = {m.name: m for m in await self.fetch_remote_modules()}

        installed: Dict[str, str] = {}
        for directory in self.module_search_paths:
            if not directory.exists():
                continue
            for module_path in directory.iterdir():
                if not module_path.is_dir() or module_path.name.startswith("."):
                    continue
                if module_path.name in installed:
                    continue  # an earlier search path already provides it
                info = self._analyze_local_module(module_path)
                if info is not None:
                    # Registry keys are folder names; METADATA['name'] may be
                    # a human-readable title instead.
                    installed[module_path.name] = info.version

        updates = sorted(
            name
            for name, version in installed.items()
            if name in registry_modules
            and _version_tuple(registry_modules[name].version)
            > _version_tuple(version)
        )
        not_installed = sorted(
            name for name in registry_modules if name not in installed
        )
        return {
            "registry_url": self.remote_registry_url,
            "registry_count": len(registry_modules),
            "installed_count": len(installed),
            "updates_available": len(updates),
            "update_modules": updates,
            "not_installed": not_installed,
        }

    async def install_module(self, module_name: str) -> bool:
        """Install a module"""
        if module_name not in self._cached_modules:
            logger.error(f"Module {module_name} not found in registry")
            return False

        module_info = self._cached_modules[module_name]

        if module_info.installed:
            logger.info(f"Module {module_name} already installed")
            return True

        try:
            if self.debug_mode:
                # Debug mode: simulate installation
                logger.info(f"[DEBUG] Simulating installation of {module_name}")
                module_info.installed = True
                module_info.local_path = str(self.modules_dir / module_name)
                self._save_cache()
                return True
            else:
                # Production mode: actual download and installation
                return await self._download_and_install_module(module_info)

        except Exception as e:
            logger.error(f"Failed to install module {module_name}: {e}")
            return False

    async def _download_and_install_module(self, module_info: ModuleInfo) -> bool:
        """Download and install module from git repository using sparse checkout"""
        target_dir = self.modules_dir / module_info.name
        temp_repo_dir = self.cache_dir / f"temp_repo_{module_info.name}"

        try:
            # Remove temp directory if it exists
            if temp_repo_dir.exists():
                shutil.rmtree(temp_repo_dir)

            # Remove target directory if it exists
            if target_dir.exists():
                shutil.rmtree(target_dir)

            logger.info(f"Cloning module {module_info.name} from {module_info.repository_url}")

            # Initialize git repository with sparse checkout
            temp_repo_dir.mkdir(parents=True, exist_ok=True)

            # Git commands to sparse checkout only the specific module
            commands = [
                ["git", "init"],
                ["git", "remote", "add", "-f", "origin", module_info.repository_url],
                ["git", "config", "core.sparseCheckout", "true"],
            ]

            # Add the module path to sparse-checkout
            sparse_checkout_file = temp_repo_dir / ".git" / "info" / "sparse-checkout"

            # Execute initial git commands
            for cmd in commands:
                result = subprocess.run(
                    cmd,
                    cwd=str(temp_repo_dir),
                    capture_output=True,
                    text=True,
                    check=True
                )
                logger.debug(f"Git command output: {result.stdout}")

            # Write sparse-checkout configuration
            sparse_checkout_file.parent.mkdir(parents=True, exist_ok=True)
            with open(sparse_checkout_file, 'w') as f:
                f.write(f"{module_info.module_path}/\n")

            # Pull the specific module
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(temp_repo_dir),
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"Git pull output: {result.stdout}")

            # Move the module from temp directory to ltts_modules
            source_module_path = temp_repo_dir / module_info.module_path
            if not source_module_path.exists():
                # Try 'master' branch if 'main' doesn't work
                logger.info("Trying 'master' branch...")
                result = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=str(temp_repo_dir),
                    capture_output=True,
                    text=True,
                    check=True
                )

            if source_module_path.exists():
                shutil.copytree(source_module_path, target_dir)
                logger.info(f"Module {module_info.name} copied to {target_dir}")
            else:
                raise FileNotFoundError(f"Module path {module_info.module_path} not found in repository")

            # Clean up temp directory
            shutil.rmtree(temp_repo_dir)

            # Update module info
            module_info.installed = True
            module_info.local_path = str(target_dir)
            self._save_cache()

            # Create virtual environment if available
            if self.venv_manager:
                logger.info(f"Creating virtual environment for {module_info.name}")
                self.venv_manager.create_module_venv(module_info.name)

                # Install requirements if they exist
                requirements_file = target_dir / "requirements.txt"
                if requirements_file.exists():
                    logger.info(f"Installing requirements for {module_info.name}")
                    self.venv_manager.install_requirements(module_info.name)

            logger.info(f"Successfully installed module {module_info.name}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Git command failed for module {module_info.name}: {e.stderr}")
            # Clean up on error
            if temp_repo_dir.exists():
                shutil.rmtree(temp_repo_dir)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            return False
        except Exception as e:
            logger.error(f"Failed to download and install module {module_info.name}: {e}")
            # Clean up on error
            if temp_repo_dir.exists():
                shutil.rmtree(temp_repo_dir)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            return False
    def uninstall_module(self, module_name: str) -> bool:
        """Uninstall a module and its virtual environment"""
        if module_name not in self._cached_modules:
            return False

        module_info = self._cached_modules[module_name]

        if not module_info.installed:
            return True

        try:
            # Remove module directory
            module_dir = self.modules_dir / module_name
            if module_dir.exists():
                shutil.rmtree(module_dir)

            # Update module info
            module_info.installed = False
            module_info.local_path = None

            # Update cache
            self._save_cache()

            logger.info(f"Successfully uninstalled module: {module_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstall module {module_name}: {e}")
            return False

# Global registry instance
_registry: Optional[ModuleRegistry] = None

def get_module_registry(project_root: str = ".", debug_mode: bool = None) -> ModuleRegistry:
    """Get global module registry instance.

    Use the configured registry by default. Local mock discovery is enabled
    only by ``LTTS_REGISTRY_DEBUG=1``.
    """
    global _registry
    if _registry is None:
        if debug_mode is None:
            debug_mode = os.environ.get("LTTS_REGISTRY_DEBUG", "").strip().lower() in (
                "1",
                "true",
                "yes",
            )
        _registry = ModuleRegistry(project_root, debug_mode)
    return _registry
