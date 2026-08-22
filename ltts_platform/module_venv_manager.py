import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Optional, List, Set
import site
import venv
import logging

logger = logging.getLogger(__name__)


class ModuleVenvManager:
    """Manages virtual environments for modules while maintaining access to shared packages"""

    def __init__(self, project_root: str):
        """
        Initialize the module virtual environment manager

        Args:
            project_root: Path to the project root directory
        """
        self.project_root = Path(project_root)
        self.shared_venv = self.project_root / "venv"
        try:
            from .module_paths import resolve_modules_dir, resolve_module_search_paths
        except ImportError:  # pragma: no cover - direct script execution
            from module_paths import resolve_modules_dir, resolve_module_search_paths
        self.modules_dir = resolve_modules_dir(self.project_root)
        self.module_search_paths = resolve_module_search_paths(self.project_root)

        # Verify shared venv exists
        if not self.shared_venv.exists():
            raise ValueError(
                f"Shared virtual environment not found at {self.shared_venv}"
            )

    def _module_dir(self, module_name: str):
        """Where this module lives - source checkout or install directory."""
        for directory in self.module_search_paths:
            candidate = directory / module_name
            if candidate.is_dir():
                return candidate
        return self.modules_dir / module_name

    def _get_shared_site_packages(self) -> str:
        """Get the path to site-packages in the shared virtual environment"""
        if sys.platform == "win32":
            return str(self.shared_venv / "Lib" / "site-packages")
        return str(
            self.shared_venv
            / "lib"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
            / "site-packages"
        )

    def _create_venv_with_shared_packages(self, venv_dir: Path) -> None:
        """
        Create a virtual environment that has access to the shared venv's packages

        Args:
            venv_dir: Path where the virtual environment should be created
        """
        # Create the virtual environment
        venv.create(venv_dir, with_pip=True)

        # Get the path to the new venv's site-packages
        if sys.platform == "win32":
            module_site_packages = venv_dir / "Lib" / "site-packages"
        else:
            module_site_packages = (
                venv_dir
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
            )

        # Create a .pth file to add the shared site-packages to the module's Python path
        shared_site_packages = self._get_shared_site_packages()
        pth_file = module_site_packages / "shared_venv.pth"
        with open(pth_file, "w") as f:
            f.write(shared_site_packages + "\n")

    def _check_requirements_installed(
        self, venv_dir: Path, requirements: List[str]
    ) -> Set[str]:
        """
        Check which requirements are not yet installed in the venv using pip list.

        Args:
            venv_dir: Path to virtual environment
            requirements: List of package specifications

        Returns:
            Set of requirements that need to be installed
        """
        if sys.platform == "win32":
            pip_path = venv_dir / "Scripts" / "pip.exe"
        else:
            pip_path = venv_dir / "bin" / "pip"

        if not pip_path.exists():
            # Venv doesn't exist yet, all requirements needed
            return set(requirements)

        try:
            # Get all installed packages at once using pip list
            result = subprocess.run(
                [str(pip_path), "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # If pip list fails, assume all packages need installation
                logger.debug(f"pip list failed, installing all requirements")
                return set(requirements)

            # Parse installed packages
            installed_packages = {}
            try:
                packages_list = json.loads(result.stdout)
                # Create a dict of installed package names (normalized to lowercase)
                installed_packages = {
                    pkg["name"].lower().replace("-", "_"): pkg["version"]
                    for pkg in packages_list
                }
            except (json.JSONDecodeError, KeyError) as e:
                logger.debug(f"Failed to parse pip list output: {e}")
                return set(requirements)

            # Check which requirements are missing
            missing = set()
            for req in requirements:
                # Parse package name (before any version specifiers)
                package_name = (
                    req.split(">=")[0]
                    .split("==")[0]
                    .split("<")[0]
                    .split(">")[0]
                    .split("[")[0]  # Handle extras like package[extra]
                    .strip()
                    .lower()
                    .replace("-", "_")
                )

                if package_name not in installed_packages:
                    missing.add(req)

            return missing

        except (subprocess.TimeoutExpired, Exception) as e:
            logger.debug(f"Error checking installed packages: {e}")
            # If check fails, assume all packages need to be installed
            return set(requirements)

    def create_module_venv(
        self, module_name: str, requirements: Optional[List[str]] = None
    ) -> None:
        """
        Create a new virtual environment for a module

        Args:
            module_name: Name of the module
            requirements: List of additional requirements for the module
        """
        module_dir = self._module_dir(module_name)
        venv_dir = module_dir / "venv"

        # Create module directory if it doesn't exist
        module_dir.mkdir(parents=True, exist_ok=True)

        # Check if venv needs to be created
        venv_exists = venv_dir.exists()
        if not venv_exists:
            print(f"creating venv... ", end="", flush=True)
            # Create virtual environment with access to shared packages
            self._create_venv_with_shared_packages(venv_dir)

        # Handle requirements
        if requirements:
            # Check which requirements are missing
            missing_reqs = self._check_requirements_installed(venv_dir, requirements)

            if missing_reqs:
                print(f"installing {len(missing_reqs)} pkg... ", end="", flush=True)

                # Create requirements.txt with missing requirements
                requirements_file = module_dir / "requirements.txt"
                with open(requirements_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(requirements))

                # Install requirements
                if sys.platform == "win32":
                    pip_path = venv_dir / "Scripts" / "pip.exe"
                else:
                    pip_path = venv_dir / "bin" / "pip"

                # Run pip with captured output (hide verbose installation messages)
                result = subprocess.run(
                    [str(pip_path), "install", "-q", "-r", str(requirements_file)],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    # Show error output if installation failed
                    error_msg = (
                        result.stderr[:200] if result.stderr else "unknown error"
                    )
                    raise subprocess.CalledProcessError(
                        result.returncode, result.args, stderr=error_msg
                    )
        else:
            logger.debug(f"ℹ No requirements specified for {module_name}")

    def get_module_python_path(self, module_name: str) -> str:
        """
        Get the path to the Python interpreter for a specific module

        Args:
            module_name: Name of the module

        Returns:
            Path to the module's Python interpreter
        """
        venv_dir = self._module_dir(module_name) / "venv"
        if sys.platform == "win32":
            return str(venv_dir / "Scripts" / "python.exe")
        return str(venv_dir / "bin" / "python")

    def install_requirements(self, module_name: str) -> None:
        """
        Install requirements for a module

        Args:
            module_name: Name of the module
        """
        module_dir = self._module_dir(module_name)
        requirements_file = module_dir / "requirements.txt"

        if not requirements_file.exists():
            return

        python_path = self.get_module_python_path(module_name)
        if sys.platform == "win32":
            pip_path = Path(python_path).parent / "pip.exe"
        else:
            pip_path = Path(python_path).parent / "pip"

        subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_file)], check=True
        )


def main():
    """Command-line interface for the module virtual environment manager"""
    import argparse

    parser = argparse.ArgumentParser(description="Manage module virtual environments")
    parser.add_argument("--module", type=str, required=True, help="Module name")
    parser.add_argument(
        "--action",
        type=str,
        choices=["create", "install"],
        required=True,
        help="Action to perform",
    )
    parser.add_argument(
        "--project-root", type=str, default=".", help="Path to project root directory"
    )

    args = parser.parse_args()

    manager = ModuleVenvManager(args.project_root)

    if args.action == "create":
        manager.create_module_venv(args.module)
    elif args.action == "install":
        manager.install_requirements(args.module)


if __name__ == "__main__":
    main()
