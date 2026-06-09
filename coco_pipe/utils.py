"""
Shared package utilities.

This module holds small helpers that are not specific to one subpackage.
"""

import datetime as dt
import importlib.metadata
import os
import platform
import shlex
import subprocess
import sys
from collections.abc import Mapping
from typing import Any, Optional

PACKAGE_VERSIONS: Mapping[str, str] = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "plotly": "plotly",
    "pydantic": "pydantic",
    "jinja2": "jinja2",
    "umap-learn": "umap-learn",
    "trimap": "trimap",
    "phate": "phate",
    "pacmap": "pacmap",
    "faiss-cpu": "faiss-cpu",
    "dask": "dask",
    "dask-ml": "dask-ml",
    "tensorflow": "tensorflow",
    "keras": "keras",
    "torch": "torch",
    "skorch": "skorch",
    "gudhi": "gudhi",
    "pydmd": "pydmd",
    "mne": "mne",
    "mne-bids": "mne-bids",
    "meegkit": "meegkit",
    "specparam": "specparam",
    "antropy": "antropy",
    "neurokit2": "neurokit2",
    "braindecode": "braindecode",
    "transformers": "transformers",
    "huggingface-hub": "huggingface-hub",
    "accelerate": "accelerate",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
}

__all__ = [
    "PACKAGE_VERSIONS",
    "get_environment_info",
    "get_git_revision_hash",
    "get_package_version",
    "import_optional_dependency",
]


def import_optional_dependency(
    loader: Any,
    feature: str,
    dependency: str,
    install_hint: Optional[str] = None,
) -> Any:
    """
    Lazily import an optional dependency with clearer failure modes.

    Parameters
    ----------
    loader : callable
        Zero-argument callable returning the imported dependency.
    feature : str
        Feature or component name using the dependency.
    dependency : str
        Human-readable dependency name.
    install_hint : str, optional
        Installation hint shown on ImportError.

    Returns
    -------
    Any
        Imported dependency returned by ``loader``.

    Raises
    ------
    ImportError
        If the dependency is not installed.
    RuntimeError
        If the dependency is installed but fails during initialization.
    """
    try:
        return loader()
    except ImportError as exc:
        msg = f"{dependency} is required for {feature}."
        if install_hint:
            msg += f" Install it with '{install_hint}'."
        raise ImportError(msg) from exc
    except Exception as exc:
        raise RuntimeError(
            f"{dependency} failed to initialize for {feature}: {exc}"
        ) from exc


def get_git_revision_hash(cwd: str | os.PathLike[str] | None = None) -> str:
    """Return the current short git hash, or ``"Unknown"`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1,
            cwd=cwd,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (
        FileNotFoundError,
        NotADirectoryError,
        PermissionError,
        subprocess.TimeoutExpired,
    ):
        pass
    return "Unknown"


def get_package_version(package_name: str) -> str:
    """Return an installed distribution version, or ``"Unknown"``."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "Unknown"


def get_environment_info(
    cwd: str | os.PathLike[str] | None = None,
    packages: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Capture runtime provenance metadata for reports and experiment results."""
    version_packages = packages or PACKAGE_VERSIONS
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "command": shlex.join(sys.argv),
        "git_hash": get_git_revision_hash(cwd=cwd),
        "coco_pipe_version": get_package_version("coco-pipe"),
        "versions": {
            label: get_package_version(distribution)
            for label, distribution in version_packages.items()
        },
    }
