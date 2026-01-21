"""
Provenance Capture
==================

Utilities for capturing reproducibility metadata (Git hash, environment, versions).
"""

import datetime
import importlib.metadata
import platform
import subprocess
import sys
from typing import Any, Dict


def get_git_revision_hash() -> str:
    """
    Return the current git hash if available.

    Returns
    -------
    str
        Short git hash (e.g. "a1b2c3d") or "Unknown".
    """
    try:
        # We need to run this from the package root ideally, or CWD
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "Unknown"


def get_package_version(package_name: str) -> str:
    """
    Safely get package version.

    Parameters
    ----------
    package_name : str
        Name of the pip package.

    Returns
    -------
    str
        Version string or "Unknown".
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "Unknown"


def get_environment_info() -> Dict[str, Any]:
    """
    Capture runtime environment information for reproducibility.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing timestamp, os, python version, git hash, etc.

    Examples
    --------
    >>> info = get_environment_info()
    >>> print(info['git_hash'])
    'a1b2c3d'
    """
    info = {
        "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "os_platform": platform.platform(),
        "python_version": platform.python_version(),
        "command": " ".join(sys.argv),
        "git_hash": get_git_revision_hash(),
        "coco_pipe_version": "0.0.1",  # TODO: fetch dynamically if setup properly
        "versions": {
            "numpy": get_package_version("numpy"),
            "pandas": get_package_version("pandas"),
            "scipy": get_package_version("scipy"),
            "plotly": get_package_version("plotly"),
        },
    }
    return info
