"""
Shared package utilities.

This module holds small helpers that are not specific to one subpackage.
"""

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Optional

import joblib

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
    "stable_hash",
    "slug",
    "resolve_n_jobs",
    "run_task_batch",
]


def stable_hash(value: Any, *, length: int = 64) -> str:
    """Return a deterministic SHA-256 prefix for a JSON-compatible value.

    Dictionaries are serialized with sorted keys and compact separators. Values
    such as paths that are not directly JSON serializable fall back to ``str``.
    """
    if not 1 <= length <= 64:
        raise ValueError("length must be between 1 and 64.")
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


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


def slug(value: object, *, max_len: int = 80) -> str:
    """Return a filesystem-safe slug from an arbitrary value.

    Collapses runs of non-alphanumeric characters (except ``.``, ``_``, ``=``,
    ``-``) into a single ``-``, strips leading/trailing punctuation, and
    truncates at *max_len*.

    Parameters
    ----------
    value:
        Any object; ``str(value)`` is used as the source text.
    max_len:
        Maximum character length of the returned slug.
    """
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9._=-]+", "-", text)
    text = text.strip("-._")
    if not text:
        text = "none"
    return text[:max_len]


def resolve_n_jobs(n_jobs: int) -> int:
    """Resolve ``n_jobs`` to a concrete positive integer.

    ``-1`` maps to ``os.cpu_count()`` (minimum 1).  Any other value must
    already be a positive integer, or :class:`ValueError` is raised.
    """
    if n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    if n_jobs < 1:
        raise ValueError("n_jobs must be -1 or a positive integer.")
    return n_jobs


def run_task_batch(
    tasks: Sequence[Any],
    worker_fn: Callable[[Any], Any],
    max_workers: int,
) -> list[Any]:
    """Execute *tasks* with *worker_fn*, optionally in parallel.

    When *max_workers* is 1 the tasks are run serially in the current process.
    For any larger value :func:`joblib.Parallel` is used with
    ``n_jobs=min(max_workers, len(tasks))`` so the pool size never exceeds the
    actual work to do.

    Parameters
    ----------
    tasks:
        Sequence of opaque task objects passed one-by-one to *worker_fn*.
    worker_fn:
        Single-argument callable that processes one task and returns a result.
    max_workers:
        Maximum number of parallel workers.  Pass ``1`` for serial execution.

    Returns
    -------
    list
        Results in the same order as *tasks*.
    """
    if not tasks:
        return []
    if max_workers == 1:
        return [worker_fn(task) for task in tasks]
    return joblib.Parallel(n_jobs=min(max_workers, len(tasks)))(
        joblib.delayed(worker_fn)(task) for task in tasks
    )
