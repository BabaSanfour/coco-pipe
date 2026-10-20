"""Low-level, domain-agnostic serialization and file-reading primitives.

Single home for the byte-level IO shared across coco-pipe:

* tabular reads (CSV / parquet) — :func:`read_table`
* JSON read / write — :func:`read_json`, :func:`write_json`
* joblib object persistence — :func:`save_object`, :func:`load_object`
* atomic NPZ writes — :func:`save_npz`

Domain modules build their schema-specific ``save`` / ``load`` on top of these
so the actual disk operations (atomic temp-file replacement, extension handling,
type checks) live in exactly one place.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def read_table(
    path: Path | str,
    sep: str | None = None,
    drop_all_empty: bool = True,
) -> pd.DataFrame:
    """Read a CSV or parquet file into a DataFrame.

    CSV delimiters are auto-detected when ``sep`` is omitted. Unnamed and
    entirely empty columns caused by trailing separators are removed.

    Parameters
    ----------
    path : Path or str
        Path to a ``.csv`` or ``.parquet`` file.
    sep : str, optional
        Explicit CSV delimiter. When omitted, pandas' Python engine detects it.
    drop_all_empty : bool, default=True
        Whether to remove entirely empty named columns. Descriptor loading
        disables this so column-pruning QC can record all-NaN features.

    Returns
    -------
    pandas.DataFrame
        The cleaned table.

    Raises
    ------
    ValueError
        If the file extension is unsupported.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        read_kwargs: dict[str, Any] = {"encoding": "utf-8"}
        if sep is None:
            read_kwargs.update({"sep": None, "engine": "python"})
        else:
            read_kwargs.update({"sep": sep, "low_memory": False})
        df = pd.read_csv(path, **read_kwargs)
    else:
        raise ValueError(
            f"Unsupported table format '{path.suffix}'. Expected .csv or .parquet."
        )

    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    return df.dropna(axis=1, how="all") if drop_all_empty else df


def read_json(path: Path | str) -> Any:
    """Read and decode a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(
    path: Path | str,
    obj: Any,
    atomic: bool = True,
    indent: int = 2,
    sort_keys: bool = False,
) -> Path:
    """Serialize ``obj`` to a UTF-8 JSON file, creating parent directories.

    When ``atomic`` is True the payload is written to a temporary sibling and
    moved into place with :func:`os.replace`, so readers never observe a
    partially written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys)
    if atomic:
        with atomic_path(path) as tmp:
            tmp.write_text(text, encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    return path


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary sibling of ``path`` to write, then move it onto ``path``.

    Each call gets its own temporary name, so processes writing the same file
    at once do not move each other's half-written file into place; the last
    one to finish wins. On an error the temporary file is removed and
    ``path`` is left as it was.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def save_object(obj: Any, path: Path | str) -> Path:
    """Serialize a Python object to disk with joblib, creating parent dirs."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_object(path: Path | str, expected_type: type | None = None) -> Any:
    """Load a joblib-serialized object, optionally enforcing its type.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If ``expected_type`` is given and the loaded object is not an instance.
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    obj = joblib.load(path)
    if expected_type is not None and not isinstance(obj, expected_type):
        raise TypeError(
            f"Loaded object is {type(obj)}, expected {expected_type.__name__}"
        )
    return obj


def save_npz(
    path: Path | str,
    atomic: bool = True,
    compressed: bool = True,
    **arrays: Any,
) -> Path:
    """Write named arrays to an ``.npz`` archive, optionally atomically.

    ``path`` should end in ``.npz`` (NumPy appends the extension otherwise,
    which would desync the returned path from the file actually written). With
    ``atomic`` the archive is built in a temporary sibling and moved into place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = np.savez_compressed if compressed else np.savez
    if atomic:
        tmp = path.with_name(f".{path.name}.tmp.npz")
        writer(tmp, **arrays)
        os.replace(tmp, path)
    else:
        writer(path, **arrays)
    return path
