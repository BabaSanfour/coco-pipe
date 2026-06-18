"""Low-level, domain-agnostic serialization and file-reading primitives.

Single home for the byte-level IO shared across coco-pipe:

* tabular reads (CSV / parquet) — :func:`read_table`
* opaque blob reads (pkl / npy / json / h5) — :func:`smart_reader`
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
from pathlib import Path
from typing import Any, Dict

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
        read_kwargs: Dict[str, Any] = {"encoding": "utf-8"}
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


def smart_reader(path: Path) -> Any:
    """Read an opaque embedding blob, dispatching on file extension.

    Supports ``.pkl``, ``.npy``, ``.json``, and ``.h5``/``.hdf5``. HDF5 files
    return the ``embeddings`` dataset, then ``data``, then the sole dataset if
    only one is present; otherwise the structure is ambiguous and a custom
    reader is required.
    """
    suffix = path.suffix.lower()
    if suffix == ".pkl":
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
    elif suffix == ".npy":
        return np.load(path)
    elif suffix == ".json":
        return read_json(path)
    elif suffix in [".h5", ".hdf5"]:
        import h5py

        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            if "embeddings" in keys:
                return f["embeddings"][:]
            elif "data" in keys:
                return f["data"][:]
            elif len(keys) == 1:
                return f[keys[0]][:]
            else:
                raise ValueError(
                    f"Ambiguous HDF5 structure: {keys}. Use custom reader."
                )
    else:
        raise ValueError(f"Unsupported extension {suffix}, utilize custom reader.")


def default_id_extractor(path: Path) -> str:
    """Extract a subject ID from a filename, honouring a ``sub-`` token."""
    parts = path.name.split("_")
    for p in parts:
        if p.startswith("sub-"):
            return p.replace("sub-", "")
    return path.stem


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
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    else:
        path.write_text(text, encoding="utf-8")
    return path


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
