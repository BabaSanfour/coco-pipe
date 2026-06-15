"""Type aliases and runtime constants shared across the io layer.

Centralises the small set of domain literals (``QualityStatus``,
``QCFlagLevel``) and the status-ordering dict so that :mod:`coco_pipe.io.quality`
stays focused on logic and tests can import the aliases without pulling in the
full module dependency tree.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

QualityStatus = Literal["OK", "WARN", "FAIL"]
"""Status value produced by :class:`~coco_pipe.io.quality.CheckResult`."""

QCFlagLevel = Literal["pass", "warn", "fail"]
"""Severity level for QC flag dicts produced by ``make_qc_flag``."""

QualityInput = pd.DataFrame | np.ndarray
"""Accepted input type for column-level quality checks."""

_STATUS_ORDER: dict[str, int] = {"pass": 0, "warn": 1, "fail": 2}
"""Ordinal ranking used by ``resolve_qc_status`` to pick the worst level."""

GROUP_BY_COLUMN: dict[str, str] = {
    "family": "family",
    "subfamily": "subfamily",
    "measure": "measure",
    "feature": "column",
}
"""Maps a ``group_by`` granularity to the ``classify_descriptor_columns`` column
that carries its label. Shared by the QC drop path and downstream callers so the
set of valid ``group_by`` values lives in one place."""
