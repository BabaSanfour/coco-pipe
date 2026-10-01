"""
Data Structures
===============

Standardized containers for passing data between Datasets, Preprocessing, and main
modules.

This module provides the `~coco_pipe.io.DataContainer`, an N-dimensional tensor wrapper
that manages metadata, coordinates, and labels alongside the raw data matrix. It serves
as the common currency for the entire pipeline.

Examples
--------
>>> import numpy as np
>>> from coco_pipe.io import DataContainer

# 1. Creating a container for EEG Epochs (N_epochs, N_channels, N_time)
>>> X = np.random.randn(10, 64, 500)
>>> container = DataContainer(
...     X=X,
...     dims=("obs", "channel", "time"),
...     coords={
...         "channel": ["Fz", "Cz", "Pz"],  # ... etc
...         "time": np.linspace(0, 1.0, 500),
...     },
...     y=np.random.randint(0, 2, 10),
...     ids=[f"sub-01_trial-{i}" for i in range(10)],
... )

# 2. Creating a container for simple Tabular Features (N_subjects, N_features)
>>> X_tab = np.random.randn(20, 5)
>>> container_tab = DataContainer(
...     X=X_tab,
...     dims=("obs", "feature"),
...     coords={"feature": ["age", "IQ", "response_time", "power_alpha", "power_beta"]},
... )
"""

import difflib
import fnmatch
import itertools
import logging
import re
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .utils import make_strata, sample_indices

logger = logging.getLogger(__name__)


@dataclass
class DataContainer:
    """
    Generic container for N-dimensional neurophysiological data.

    Acts as a lightweight labelled array (like xarray but simpler), managing
    dimensions, coordinates, and associated target labels (y) and IDs.

    Attributes
    ----------
    X : np.ndarray
        The primary data tensor. Shape must match `dims`.
    dims : Tuple[str, ...]
        Labels for each dimension of X.
        Examples: ('obs', 'feature'), ('obs', 'channel', 'time').
        Note: The 'obs' dimension is special and typically represents independent
        samples.
    coords : Dict[str, Union[List, np.ndarray]]
        Coordinates/Labels for dimensions. Keys must be in `dims`.
        Values must match the length of the corresponding dimension in X.
    y : Optional[np.ndarray], optional
        Target labels corresponding to the 'obs' dimension.
        Used for supervised learning or coloring plots.
    ids : Optional[np.ndarray], optional
        Identifiers for observations (e.g., subject IDs, trial names).
        Should correspond to 'obs' dim in coords if provided.
        Kept separate from coords for convenient tracking.
    meta : Dict[str, Any]
        Arbitrary metadata (sfreq, units, source path, etc).

    Examples
    --------
    Accessing data:
    >>> container.X.shape
    (10, 64, 500)

    Accessing coordinates:
    >>> container.coords["channel"][:3]
    ['Fz', 'Cz', 'Pz']
    """

    X: np.ndarray
    dims: tuple[str, ...]
    coords: dict[str, list | np.ndarray | Sequence] = field(default_factory=dict)
    y: np.ndarray | None = None
    ids: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validation
        if self.X.ndim != len(self.dims):
            raise ValueError(
                f"Shape mismatch: X has {self.X.ndim} dims {self.X.shape}, "
                f"but `dims` has {len(self.dims)} labels {self.dims}."
            )

        # Check coords lengths.
        for dim, labels in self.coords.items():
            if dim in self.dims:
                axis = self.dims.index(dim)
                if self.X.shape[axis] != len(labels):
                    logger.debug(
                        f"Coord '{dim}' length ({len(labels)}) does not match "
                        f"X dimension {axis} ({self.X.shape[axis]})."
                    )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.X.shape

    def _coord_axis(self, name: str, labels: Any, obs_dim_idx: int) -> int:
        """Resolve the data axis a coordinate aligns to, or ``-1`` if none.

        Resolution order:

        1. A coordinate named exactly like a dimension aligns to that dimension.
        2. A *prefixed* coordinate ``{dim}_{suffix}`` (e.g. ``feature_family``)
           is metadata for ``{dim}`` and aligns to that axis regardless of
           lengths. This keeps auxiliary feature/sensor metadata bound to the
           right axis even when two dimensions share a length.
        3. Otherwise alignment is inferred by length, preferring the ``obs``
           axis on ties so observation-level metadata is never mis-assigned.
        """
        if name in self.dims:
            return self.dims.index(name)
        prefix_dim = max(
            (dim for dim in self.dims if name.startswith(f"{dim}_")),
            key=len,
            default=None,
        )
        if prefix_dim is not None:
            return self.dims.index(prefix_dim)
        length = len(labels)
        if obs_dim_idx != -1 and length == self.X.shape[obs_dim_idx]:
            return obs_dim_idx
        for axis, dim_len in enumerate(self.X.shape):
            if length == dim_len:
                return axis
        return -1

    def save(self, path: str | Any) -> None:
        """
        Save the DataContainer to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        from ._serialization import save_object

        saved = save_object(self, path)
        logger.info(f"DataContainer saved to {saved}")

    def observation_frame(self) -> pd.DataFrame:
        """Return observation-aligned coordinates and stable sample IDs."""
        if "obs" not in self.dims:
            raise ValueError("DataContainer has no 'obs' dimension.")
        obs_dim_idx = self.dims.index("obs")
        n_obs = self.X.shape[obs_dim_idx]
        columns: dict[str, Any] = {}
        for key, values in self.coords.items():
            array = np.asarray(values)
            if (
                array.ndim == 1
                and len(array) == n_obs
                and self._coord_axis(key, array, obs_dim_idx) == obs_dim_idx
            ):
                columns[str(key)] = array
        if self.ids is not None:
            columns["sample_id"] = np.asarray(self.ids).astype(str)
        elif "sample_id" not in columns:
            columns["sample_id"] = [f"sample-{idx:06d}" for idx in range(n_obs)]
        return pd.DataFrame(columns, index=np.arange(n_obs))

    @classmethod
    def load(cls, path: str | Any) -> "DataContainer":
        """
        Load a DataContainer from disk.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        DataContainer
        """
        from ._serialization import load_object

        return load_object(path, expected_type=cls)

    @classmethod
    def concat(
        cls,
        containers: Sequence["DataContainer"],
        fill_condition_from_meta: bool = True,
    ) -> "DataContainer":
        """Concatenate containers along their observation axis.

        All containers must use matching dimensions and non-observation
        shapes. Observation-aligned coordinates are concatenated, filling
        missing entries with ``None`` when a coordinate is absent from a
        container. Non-observation coordinates are copied from the first
        container.

        Parameters
        ----------
        containers : sequence of DataContainer
            One or more containers to concatenate.
        fill_condition_from_meta : bool, default=True
            If no observation-level ``condition`` coordinate is available,
            create one from each container's ``meta["condition"]`` value.

        Returns
        -------
        DataContainer
            The concatenated container.
        """
        containers = list(containers)
        if not containers:
            raise ValueError("Need at least one container to concatenate.")

        base = containers[0]
        if "obs" not in base.dims:
            raise ValueError("Containers must include an 'obs' dimension.")
        if any(container.dims != base.dims for container in containers[1:]):
            raise ValueError("All containers must have matching dims.")

        obs_axis = base.dims.index("obs")
        non_obs_shape = tuple(
            size for axis, size in enumerate(base.X.shape) if axis != obs_axis
        )
        for container in containers[1:]:
            candidate_shape = tuple(
                size for axis, size in enumerate(container.X.shape) if axis != obs_axis
            )
            if candidate_shape != non_obs_shape:
                raise ValueError(
                    "All containers must have matching non-obs dimensions."
                )

        coords: dict[str, np.ndarray] = {}
        base_n_obs = base.X.shape[obs_axis]

        # Dimension coordinates are non-observation coordinates by definition.
        # Auxiliary vectors are non-observation metadata when any container
        # where they are present shows that they are not obs-aligned.
        non_obs_coord_keys = set()
        for key, values in base.coords.items():
            arr = np.asarray(values)
            if key in base.dims and key != "obs":
                coords[key] = arr
                non_obs_coord_keys.add(key)
                continue
            if arr.ndim != 1:
                continue

            present_values = [
                (
                    np.asarray(container.coords[key]),
                    container.X.shape[obs_axis],
                )
                for container in containers
                if key in container.coords
            ]
            has_non_obs_alignment = any(
                candidate.ndim != 1 or len(candidate) != n_obs
                for candidate, n_obs in present_values
            )
            if len(arr) != base_n_obs or has_non_obs_alignment:
                coords[key] = arr
                non_obs_coord_keys.add(key)

        obs_keys = set()
        for container in containers:
            n_obs = container.X.shape[obs_axis]
            for key, values in container.coords.items():
                arr = np.asarray(values)
                if (
                    key not in non_obs_coord_keys
                    and arr.ndim == 1
                    and len(arr) == n_obs
                ):
                    obs_keys.add(key)

        for key in sorted(obs_keys):
            parts = []
            for container in containers:
                n_obs = container.X.shape[obs_axis]
                if key not in container.coords:
                    parts.append(np.full(n_obs, None, dtype=object))
                    continue

                arr = np.asarray(container.coords[key])
                if arr.ndim != 1 or len(arr) != n_obs:
                    raise ValueError(
                        f"Observation coordinate '{key}' must be 1D and aligned "
                        "to the 'obs' dimension in every container where present."
                    )
                parts.append(arr)
            coords[key] = np.concatenate(parts)

        if fill_condition_from_meta and "condition" not in coords:
            coords["condition"] = np.concatenate(
                [
                    np.full(
                        container.X.shape[obs_axis],
                        container.meta.get("condition"),
                        dtype=object,
                    )
                    for container in containers
                ]
            )

        y = None
        if all(container.y is not None for container in containers):
            y = np.concatenate([np.asarray(container.y) for container in containers])

        ids = None
        if all(container.ids is not None for container in containers):
            ids = np.concatenate(
                [np.asarray(container.ids) for container in containers]
            )

        return cls(
            X=np.concatenate(
                [np.asarray(container.X) for container in containers],
                axis=obs_axis,
            ),
            dims=base.dims,
            coords=coords,
            y=y,
            ids=ids,
            meta={
                "source": "concat",
                "conditions": [
                    container.meta.get("condition") for container in containers
                ],
            },
        )

    def __repr__(self) -> str:
        dim_strs = [f"{d}={s}" for d, s in zip(self.dims, self.X.shape, strict=False)]
        return (
            f"<DataContainer [{' x '.join(dim_strs)}], "
            f"coords={list(self.coords.keys())}>"
        )

    def obs_table(
        self,
        include_ids: bool = False,
        id_col: str = "obs_id",
        include_y: bool = False,
        y_col: str = "y",
        include_obs_coord: bool = False,
    ) -> pd.DataFrame:
        """
        Return one-dimensional coordinates aligned to the observation axis.

        This helper is useful when exporting a row-wise table from a container.
        It only materializes metadata that can map cleanly to one row per
        observation, skipping coordinates that belong to other axes such as
        ``channel``, ``time``, ``feature``, or ``stat``.

        Parameters
        ----------
        include_ids : bool, default=False
            If True, include ``self.ids`` as the first column.
        id_col : str, default="obs_id"
            Column name used when exporting ``self.ids``.
        include_y : bool, default=False
            If True, include ``self.y`` as a column when present.
        y_col : str, default="y"
            Column name used when exporting ``self.y``.
        include_obs_coord : bool, default=False
            If True, include ``coords["obs"]`` when present.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing only one-dimensional observation-aligned
            metadata columns.

        Raises
        ------
        ValueError
            If the container has no ``obs`` dimension, or if ``include_ids`` is
            requested when ``self.ids`` is missing.
        """
        if "obs" not in self.dims:
            raise ValueError("Observation metadata export requires an 'obs' dimension.")

        obs_len = self.X.shape[self.dims.index("obs")]
        data: dict[str, np.ndarray] = {}

        if include_ids:
            if self.ids is None:
                raise ValueError("`include_ids=True` requires `DataContainer.ids`.")
            ids = np.asarray(self.ids, dtype=object)
            if ids.ndim != 1 or len(ids) != obs_len:
                raise ValueError("`DataContainer.ids` must be 1D and aligned to 'obs'.")
            data[id_col] = ids

        if include_obs_coord and "obs" in self.coords:
            obs_coord = np.asarray(self.coords["obs"], dtype=object)
            if obs_coord.ndim == 1 and len(obs_coord) == obs_len:
                data["obs"] = obs_coord

        for key, values in self.coords.items():
            if key == "obs":
                continue
            arr = np.asarray(values, dtype=object)
            if arr.ndim == 1 and len(arr) == obs_len:
                data[key] = arr

        if include_y and self.y is not None:
            y = np.asarray(self.y, dtype=object)
            if y.ndim != 1 or len(y) != obs_len:
                raise ValueError("`DataContainer.y` must be 1D and aligned to 'obs'.")
            if y_col not in data:
                data[y_col] = y

        return pd.DataFrame(data)

    def isel(self, **indexers) -> "DataContainer":
        """
        Select data by integer indices on specified dimensions.

        This method is the integer-index equivalent of `select`. It operates
        directly on the dimensions of the data tensor `X`. It is robust and
        handles metadata splitting/alignment automatically.

        Parameters
        ----------
        **indexers : dict
            Dimension names mapped to the desired index. The index can be:

            - List or numpy array of integers: [0, 1, 5]
            - Slice object: slice(0, 10)
            - Single integer: 0

            Note: If you provide a list of indices with repeats (e.g., [0, 0, 1]),
            the output will be oversampled accordingly.

        Returns
        -------
        DataContainer
            A new DataContainer instance with the sliced data and coordinates.

        Examples
        --------
        >>> # Select first 10 observations
        >>> subset = container.isel(obs=slice(0, 10))

        >>> # Select specific channels by index
        >>> subset = container.isel(channel=[0, 5, 12])

        >>> # Select time range by index
        >>> subset = container.isel(time=slice(100, 200))

        >>> # Bootstrap/Resample (Select index 0 five times)
        >>> bootstrap = container.isel(obs=[0, 0, 0, 0, 0])
        """
        if not indexers:
            return self

        slices = [slice(None)] * self.X.ndim
        obs_dim_idx = self.dims.index("obs") if "obs" in self.dims else -1

        # Build the per-axis slicers from the requested indexers.
        for dim_name, indices in indexers.items():
            if dim_name not in self.dims:
                logger.warning(
                    f"Dimension {dim_name} not in {self.dims}, skipping isel."
                )
                continue
            # Normalize int to list to preserve the dimension.
            if isinstance(indices, int):
                indices = [indices]
            slices[self.dims.index(dim_name)] = indices

        # Re-slice each coordinate exactly once, using its resolved axis. This
        # avoids mis-slicing aux coordinates when two dimensions share a length.
        new_coords = self.coords.copy()
        for name, labels in self.coords.items():
            axis = self._coord_axis(name, labels, obs_dim_idx)
            if axis == -1:
                continue
            sl = slices[axis]
            if not (isinstance(sl, slice) and sl == slice(None)):
                new_coords[name] = np.array(labels)[sl]

        # Orthogonal Application
        try:
            new_X = self.X
            for axis, sl in enumerate(slices):
                if isinstance(sl, slice) and sl == slice(None):
                    continue
                indexer = [slice(None)] * new_X.ndim
                indexer[axis] = sl
                new_X = new_X[tuple(indexer)]
        except Exception as e:
            logger.error(f"Slicing failed with slices {slices}: {e}")
            raise

        new_y = self.y
        new_ids = self.ids

        if obs_dim_idx != -1:
            obs_sl = slices[obs_dim_idx]
            # Slicing y/ids if they exist
            if not (isinstance(obs_sl, slice) and obs_sl == slice(None)):
                if self.y is not None:
                    new_y = self.y[obs_sl]
                if self.ids is not None:
                    new_ids = self.ids[obs_sl]  # ids is numpy array

        return replace(
            self,
            X=new_X,
            y=new_y,
            ids=new_ids,
            coords=new_coords,
            meta=deepcopy(self.meta),
        )

    def balance(
        self,
        target: str = "y",
        strategy: str = "undersample",
        covariates: list[str] | None = None,
        random_state: int = 42,
        **kwargs,
    ) -> "DataContainer":
        """
        Balance the dataset classes using undersampling or oversampling.

        This method adjusts the number of observations (rows) in the container
        so that class counts in `target` are equalized. It supports simple
        random sampling and stratified sampling based on covariates.

        Parameters
        ----------
        target : str or array-like
            The target vector to balance against:

            - 'y': Uses `self.y`.
            - Any other string: Looks for the variable in `self.coords`.
            - Array-like: Direct labels to use.
        method : str, default='auto'
            Balancing strategy:

            - 'undersample': Downsample majority classes to match the minority
              class frequency. Uses `self.ids` to ensure repeatability.
            - 'oversample': Upsample minority classes (with replacement) to match
              the majority frequency.
            - 'auto': Heuristic choice. Uses undersampling if total size remains >
              20% of original, else oversampling.
        covariates : list of str, optional
            List of covariate names in `self.coords` to preserve distribution of.
            If provided, the balancing is performed *within* strata defined by these
            covariates.
        random_state : int, default=42
            Seed for the random number generator.
            Change this value to produce different random subsets (e.g., for bagging).
        **kwargs : dict
            Additional arguments passed to internal logic:

            - n_bins (int): Number of bins for continuous covariates (default 5).
            - binning (str): 'quantile' (default) or 'uniform' binning.
            - prefer_clean_rows (bool): If True, weighs sampling to prefer rows
              with fewer NaNs/artifacts.

        Returns
        -------
        DataContainer
            A new DataContainer instance with balanced classes.

        Examples
        --------
        >>> # 1. Simple Undersampling of 'y'
        >>> balanced = container.balance(strategy="undersample")

        >>> # 2. Balance based on a metadata column 'condition'
        >>> balanced = container.balance(target="condition")

        >>> # 3. Stratified Balancing (Balance 'y' while preserving 'sex' and 'age'
        >>> #    ratios)
        >>> balanced = container.balance(target="y", covariates=["sex", "age"])

        >>> # 4. Iterative Bootstrapping (Different seeds)
        >>> for seed in [1, 2, 3]:
        ...     subset = container.balance(strategy="undersample", random_state=seed)
        ...     # process subset...
        """
        # 1. Construct temporary DataFrame for Metadata
        data_dict = {}

        # Target
        if target == "y":
            if self.y is None:
                raise ValueError("Container has no y data.")
            data_dict["y"] = self.y
        elif target in self.coords:
            data_dict[target] = self.coords[target]
        else:
            raise ValueError(f"Target '{target}' not found in y or coords.")

        # Covariates
        if covariates:
            for c in covariates:
                if c not in self.coords:
                    raise ValueError(f"Covariate '{c}' not found in coords.")
                data_dict[c] = self.coords[c]

        df_meta = pd.DataFrame(data_dict)
        # Index is implicitly RangeIndex (0..N-1), which matches DataContainer
        # positional indices

        # 2. Get Indices
        if target not in df_meta.columns:
            raise ValueError(f"Target '{target}' missing")
        strategy = strategy.lower()
        counts = df_meta[target].value_counts()
        min_c, max_c = int(counts.min()), int(counts.max())

        if strategy == "auto":
            strategy = (
                "undersample"
                if (min_c * counts.size) >= len(df_meta) / 2
                else "oversample"
            )

        exclude = [target] + (covariates or [])
        rng = np.random.default_rng(random_state)

        indices_val = None

        # 1. Simple Case (No Covariates)
        if not covariates:
            size = dict.fromkeys(
                counts.index, min_c if strategy == "undersample" else max_c
            )
            indices_val = sample_indices(
                df_meta,
                target,
                size,
                rng,
                strategy != "undersample",
                kwargs.get("prefer_clean_rows", False),
                exclude,
            )

        else:
            # 2. Covariate Balancing (Stratified)
            # kwargs.get('n_bins', 5), kwargs.get('binning', 'quantile')
            strata_s = make_strata(
                df_meta,
                covariates,
                kwargs.get("n_bins", 5),
                kwargs.get("binning", "quantile"),
            )
            tmp = df_meta.assign(__strata__=strata_s)
            indices_parts = []

            for _, g in tmp.groupby("__strata__", observed=False):
                sc = g[target].value_counts()
                if len(sc) <= 1:
                    # Cannot balance within a single-class stratum
                    if strategy != "undersample":
                        M = int(counts.max())
                        size_map = dict.fromkeys(sc.index, M)
                        indices_parts.append(
                            sample_indices(
                                g,
                                target,
                                size_map,
                                rng,
                                True,
                                kwargs.get("prefer_clean_rows", False),
                                exclude,
                            )
                        )
                    continue

                # Balance locally within stratum
                sz = {
                    c: int(sc.min()) if strategy == "undersample" else int(sc.max())
                    for c in sc.index
                }
                indices_parts.append(
                    sample_indices(
                        g,
                        target,
                        sz,
                        rng,
                        strategy != "undersample",
                        kwargs.get("prefer_clean_rows", False),
                        exclude,
                    )
                )

            if not indices_parts:
                # Fallback: global balance
                sz = dict.fromkeys(
                    counts.index, min_c if strategy == "undersample" else max_c
                )
                indices_val = sample_indices(
                    df_meta,
                    target,
                    sz,
                    rng,
                    strategy != "undersample",
                    kwargs.get("prefer_clean_rows", False),
                    exclude,
                )
            else:
                combined = pd.concat([pd.Series(i) for i in indices_parts]).sample(
                    frac=1.0, random_state=rng
                )
                indices_val = pd.Index(combined.values)

        # 3. Apply Indexing
        # indices is a pandas Index (Int64 or similar). Convert to values for safe
        # numpy indexing.
        return self.isel(obs=indices_val.values)

    def select(
        self, ignore_case: bool = False, fuzzy: bool = False, **selections
    ) -> "DataContainer":
        """
        Select data subsets based on coordinates, ids, or y.

        This method supports exact matching, wildcard matching, operator-based
        filtering, and custom callable filters.

        Parameters
        ----------
        ignore_case : bool, default=False
            If True, string matching is case-insensitive (e.g., 'fz' matches 'Fz').
        fuzzy : bool, default=False
            If True, uses `difflib` to find closest matches for string queries
            (e.g., 'Alpha' matches 'alpha'). Useful for handling typos.
        **selections : dict
            Key is the dimension name (or special keys 'y', 'ids').
            Value is the query. Supported query types:

            1. **List/Array (Exact or Wildcard)**:
               Matches values present in the list. Strings can use shell-style
               wildcards ('*', '?').

            2. **Dictionary (Operator Queries)**:
               Filters numerical or string values using operators.
               Keys: '>', '<', '>=', '<=', '==', '!=', 'in'.

            3. **Callable**:
               A function taking the coordinate array and returning a boolean mask.

        Returns
        -------
        DataContainer
            A new DataContainer instance containing the selected subset.

        Examples
        --------
        >>> # 1. Exact Selection (Sensors)
        >>> sub = container.select(channel=["Fz", "Cz"])

        >>> # 2. Wildcard Selection (All Alpha features)
        >>> sub = container.select(feature="*alpha*")

        >>> # 3. Range Selection (Time)
        >>> sub = container.select(time={">=": 0.1, "<": 0.5})

        >>> # 4. Case-Insensitive Fuzzy Matching
        >>> sub = container.select(channel=["fz"], ignore_case=True)

        >>> # 5. Filter by Target (y)
        >>> sub = container.select(y=["Patient"])

        >>> # 6. Complex Logic (Subjects 1-5 via Operator)
        >>> sub = container.select(subject_id={">=": 1, "<=": 5})

        >>> # 7. Stratified Selection (First 2 epochs per subject via Callable)
        >>> def first_n(ids, n=2):
        ...     # ... logic ...
        ...     return mask
        >>> sub = container.select(ids=first_n)
        """
        slices = [slice(None)] * self.X.ndim
        obs_dim_idx = self.dims.index("obs") if "obs" in self.dims else -1

        for key, query in selections.items():
            # Determine target array and axis
            target_arr = None
            axis = -1
            target_dim = None

            if key == "y" and self.y is not None:
                target_arr = self.y
                target_dim = "obs"
            elif key == "ids" and self.ids is not None:
                target_arr = self.ids
                target_dim = "obs"
            elif key in self.dims:
                target_dim = key
                target_arr = np.array(self.coords.get(key, []))

            elif key in self.coords:
                target_arr = np.array(self.coords[key])
                matched_dim = None
                for d_i, d_name in enumerate(self.dims):
                    if self.X.shape[d_i] == len(target_arr):
                        matched_dim = d_name
                        break

                if matched_dim:
                    target_dim = matched_dim
                else:
                    logger.warning(
                        f"Aux coordinate '{key}' len={len(target_arr)} matches no "
                        f"dimension shape {self.X.shape}. Ignoring selection."
                    )
                    continue

            if target_arr is None:
                if key in self.dims and key not in self.coords:
                    logger.warning(
                        f"Selection on dim '{key}' ignored (no coordinates)."
                    )
                    continue
                logger.warning(
                    f"Selection key '{key}' not found in logical dims, y, ids, or "
                    f"aux coords. Ignoring."
                )
                continue

            if target_dim in self.dims:
                axis = self.dims.index(target_dim)

            if len(target_arr) == 0:
                logger.warning(
                    f"Target array for '{key}' is empty. Skipping selection."
                )
                continue

            mask = np.zeros(len(target_arr), dtype=bool)

            # Handle different query types
            if callable(query):
                mask = query(target_arr)
                if not isinstance(mask, (np.ndarray, list)) or len(mask) != len(
                    target_arr
                ):
                    raise ValueError(
                        f"Callable query for '{key}' must return boolean array of "
                        f"shape {target_arr.shape}."
                    )
                mask = np.array(mask, dtype=bool)

            elif isinstance(query, dict):
                # Operator mode: {'>': 5}
                mask = np.ones(len(target_arr), dtype=bool)
                ops = {
                    ">": lambda a, b: a > b,
                    "<": lambda a, b: a < b,
                    ">=": lambda a, b: a >= b,
                    "<=": lambda a, b: a <= b,
                    "==": lambda a, b: a == b,
                    "!=": lambda a, b: a != b,
                    "in": np.isin,
                }
                for op, val in query.items():
                    if op not in ops:
                        raise ValueError(
                            f"Unknown operator '{op}'. Supported: {list(ops.keys())}"
                        )
                    mask &= ops[op](target_arr, val)

            else:
                # Standard List/Value Match (Exact / Wildcard / Fuzzy)
                query_arr = np.array(query, ndmin=1)

                # Pre-processing for String Matching
                is_str_target = target_arr.dtype.kind in ("U", "S", "O")

                if is_str_target:
                    # String Matching
                    target_list = target_arr.tolist()
                    query_list = query_arr.tolist()

                    if ignore_case:
                        target_lookup = [str(x).lower() for x in target_list]
                        query_list_proc = [str(q).lower() for q in query_list]
                    else:
                        target_lookup = [str(x) for x in target_list]
                        query_list_proc = [str(q) for q in query_list]

                    # 1. Fuzzy Choice?
                    final_queries = set()
                    if fuzzy:
                        for q in query_list_proc:
                            matches = difflib.get_close_matches(
                                q, target_lookup, n=3, cutoff=0.6
                            )
                            final_queries.update(matches)
                            if not matches:
                                logger.warning(f"No fuzzy match found for '{q}'.")
                    else:
                        final_queries = set(query_list_proc)

                    # 2. Pattern vs Exact
                    patterns = [q for q in final_queries if "*" in q or "?" in q]
                    exacts = [q for q in final_queries if q not in patterns]

                    target_lookup_arr = np.array(target_lookup)

                    if exacts:
                        mask |= np.isin(target_lookup_arr, exacts)

                    # Wildcards
                    for pat in patterns:
                        regex = re.compile(fnmatch.translate(pat))
                        matches = [bool(regex.match(x)) for x in target_lookup]
                        mask |= np.array(matches)

                else:
                    # Numeric Exact Match
                    mask = np.isin(target_arr, query_arr)

            indices = np.where(mask)[0]
            if len(indices) == 0:
                raise ValueError(
                    f"Selection for '{key}' resulted in empty set. Query: {query}"
                )

            # Apply Slicing Logic (Intersect with current)
            existing_slice = slices[axis]
            if isinstance(existing_slice, slice) and existing_slice == slice(None):
                slices[axis] = indices
            else:
                common = np.intersect1d(existing_slice, indices)
                if len(common) == 0:
                    raise ValueError(
                        f"Conflicting selections for axis {axis} ({key}) resulted "
                        f"in empty set."
                    )
                slices[axis] = common

        # Final Application (Orthogonal Indexing)
        # Apply slices sequentially to avoid broadcasting issues
        X_new = self.X
        for axis, sl in enumerate(slices):
            if isinstance(sl, slice) and sl == slice(None):
                continue

            indexer = [slice(None)] * X_new.ndim
            indexer[axis] = sl
            X_new = X_new[tuple(indexer)]

        # Update coordinates to match new X. Each coord is mapped to its axis
        # once (obs-priority on ties); coords matching no axis are dropped.
        final_coords = {}
        for coord_name, labels in self.coords.items():
            aligned_dim_idx = self._coord_axis(coord_name, labels, obs_dim_idx)
            if aligned_dim_idx != -1:
                final_coords[coord_name] = np.array(labels)[slices[aligned_dim_idx]]

        # Update y/ids
        y_new = self.y
        ids_new = self.ids

        # If obs was sliced (even indirectly via y/ids)
        if obs_dim_idx != -1:
            obs_sl = slices[obs_dim_idx]
            if not isinstance(obs_sl, slice) or obs_sl != slice(None):
                if y_new is not None:
                    y_new = y_new[obs_sl]
                if ids_new is not None:
                    ids_new = ids_new[obs_sl]

        return replace(self, X=X_new, coords=final_coords, y=y_new, ids=ids_new)

    def flatten(
        self,
        preserve: str | list[str] = "obs",
        sep: str = "_",
    ) -> "DataContainer":
        """
        Flatten dimensions NOT in `preserve` into a single 'feature' dimension.

        This is useful for preparing N-dimensional data for standard 2D machine
        learning algorithms (scikit-learn). It automatically generates composite
        feature names (e.g., 'Fz_0.1s') for tracking.

        Parameters
        ----------
        preserve : str or list of str, default="obs"
            Dimensions to preserve. All other dimensions are flattened into a new
            dimension 'feature'.

            - 'obs': Result shape (N_obs, N_features). Standard specification.
            - ['obs', 'time']: Result shape (N_obs, N_time, N_features).
              Useful for time-resolved decoding distributions.
        sep : str, default="_"
            Separator used when generating composite feature names.

        Returns
        -------
        DataContainer
            A new DataContainer with reshaped X and generated 'feature' coordinates.

        Examples
        --------
        >>> # Flatten (10, 64, 500) -> (10, 32000)
        >>> flat = container.flatten(preserve="obs")
        >>> flat.shape
        (10, 32000)
        >>> flat.coords["feature"][0]
        'Fz_0.0'

        >>> # Flatten spatial only, keep time (10, 64, 500) -> (10, 500, 64)
        >>> time_resolved = container.flatten(preserve=["obs", "time"])
        """
        if isinstance(preserve, str):
            preserve = [preserve]

        # Verify dims exist
        for p in preserve:
            if p not in self.dims:
                raise ValueError(f"Dimension '{p}' not found in {self.dims}.")

        # Dimensions to flatten
        to_flatten = [d for d in self.dims if d not in preserve]

        if not to_flatten:
            return self  # Nothing to do

        # Move preserved dims to front
        # Current indices
        permute_order = [self.dims.index(p) for p in preserve] + [
            self.dims.index(d) for d in to_flatten
        ]

        X_trans = np.transpose(self.X, axes=permute_order)

        # New shape: (*preserved_shapes, product(flattened_shapes))
        preserved_shape = [self.X.shape[self.dims.index(p)] for p in preserve]
        flattened_len = int(
            np.prod([self.X.shape[self.dims.index(d)] for d in to_flatten])
        )

        new_shape = (*tuple(preserved_shape), flattened_len)
        X_flat = X_trans.reshape(new_shape)

        # New Dims
        new_dims = (*tuple(preserve), "feature")

        # New Coords
        # We keep coords for preserved dimensions.
        new_coords = {k: v for k, v in self.coords.items() if k in preserve}
        if "obs" in preserve and "obs" in self.dims:
            n_obs = self.X.shape[self.dims.index("obs")]
            for k, v in self.coords.items():
                if k not in self.dims and len(v) == n_obs:
                    new_coords[k] = v

        primary_coords = []
        parallel_by_dim: dict[str, dict[str, Any]] = {}
        obs_dim_idx = self.dims.index("obs") if "obs" in self.dims else -1
        for d in to_flatten:
            axis = self.dims.index(d)
            primary = self.coords.get(d)
            if primary is None:
                primary = np.arange(self.X.shape[axis])
            primary_coords.append(primary)
            extras = {
                name: vals
                for name, vals in self.coords.items()
                if name not in self.dims
                and self._coord_axis(name, vals, obs_dim_idx) == axis
            }
            if extras:
                parallel_by_dim[d] = extras

        # Create Cartesian product
        if primary_coords:
            # Check size first to avoid memory explosion?
            total_size = np.prod([len(x) for x in primary_coords])
            if total_size < 200000:  # Limit to ~200k features strings
                index_product = list(
                    itertools.product(*[range(len(c)) for c in primary_coords])
                )
                combo_labels = [
                    sep.join(str(primary_coords[ax][i]) for ax, i in enumerate(combo))
                    for combo in index_product
                ]
                new_coords["feature"] = combo_labels
                for axis, d in enumerate(to_flatten):
                    for name, vals in parallel_by_dim.get(d, {}).items():
                        new_coords[name] = [
                            vals[combo[axis]] for combo in index_product
                        ]
            else:
                warnings.warn(
                    f"flatten(): {total_size} features exceeds the 200000 label cap; "
                    "'feature' coordinate names were not generated.",
                    stacklevel=2,
                )

        return replace(
            self,
            X=X_flat,
            dims=new_dims,
            coords=new_coords,
            meta={**self.meta, "flattened_from": self.dims},
        )

    def feature_schema(self) -> "pd.DataFrame | None":
        """Return feature-axis metadata, or None when no feature coord exists.

        Only coordinates aligned to the feature axis are included. ``feature_*``
        metadata is mapped to canonical schema names such as ``family`` and
        ``measure``; primary dimension coords folded by ``flatten()`` are used
        in the feature labels and are not recovered as structured metadata.
        """
        if "feature" not in self.dims or "feature" not in self.coords:
            return None
        feature_axis = self.dims.index("feature")
        obs_dim_idx = self.dims.index("obs") if "obs" in self.dims else -1
        n = len(self.coords["feature"])
        aliases = {
            "feature_family": "family",
            "feature_scope": "scope",
            "feature_channel": "channel",
            "feature_sensor": "channel",
            "feature_measure": "measure",
            "feature_subfamily": "subfamily",
            "feature_descriptor": "descriptor",
        }
        data = {"column": list(self.coords["feature"])}
        for name, vals in self.coords.items():
            if name == "feature" or name in self.dims or len(vals) != n:
                continue
            if self._coord_axis(name, vals, obs_dim_idx) != feature_axis:
                continue
            schema_name = aliases.get(name, name)
            if schema_name not in data:
                data[schema_name] = list(vals)
        return pd.DataFrame(data)

    def stack(self, dims: Sequence[str], new_dim: str = "obs") -> "DataContainer":
        """
        Stack multiple dimensions into a single new dimension.

        This reshapes N-dimensional data into (N-K) dimensions by combining
        specified dimensions. It is useful for transforming spatiotemporal data
        (Trials, Channels, Time) -> (Trials*Time, Channels) for trajectory analysis.

        Parameters
        ----------
        dims : sequence of str
            Dimensions to stack. The order determines the nesting (slowest to fastest).
            e.g., ('obs', 'time') means 'obs' changes slowly, 'time' cycles fast.
        new_dim : str, default='obs'
            Name of the resulting stacked dimension.

        Returns
        -------
        DataContainer
            New container with stacked dimension. Metadata (coords/ids) are
            expanded/tiled to match the new shape.

        Examples
        --------
        >>> # Stack time into observations:
        >>> # (10 obs, 64 ch, 500 time) -> (5000 obs, 64 ch)
        >>> stacked = container.stack(dims=("obs", "time"), new_dim="obs")
        >>> stacked.shape
        (5000, 64)
        """
        for d in dims:
            if d not in self.dims:
                raise ValueError(f"Dimension '{d}' not found in {self.dims}")

        # 1. Permute
        preserved = [d for d in self.dims if d not in dims]
        stack_indices = [self.dims.index(d) for d in dims]
        preserve_indices = [self.dims.index(d) for d in preserved]

        permute_order = stack_indices + preserve_indices
        X_trans = np.transpose(self.X, axes=permute_order)

        # 2. Reshape
        stack_shape = [self.X.shape[i] for i in stack_indices]
        prod_len = int(np.prod(stack_shape))
        preserved_shape = [self.X.shape[i] for i in preserve_indices]

        new_shape = (prod_len, *tuple(preserved_shape))
        X_new = X_trans.reshape(new_shape)

        # 3. Handle Metadata Expansion (if new_dim is 'obs' or overrides it)
        new_ids = None
        new_y = None
        new_coords = self.coords.copy()

        # Drop old coords keys that are being stacked
        for d in dims:
            if d in new_coords:
                del new_coords[d]

        # Logic for IDs/Y expansion if 'obs' is involved
        if "obs" in dims and new_dim == "obs":
            obs_idx = dims.index("obs")
            n_obs = self.X.shape[self.dims.index("obs")]

            # Repeats (inner) and Tiles (outer) logic
            # product(dims after obs) -> repeats
            # product(dims before obs) -> tiles
            n_repeats = int(
                np.prod([self.X.shape[self.dims.index(d)] for d in dims[obs_idx + 1 :]])
            )
            n_tiles = int(
                np.prod([self.X.shape[self.dims.index(d)] for d in dims[:obs_idx]])
            )

            # Expand Y
            if self.y is not None:
                new_y = np.tile(np.repeat(self.y, n_repeats), n_tiles)

            for k, v in self.coords.items():
                if k not in self.dims and len(v) == n_obs:
                    new_coords[k] = np.tile(np.repeat(np.array(v), n_repeats), n_tiles)

            # Expand IDs
            if self.ids is not None:
                # We want composite IDs: "sub-0_t-0", "sub-0_t-1"
                # Construct MultiIndex details
                idx_components = []
                for d in dims:
                    if d == "obs":
                        idx_components.append(self.ids)
                    else:
                        # Use coordinate labels if available, else range
                        c = self.coords.get(d)
                        if c is None:
                            c = np.arange(self.X.shape[self.dims.index(d)])
                        idx_components.append(c)

                # Cartesian Product
                # Use pandas for robust string joining
                mi = pd.MultiIndex.from_product(idx_components, names=dims)
                new_ids = (
                    mi.to_frame(index=False).astype(str).agg("_".join, axis=1).values
                )

        new_dims_final = (new_dim, *tuple(preserved))

        n_obs_orig = (
            self.X.shape[self.dims.index("obs")] if "obs" in self.dims else None
        )
        snapshot_coords: dict[str, np.ndarray] = {
            d: np.asarray(self.coords[d]).copy() for d in dims if d in self.coords
        }
        for coord_name, values in self.coords.items():
            if coord_name in dims or coord_name in self.dims:
                continue
            arr = np.asarray(values)
            if n_obs_orig is not None and arr.shape[0] == n_obs_orig:
                snapshot_coords.setdefault(coord_name, arr.copy())

        snapshot = {
            "y": (None if self.y is None else np.asarray(self.y).copy()),
            "ids": (None if self.ids is None else np.asarray(self.ids).copy()),
            "coords": snapshot_coords,
            "original_dims": tuple(self.dims),
        }

        return replace(
            self,
            X=X_new,
            dims=new_dims_final,
            ids=new_ids,
            y=new_y,
            coords=new_coords,
            meta={
                **self.meta,
                "stacked_from": dims,
                "stacked_shapes": tuple(stack_shape),
                "_stacked_snapshot": snapshot,
            },
        )

    def unstack(self, dim: str) -> "DataContainer":
        """
        Unstack a dimension into multiple dimensions.

        Inverse operation of `stack`. Reshapes the data tensor by splitting one
        dimension into multiple using metadata stored during the `stack` operation.

        Parameters
        ----------
        dim : str
            Dimension to unstack (e.g. 'obs').

        Returns
        -------
        DataContainer
            New container with unstacked dimensions.

        Raises
        ------
        ValueError
            If the container was not previously stacked (missing metadata).

        Examples
        --------
        >>> # Stack 'trials' and 'time' -> 'obs'
        >>> stacked = container.stack(("trials", "time"), new_dim="obs")
        >>> # Unstack 'obs' -> ('trials', 'time') (automatically inferred)
        >>> unstacked = stacked.unstack("obs")
        """
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in {self.dims}")

        # Strict Metadata Check
        if "stacked_from" not in self.meta or "stacked_shapes" not in self.meta:
            raise ValueError(
                "Cannot unstack: Metadata 'stacked_from' or 'stacked_shapes' not found."
                "Ensure data was processed with .stack() or metadata is preserved."
            )

        new_dims = self.meta["stacked_from"]
        new_shapes = self.meta["stacked_shapes"]

        dim_idx = self.dims.index(dim)
        current_len = self.X.shape[dim_idx]
        target_len = int(np.prod(new_shapes))

        if target_len != current_len:
            raise ValueError(
                f"Shape mismatch: {dim} has length {current_len}, "
                f"but product of new_shapes {new_shapes} is {target_len}"
            )

        # 1. Reshape: Move target dim to front, reshape, then permute back
        # Move 'dim' to axis 0: (dim, ...)
        X_moved = np.moveaxis(self.X, dim_idx, 0)

        # Reshape to (new_d1, new_d2, ..., other_dims...)
        X_reshaped = X_moved.reshape(*new_shapes, *X_moved.shape[1:])

        # Permute to insert new dimensions at original position
        # new dims are at [0...k-1]. We want them at [dim_idx...dim_idx+k-1]
        k = len(new_dims)
        # Construct permutation:
        # [k...k+dim_idx-1] + [0...k-1] + [k+dim_idx...]
        # axes before dim + new axes + axes after
        perm = (
            list(range(k, k + dim_idx))
            + list(range(k))
            + list(range(k + dim_idx, X_reshaped.ndim))
        )
        X_final = np.transpose(X_reshaped, perm)

        # 2. Update Metadata
        final_dims = []
        for d in self.dims:
            if d == dim:
                final_dims.extend(new_dims)
            else:
                final_dims.append(d)

        new_coords = {k: v for k, v in self.coords.items() if k != dim}

        # Drop y/ids if they matched the unstacked dimension length
        new_y = self.y if (self.y is None or len(self.y) != current_len) else None
        new_ids = (
            self.ids if (self.ids is None or len(self.ids) != current_len) else None
        )

        snapshot = self.meta.get("_stacked_snapshot")
        if snapshot is not None:
            if snapshot.get("y") is not None:
                new_y = snapshot["y"]
            if snapshot.get("ids") is not None:
                new_ids = snapshot["ids"]
            for coord_name, values in snapshot.get("coords", {}).items():
                new_coords[coord_name] = values

            original_dims = snapshot.get("original_dims")
            if original_dims is not None and set(original_dims) == set(final_dims):
                # Restore the original dim order by transposing.
                current_dims = tuple(final_dims)
                permutation = [current_dims.index(d) for d in original_dims]
                X_final = np.transpose(X_final, permutation)
                final_dims = list(original_dims)

        # Drop the snapshot once consumed so downstream stacks don't carry it.
        new_meta = {k: v for k, v in self.meta.items() if k != "_stacked_snapshot"}
        new_meta["unstacked_from"] = dim

        return replace(
            self,
            X=X_final,
            dims=tuple(final_dims),
            y=new_y,
            ids=new_ids,
            coords=new_coords,
            meta=new_meta,
        )

    def with_features(
        self,
        X: np.ndarray,
        names: Sequence[str] | None = None,
        feature_dim: str | None = None,
        new_dim_name: str = "component",
    ) -> "DataContainer":
        """
        Return a new container with the feature axis replaced.

        Typical use: re-attach reduced-dimensionality scores (e.g. PCA
        components) to a container, so downstream operations (``unstack``,
        ``aggregate``, plotting) keep working with proper coordinates.

        Parameters
        ----------
        X : np.ndarray
            New data array. The leading axes must match the container's
            non-feature axes; the trailing axis is the new feature axis.
        names : sequence of str, optional
            Coordinate labels for the new feature axis. When ``None``,
            integer indices are used. Must have length ``X.shape[-1]``.
        feature_dim : str, optional
            Name of the dimension being replaced. Defaults to the last
            dimension of the container.
        new_dim_name : str, default='component'
            Dimension name to assign to the replaced axis when
            ``feature_dim`` is not present in ``self.dims`` (e.g., when
            replacing ``channel`` with ``component`` after PCA).

        Returns
        -------
        DataContainer
            New container with ``X`` replaced and the feature-axis coord
            updated. All other dims, coords, ``y``, ``ids``, and meta are
            preserved.

        Raises
        ------
        ValueError
            If ``X``'s leading shape doesn't match the container, or if
            ``names`` has the wrong length.

        Examples
        --------
        >>> # After fitting PCA on stacked data:
        >>> scores = reducer.fit_transform(c_stacked.X)  # (n_obs, 3)
        >>> c_pc = c_stacked.with_features(
        ...     scores,
        ...     names=["PC1", "PC2", "PC3"],
        ...     new_dim_name="component",
        ... )
        >>> c_pc.dims
        ('obs', 'component')
        """
        X = np.asarray(X)
        if X.ndim != self.X.ndim:
            raise ValueError(
                f"`X` must have the same number of dimensions as the container "
                f"({self.X.ndim}); got {X.ndim}."
            )
        if X.shape[:-1] != self.X.shape[:-1]:
            raise ValueError(
                f"`X`'s leading axes {X.shape[:-1]} must match the container's "
                f"non-feature axes {self.X.shape[:-1]}."
            )

        if feature_dim is None:
            feature_dim = self.dims[-1]

        # The output axis is renamed to ``new_dim_name`` unconditionally.
        # When the caller wants to keep the original dim name, they pass
        # ``new_dim_name=feature_dim`` (or rely on the same name).
        new_dim = new_dim_name

        n_features_new = X.shape[-1]
        if names is not None:
            names_arr = np.asarray(names)
            if names_arr.shape[0] != n_features_new:
                raise ValueError(
                    f"`names` length {names_arr.shape[0]} does not match new "
                    f"feature count {n_features_new}."
                )
        else:
            names_arr = np.arange(n_features_new)

        # Build new dims tuple, swapping feature_dim → new_dim if changed
        new_dims = tuple(new_dim if d == feature_dim else d for d in self.dims)

        # Build new coords: drop the old feature coord (if present) and set
        # the new one
        new_coords = {k: v for k, v in self.coords.items() if k != feature_dim}
        new_coords[new_dim] = names_arr

        return replace(
            self,
            X=X,
            dims=new_dims,
            coords=new_coords,
            meta={
                **self.meta,
                "with_features_from": feature_dim,
                "with_features_to": new_dim,
            },
        )

    def center(
        self, dim: str | Sequence[str] = "time", inplace: bool = False
    ) -> "DataContainer":
        """
        Remove mean along a specified dimension (Centering/Baseline Correction).

        This operation computes the mean along `dim` (ignoring NaNs) and subtracts it.
        Commonly used in EEG for baseline correction (subtracting mean of
        pre-stimulus interval) or centering features before covariance calculation.

        Parameters
        ----------
        dim : str or sequence of str, default='time'
            Dimension name(s) to center over (e.g., 'time', 'channel', 'obs',
            or ('obs', 'time')).
        inplace : bool, default=False
            If True, modifies X in-place to save memory.
            Returns self.

        Returns
        -------
        DataContainer
            Container with centered data.

        Examples
        --------
        >>> # Baseline correction over time
        >>> container.center(dim="time")
        """
        dims_list = [dim] if isinstance(dim, str) else dim
        for d in dims_list:
            if d not in self.dims:
                raise ValueError(f"Dimension '{d}' not found in {self.dims}")

        axes = tuple(self.dims.index(d) for d in dims_list)
        X = self.X if inplace else self.X.copy()

        mean = np.nanmean(X, axis=axes, keepdims=True)
        X -= mean

        if inplace:
            return self
        return replace(self, X=X)

    def zscore(
        self,
        dim: str | Sequence[str] = "time",
        eps: float = 1e-8,
        inplace: bool = False,
    ) -> "DataContainer":
        """
        Standardize (Z-score) along a specified dimension.

        Computes `(X - mean) / std` along the given dimension. Robust to NaNs.
        Useful for normalizing features or standardizing temporal dynamics.

        Parameters
        ----------
        dim : str or sequence of str
            Dimension(s) to standardize.
        eps : float
            Stability epsilon to avoid division by zero.
        inplace : bool

        Returns
        -------
        DataContainer

        Examples
        --------
        >>> # Standardize each channel's timecourse
        >>> container.zscore(dim="time")
        """
        dims_list = [dim] if isinstance(dim, str) else dim
        for d in dims_list:
            if d not in self.dims:
                raise ValueError(f"Dimension '{d}' not found in {self.dims}")

        axes = tuple(self.dims.index(d) for d in dims_list)
        X = self.X if inplace else self.X.copy()

        mean = np.nanmean(X, axis=axes, keepdims=True)
        std = np.nanstd(X, axis=axes, keepdims=True)

        X -= mean
        X /= std + eps

        if inplace:
            return self
        return replace(self, X=X)

    def rms_scale(
        self,
        dim: str | Sequence[str] = "time",
        eps: float = 1e-8,
        inplace: bool = False,
    ) -> "DataContainer":
        """
        Scale by Root Mean Square (RMS) amplitude along a dimension.

        Divides data by `sqrt(mean(X**2))` along the dimension.
        Preserves relative shape but normalizes energy.

        Parameters
        ----------
        dim : str or sequence of str
            Dimension(s) to scale.
        eps : float
            Stability epsilon.
        inplace : bool

        Returns
        -------
        DataContainer
        """
        dims_list = [dim] if isinstance(dim, str) else dim
        for d in dims_list:
            if d not in self.dims:
                raise ValueError(f"Dimension '{d}' not found in {self.dims}")

        axes = tuple(self.dims.index(d) for d in dims_list)
        X = self.X if inplace else self.X.copy()

        mean_sq = np.nanmean(X**2, axis=axes, keepdims=True)
        rms = np.sqrt(mean_sq)

        X /= rms + eps

        if inplace:
            return self
        return replace(self, X=X)

    def baseline_correction(
        self, dim: str | Sequence[str] = "time", inplace: bool = False
    ) -> "DataContainer":
        """Alias for center(). Common in EEG."""
        return self.center(dim=dim, inplace=inplace)

    def combine_coords(
        self,
        keys: Sequence[str],
        name: str,
        *,
        sep: str = "_",
        pair_sep: str | None = "-",
        overwrite: bool = False,
    ) -> "DataContainer":
        """Return a copy with a new coordinate combining several existing coords.

        Each element of the new coordinate joins the corresponding elements of
        ``keys`` in the order given. This materializes a single composite key
        (e.g. a ``recording_id`` from ``subject``/``session``/``run``) that can
        then be used anywhere a single coordinate is expected — most notably as
        the ``by`` argument to :meth:`aggregate`, but also for ``select``,
        ``observation_frame``, or provenance labels.

        Parameters
        ----------
        keys : sequence of str
            Names of existing coordinates to combine. All must be present in
            ``coords`` and share the same length.
        name : str
            Name of the new coordinate to create.
        sep : str, default="_"
            Separator placed between components.
        pair_sep : str or None, default="-"
            Separator between a component's source name and its value, yielding
            ``"<key><pair_sep><value>"`` (e.g. ``"subject-0001"``). When
            ``None``, names are omitted and only the values are joined.
        overwrite : bool, default=False
            Whether to replace ``name`` if a coordinate by that name exists.

        Returns
        -------
        DataContainer
            A copy with the new composite coordinate added.

        Raises
        ------
        ValueError
            If ``keys`` is empty, any key is missing, the keys differ in
            length, or ``name`` already exists and ``overwrite`` is False.

        Examples
        --------
        >>> grouped = container.combine_coords(
        ...     ["subject", "session", "run"], "recording_id"
        ... ).aggregate(by="recording_id", stats="mean")
        """
        if not keys:
            raise ValueError("`keys` must contain at least one coordinate name.")
        missing = [k for k in keys if k not in self.coords]
        if missing:
            raise ValueError(
                f"Coordinates not found: {missing}. Available: {sorted(self.coords)}."
            )
        if name in self.coords and not overwrite:
            raise ValueError(
                f"Coordinate '{name}' already exists. "
                "Pass overwrite=True to replace it."
            )

        arrays = [np.asarray(self.coords[k], dtype=object) for k in keys]
        lengths = {arr.shape[0] for arr in arrays}
        if len(lengths) != 1:
            sizes = {k: arr.shape[0] for k, arr in zip(keys, arrays, strict=True)}
            raise ValueError(
                f"Coordinates to combine must share a length; got {sizes}."
            )

        n = arrays[0].shape[0]
        combined = np.empty(n, dtype=object)
        for i in range(n):
            if pair_sep is None:
                combined[i] = sep.join(str(arr[i]) for arr in arrays)
            else:
                combined[i] = sep.join(
                    f"{k}{pair_sep}{arr[i]}"
                    for k, arr in zip(keys, arrays, strict=True)
                )

        new_coords = dict(self.coords)
        new_coords[name] = combined
        return replace(self, coords=new_coords)

    def aggregate(
        self,
        by: str | np.ndarray | list[Any],
        stats: str | Sequence[str] = "mean",
        min_count: int = 1,
        on_insufficient: str = "raise",
    ) -> "DataContainer":
        """
        Aggregate observations into grouped summaries along the ``obs`` axis.

        Parameters
        ----------
        by : str or array-like
            Grouping definition.

            - If str: resolve the key from ``self.coords`` or from ``self.y``
              (if "y" is passed).
            - If array-like: explicit group labels aligned with ``obs``.
        stats : str or sequence of str, default="mean"
            Aggregation statistic or ordered list of statistics. Supported
            tokens are ``"mean"``, ``"median"``, ``"std"``, ``"var"``,
            ``"sem"``, ``"mad"``, ``"iqr"``, ``"min"``, ``"max"``,
            ``"count"``, and ``"first"``. Legacy ``"obs-*"`` aliases are
            accepted and normalized.
        min_count : int, default=1
            Minimum number of valid observations required per group. A valid
            observation is one with at least one finite value across the
            non-observation axes.
        on_insufficient : {"raise", "warn", "collect"}, default="raise"
            Policy applied when a group has fewer than ``min_count`` valid
            observations.

        Returns
        -------
        DataContainer
            Aggregated container with grouped observations on the ``obs`` axis.
            When multiple stats are requested, a ``stat`` dimension is inserted
            immediately after ``obs``.

        Raises
        ------
        ValueError
            If the container has no ``obs`` dimension, grouping is invalid,
            requested stats are unsupported, or ``min_count`` /
            ``on_insufficient`` are invalid.
        """
        if "obs" not in self.dims:
            raise ValueError("Aggregation requires 'obs' dimension.")

        obs_idx = self.dims.index("obs")
        n_obs = self.X.shape[obs_idx]

        if min_count < 1:
            raise ValueError("`min_count` must be at least 1.")
        if on_insufficient not in {"raise", "warn", "collect"}:
            raise ValueError("`on_insufficient` must be one of: raise, warn, collect.")

        stat_aliases = {
            "obs-mean": "mean",
            "obs-median": "median",
            "obs-std": "std",
            "obs-var": "var",
            "obs-sem": "sem",
            "obs-mad": "mad",
            "obs-iqr": "iqr",
            "obs-min": "min",
            "obs-max": "max",
            "obs-count": "count",
        }
        supported_stats = {
            "mean",
            "median",
            "std",
            "var",
            "sem",
            "mad",
            "iqr",
            "min",
            "max",
            "count",
            "first",
        }
        if isinstance(stats, str):
            stats_out = [stat_aliases.get(stats, stats)]
        else:
            stats_out = [stat_aliases.get(str(stat), str(stat)) for stat in stats]
        if not stats_out:
            raise ValueError("`stats` must not be empty.")
        invalid_stats = sorted(set(stats_out) - supported_stats)
        if invalid_stats:
            raise ValueError(
                f"Unknown stats: {invalid_stats}. Supported stats are: "
                f"{sorted(supported_stats)}"
            )

        if isinstance(by, str):
            if by == "y" and self.y is not None:
                groups_raw = self.y
            elif by in self.coords:
                groups_raw = self.coords[by]
            else:
                raise ValueError(f"Grouping key '{by}' not found in coords or y.")
        else:
            groups_raw = by

        labels_list = list(groups_raw)
        groups = np.empty(len(labels_list), dtype=object)
        groups[:] = labels_list

        if len(groups) != n_obs:
            raise ValueError(
                f"Grouping array length {len(groups)} must match obs length {n_obs}."
            )

        X_moved = np.moveaxis(self.X, obs_idx, 0) if obs_idx != 0 else self.X

        other_dims = tuple(dim for dim in self.dims if dim != "obs")
        group_positions: dict[Any, list[int]] = {}
        ordered_groups: list[Any] = []
        for obs_position, group_id in enumerate(groups.tolist()):
            if group_id not in group_positions:
                ordered_groups.append(group_id)
                group_positions[group_id] = []
            group_positions[group_id].append(obs_position)

        def _reshape_reduced(values_flat: np.ndarray) -> np.ndarray | np.float64:
            if rest_shape:
                return np.asarray(values_flat, dtype=np.float64).reshape(rest_shape)
            return np.asarray(values_flat, dtype=np.float64)[0]

        def _reduce_group(
            group_X: np.ndarray,
            group_X_flat: np.ndarray,
            counts_flat: np.ndarray,
            stat: str,
        ) -> np.ndarray | np.float64:
            if stat == "count":
                return _reshape_reduced(counts_flat)
            if stat == "first":
                return np.asarray(group_X[0], dtype=np.float64)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if stat == "mean":
                    values_flat = np.nanmean(group_X_flat, axis=0)
                elif stat == "median":
                    values_flat = np.nanmedian(group_X_flat, axis=0)
                elif stat == "std":
                    values_flat = np.nanstd(group_X_flat, axis=0)
                elif stat == "var":
                    values_flat = np.nanvar(group_X_flat, axis=0)
                elif stat == "sem":
                    values_flat = np.nanstd(group_X_flat, axis=0) / np.sqrt(
                        counts_flat.astype(np.float64)
                    )
                elif stat == "mad":
                    medians_flat = np.nanmedian(group_X_flat, axis=0)
                    values_flat = np.nanmedian(
                        np.abs(group_X_flat - medians_flat),
                        axis=0,
                    )
                elif stat == "iqr":
                    values_flat = np.nanpercentile(
                        group_X_flat,
                        75,
                        axis=0,
                    ) - np.nanpercentile(
                        group_X_flat,
                        25,
                        axis=0,
                    )
                elif stat == "min":
                    values_flat = np.nanmin(group_X_flat, axis=0)
                elif stat == "max":
                    values_flat = np.nanmax(group_X_flat, axis=0)
                else:  # pragma: no cover - guarded above
                    raise ValueError(f"Unknown stat '{stat}'")

            values_flat = np.asarray(values_flat, dtype=np.float64)
            if counts_flat.size:
                values_flat = np.where(counts_flat == 0, np.nan, values_flat)
            return _reshape_reduced(values_flat)

        def _failure_record(
            group_id: Any,
            group_index: int,
            row_count: int,
            valid_row_count: int,
            message: str,
        ) -> dict[str, Any]:
            return {
                "group_id": group_id,
                "group_index": group_index,
                "row_count": row_count,
                "valid_row_count": valid_row_count,
                "exception_type": "InsufficientObservations",
                "message": message,
            }

        n_groups = len(ordered_groups)
        rest_shape = X_moved.shape[1:]
        reduced_shape = (n_groups, len(stats_out), *rest_shape)
        agg_moved = np.empty(reduced_shape, dtype=np.float64)
        epoch_counts = np.empty(n_groups, dtype=np.int64)
        failures: list[dict[str, Any]] = []

        for group_index, group_id in enumerate(ordered_groups):
            obs_positions = np.asarray(group_positions[group_id], dtype=int)
            group_X = X_moved[obs_positions]
            row_count = int(obs_positions.size)
            epoch_counts[group_index] = row_count

            if rest_shape:
                group_X_flat = group_X.reshape(row_count, -1)
            else:
                group_X_flat = group_X.reshape(row_count, 1)
            if group_X_flat.shape[1] == 0:
                valid_row_count = row_count
            else:
                valid_row_count = int(np.isfinite(group_X_flat).any(axis=1).sum())
            if valid_row_count < min_count:
                message = (
                    f"Group {group_id!r} has {valid_row_count} valid rows, "
                    f"requires at least {min_count}."
                )
                failure = _failure_record(
                    group_id=group_id,
                    group_index=group_index,
                    row_count=row_count,
                    valid_row_count=valid_row_count,
                    message=message,
                )
                if on_insufficient == "raise":
                    raise ValueError(message)
                if on_insufficient == "warn":
                    warnings.warn(message, stacklevel=2)
                failures.append(failure)
                agg_moved[group_index] = np.full((len(stats_out), *rest_shape), np.nan)
                continue

            counts_flat = np.isfinite(group_X_flat).sum(axis=0, dtype=np.int64)
            for stat_index, stat in enumerate(stats_out):
                agg_moved[group_index, stat_index] = _reduce_group(
                    group_X=group_X,
                    group_X_flat=group_X_flat,
                    counts_flat=counts_flat,
                    stat=stat,
                )

        if len(stats_out) == 1:
            moved_dims = ("obs", *other_dims)
            final_dims = self.dims
            agg_values = agg_moved[:, 0, ...]
        else:
            moved_dims = ("obs", "stat", *other_dims)
            final_dims_list: list[str] = []
            for dim in self.dims:
                final_dims_list.append(dim)
                if dim == "obs":
                    final_dims_list.append("stat")
            final_dims = tuple(final_dims_list)
            agg_values = agg_moved

        permutation = [moved_dims.index(dim) for dim in final_dims]
        X_agg = np.transpose(agg_values, axes=permutation)

        unique_groups = np.empty(n_groups, dtype=object)
        unique_groups[:] = ordered_groups

        new_y = None
        if self.y is not None:
            grouped_y: list[Any] = []
            y_consistent = True
            for group_id in ordered_groups:
                values = np.asarray(self.y)[group_positions[group_id]]
                if len(set(values.tolist())) != 1:
                    y_consistent = False
                    break
                grouped_y.append(values[0])
            if y_consistent:
                new_y = np.asarray(grouped_y)

        new_coords = {
            dim: deepcopy(values)
            for dim, values in self.coords.items()
            if dim in self.dims and dim != "obs"
        }
        new_coords["obs"] = unique_groups
        if len(stats_out) > 1:
            new_coords["stat"] = np.asarray(stats_out, dtype=object)
        new_coords["epoch_count"] = epoch_counts

        for key, values in self.coords.items():
            if key == "obs" or key in self.dims:
                continue
            if len(values) != n_obs:
                continue
            grouped_values: list[Any] = []
            consistent = True
            values_array = np.asarray(values, dtype=object)
            for group_id in ordered_groups:
                group_values = values_array[group_positions[group_id]]
                if len(set(group_values.tolist())) != 1:
                    consistent = False
                    break
                grouped_values.append(group_values[0])
            if consistent:
                coord_out = np.empty(n_groups, dtype=object)
                coord_out[:] = grouped_values
                new_coords[key] = coord_out

        meta = deepcopy(self.meta)
        meta.update(
            {
                "aggregated": True,
                "agg_by": by if isinstance(by, str) else None,
                "agg_stats": list(stats_out),
                "min_count": int(min_count),
            }
        )
        if failures:
            meta["aggregate_failures"] = failures

        return replace(
            self,
            X=X_agg,
            y=new_y,
            dims=final_dims,
            ids=unique_groups,
            coords=new_coords,
            meta=meta,
        )

    def aggregate_groups(
        self,
        by: str | np.ndarray | list[Any],
        groups: Sequence[dict[str, Any]],
        min_count: int = 1,
        on_insufficient: str = "raise",
        skip_empty: bool = True,
    ) -> "DataContainer":
        """
        Aggregate selected feature groups with different statistics.

        This is a thin wrapper around :meth:`aggregate` for tabular feature
        containers. Each group spec selects a subset of feature columns and
        applies one or more stats to that subset. The outputs are concatenated
        along the ``feature`` dimension, and each resulting feature name is
        prefixed with its stat (for example ``"mean_band_log_abs_alpha"``).

        Parameters
        ----------
        by : str or array-like
            Group definition for the observation axis. Passed through to
            :meth:`aggregate`.
        groups : sequence of dict
            Ordered group specifications. Each group must provide ``"stats"``
            and may optionally provide include/exclude selectors:

            - ``names`` / ``exclude_names``
            - ``prefixes`` / ``exclude_prefixes``
            - ``suffixes`` / ``exclude_suffixes``
            - ``contains`` / ``exclude_contains``
            - ``regex`` / ``exclude_regex``

            If a group provides no include selectors, it starts from all
            features and then applies exclusions.
        min_count : int, default=1
            Minimum number of valid observations required per group. Passed
            through to :meth:`aggregate`.
        on_insufficient : {"raise", "warn", "collect"}, default="raise"
            Policy applied when a group has fewer than ``min_count`` valid
            observations. Passed through to :meth:`aggregate`.
        skip_empty : bool, default=True
            If True, silently skip group specs that match no features. If
            False, raise a ``ValueError`` when a group matches nothing.

        Returns
        -------
        DataContainer
            Aggregated container with dims ``("obs", "feature")`` and
            stat-prefixed feature names.

        Raises
        ------
        ValueError
            If the container lacks a ``feature`` dimension or coord, no groups
            are provided, a group spec is invalid, multiple groups would emit
            the same output feature name, or no non-empty grouped outputs are
            produced.
        """
        if "feature" not in self.dims:
            raise ValueError("aggregate_groups requires a 'feature' dimension.")
        if "feature" not in self.coords:
            raise ValueError("aggregate_groups requires a 'feature' coordinate.")
        if not groups:
            raise ValueError("`groups` must not be empty.")

        feature_names = np.asarray(self.coords["feature"], dtype=object)
        feature_axis = self.dims.index("feature")
        include_keys = ("names", "prefixes", "suffixes", "contains", "regex")
        exclude_keys = tuple(f"exclude_{key}" for key in include_keys)
        allowed_group_keys = {"name", "stats", *include_keys, *exclude_keys}

        def _normalize_patterns(value: Any) -> tuple[str, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                return (value,)
            return tuple(str(item) for item in value)

        def _selector_mask(spec: dict[str, Any], *, exclude: bool) -> np.ndarray:
            selector_keys = exclude_keys if exclude else include_keys
            mask = np.zeros(feature_names.size, dtype=bool)
            for key in selector_keys:
                patterns = _normalize_patterns(spec.get(key))
                if not patterns:
                    continue
                base_key = key.removeprefix("exclude_")
                if base_key == "names":
                    mask |= np.isin(feature_names.astype(str), patterns)
                elif base_key == "prefixes":
                    mask |= np.array(
                        [
                            any(str(name).startswith(pattern) for pattern in patterns)
                            for name in feature_names
                        ],
                        dtype=bool,
                    )
                elif base_key == "suffixes":
                    mask |= np.array(
                        [
                            any(str(name).endswith(pattern) for pattern in patterns)
                            for name in feature_names
                        ],
                        dtype=bool,
                    )
                elif base_key == "contains":
                    mask |= np.array(
                        [
                            any(pattern in str(name) for pattern in patterns)
                            for name in feature_names
                        ],
                        dtype=bool,
                    )
                elif base_key == "regex":
                    compiled = [re.compile(pattern) for pattern in patterns]
                    mask |= np.array(
                        [
                            any(pattern.search(str(name)) for pattern in compiled)
                            for name in feature_names
                        ],
                        dtype=bool,
                    )
            return mask

        combined_parts: list[np.ndarray] = []
        combined_feature_names: list[str] = []
        aggregate_failures: list[dict[str, Any]] = []
        base_agg: DataContainer | None = None

        for group_index, group in enumerate(groups):
            if not isinstance(group, dict):
                raise ValueError("Each entry in `groups` must be a dict.")

            unknown_keys = sorted(set(group) - allowed_group_keys)
            if unknown_keys:
                raise ValueError(
                    f"Unknown aggregate_groups keys: {unknown_keys}. "
                    f"Supported keys are: {sorted(allowed_group_keys)}"
                )
            if "stats" not in group:
                raise ValueError("Each aggregate_groups spec must include `stats`.")

            stats_spec = group["stats"]
            if isinstance(stats_spec, str):
                stats_out = [stats_spec]
            else:
                stats_out = [str(stat) for stat in stats_spec]
            if not stats_out:
                raise ValueError(
                    "Each aggregate_groups spec must include at least one stat."
                )

            include_mask = _selector_mask(group, exclude=False)
            has_include_selectors = any(
                group.get(key) is not None for key in include_keys
            )
            if not has_include_selectors:
                include_mask = np.ones(feature_names.size, dtype=bool)
            exclude_mask = _selector_mask(group, exclude=True)
            selected_mask = include_mask & ~exclude_mask
            feature_indices = np.flatnonzero(selected_mask)
            if feature_indices.size == 0:
                if skip_empty:
                    continue
                group_name = group.get("name", f"index {group_index}")
                raise ValueError(
                    f"aggregate_groups spec {group_name!r} matched no features."
                )

            subset = self.isel(feature=feature_indices.tolist())
            for stat in stats_out:
                grouped = subset.aggregate(
                    by=by,
                    stats=stat,
                    min_count=min_count,
                    on_insufficient=on_insufficient,
                )
                prefixed_names = [
                    f"{stat}_{name}"
                    for name in np.asarray(grouped.coords["feature"], dtype=object)
                ]
                duplicate_names = sorted(
                    set(prefixed_names).intersection(combined_feature_names)
                )
                if duplicate_names:
                    raise ValueError(
                        "aggregate_groups would emit duplicate feature names: "
                        f"{duplicate_names}"
                    )

                if base_agg is None:
                    base_agg = grouped
                else:
                    if grouped.dims != base_agg.dims:
                        raise ValueError(
                            "aggregate_groups requires all grouped outputs to "
                            "share the same dimensions."
                        )
                    if not np.array_equal(grouped.ids, base_agg.ids):
                        raise ValueError(
                            "aggregate_groups requires all grouped outputs to "
                            "share the same grouped observation ids."
                        )

                combined_parts.append(np.asarray(grouped.X, dtype=np.float64))
                combined_feature_names.extend(prefixed_names)

                failures = grouped.meta.get("aggregate_failures", [])
                for failure in failures:
                    failure_out = deepcopy(failure)
                    failure_out["aggregate_group_index"] = group_index
                    if "name" in group:
                        failure_out["aggregate_group_name"] = group["name"]
                    failure_out["aggregate_stat"] = stat
                    aggregate_failures.append(failure_out)

        if base_agg is None:  # pragma: no cover - guarded above
            raise ValueError("aggregate_groups produced no aggregated outputs.")

        new_coords = deepcopy(base_agg.coords)
        new_coords["feature"] = np.asarray(combined_feature_names, dtype=object)
        meta = deepcopy(self.meta)
        unique_stats = list(
            dict.fromkeys(name.split("_", 1)[0] for name in combined_feature_names)
        )
        meta.update(
            {
                "aggregated": True,
                "agg_by": by if isinstance(by, str) else None,
                "agg_stats": unique_stats,
                "agg_groups": deepcopy(list(groups)),
                "min_count": int(min_count),
            }
        )
        if aggregate_failures:
            meta["aggregate_failures"] = aggregate_failures
        elif "aggregate_failures" in meta:
            del meta["aggregate_failures"]

        X_out = np.concatenate(combined_parts, axis=feature_axis)
        return replace(
            base_agg,
            X=X_out,
            coords=new_coords,
            meta=meta,
        )
