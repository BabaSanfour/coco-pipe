"""
Data Structures
===============

Standardized containers for passing data between Datasets, Preprocessing, and main
modules.

This module provides the `DataContainer`, an N-dimensional tensor wrapper that manages
metadata, coordinates, and labels alongside the raw data matrix. It serves as the
common currency for the entire pipeline.

Examples
--------
>>> import numpy as np
>>> from coco_pipe.io import DataContainer

# 1. Creating a container for EEG Epochs (N_epochs, N_channels, N_time)
>>> X = np.random.randn(10, 64, 500)
>>> container = DataContainer(
...     X=X,
...     dims=('obs', 'channel', 'time'),
...     coords={
...         'channel': ['Fz', 'Cz', 'Pz'], # ... etc
...         'time': np.linspace(0, 1.0, 500)
...     },
...     y=np.random.randint(0, 2, 10),
...     ids=[f'sub-01_trial-{i}' for i in range(10)]
... )

# 2. Creating a container for simple Tabular Features (N_subjects, N_features)
>>> X_tab = np.random.randn(20, 5)
>>> container_tab = DataContainer(
...     X=X_tab,
...     dims=('obs', 'feature'),
...     coords={'feature': ['age', 'IQ', 'response_time', 'power_alpha', 'power_beta']}
... )
"""

import difflib
import fnmatch
import itertools
import logging
import re
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
        Note: The 'obs' dimension is special and typically represents independent samples.
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
    >>> container.coords['channel'][:3]
    ['Fz', 'Cz', 'Pz']
    """

    X: np.ndarray
    dims: Tuple[str, ...]
    coords: Dict[str, Union[List, np.ndarray, Sequence]] = field(default_factory=dict)
    y: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validation
        if self.X.ndim != len(self.dims):
            raise ValueError(
                f"Shape mismatch: X has {self.X.ndim} dims {self.X.shape}, "
                f"but `dims` has {len(self.dims)} labels {self.dims}."
            )

        # Check coords lengths
        for dim, labels in self.coords.items():
            if dim in self.dims:
                axis = self.dims.index(dim)
                if self.X.shape[axis] != len(labels):
                    logger.debug(
                        f"Coord '{dim}' length ({len(labels)}) does not match "
                        f"X dimension {axis} ({self.X.shape[axis]})."
                    )

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.X.shape

    def save(self, path: Union[str, Any]) -> None:
        """
        Save the DataContainer to disk using joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        from pathlib import Path

        import joblib

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        logger.info(f"DataContainer saved to {p}")

    @classmethod
    def load(cls, path: Union[str, Any]) -> "DataContainer":
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
        from pathlib import Path

        import joblib

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        obj = joblib.load(p)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is {type(obj)}, expected {cls.__name__}")

        return obj

    def __repr__(self) -> str:
        dim_strs = [f"{d}={s}" for d, s in zip(self.dims, self.X.shape)]
        return f"<DataContainer [{' x '.join(dim_strs)}], coords={list(self.coords.keys())}>"

    def isel(self, **indexers) -> "DataContainer":
        """
        Select data by integer indices on specified dimensions.

        This method is the integer-index equivalent of `select`. It operates
        directly on the dimensions of the data tensor `X`. It is robust and
        handles metadata splitting/alignment automatically.

        Parameters
        ----------
        **indexers : dict
            Key: Dimension name (e.g., 'obs', 'channel', 'time').
            Value: Integer indices to select. Can be:
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

        self.X.shape[0] if "obs" in self.dims else 0
        obs_dim_idx = self.dims.index("obs") if "obs" in self.dims else -1

        new_coords = self.coords.copy()

        # Apply slicers
        for dim_name, indices in indexers.items():
            if dim_name not in self.dims:
                logger.warning(
                    f"Dimension {dim_name} not in {self.dims}, skipping isel."
                )
                continue

            d_idx = self.dims.index(dim_name)

            # Normalize int to list to preserve dimension
            if isinstance(indices, int):
                indices = [indices]

            # Update specific dim slice
            slices[d_idx] = indices

            # Handle metadata alignment
            dim_len_old = self.X.shape[d_idx]

            # We must be careful not to update coords twice if orthogonal slicing
            # But here we just prepare new_coords values

            for k, v in self.coords.items():
                if dim_name in self.dims and k == dim_name:
                    # This IS the coordinate for this dimension
                    new_coords[k] = np.array(v)[indices]
                elif (
                    len(v) == dim_len_old and k not in self.dims
                ):  # Don't overwrite other dim labels
                    # Heuristic match
                    new_coords[k] = np.array(v)[indices]

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
        covariates: Optional[List[str]] = None,
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
        target : str, default='y'
            Name of the target variable.
            - 'y': Uses `self.y`.
            - Any other string: Looks for the variable in `self.coords`.
        strategy : {'undersample', 'oversample', 'auto'}, default='undersample'
            - 'undersample': Downsample majority classes to match the minority class count.
            - 'oversample': Upsample minority classes (with replacement) to match the majority class.
            - 'auto': Heuristic choice. Uses undersampling if total size remains > 50% of original, else oversampling.
        covariates : list of str, optional
            List of covariate names in `self.coords` to preserve distribution of.
            If provided, the balancing is performed *within* strata defined by these covariates.
        random_state : int, default=42
            Seed for the random number generator.
            Change this value to produce different random subsets (e.g., for bagging).
        **kwargs : dict
            Additional arguments passed to internal logic:
            - n_bins (int): Number of bins for continuous covariates (default 5).
            - binning (str): 'quantile' (default) or 'uniform' binning.
            - prefer_clean_rows (bool): If True, weighs sampling to prefer rows with fewer NaNs/artifacts.

        Returns
        -------
        DataContainer
            A new DataContainer instance with balanced classes.

        Examples
        --------
        >>> # 1. Simple Undersampling of 'y'
        >>> balanced = container.balance(strategy='undersample')

        >>> # 2. Balance based on a metadata column 'condition'
        >>> balanced = container.balance(target='condition')

        >>> # 3. Stratified Balancing (Balance 'y' while preserving 'sex' and 'age' ratios)
        >>> balanced = container.balance(target='y', covariates=['sex', 'age'])

        >>> # 4. Iterative Bootstrapping (Different seeds)
        >>> for seed in [1, 2, 3]:
        ...     subset = container.balance(strategy='undersample', random_state=seed)
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
        # Index is implicitly RangeIndex (0..N-1), which matches DataContainer positional indices

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
            size = {
                c: min_c if strategy == "undersample" else max_c for c in counts.index
            }
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

            for _, g in tmp.groupby("__strata__"):
                sc = g[target].value_counts()
                if len(sc) <= 1:
                    # Cannot balance within a single-class stratum
                    if strategy != "undersample":
                        M = int(counts.max())
                        size_map = {c: M for c in sc.index}
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
                sz = {
                    c: min_c if strategy == "undersample" else max_c
                    for c in counts.index
                }
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
        # indices is a pandas Index (Int64 or similar). Convert to values for safe numpy indexing.
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
        >>> sub = container.select(channel=['Fz', 'Cz'])

        >>> # 2. Wildcard Selection (All Alpha features)
        >>> sub = container.select(feature='*alpha*')

        >>> # 3. Range Selection (Time)
        >>> sub = container.select(time={'>=': 0.1, '<': 0.5})

        >>> # 4. Case-Insensitive Fuzzy Matching
        >>> sub = container.select(channel=['fz'], ignore_case=True)

        >>> # 5. Filter by Target (y)
        >>> sub = container.select(y=['Patient'])

        >>> # 6. Complex Logic (Subjects 1-5 via Operator)
        >>> sub = container.select(subject_id={'>=': 1, '<=': 5})

        >>> # 7. Stratified Selection (First 2 epochs per subject via Callable)
        >>> def first_n(ids, n=2):
        ...     # ... logic ...
        ...     return mask
        >>> sub = container.select(ids=first_n)
        """
        slices = [slice(None)] * self.X.ndim
        self.coords.copy()

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
                        f"Aux coordinate '{key}' len={len(target_arr)} matches no dimension shape {self.X.shape}. Ignoring selection."
                    )
                    continue

            if target_arr is None:
                if key in self.dims and key not in self.coords:
                    logger.warning(
                        f"Selection on dim '{key}' ignored (no coordinates)."
                    )
                    continue
                logger.warning(
                    f"Selection key '{key}' not found in logical dims, y, ids, or aux coords. Ignoring."
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
                        f"Callable query for '{key}' must return boolean array of shape {target_arr.shape}."
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
                    "in": lambda a, b: np.isin(a, b),
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
                        f"Conflicting selections for axis {axis} ({key}) resulted in empty set."
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

        # Update coordinates to match new X
        final_coords = {}
        for coord_name, labels in self.coords.items():
            # Check if coordinate aligns with any dimension
            aligned_dim_idx = -1

            if coord_name in self.dims:
                aligned_dim_idx = self.dims.index(coord_name)
            else:
                # Heuristic: Find matching dimension length
                # Note: Ambiguity if multiple dims have same length.
                # We prioritize 'obs' if length matches, then others.

                # Check obs first
                if obs_dim_idx != -1 and len(labels) == self.X.shape[obs_dim_idx]:
                    aligned_dim_idx = obs_dim_idx
                else:
                    for d_i, d_len in enumerate(self.X.shape):
                        if len(labels) == d_len:
                            aligned_dim_idx = d_i
                            break

            if aligned_dim_idx != -1:
                sl = slices[aligned_dim_idx]
                if isinstance(sl, slice):
                    final_coords[coord_name] = np.array(labels)[sl]
                else:
                    final_coords[coord_name] = np.array(labels)[sl]
            else:
                # Coordinate didn't match any dimension? Drop it to be safe, or keep?
                # If validation passes, this shouldn't happen unless corrupt.
                pass

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

    def flatten(self, preserve: Union[str, List[str]] = "obs") -> "DataContainer":
        """
        Flatten dimensions NOT in `preserve` into a single 'feature' dimension.

        This is useful for preparing N-dimensional data for standard 2D machine
        learning algorithms (scikit-learn). It automatically generates composite
        feature names (e.g., 'Fz_0.1s') for tracking.

        Parameters
        ----------
        preserve : str or List[str], default='obs'
            Dimensions to keep. All other dimensions will be collapsed into a
            single 'feature' dimension.
            - 'obs': Result shape (N_obs, N_features). Standard specifiction.
            - ['obs', 'time']: Result shape (N_obs, N_time, N_features).
              Useful for time-resolved decoding distributions.

        Returns
        -------
        DataContainer
            A new DataContainer with reshaped X and generated 'feature' coordinates.

        Examples
        --------
        >>> # Flatten (10, 64, 500) -> (10, 32000)
        >>> flat = container.flatten(preserve='obs')
        >>> flat.shape
        (10, 32000)
        >>> flat.coords['feature'][0]
        'Fz_0.0'

        >>> # Flatten spatial only, keep time (10, 64, 500) -> (10, 500, 64)
        >>> time_resolved = container.flatten(preserve=['obs', 'time'])
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

        new_shape = tuple(preserved_shape) + (flattened_len,)
        X_flat = X_trans.reshape(new_shape)

        # New Dims
        new_dims = tuple(preserve) + ("feature",)

        # New Coords
        # We keep coords for preserved dimensions.
        new_coords = {k: v for k, v in self.coords.items() if k in preserve}

        flat_coords_list = []
        for d in to_flatten:
            c = self.coords.get(d)
            if c is not None:
                flat_coords_list.append(c)
            else:
                flat_coords_list.append(np.arange(self.X.shape[self.dims.index(d)]))

        # Create Cartesian product
        if flat_coords_list:
            # Check size first to avoid memory explosion?
            total_size = np.prod([len(x) for x in flat_coords_list])
            if total_size < 200000:  # Limit to ~200k features strings
                combo_labels = [
                    "_".join(map(str, x)) for x in itertools.product(*flat_coords_list)
                ]
                new_coords["feature"] = combo_labels

        return replace(
            self,
            X=X_flat,
            dims=new_dims,
            coords=new_coords,
            meta={**self.meta, "flattened_from": self.dims},
        )

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
        >>> # Stack time into observations: (10 obs, 64 ch, 500 time) -> (5000 obs, 64 ch)
        >>> stacked = container.stack(dims=('obs', 'time'), new_dim='obs')
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

        new_shape = (prod_len,) + tuple(preserved_shape)
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

        new_dims_final = (new_dim,) + tuple(preserved)

        return replace(
            self,
            X=X_new,
            dims=new_dims_final,
            ids=new_ids,
            y=new_y,
            coords=new_coords,
            meta={**self.meta, "stacked_from": dims},
        )

    def center(self, dim: str = "time", inplace: bool = False) -> "DataContainer":
        """
        Remove mean along a specified dimension (Centering/Baseline Correction).

        This operation computes the mean along `dim` (ignoring NaNs) and subtracts it.
        Commonly used in EEG for baseline correction (subtracting mean of pre-stimulus interval)
        or centering features before covariance calculation.

        Parameters
        ----------
        dim : str, default='time'
            Dimension name to center over (e.g., 'time', 'channel', 'obs').
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
        >>> container.center(dim='time')
        """
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in {self.dims}")

        axis = self.dims.index(dim)
        X = self.X if inplace else self.X.copy()

        mean = np.nanmean(X, axis=axis, keepdims=True)
        X -= mean

        if inplace:
            return self
        else:
            return replace(self, X=X)

    def zscore(
        self, dim: str = "time", eps: float = 1e-8, inplace: bool = False
    ) -> "DataContainer":
        """
        Standardize (Z-score) along a specified dimension.

        Computes `(X - mean) / std` along the given dimension. Robust to NaNs.
        Useful for normalizing features or standardizing temporal dynamics.

        Parameters
        ----------
        dim : str
            Dimension to standardize.
        eps : float
            Stability epsilon to avoid division by zero.
        inplace : bool

        Returns
        -------
        DataContainer

        Examples
        --------
        >>> # Standardize each channel's timecourse
        >>> container.zscore(dim='time')
        """
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in {self.dims}")

        axis = self.dims.index(dim)
        X = self.X if inplace else self.X.copy()

        mean = np.nanmean(X, axis=axis, keepdims=True)
        std = np.nanstd(X, axis=axis, keepdims=True)

        X -= mean
        X /= std + eps

        if inplace:
            return self
        else:
            return replace(self, X=X)

    def rms_scale(
        self, dim: str = "time", eps: float = 1e-8, inplace: bool = False
    ) -> "DataContainer":
        """
        Scale by Root Mean Square (RMS) amplitude along a dimension.

        Divides data by `sqrt(mean(X**2))` along the dimension.
        Preserves relative shape but normalizes energy.

        Parameters
        ----------
        dim : str
            Dimension to scale.
        eps : float
            Stability epsilon.
        inplace : bool

        Returns
        -------
        DataContainer
        """
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in {self.dims}")

        axis = self.dims.index(dim)
        X = self.X if inplace else self.X.copy()

        mean_sq = np.nanmean(X**2, axis=axis, keepdims=True)
        rms = np.sqrt(mean_sq)

        X /= rms + eps

        if inplace:
            return self
        else:
            return replace(self, X=X)

    def baseline_correction(
        self, dim: str = "time", inplace: bool = False
    ) -> "DataContainer":
        """Alias for center(). Common in EEG."""
        return self.center(dim=dim, inplace=inplace)

    def aggregate(
        self, by: Union[str, np.ndarray, List[Any]], method: str = "mean"
    ) -> "DataContainer":
        """
        Aggregate observations by groups (vectorized implementation).

        Parameters
        ----------
        by : str or array-like
            The key defining the groups.
            - If str: It looks up the key in `self.coords` (e.g., 'subject_id')
              or checks `self.y` if by='y'.
            - If array: A sequence of labels matching the length of the 'obs'
              dimension.
        method : str, default='mean'
            Aggregation method. Options: 'mean', 'median', 'std'.

        Returns
        -------
        DataContainer
             DataContainer with aggregated 'obs' dimension.
        """
        if "obs" not in self.dims:
            raise ValueError("Aggregation requires 'obs' dimension.")

        obs_idx = self.dims.index("obs")
        n_obs = self.X.shape[obs_idx]

        # Resolve 'by' to labels
        if isinstance(by, str):
            if by == "y" and self.y is not None:
                groups = self.y
            elif by in self.coords:
                groups = self.coords[by]
            else:
                raise ValueError(f"Grouping key '{by}' not found in coords or y.")
        else:
            groups = np.array(by)

        if len(groups) != n_obs:
            raise ValueError(
                f"Grouping array length {len(groups)} must match obs length {n_obs}."
            )

        # Transform inputs for DataFrame-based GroupBy (Vectorized)
        # 1. Flatten X to (n_obs, n_features_flat)
        # We need to reshape specifically so obs is axis 0, and rest is flattened
        if obs_idx != 0:
            # Move obs to front if not already
            X_moved = np.moveaxis(self.X, obs_idx, 0)
        else:
            X_moved = self.X

        original_shape = X_moved.shape
        X_flat = X_moved.reshape(original_shape[0], -1)

        # 2. Create DataFrame
        # Using numeric index for columns to avoid overhead
        df = pd.DataFrame(X_flat)
        df["__group__"] = groups

        # 3. Groupby & Aggregate
        grouped = df.groupby("__group__")

        if method == "mean":
            agg_df = grouped.mean()
        elif method == "median":
            agg_df = grouped.median()
        elif method == "std":
            agg_df = grouped.std()
        elif method == "first":
            agg_df = grouped.first()
        else:
            raise ValueError(f"Unknown method '{method}'")

        # 4. Extract Result
        # agg_df index is the unique groups (sorted)
        unique_groups = agg_df.index.to_numpy()
        X_agg_flat = agg_df.values  # (n_groups, n_features_flat)

        # 5. Reshape back
        new_shape = (len(unique_groups),) + original_shape[1:]
        X_agg_moved = X_agg_flat.reshape(new_shape)

        if obs_idx != 0:
            X_agg = np.moveaxis(X_agg_moved, 0, obs_idx)
        else:
            X_agg = X_agg_moved

        # 6. Metadata Handling (y consistency)
        new_y = None
        if self.y is not None:
            # Check if y is consistent per group
            y_df = pd.DataFrame({"g": groups, "y": self.y})
            y_nunique = y_df.groupby("g")["y"].nunique()
            if y_nunique.max() == 1:
                # Take first
                new_y = y_df.groupby("g")["y"].first().reindex(unique_groups).values

        # 7. Update Coords
        new_coords = self.coords.copy()
        new_coords["obs"] = unique_groups

        # Remove aux coords that were length n_obs (unless consistent?)
        keys_to_drop = []
        for k, v in self.coords.items():
            if isinstance(by, str) and k == by:
                continue
            if k == "obs":
                continue
            if len(v) == n_obs and k not in self.dims:
                keys_to_drop.append(k)

        for k in keys_to_drop:
            del new_coords[k]

        return replace(
            self,
            X=X_agg,
            y=new_y,
            ids=unique_groups if method != "std" else None,
            coords=new_coords,
            meta={
                **self.meta,
                "aggregated": True,
                "agg_by": str(by),
                "agg_method": method,
            },
        )
