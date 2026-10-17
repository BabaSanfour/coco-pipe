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

import fnmatch
import logging
import re
import warnings
from collections.abc import Sequence
from copy import deepcopy
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .utils import make_strata, sample_indices

logger = logging.getLogger(__name__)


def _infer_dim(
    name: str, values: Any, sizes: dict[str, int], obs_dim: str
) -> str | None:
    """Return the dimension a coordinate belongs to, or None."""
    if name in sizes:
        return name
    if isinstance(values, str) or np.ndim(values) != 1:
        return None
    length = len(values)
    prefixed = [
        dim for dim in sizes if name.startswith(f"{dim}_") and sizes[dim] == length
    ]
    if prefixed:
        return max(prefixed, key=len)
    if sizes[obs_dim] == length:
        return obs_dim
    return next((dim for dim in sizes if sizes[dim] == length), None)


_OPERATORS = {
    ">": np.greater,
    "<": np.less,
    ">=": np.greater_equal,
    "<=": np.less_equal,
    "==": np.equal,
    "!=": np.not_equal,
    "in": np.isin,
}


def _match(key: str, values: np.ndarray, query: Any, ignore_case: bool) -> np.ndarray:
    """Return the boolean mask of ``values`` matched by a ``select`` query."""
    if callable(query):
        mask = np.asarray(query(values))
        if mask.shape != values.shape:
            raise ValueError(
                f"Callable query for '{key}' must return boolean array of "
                f"shape {values.shape}."
            )
        return mask.astype(bool)
    if isinstance(query, dict):
        mask = np.ones(len(values), dtype=bool)
        for operator, value in query.items():
            if operator not in _OPERATORS:
                raise ValueError(
                    f"Unknown operator '{operator}'. Supported: {list(_OPERATORS)}"
                )
            mask &= _OPERATORS[operator](values, value)
        return mask
    queries = np.array(query, ndmin=1).tolist()
    if values.dtype.kind not in "USO":
        return np.isin(values, queries)
    labels = np.array([str(v).lower() if ignore_case else str(v) for v in values])
    queries = [str(q).lower() if ignore_case else str(q) for q in queries]
    mask = np.isin(labels, queries)
    for pattern in queries:
        if "*" in pattern or "?" in pattern:
            regex = re.compile(fnmatch.translate(pattern))
            mask |= np.array([bool(regex.match(label)) for label in labels])
    return mask


def _mad(grouped: Any, counts: xr.DataArray) -> xr.DataArray:
    return grouped.map(lambda x: abs(x - x.median("obs")).median("obs"))


def _iqr(grouped: Any, counts: xr.DataArray) -> xr.DataArray:
    quantiles = grouped.quantile([0.25, 0.75])
    return quantiles.sel(quantile=0.75, drop=True) - quantiles.sel(
        quantile=0.25, drop=True
    )


# Each statistic takes the grouped data and the number of finite values per cell.
_AGGREGATE_STATS = {
    "mean": lambda grouped, counts: grouped.mean(),
    "median": lambda grouped, counts: grouped.median(),
    "std": lambda grouped, counts: grouped.std(),
    "var": lambda grouped, counts: grouped.var(),
    "sem": lambda grouped, counts: grouped.std() / np.sqrt(counts),
    "mad": _mad,
    "iqr": _iqr,
    "min": lambda grouped, counts: grouped.min(),
    "max": lambda grouped, counts: grouped.max(),
    "count": lambda grouped, counts: counts,
    "first": lambda grouped, counts: grouped.first(skipna=False),
}


_SAVE_FORMAT = "coco_pipe.DataContainer/1"


class DataContainer:
    """
    Generic container for N-dimensional neurophysiological data.

    A thin layer over one :class:`xarray.DataArray` (``data``): every
    coordinate, ``y`` and ``ids`` belong to one dimension, and ``meta`` is its
    ``attrs``. ``X``, ``dims``, ``coords``, ``y``, ``ids`` and ``meta`` are views
    onto it.

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
        Labels along one dimension. A coordinate belongs to the dimension it is
        named after, else to the dimension it is prefixed with (e.g.
        ``feature_family``) or, failing that, to the observation dimension or
        the first other dimension of the same length. A value that fits no
        dimension is stored in `meta`.
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

    def __init__(
        self,
        X: np.ndarray | xr.DataArray,
        dims: Sequence[str] | None = None,
        coords: dict[str, Any] | None = None,
        y: np.ndarray | None = None,
        ids: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        *,
        _coord_dims: dict[str, str] | None = None,
    ):
        if isinstance(X, xr.DataArray):
            self.data = X
            return
        X = np.asarray(X)
        dims = tuple(dims)
        if X.ndim != len(dims):
            raise ValueError(
                f"Shape mismatch: X has {X.ndim} dims {X.shape}, "
                f"but `dims` has {len(dims)} labels {dims}."
            )
        sizes = dict(zip(dims, X.shape, strict=True))
        obs_dim = "obs" if "obs" in sizes else dims[0]
        attached = {}
        meta = dict(meta or {})
        for name, values in (coords or {}).items():
            dim = (_coord_dims or {}).get(name)
            if dim not in sizes or np.ndim(values) != 1 or len(values) != sizes[dim]:
                dim = _infer_dim(name, values, sizes, obs_dim)
            if dim is None:
                meta[name] = values
            else:
                attached[name] = (dim, values)
        for name, values in (("y", y), ("ids", ids)):
            if values is not None:
                if name in attached:
                    raise ValueError(f"'{name}' cannot also be a coordinate name.")
                attached[name] = (obs_dim, values)
        self.data = xr.DataArray(X, dims=dims, coords=attached, attrs=meta)

    @property
    def X(self) -> np.ndarray:
        return self.data.values

    @property
    def dims(self) -> tuple[str, ...]:
        return tuple(self.data.dims)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def coords(self) -> MappingProxyType:
        return MappingProxyType(
            {
                name: coord.values
                for name, coord in self.data.coords.items()
                if name not in ("y", "ids")
            }
        )

    @property
    def y(self) -> np.ndarray | None:
        return self.data.coords["y"].values if "y" in self.data.coords else None

    @property
    def ids(self) -> np.ndarray | None:
        return self.data.coords["ids"].values if "ids" in self.data.coords else None

    @property
    def meta(self) -> dict[str, Any]:
        return self.data.attrs

    @meta.setter
    def meta(self, value: dict[str, Any]) -> None:
        self.data.attrs = dict(value)

    def to_xarray(self) -> xr.DataArray:
        """Return the underlying :class:`xarray.DataArray`.

        ``y`` and ``ids`` are coordinates of the observation dimension, and
        ``meta`` is ``attrs``.
        """
        return self.data

    def replace(self, **changes: Any) -> "DataContainer":
        """Return a copy with some of X, dims, coords, y, ids and meta replaced."""
        fields = {
            "X": self.X,
            "dims": self.dims,
            "coords": self.coords,
            "y": self.y,
            "ids": self.ids,
            "meta": self.meta,
            **changes,
        }
        coord_dims = {name: coord.dims[0] for name, coord in self.data.coords.items()}
        return DataContainer(**fields, _coord_dims=coord_dims)

    def save(self, path: str | Any) -> None:
        """
        Save the container to ``path`` with joblib.

        The file holds plain arrays and dictionaries (the data, each
        coordinate with its dimension, ``y``, ``ids`` and ``meta``), not a
        pickled instance, so it can still be read after this class changes.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        from ._serialization import save_object

        save_object(
            {
                "format": _SAVE_FORMAT,
                "X": self.X,
                "dims": self.dims,
                "coords": {
                    name: (coord.dims, coord.values)
                    for name, coord in self.data.coords.items()
                },
                "meta": dict(self.meta),
            },
            path,
        )

    def observation_frame(self) -> pd.DataFrame:
        """Return observation-aligned coordinates and stable sample IDs."""
        if "obs" not in self.dims:
            raise ValueError("DataContainer has no 'obs' dimension.")
        n_obs = self.data.sizes["obs"]
        frame = pd.DataFrame(self._obs_coords(), index=np.arange(n_obs))
        frame.columns = frame.columns.astype(str)
        if self.ids is not None:
            frame["sample_id"] = self.ids.astype(str)
        elif "sample_id" not in frame:
            frame["sample_id"] = [f"sample-{idx:06d}" for idx in range(n_obs)]
        return frame

    def _obs_coords(self) -> dict[str, np.ndarray]:
        """Return the coordinates of the observation dimension."""
        return {
            name: coord.values
            for name, coord in self.data.coords.items()
            if coord.dims == ("obs",) and name not in ("y", "ids")
        }

    @classmethod
    def load(cls, path: str | Any) -> "DataContainer":
        """
        Load a container saved with :meth:`save`.

        Parameters
        ----------
        path : str or Path
            Source file path.

        Returns
        -------
        DataContainer

        Raises
        ------
        FileNotFoundError
            If ``path`` does not exist.
        ValueError
            If the file does not hold a saved container.
        """
        from ._serialization import load_object

        payload = load_object(path)
        if not isinstance(payload, dict) or payload.get("format") != _SAVE_FORMAT:
            raise ValueError(f"{path} does not hold a saved DataContainer.")
        return cls(
            xr.DataArray(
                payload["X"],
                dims=payload["dims"],
                coords=payload["coords"],
                attrs=payload["meta"],
            )
        )

    @classmethod
    def concat(
        cls,
        containers: Sequence["DataContainer"],
        fill_condition_from_meta: bool = True,
        join: str = "exact",
    ) -> "DataContainer":
        """Concatenate containers along their observation axis.

        All containers must use matching dimensions. A labelled dimension
        (``channel``, ``feature``, ...) is matched by label, not by position:
        each container is put in the label order of the first. An unlabelled
        dimension must have the same size in every container.
        Observation-aligned coordinates are concatenated, filling missing
        entries with ``None`` when a coordinate is absent from a container.

        Parameters
        ----------
        containers : sequence of DataContainer
            One or more containers to concatenate.
        fill_condition_from_meta : bool, default=True
            If no observation-level ``condition`` coordinate is available,
            create one from each container's ``meta["condition"]`` value.
        join : {"exact", "inner"}, default="exact"
            How the labels of the other dimensions must agree. ``"exact"``:
            every container has the same labels, in any order. ``"inner"``:
            only the labels all containers share are kept.

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
        if join not in {"exact", "inner"}:
            raise ValueError("`join` must be 'exact' or 'inner'.")

        arrays = [container.data for container in containers]
        for dim in base.dims:
            if dim == "obs" or dim not in base.data.indexes:
                continue
            if any(dim not in array.indexes for array in arrays):
                raise ValueError(f"Some containers have no '{dim}' labels.")
            labels = [array.indexes[dim].tolist() for array in arrays]
            if join == "inner":
                shared = set.intersection(*(set(values) for values in labels))
                keep = [value for value in labels[0] if value in shared]
                if not keep:
                    raise ValueError(f"The containers share no '{dim}' label.")
            elif any(set(values) != set(labels[0]) for values in labels[1:]):
                raise ValueError(
                    f"The containers have different '{dim}' labels; pass "
                    "join='inner' to keep the labels they share."
                )
            else:
                keep = labels[0]
            arrays = [
                array if values == keep else array.sel({dim: keep})
                for array, values in zip(arrays, labels, strict=True)
            ]
        other_sizes = {dim: n for dim, n in arrays[0].sizes.items() if dim != "obs"}
        if any(
            {dim: n for dim, n in array.sizes.items() if dim != "obs"} != other_sizes
            for array in arrays[1:]
        ):
            raise ValueError("All containers must have matching non-obs dimensions.")

        if fill_condition_from_meta and not any(
            "condition" in array.coords for array in arrays
        ):
            arrays = [
                array.assign_coords(
                    condition=(
                        "obs",
                        [array.attrs.get("condition")] * array.sizes["obs"],
                    )
                )
                for array in arrays
            ]
        # y and ids are kept only when every container has them; any other
        # observation coordinate missing from a container is filled with None.
        obs_names = {
            name
            for array in arrays
            for name, coord in array.coords.items()
            if coord.dims == ("obs",)
        }
        for name in ("y", "ids"):
            if not all(name in array.coords for array in arrays):
                arrays = [array.drop_vars(name, errors="ignore") for array in arrays]
                obs_names.discard(name)
        arrays = [
            array.assign_coords(
                {
                    name: ("obs", np.full(array.sizes["obs"], None, dtype=object))
                    for name in obs_names - set(array.coords)
                }
            )
            for array in arrays
        ]
        data = xr.concat(
            arrays,
            dim="obs",
            coords="minimal",
            compat="override",
            join="override",
            combine_attrs="drop",
        )
        data.attrs = {
            "source": "concat",
            "conditions": [container.meta.get("condition") for container in containers],
        }
        return cls(data)

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
        data: dict[str, np.ndarray] = {}
        if include_ids:
            if self.ids is None:
                raise ValueError("`include_ids=True` requires `DataContainer.ids`.")
            data[id_col] = self.ids
        coords = self._obs_coords()
        obs_labels = coords.pop("obs", None)
        if include_obs_coord and obs_labels is not None:
            data["obs"] = obs_labels
        data.update(coords)
        if include_y and self.y is not None:
            data.setdefault(y_col, self.y)
        return pd.DataFrame(
            {name: values.astype(object) for name, values in data.items()}
        )

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
        # An integer keeps its dimension, with length one.
        indexers = {
            dim: [index] if isinstance(index, (int, np.integer)) else index
            for dim, index in indexers.items()
        }
        return DataContainer(self.data.isel(indexers))

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

    def select(self, ignore_case: bool = False, **selections) -> "DataContainer":
        """
        Select the entries whose coordinate values match a query.

        Parameters
        ----------
        ignore_case : bool, default=False
            If True, string matching is case-insensitive (e.g., 'fz' matches 'Fz').
        **selections : dict
            Key is the name of a coordinate, 'y' or 'ids'. Value is the query:

            1. **List or single value**: the values to keep. Strings can use
               shell-style wildcards ('*', '?').
            2. **Dictionary**: operators mapped to values, all of which must
               hold. Keys: '>', '<', '>=', '<=', '==', '!=', 'in'.
            3. **Callable**: a function taking the coordinate array and
               returning a boolean mask.

        Returns
        -------
        DataContainer
            The selected subset.

        Raises
        ------
        KeyError
            If a key is not a coordinate of the container.
        ValueError
            If a selection matches nothing.

        Examples
        --------
        >>> sub = container.select(channel=["Fz", "Cz"])
        >>> sub = container.select(feature="*alpha*")
        >>> sub = container.select(time={">=": 0.1, "<": 0.5})
        >>> sub = container.select(channel=["fz"], ignore_case=True)
        >>> sub = container.select(y=["Patient"])
        """
        masks: dict[str, np.ndarray] = {}
        for key, query in selections.items():
            if key not in self.data.coords:
                raise KeyError(
                    f"'{key}' is not a coordinate of this container. "
                    f"Available: {sorted(self.data.coords)}."
                )
            coord = self.data.coords[key]
            mask = _match(key, coord.values, query, ignore_case)
            dim = coord.dims[0]
            masks[dim] = masks.get(dim, True) & mask
            if not masks[dim].any():
                raise ValueError(
                    f"Selection for '{key}' resulted in empty set. Query: {query}"
                )
        return DataContainer(self.data.isel(masks))

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
        preserve = [preserve] if isinstance(preserve, str) else list(preserve)
        for dim in preserve:
            if dim not in self.dims:
                raise ValueError(f"Dimension '{dim}' not found in {self.dims}.")
        to_flatten = [dim for dim in self.dims if dim not in preserve]
        if not to_flatten:
            return self

        stacked = self._labelled(to_flatten).stack(_flat=to_flatten)
        index = stacked.indexes["_flat"]
        stacked = (
            stacked.reset_index("_flat", drop=True)
            .rename(_flat="feature")
            .transpose(*preserve, "feature")
        )
        if len(index) < 200000:
            stacked = stacked.assign_coords(
                feature=[sep.join(map(str, labels)) for labels in index]
            )
        else:
            warnings.warn(
                f"flatten(): {len(index)} features exceeds the 200000 label cap; "
                "'feature' coordinate names were not generated.",
                stacklevel=2,
            )
        stacked.attrs = {**self.meta, "flattened_from": self.dims}
        return DataContainer(stacked)

    def _labelled(self, dims: Sequence[str]) -> xr.DataArray:
        """Return the data with positions as labels on the unlabelled ``dims``."""
        return self.data.assign_coords(
            {
                dim: np.arange(self.data.sizes[dim])
                for dim in dims
                if dim not in self.data.coords
            }
        )

    def feature_schema(self) -> "pd.DataFrame | None":
        """Return feature-axis metadata, or None when no feature coord exists.

        Only coordinates aligned to the feature axis are included. ``feature_*``
        metadata is mapped to canonical schema names such as ``family`` and
        ``measure``; primary dimension coords folded by ``flatten()`` are used
        in the feature labels and are not recovered as structured metadata.
        """
        if "feature" not in self.data.coords:
            return None
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
        for name, coord in self.data.coords.items():
            if coord.dims == ("feature",) and name != "feature":
                data.setdefault(aliases.get(name, name), list(coord.values))
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
        dims = list(dims)
        for dim in dims:
            if dim not in self.dims:
                raise ValueError(f"Dimension '{dim}' not found in {self.dims}")

        stacked = self._labelled(dims).stack(_stacked=dims)
        index = stacked.indexes["_stacked"]
        stacked = stacked.reset_index("_stacked", drop=True)
        if "ids" in stacked.coords and "obs" in dims:
            # Composite IDs such as "sub-0_t-0": the ID of the observation, then
            # the label along each of the other stacked dimensions.
            parts = [
                stacked.coords["ids"].values
                if dim == "obs"
                else index.get_level_values(dim)
                for dim in dims
            ]
            stacked = stacked.assign_coords(
                ids=(
                    "_stacked",
                    ["_".join(map(str, row)) for row in zip(*parts, strict=True)],
                )
            )
        stacked = stacked.rename(_stacked=new_dim).transpose(new_dim, ...)
        stacked.attrs = {
            **self.meta,
            "stacked_from": tuple(dims),
            "stacked_shapes": tuple(self.data.sizes[dim] for dim in dims),
            "_stacked_snapshot": {
                "dims": self.dims,
                "coords": {
                    name: (coord.dims, coord.values)
                    for name, coord in self.data.coords.items()
                    if coord.dims[0] in dims
                },
            },
        }
        return DataContainer(stacked)

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
        if "stacked_from" not in self.meta or "stacked_shapes" not in self.meta:
            raise ValueError(
                "Cannot unstack: Metadata 'stacked_from' or 'stacked_shapes' not found."
                "Ensure data was processed with .stack() or metadata is preserved."
            )
        new_dims = list(self.meta["stacked_from"])
        new_shapes = self.meta["stacked_shapes"]
        if int(np.prod(new_shapes)) != self.data.sizes[dim]:
            raise ValueError(
                f"Shape mismatch: {dim} has length {self.data.sizes[dim]}, "
                f"but product of new_shapes {new_shapes} is {int(np.prod(new_shapes))}"
            )
        snapshot = self.meta.get("_stacked_snapshot", {})

        # Coordinates along `dim` describe stacked entries: they are replaced by
        # those the dimensions had before stacking.
        levels = [f"_{name}" for name in new_dims]
        index = pd.MultiIndex.from_product([range(n) for n in new_shapes], names=levels)
        data = self.data.drop_vars(
            [name for name, coord in self.data.coords.items() if coord.dims == (dim,)]
        )
        data = (
            data.assign_coords(xr.Coordinates.from_pandas_multiindex(index, dim))
            .unstack(dim)
            .drop_vars(levels)
            .rename(dict(zip(levels, new_dims, strict=True)))
            .assign_coords(snapshot.get("coords", {}))
        )
        order = [
            name for old in self.dims for name in (new_dims if old == dim else [old])
        ]
        if set(snapshot.get("dims", ())) == set(order):
            order = snapshot["dims"]
        data = data.transpose(*order)
        data.attrs = {
            **{
                key: value
                for key, value in self.meta.items()
                if key != "_stacked_snapshot"
            },
            "unstacked_from": dim,
        }
        return DataContainer(data)

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

        feature_dim = feature_dim or self.dims[-1]
        if names is None:
            names = np.arange(X.shape[-1])
        elif len(names) != X.shape[-1]:
            raise ValueError(
                f"`names` length {len(names)} does not match new "
                f"feature count {X.shape[-1]}."
            )
        # Everything not on the replaced dimension is kept as it is.
        kept = self.data.isel({feature_dim: 0}, drop=True).coords
        data = xr.DataArray(
            X,
            dims=[new_dim_name if dim == feature_dim else dim for dim in self.dims],
            coords=kept,
            attrs={
                **self.meta,
                "with_features_from": feature_dim,
                "with_features_to": new_dim_name,
            },
        )
        return DataContainer(data.assign_coords({new_dim_name: np.asarray(names)}))

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
        n_obs = self.data.sizes["obs"]
        if min_count < 1:
            raise ValueError("`min_count` must be at least 1.")
        if on_insufficient not in {"raise", "warn", "collect"}:
            raise ValueError("`on_insufficient` must be one of: raise, warn, collect.")

        stats = [stats] if isinstance(stats, str) else [str(stat) for stat in stats]
        stats = [stat.removeprefix("obs-") for stat in stats]
        if not stats:
            raise ValueError("`stats` must not be empty.")
        invalid_stats = sorted(set(stats) - set(_AGGREGATE_STATS))
        if invalid_stats:
            raise ValueError(
                f"Unknown stats: {invalid_stats}. Supported stats are: "
                f"{sorted(_AGGREGATE_STATS)}"
            )

        if not isinstance(by, str):
            groups = list(by)
        elif by == "y" and self.y is not None:
            groups = list(self.y)
        elif by in self.coords:
            groups = list(self.coords[by])
        else:
            raise ValueError(f"Grouping key '{by}' not found in coords or y.")
        if len(groups) != n_obs:
            raise ValueError(
                f"Grouping array length {len(groups)} must match obs length {n_obs}."
            )
        labels = np.empty(n_obs, dtype=object)
        labels[:] = groups
        # Groups are numbered in order of first appearance.
        codes, group_ids = pd.factorize(labels, use_na_sentinel=False)
        first_rows = np.unique(codes, return_index=True)[1]
        row_counts = np.bincount(codes)
        group = xr.DataArray(codes, dims="obs", name="_group")

        other_dims = [dim for dim in self.dims if dim != "obs"]
        finite = np.isfinite(self.data.drop_vars(self.data.coords))
        cell_counts = finite.groupby(group).sum()
        valid_rows = finite.any(other_dims) if finite.size else finite.all(other_dims)
        valid_counts = valid_rows.groupby(group).sum().values

        failures: list[dict[str, Any]] = []
        for index in np.flatnonzero(valid_counts < min_count):
            message = (
                f"Group {group_ids[index]!r} has {valid_counts[index]} valid rows, "
                f"requires at least {min_count}."
            )
            if on_insufficient == "raise":
                raise ValueError(message)
            if on_insufficient == "warn":
                warnings.warn(message, stacklevel=2)
            failures.append(
                {
                    "group_id": group_ids[index],
                    "group_index": int(index),
                    "row_count": int(row_counts[index]),
                    "valid_row_count": int(valid_counts[index]),
                    "exception_type": "InsufficientObservations",
                    "message": message,
                }
            )

        grouped = self.data.drop_vars(self.data.coords).groupby(group)
        # Float data keeps its precision (float32 embeddings stay float32).
        dtype = self.X.dtype if np.issubdtype(self.X.dtype, np.floating) else np.float64
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            results = [
                _AGGREGATE_STATS[stat](grouped, cell_counts).astype(dtype)
                for stat in stats
            ]
        results = [
            result if stat in ("count", "first") else result.where(cell_counts > 0)
            for stat, result in zip(stats, results, strict=True)
        ]
        if len(stats) == 1:
            data = results[0]
            dims = list(self.dims)
        else:
            data = xr.concat(results, dim=pd.Index(stats, dtype=object, name="stat"))
            dims = [
                name
                for dim in self.dims
                for name in ((dim, "stat") if dim == "obs" else (dim,))
            ]
        sufficient = xr.DataArray(valid_counts >= min_count, dims="_group")
        data = data.where(sufficient).rename(_group="obs").transpose(*dims)

        # A coordinate of the observations is kept when it has one value per group.
        frame = pd.DataFrame(self._obs_coords()).drop(columns="obs", errors="ignore")
        if self.y is not None:
            frame["y"] = self.y
        constant = [
            name
            for name in frame
            if (frame[name].groupby(codes).nunique(dropna=False) == 1).all()
        ]
        unique_groups = np.empty(len(group_ids), dtype=object)
        unique_groups[:] = list(group_ids)
        data = data.assign_coords(
            {
                **{
                    name: coord
                    for name, coord in self.data.coords.items()
                    if "obs" not in coord.dims
                },
                "obs": unique_groups,
                "ids": ("obs", unique_groups),
                "epoch_count": ("obs", row_counts),
                **{
                    name: ("obs", frame[name].to_numpy()[first_rows])
                    for name in constant
                },
            }
        )
        data.attrs = {
            **self.meta,
            "aggregated": True,
            "agg_by": by if isinstance(by, str) else None,
            "agg_stats": stats,
            "min_count": int(min_count),
        }
        if failures:
            data.attrs["aggregate_failures"] = failures
        return DataContainer(data)

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

                combined_parts.append(np.asarray(grouped.X))
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

        new_coords = dict(base_agg.coords)
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
        return base_agg.replace(
            X=X_out,
            coords=new_coords,
            meta=meta,
        )
