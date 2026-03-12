"""
Spatiotemporal dimensionality reduction reducers.

This module provides reducers for structured signals where time, trials, or
snapshots are part of the data layout. These reducers follow the shared
`BaseReducer` contract while declaring nonstandard input layouts through the
`capabilities` mapping.

Classes
-------
DMDReducer
    Dynamic Mode Decomposition wrapper based on `pydmd.DMD`.
TRCAReducer
    Task-Related Component Analysis wrapper based on `meegkit.trca.TRCA`.

References
----------
.. [1] Schmid, P. J. (2010). "Dynamic mode decomposition of numerical and
       experimental data". Journal of Fluid Mechanics, 656, 5-28.
.. [2] PyDMD documentation:
       https://github.com/mathLab/PyDMD
.. [3] Nakanishi, M., Wang, Y., Chen, X., Wang, Y.-T., Gao, X., and Jung, T.-P.
       (2018). "Enhancing detection of SSVEPs for a high-speed brain speller
       using task-related component analysis". IEEE Transactions on Biomedical
       Engineering, 65(1), 104-112.
.. [4] MEEGkit documentation:
       https://github.com/nbara/python-meegkit

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from typing import Optional

import numpy as np

from ...utils import import_optional_dependency
from .base import ArrayLike, BaseReducer

__all__ = ["DMDReducer", "TRCAReducer"]

_DMD_DIAGNOSTIC_ATTRS = ("eigs_", "modes_", "reconstructed_data_")
_TRCA_DIAGNOSTIC_ATTRS = ("coef_",)


class DMDReducer(BaseReducer):
    """
    Dynamic Mode Decomposition reducer.

    DMD decomposes a snapshot matrix into dynamic modes that capture coherent
    spatial patterns and their temporal evolution. It is useful for
    spatiotemporal systems such as fluid flows, simulation outputs, and
    structured neural trajectories when data are arranged as
    `(n_features, n_snapshots)`.

    Parameters
    ----------
    n_components : int, default=0
        Number of modes to keep. This is forwarded to PyDMD as `svd_rank`.
        A value of `0` keeps all modes.
    force_transpose : bool, default=False
        If ``True``, transpose incoming arrays from `(n_snapshots, n_features)`
        to `(n_features, n_snapshots)` before fitting and transforming.
    **kwargs : dict
        Additional keyword arguments forwarded to `pydmd.DMD` after signature
        filtering. Common options include `tlsq_rank`, `exact`, and `opt`.

    Attributes
    ----------
    model : pydmd.DMD or None
        Fitted DMD estimator after `fit`.

    Notes
    -----
    Unlike most reducers in this package, DMD expects columns to represent
    time snapshots. This is declared through
    `capabilities["input_layout"] = "features_snapshots"`.

    See Also
    --------
    TRCAReducer : Trial-structured spatiotemporal reducer for labeled repeated signals.
    PHATEReducer : Nonlinear embedding often useful for smooth trajectories.
    UMAPReducer : Nonlinear neighborhood-preserving reducer for tabular inputs.
    PCAReducer : Linear baseline for sample-feature matrices.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import DMDReducer
    >>> x = np.linspace(0, 2 * np.pi, 20)
    >>> t = np.linspace(0, 4 * np.pi, 40)
    >>> X = np.sin(x)[:, None] * np.cos(t)[None, :]
    >>> reducer = DMDReducer(n_components=2)
    >>> _ = reducer.fit(X)
    >>> reducer.eigs_.shape
    (2,)
    >>> reducer.transform(X).shape
    (40, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for DMD.

        Returns
        -------
        dict
            Capability mapping describing DMD as a linear reducer operating on
            `(n_features, n_snapshots)` inputs.
        """
        return self._merge_capabilities(
            super().capabilities,
            input_ndim=2,
            input_layout="features_snapshots",
            has_transform=True,
            has_components=True,
            supported_diagnostics=_DMD_DIAGNOSTIC_ATTRS,
            supported_metadata=("svd_rank", "force_transpose", "n_modes_"),
            has_native_plot=True,
            is_linear=True,
        )

    def __init__(self, n_components: int = 0, force_transpose: bool = False, **kwargs):
        """
        Initialize the DMD reducer.

        Parameters
        ----------
        n_components : int, default=0
            Number of modes to keep. Forwarded as `svd_rank`.
        force_transpose : bool, default=False
            Whether to transpose input arrays before fit/transform.
        **kwargs : dict
            Additional keyword arguments forwarded to `pydmd.DMD` after
            filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.force_transpose = force_transpose

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "DMDReducer":
        """
        Fit DMD on the input snapshot matrix.

        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_snapshots)
            Training data. If `force_transpose=True`, input may instead be
            provided as `(n_snapshots, n_features)`.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        DMDReducer
            Fitted reducer instance.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import DMDReducer
        >>> X = np.random.rand(5, 20)
        >>> reducer = DMDReducer(n_components=2)
        >>> _ = reducer.fit(X)
        >>> reducer.model is not None
        True
        """
        DMD = import_optional_dependency(
            lambda: __import__("pydmd", fromlist=["DMD"]).DMD,
            feature="DMDReducer",
            dependency="pydmd",
            install_hint="pip install coco-pipe[spatiotemporal]",
        )

        params = {key: value for key, value in self.params.items() if key != "svd_rank"}
        self.model = self._build_estimator(
            DMD,
            params=params,
            component_param="svd_rank",
        )

        X_arr = np.asarray(X)
        if self.force_transpose:
            X_arr = X_arr.T

        self.model.fit(X_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project snapshots onto the fitted DMD modes.

        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_snapshots)
            Data to project. If `force_transpose=True`, input may instead be
            provided as `(n_snapshots, n_features)`.

        Returns
        -------
        np.ndarray of shape (n_snapshots, n_components)
            Time-evolution amplitudes projected onto the fitted modes.
        """
        model = self._require_fitted()

        X_arr = np.asarray(X)
        if self.force_transpose:
            X_arr = X_arr.T

        return np.real(X_arr.T @ model.modes)

    @property
    def svd_rank(self) -> int:
        """
        Return the SVD rank used for the DMD decomposition.

        Returns
        -------
        int
            SVD rank.
        """
        return self.n_components

    @property
    def n_modes_(self) -> Optional[int]:
        """
        Return the number of fitted DMD modes.

        Returns
        -------
        int or None
            Mode count or None if not fitted.
        """
        if self.model is not None and hasattr(self.model, "modes"):
            return int(np.asarray(self.model.modes).shape[1])
        return None

    @property
    def eigs_(self) -> np.ndarray:
        """
        Return the DMD eigenvalues.

        Returns
        -------
        np.ndarray
            Eigenvalues associated with the fitted modes.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.eigs

    @property
    def modes_(self) -> np.ndarray:
        """
        Return the DMD spatial modes.

        Returns
        -------
        np.ndarray
            Spatial mode matrix exposed by the fitted DMD model.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.modes

    def get_components(self) -> np.ndarray:
        """
        Return DMD modes in component-major layout.

        Returns
        -------
        np.ndarray
            Mode matrix transposed to `(n_components, n_features)`.
        """
        return self.modes_.T

    @property
    def reconstructed_data_(self) -> np.ndarray:
        """
        Return the reconstructed snapshot matrix from the fitted DMD model.

        Returns
        -------
        np.ndarray
            Reconstructed data exposed by the fitted DMD backend.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.reconstructed_data


class TRCAReducer(BaseReducer):
    """
    Task-Related Component Analysis reducer.

    TRCA learns spatial filters that maximize reproducibility across repeated
    labeled trials. It is primarily useful for trial-based biosignal data such
    as SSVEP or ERP analyses, but the reducer contract is expressed in terms of
    generic `(n_trials, n_channels, n_times)` arrays rather than any specific
    domain object.

    Parameters
    ----------
    n_components : int, default=1
        Number of output components to keep after projection. The underlying
        TRCA backend may produce more `(band, class)` filters; this wrapper
        truncates the projected output to the requested count.
    sfreq : float, default=250.0
        Sampling frequency in Hertz.
    filterbank : list, optional
        Filterbank definition passed to `meegkit.trca.TRCA`. If omitted, a
        single broad band `[(8, 30), (7, 35)]` is used.
    **kwargs : dict
        Additional keyword arguments forwarded to `meegkit.trca.TRCA` after
        signature filtering.

    Attributes
    ----------
    model : meegkit.trca.TRCA or None
        Fitted TRCA estimator after `fit`.

    Notes
    -----
    TRCA requires class labels during fitting. The `y` argument is not
    optional in practice even though it remains optional in the shared reducer
    interface.

    See Also
    --------
    DMDReducer : Snapshot-based spatiotemporal decomposition.
    PCAReducer : Linear reducer for standard sample-feature matrices.
    UMAPReducer : Nonlinear reducer for standard sample-feature matrices.
    PHATEReducer : Nonlinear reducer often used for continuous trajectories.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import TRCAReducer
    >>> X = np.random.rand(8, 4, 50)
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> reducer = TRCAReducer(n_components=1, sfreq=100.0)
    >>> _ = reducer.fit(X, y=y)
    >>> reducer.transform(X).shape
    (8, 1, 50)
    >>> reducer.get_diagnostics()["coef_"].shape[0] >= 1
    True
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for TRCA.

        Returns
        -------
        dict
            Capability mapping describing TRCA as a linear reducer operating on
            `(n_trials, n_channels, n_times)` inputs.
        """
        return self._merge_capabilities(
            super().capabilities,
            input_ndim=3,
            input_layout="trials_channels_times",
            has_transform=True,
            has_components=True,
            supported_diagnostics=_TRCA_DIAGNOSTIC_ATTRS,
            supported_metadata=("n_components", "sfreq", "n_bands", "n_classes"),
            is_linear=True,
        )

    def __init__(
        self,
        n_components: int = 1,
        sfreq: float = 250.0,
        filterbank: Optional[list] = None,
        **kwargs,
    ):
        """
        Initialize the TRCA reducer.

        Parameters
        ----------
        n_components : int, default=1
            Number of projected components to keep.
        sfreq : float, default=250.0
            Sampling frequency in Hertz.
        filterbank : list, optional
            Filterbank definition passed to the TRCA backend.
        **kwargs : dict
            Additional keyword arguments forwarded to `TRCA` after filtering.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.sfreq = sfreq
        self.filterbank = filterbank or [[(8, 30), (7, 35)]]

    @property
    def n_bands(self) -> int:
        """
        Return the number of filter bands.

        Returns
        -------
        int
            Band count.
        """
        return len(self.filterbank)

    @property
    def n_classes(self) -> Optional[int]:
        """
        Return the number of classes identified by TRCA.

        Returns
        -------
        int or None
            Class count or None if not fitted.
        """
        if self.model is not None and hasattr(self.model, "coef_"):
            return int(np.asarray(self.model.coef_).shape[1])
        return None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TRCAReducer":
        """
        Fit TRCA on labeled trial data.

        Parameters
        ----------
        X : ArrayLike of shape (n_trials, n_channels, n_times)
            Training data.
        y : ArrayLike of shape (n_trials,)
            Class labels aligned with trials. This argument is required.

        Returns
        -------
        TRCAReducer
            Fitted reducer instance.

        Raises
        ------
        ValueError
            If the input is not 3-dimensional, if `y` is missing, or if label
            length does not match the number of trials.

        Examples
        --------
        >>> import numpy as np
        >>> from coco_pipe.dim_reduction import TRCAReducer
        >>> X = np.random.rand(6, 3, 40)
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> reducer = TRCAReducer(n_components=1, sfreq=100.0)
        >>> _ = reducer.fit(X, y=y)
        >>> reducer.model is not None
        True
        """
        X_arr = np.asarray(X)
        if X_arr.ndim != 3:
            raise ValueError("TRCA requires 3D input: (n_trials, n_channels, n_times)")
        if y is None:
            raise ValueError("TRCA requires labels `y` during fit().")

        y_arr = np.asarray(y)
        if y_arr.shape[0] != X_arr.shape[0]:
            raise ValueError("TRCA requires one label per trial.")

        TRCA = import_optional_dependency(
            lambda: __import__("meegkit.trca", fromlist=["TRCA"]).TRCA,
            feature="TRCAReducer",
            dependency="meegkit",
            install_hint="pip install coco-pipe[spatiotemporal]",
        )

        self.model = self._build_estimator(
            TRCA,
            params={
                **self.params,
                "sfreq": self.sfreq,
                "filterbank": self.filterbank,
            },
            component_param=None,
        )

        X_transposed = np.transpose(X_arr, (2, 1, 0))
        self.model.fit(X_transposed, y_arr)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Project trial data using the fitted TRCA spatial filters.

        Parameters
        ----------
        X : ArrayLike of shape (n_trials, n_channels, n_times)
            New data to project.

        Returns
        -------
        np.ndarray of shape (n_trials, n_components, n_times)
            Projected trial signals, truncated to `n_components`.

        Raises
        ------
        ValueError
            If the input is not 3-dimensional.
        """
        model = self._require_fitted()

        X_arr = np.asarray(X)
        if X_arr.ndim != 3:
            raise ValueError("TRCA requires 3D input: (n_trials, n_channels, n_times)")

        n_bands, n_classes, _ = model.coef_.shape

        bandpass = import_optional_dependency(
            lambda: __import__("meegkit.utils.trca", fromlist=["bandpass"]).bandpass,
            feature="TRCAReducer.transform",
            dependency="meegkit",
            install_hint="pip install coco-pipe[spatiotemporal]",
        )

        projected = []
        X_tmp = X_arr.transpose(2, 1, 0)

        for band_idx in range(n_bands):
            wp = model.filterbank[band_idx][0]
            ws = model.filterbank[band_idx][1]
            X_filt = bandpass(X_tmp, model.sfreq, Wp=wp, Ws=ws)

            for class_idx in range(n_classes):
                weights = model.coef_[band_idx, class_idx, :]
                signal = np.einsum("c,tcr->tr", weights, X_filt)
                projected.append(signal.T)

        result = np.stack(projected, axis=1)
        if self.n_components and self.n_components < result.shape[1]:
            result = result[:, : self.n_components, :]
        return result

    def get_components(self) -> np.ndarray:
        """
        Return the learned TRCA spatial filters.

        Returns
        -------
        np.ndarray
            Spatial filter tensor with shape determined by the TRCA backend.
        """
        model = self._require_fitted(method_name="get_components")
        return model.coef_
