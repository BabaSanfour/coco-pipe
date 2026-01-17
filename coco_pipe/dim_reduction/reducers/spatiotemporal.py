"""
Spatiotemporal Reducers
=======================

This module implements dimensionality reduction techniques specifically designed
for spatiotemporal data (e.g., M/EEG, fluid dynamics), such as Dynamic Mode
Decomposition (DMD) and Task-Related Component Analysis (TRCA).

Classes
-------
DMDReducer
    Dynamic Mode Decomposition (pydmd wrapper).
TRCAReducer
    Task-Related Component Analysis (meegkit wrapper).

References
----------
.. [1]  Schmid, P. J. (2010). Dynamic mode decomposition of numerical and experimental data.
        Journal of fluid mechanics, 656, 5-28.
.. [2]  PyDMD: Python Dynamic Mode Decomposition. https://github.com/mathLab/PyDMD
.. [3]  Nakanishi, M., et al. (2017). "Enhancing detection of SSVEPs for a high-speed
        brain-speller using task-related component analysis". IEEE Transactions on
        Biomedical Engineering.
.. [4]  MEEGkit: https://github.com/nbara/python-meegkit

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-06
"""

from typing import Optional

import numpy as np
from meegkit.trca import TRCA
from meegkit.utils.trca import bandpass
from pydmd import DMD

from .base import ArrayLike, BaseReducer


class DMDReducer(BaseReducer):
    """
    Dynamic Mode Decomposition (DMD) reducer.

    DMD decomposes a time-series dataset into dynamic modes, capturing the
    temporal evolution of spatial patterns. It is particularly useful for
    extracting coherent structures from fluid flows or neural activity.

    Power User Access
    -----------------
    The underlying pydmd model is accessible via ``self.model``.
    Useful attributes include:
    - ``self.model.eigs``: Eigenvalues of the dynamic modes.
    - ``self.model.modes``: Spatial modes (dynamic modes).
    - ``self.model.reconstructed_data``: Reconstructed data.

    Parameters
    ----------
    n_components : int, default=0
        Number of modes to keep (svd_rank in pydmd). If 0, keeps all.
    **kwargs : dict
        Additional arguments passed to pydmd.DMD.
        Common arguments:
        - tlsq_rank : int, default=0
        - exact : bool, default=False
        - opt : bool, default=False

    Attributes
    ----------
    model : pydmd.DMD
        The underlying fitted DMD estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.spatiotemporal import DMDReducer
    >>> # Create a simple spatiotemporal signal
    >>> # X shape: (n_features, n_snapshots)
    >>> x = np.linspace(0, 10, 100)
    >>> t = np.linspace(0, 10, 100)
    >>> X = np.sin(x)[:, None] * np.cos(t)[None, :]
    >>> reducer = DMDReducer(n_components=2)
    >>> reducer.fit(X)
    >>> print(f"{reducer.eigs_.shape}")
    (2,)
    """

    def __init__(self, n_components: int = 0, **kwargs):
        # pydmd uses 'svd_rank' for n_components
        kwargs["svd_rank"] = n_components
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "DMDReducer":
        """
        Fit the DMD model.

        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_snapshots)
            Training data.
            **CRITICAL NOTE**: PyDMD expects the input matrix such that columns
            are snapshots in time. This input shape (features x samples) differs
            from the standard sklearn convention (samples x features).

        y : Ignored
            Not used.

        Returns
        -------
        self : DMDReducer
            Returns the instance itself.
        """
        # Initialize DMD
        self.model = DMD(**self.params)

        # Fit expects (n_features, n_snapshots)
        self.model.fit(X)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.

        Parameters
        ----------
        X : ArrayLike of shape (n_features, n_snapshots)
            New data. Must match the feature dimension of training data.

        Returns
        -------
        X_new : np.ndarray of shape (n_snapshots, n_components)
             Projected data (time dynamics amplitudes).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("DMDReducer must be fitted before calling transform().")

        modes = self.model.modes

        X_arr = np.array(X)
        return np.real(X_arr.T @ modes)

    @property
    def eigs_(self) -> np.ndarray:
        """
        Eigenvalues of the dynamic modes.

        Returns
        -------
        eigs_ : np.ndarray
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.eigs

    @property
    def modes_(self) -> np.ndarray:
        """
        Spatial modes (dynamic modes).

        Returns
        -------
        modes_ : np.ndarray
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.modes


class TRCAReducer(BaseReducer):
    """
    Task-Related Component Analysis (TRCA) reducer.

    TRCA finds linear combinations of channels that maximize the reproducibility
    of the signal components across trials. Excellent for SSVEP-based BCI and
    ERP analysis.

    Parameters
    ----------
    n_components : int, default=1
        Number of components to keep.
    **kwargs : dict
        Additional arguments passed to TRCA.

    Attributes
    ----------
    model : meegkit.trca.TRCA
        The underlying TRCA estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.spatiotemporal import TRCAReducer
    >>> # Mock data: 10 trials, 5 channels, 100 timepoints
    >>> # Standard MNE shape: (n_trials, n_channels, n_times)
    >>> X = np.random.rand(10, 5, 100)
    >>> reducer = TRCAReducer(n_components=1)
    >>> reducer.fit(X)
    >>> X_tr = reducer.transform(X)
    >>> print(X_tr.shape)
    (10, 1, 100)
    """

    def __init__(self, n_components: int = 1, **kwargs):
        super().__init__(n_components=n_components, **kwargs)
        self.model = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TRCAReducer":
        """
        Fit the TRCA model.

        Parameters
        ----------
        X : ArrayLike of shape (n_trials, n_channels, n_times)
            Training data. TRCA requires 3D structure (Trials x Channels x Time).
            **CRITICAL NOTE**: This deviates from sklearn's (n_samples, n_features).
            Typically, X comes from an MNE Epochs object.
        y : Ignored
            Not used.

        Returns
        -------
        self : TRCAReducer
            Returns the instance itself.
        """
        X_arr = np.array(X)
        if X_arr.ndim != 3:
            raise ValueError("TRCA requires 3D input: (n_trials, n_channels, n_times)")

        # Initialize TRCA
        self.model = TRCA(**self.params)

        # Input X from MNE is (n_trials, n_channels, n_times).
        # We transform (trials, chans, times) -> (times, chans, trials)
        X_transposed = np.transpose(X_arr, (2, 1, 0))

        self.model.fit(X_transposed, y)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X.

        Parameters
        ----------
        X : ArrayLike of shape (n_trials, n_channels, n_times)
             New data.

        Returns
        -------
        X_new : np.ndarray of shape (n_trials, n_components, n_times)
             Transformed data.

        Raises
        ------
        RuntimeError
             If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("TRCAReducer must be fitted before calling transform().")

        # X is (n_trials, n_channels, n_times)
        X_arr = np.array(X)
        n_trials, n_chans, n_times = X_arr.shape

        # We need to apply the spatial filters learned by TRCA
        # self.model.coef_ shape: (n_bands, n_classes, n_chans)
        n_bands, n_classes, _ = self.model.coef_.shape

        # Prepare output container
        # We will produce one component per (band, class) combination
        # Total components = n_bands * n_classes
        # Output shape: (n_trials, n_total_components, n_times)
        X_out = []

        for b in range(n_bands):
            # 1. Filter data for this band
            X_tmp = X_arr.transpose(2, 1, 0)  # (times, chans, trials)

            # Wp, Ws from filterbank
            # filterbank structure: [[(pass_low, pass_high), (stop_low, stop_high)], ...]
            wp = self.model.filterbank[b][0]
            ws = self.model.filterbank[b][1]

            X_filt = bandpass(X_tmp, self.model.sfreq, Wp=wp, Ws=ws)
            # X_filt: (times, chans, trials)

            for k in range(n_classes):
                w = self.model.coef_[b, k, :]  # (n_chans,)

                # Project: w^T * X
                # w is spatial filter.
                # X_filt[t, c, tr]
                # Result should be (times, trials) -> (trials, times)

                # Using einsum for clarity:
                # 'ct,tcr->tr' (if c=chans, t=times, r=trials)? No.
                # X_filt indices: t (time), c (chan), r (trial)
                # w indices: c (chan)
                # target: t, r

                proj = np.einsum("c,tcr->tr", w, X_filt)  # (times, trials)
                X_out.append(proj.T)  # (trials, times)

        # Stack predictions
        # X_out is list of (trials, times)
        # We want (trials, n_components, n_times)
        return np.stack(X_out, axis=1)

    @property
    def coef_(self) -> np.ndarray:
        """
        Spatial filters per component.

        Returns
        -------
        coef_ : np.ndarray
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.model.coef_
