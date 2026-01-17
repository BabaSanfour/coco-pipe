"""
coco_pipe/io/transform.py
-------------------------
Stateful transformers compatible with DataContainer.
Wraps scikit-learn transformers and implements M/EEG-specific whitening.

This module provides classes that adhere to the Scikit-Learn Transformer API 
but operate natively on `DataContainer` objects, preserving metadata (IDs, 
coordinates) throughout the transformation pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional, Union, List, Tuple, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import OAS

from .structures import DataContainer

logger = logging.getLogger(__name__)


def _rebuild_container(old_container: DataContainer, new_X: np.ndarray) -> DataContainer:
    """Helper to reconstruct DataContainer with new data and propagated metadata."""
    return replace(old_container, X=new_X)


def _check_container(container: DataContainer):
    """Helper to validate input."""
    if not isinstance(container, DataContainer):
        raise TypeError(f"Transformer expects DataContainer, got {type(container)}")


class SklearnWrapper(BaseEstimator, TransformerMixin):
    """
    Generic wrapper for ANY scikit-learn transformer (Scaler, PCA, etc.).
    
    This wrapper applies a standard scikit-learn transformer to the `.X` data 
    matrix of a `DataContainer`, ensuring that the resulting container has 
    correctly updated data while checking for dimension compatibility.

    Parameters
    ----------
    transformer : BaseEstimator
        An instantiated scikit-learn transformer (e.g., `StandardScaler()`, `PCA(n_components=10)`).
        
    Attributes
    ----------
    estimator_ : BaseEstimator
        The fitted scikit-learn estimator.
        
    Examples
    --------
    >>> from sklearn.preprocessing import RobustScaler
    >>> from coco_pipe.io import DataContainer, SklearnWrapper
    >>> import numpy as np
    
    >>> # Create formatted data (100 obs, 10 features)
    >>> X = np.random.randn(100, 10)
    >>> container = DataContainer(X, dims=('obs', 'feature'))
    
    >>> # Wrap a Scaler
    >>> scaler = SklearnWrapper(RobustScaler())
    >>> scaled_container = scaler.fit_transform(container)
    
    >>> # Metadata is preserved
    >>> scaled_container.dims == container.dims
    True
    """
    def __init__(self, transformer: BaseEstimator):
        self.transformer = transformer
        self.estimator_ = None 

    def fit(self, container: DataContainer, y=None):
        _check_container(container)
        if container.X.ndim != 2:
             raise ValueError(f"SklearnWrapper expects 2D input (Obs, Feat). Got {container.shape} with dims {container.dims}. Use .flatten() or .stack() first.")
        
        self.estimator_ = clone(self.transformer)
        self.estimator_.fit(container.X, y)
        return self

    def transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        if container.X.ndim != 2: raise ValueError("SklearnWrapper expects 2D input.")
        if self.estimator_ is None: raise RuntimeError("Transformer not fitted.")
        
        X_new = self.estimator_.transform(container.X)
        return _rebuild_container(container, X_new) # Use Helper

    def fit_transform(self, container: DataContainer, y=None):
        return self.fit(container, y).transform(container)

    def inverse_transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        if container.X.ndim != 2: raise ValueError("SklearnWrapper expects 2D input.")
        if self.estimator_ is None: raise RuntimeError("Transformer not fitted.")
        if not hasattr(self.estimator_, 'inverse_transform'):
             raise NotImplementedError(f"Wrapped estimator {type(self.estimator_)} has no inverse_transform.")
        
        X_orig = self.estimator_.inverse_transform(container.X)
        return _rebuild_container(container, X_orig) # Use Helper


class SpatialWhitener(BaseEstimator, TransformerMixin):
    """
    M/EEG Spatial Whitening using Covariance Decorrelation.
    
    This transformer removes spatial correlations between channels, effectively 
    transforming the noise covariance matrix towards the identity matrix. It supports 
    standard PCA, ZCA (Zero-phase Component Analysis which preserves topography), 
    and robust shrinkage covariance estimation (OAS).

    It requires a dimension named 'channel' in the input `DataContainer`.
    The operation is performed spatially: :math:`X_{white} = X \\cdot W^T`

    Parameters
    ----------
    method : {'pca', 'zca', 'shrinkage'}, default='pca'
        Shape of the transformation:
        - 'pca': Principal Component Analysis. Rotates data to principal axes and scales to unit variance.
        - 'zca': Zero-phase Component Analysis. Rotates, scales, and rotates back. Preserves spatial topography (sensors stay in place).
        - 'shrinkage': Uses Oracle Approximating Shrinkage (OAS) for robust covariance estimation in high dimensions.
    n_components : int or float, optional
        Number of components to keep (only for 'pca'/'zca' methods). If None, all matches are kept.
        
    Attributes
    ----------
    whitener_ : np.ndarray
        The estimated whitening matrix (W). Shape (n_components, n_channels).
    mean_ : np.ndarray
        Per-channel mean vector.
    inverse_whitener_ : np.ndarray
        The inverse matrix used to project back to sensor space.
        
    Examples
    --------
    >>> # Whitening EEG epochs (100 epochs, 64 channels, 500 times)
    >>> container = DataContainer(np.random.randn(100, 64, 500), dims=('obs', 'channel', 'time'))
    
    >>> # Use Shrinkage for robust covariance
    >>> whitener = SpatialWhitener(method='shrinkage')
    >>> white_data = whitener.fit_transform(container)
    
    >>> # Project back to sensor space for plotting
    >>> sensor_data = whitener.inverse_transform(white_data)
    """
    def __init__(self, method: str = 'pca', n_components: Optional[Union[int, float]] = None):
        self.method = method
        self.n_components = n_components
        self.whitener_ = None # W matrix (n_components, n_channels)
        self.mean_ = None     # Mean vector (n_channels,)
        self.inverse_whitener_ = None # W_inv matrix (n_channels, n_components)

    def fit(self, container: DataContainer, y=None):
        _check_container(container)
        
        # 1. Identify Channel Dimension
        if 'channel' not in container.dims:
            raise ValueError(f"SpatialWhitener requires 'channel' dimension. Found: {container.dims}")
        
        ch_idx = container.dims.index('channel')
        X = container.X
        
        # 2. Reshape to (N_samples, N_channels) for covariance computation
        # Move channel to last dim
        X_perm = np.moveaxis(X, ch_idx, -1)
        X_flat = X_perm.reshape(-1, X.shape[ch_idx])
        
        # 3. Compute Whitening Matrix
        if self.method in ['pca', 'zca']:
            self._fit_pca(X_flat)
        elif self.method == 'shrinkage':
            self._fit_shrinkage(X_flat)
            
        return self
    
    def _fit_pca(self, X_flat: np.ndarray):
        pca = PCA(n_components=self.n_components, whiten=True)
        pca.fit(X_flat)
        self.mean_ = pca.mean_
        
        # W_pca = scale * rotation = diag(1/sqrt(S)) * V
        V = pca.components_ 
        S = pca.explained_variance_
        
        # Whitening matrix W (n_comp, n_ch)
        W_pca = (V.T * (1.0 / np.sqrt(S))).T # (n_comp, n_ch)
        
        if self.method == 'pca':
            self.whitener_ = W_pca
            self.inverse_whitener_ = (V.T * np.sqrt(S))
            
        elif self.method == 'zca':
            # W_zca = V.T * diag(1/sqrt(S)) * V
            W_zca = V.T @ W_pca
            self.whitener_ = W_zca.T 
            
            # W_inv = V.T * diag(sqrt(S)) * V
            W_inv_zca = V.T @ np.diag(np.sqrt(S)) @ V
            self.inverse_whitener_ = W_inv_zca.T

    def _fit_shrinkage(self, X_flat: np.ndarray):
        # OAS
        oas = OAS(store_precision=True, assume_centered=False)
        oas.fit(X_flat)
        self.mean_ = oas.location_
        cov = oas.covariance_
        
        # W = Cov^(-1/2) via SVD
        import scipy.linalg
        U, S, _ = scipy.linalg.svd(cov)
        
        # Full rank ZCA
        W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S)), U.T))
        self.whitener_ = W 
        
        # Inverse W = Cov^(1/2)
        W_inv = np.dot(U, np.dot(np.diag(np.sqrt(S)), U.T))
        self.inverse_whitener_ = W_inv

    def transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        X_new = self._apply_linear_op(container, self.whitener_, self.mean_)
        return _rebuild_container(container, X_new) # Use Helper
        
    def fit_transform(self, container: DataContainer, y=None):
        return self.fit(container, y).transform(container)

    def inverse_transform(self, container: DataContainer) -> DataContainer:
        _check_container(container)
        
        X_recon = self._apply_linear_op(container, self.inverse_whitener_, mean=None)
        
        # Add mean (broadcast)
        ch_idx = container.dims.index('channel')
        mean_shape = [1] * container.X.ndim
        mean_shape[ch_idx] = -1
        X_final = X_recon + self.mean_.reshape(mean_shape)
        
        return _rebuild_container(container, X_final) # Use Helper

    def _apply_linear_op(self, container: DataContainer, W: np.ndarray, mean: Optional[np.ndarray]) -> np.ndarray:
        # Internal optimized linear operator logic (X @ W.T)
        if 'channel' not in container.dims:
             raise ValueError("Dimension 'channel' missing.")
             
        ch_idx = container.dims.index('channel')
        X = container.X
        
        # 1. Center
        if mean is not None:
            mean_shape = [1] * X.ndim
            mean_shape[ch_idx] = -1
            X = X - mean.reshape(mean_shape)
        
        # 2. Permute -> MatMul -> Permute Back
        X_perm = np.moveaxis(X, ch_idx, -1)
        X_out = X_perm @ W.T
        X_final = np.moveaxis(X_out, -1, ch_idx)
        
        return X_final
