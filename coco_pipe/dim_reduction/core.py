"""
Dimensionality Reduction Core
=============================

This module provides the main entry point for dimensionality reduction workflow.
It consolidates method instantiation, execution, validation, visualization, and
benchmarking into a single high-level `DimReduction` class.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-07
"""

import numpy as np
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

from .config import METHODS, METHODS_DICT
from .reducers.base import BaseReducer
from .benchmark import metrics

class DimReduction:
    """
    Main manager for Dimensionality Reduction.

    This class orchestrates the entire workflow:
    1.  **Factory**: Instantiates the correct reducer (PCA, UMAP, TRCA, etc.).
    2.  **Validation**: Verifies input data shapes match method requirements.
    3.  **Execution**: fit/transform/fit_transform.
    4.  **Benchmarking**: Calculates quality metrics (Trustworthiness, Continuity).
    5.  **Visualization**: Integrated plotting.

    Parameters
    ----------
    method : str
        Name of the reduction method (e.g., 'UMAP', 'PCA', 'TRCA').
        Case-insensitive.
    n_components : int, default=2
        Number of dimensions to reduce to.
    params : dict, optional
        Additional keyword arguments passed to the underlying reducer.

    Attributes
    ----------
    method : str
        The reduction method used.
    n_components : int
        Number of dimensions reduced to.
    reducer : BaseReducer
        The instantiated reducer object.
    embedding_ : np.ndarray
        The reduced embedding.

    Methods
    -------
    fit(X, y=None)
        Fit the reducer to the data.
    transform(X)
        Transform the data.
    fit_transform(X, y=None)
        Fit and transform the data.
    score(X, X_emb=None, n_neighbors=5)
        Calculate quality metrics.
    plot(labels=None, mode='embedding', **kwargs)
        Visualize the result.
    save(path)
        Save the reducer to disk.
    load(path, method)
        Load a reducer from disk.
    from_config(config)
        Initialize from configuration dictionary.
    
    Examples
    --------
    >>> reducer = DimReduction('UMAP', n_components=2)
    >>> reducer.fit_transform(X)
    >>> reducer.score(X)
    >>> reducer.plot()
    >>> reducer.save('reducer.pkl')
    >>> reducer = DimReduction.load('reducer.pkl', 'UMAP')
    >>> reducer.from_config({'method': 'UMAP', 'n_components': 2})
    """

    def __init__(self, method: str, n_components: int = 2, params: Optional[Dict[str, Any]] = None, **kwargs):
        self.method = method.upper()
        if self.method not in METHODS:
            valid = ", ".join(METHODS)
            raise ValueError(f"Unknown method '{method}'. Valid options are: {valid}")

        self.n_components = n_components
        
        reducer_kwargs = params.copy() if params else {}
        reducer_kwargs.update(kwargs)
        self.reducer_kwargs = reducer_kwargs

        ReducerCls = METHODS_DICT[self.method]
        self.reducer: BaseReducer = ReducerCls(
            n_components=n_components,
            **self.reducer_kwargs
        )
        
        self.embedding_ = None

    def _validate_input(self, X: Any) -> np.ndarray:
        """
        Validate input data shape and type.
        
        TRCA/DMD require 3D input: (n_trials, n_channels, n_times).
        Others (PCA, UMAP, etc.) require 2D: (n_samples, n_features).

        Parameters
        ----------
        X : array-like or MNE object
            Input data.
        
        Returns
        -------
        X_arr : np.ndarray
            Validated input data as a numpy array.
        """
        if hasattr(X, "get_data"): # Handle MNE objects
             X = X.get_data()
        
        X_arr = np.array(X)
        
        # Define 3D-required methods
        spatiotemporal_methods = {"TRCA", "DMD"}
        
        if self.method in spatiotemporal_methods:
            if X_arr.ndim != 3:
                raise ValueError(f"Method '{self.method}' requires 3D input (Trials x Channels x Time), got shape {X_arr.shape}.")
        else:
            if X_arr.ndim != 2:
                raise ValueError(f"Method '{self.method}' requires 2D input (Samples x Features), got shape {X_arr.shape}. "
                                 "Consider flattening your data.")
                                 
        return X_arr

    def fit(self, X: Any, y: Optional[Any] = None) -> "DimReduction":
        """
        Fit the reducer.

        Parameters
        ----------
        X : array-like or MNE object
            Input data.
        y : array-like, optional
            Target labels (for supervised methods like LDA, TRCA).

        Returns
        -------
        self : DimReduction
            The fitted reducer.
        """
        X_arr = self._validate_input(X)
        self.reducer.fit(X_arr, y=y)
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Transform new data using the fitted reducer.

        Parameters
        ----------
        X : array-like or MNE object
            Input data.

        Returns
        -------
        X_emb : np.ndarray
            Transformed data.
        """
        X_arr = self._validate_input(X)
        return self.reducer.transform(X_arr)

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """
        Fit and transform in one step.
        Updates self.embedding_.

        Parameters
        ----------
        X : array-like or MNE object
            Input data.
        y : array-like, optional
            Target labels (for supervised methods like LDA, TRCA).

        Returns
        -------
        X_emb : np.ndarray
            Transformed data.
        """
        X_arr = self._validate_input(X)
        self.embedding_ = self.reducer.fit_transform(X_arr, y=y)
        return self.embedding_

    def score(self, X: Any, X_emb: Optional[np.ndarray] = None, n_neighbors: int = 5) -> Dict[str, float]:
        """
        Compute quality metrics for the embedding.
        
        Metrics:
        - Trustworthiness: Preservation of local neighborhoods (penalizes false positives).
        - Continuity: Preservation of local neighborhoods (penalizes missing neighbors).
        
        Parameters
        ----------
        X : array-like
            Original high-dimensional data.
        X_emb : array-like, optional
            Embedded data. If None, uses cached self.embedding_.
        n_neighbors : int, default=5
            K-nearest neighbors size for metric computation.

        Returns
        -------
        scores : dict
        """
        X_arr = self._validate_input(X)
        
        if X_emb is None:
            if self.embedding_ is None:
                raise RuntimeError("No embedding available. Call fit_transform() first or provide X_emb.")
            X_emb = self.embedding_

        if X_emb.ndim == 3 or X_arr.ndim == 3:
             # Skip standard metrics for spatiotemporal for now, or implement specialized ones.
             return {"trustworthiness": np.nan, "continuity": np.nan, "note": "Metrics undefined for 3D spatiotemporal data"}

        from scipy.stats import spearmanr
        
        scores = {
            "trustworthiness": metrics.trustworthiness(X_arr, X_emb, n_neighbors=n_neighbors),
            "continuity": metrics.continuity(X_arr, X_emb, n_neighbors=n_neighbors),
            "lcmc": metrics.lcmc(X_arr, X_emb, n_neighbors=n_neighbors)
        }
        
        # Shepard Correlation (Global Distance Preservation)
        # We sample 1000 points max to keep it efficient
        d_orig, d_emb = metrics.shepard_diagram_data(X_arr, X_emb, sample_size=1000)
        if len(d_orig) > 1:
            # Spearman correlation deals with non-linear monotonic relationships
            corr, _ = spearmanr(d_orig, d_emb)
            scores["shepard_correlation"] = float(corr)
        else:
            scores["shepard_correlation"] = np.nan
        
        
        # Scrape reducer-specific metrics (attributes ending in _ or in allow-list)
        # Check both the wrapper (BaseReducer subclass) and the internal model
        sources = [self.reducer]
        if hasattr(self.reducer, 'model'):
             sources.append(self.reducer.model)
        
        # Attributes that don't follow the underscore convention but are valuable
        allow_list = {
            "eigs", "modes", "dynamics", "diff_potential",
            "reconstruction_error", "n_features_in", "coef", "loss_history"
        }

        for source in sources:
            for attr in dir(source):
                is_allowed = attr in allow_list
                is_standard = attr.endswith("_") and not attr.startswith("_") and not attr.endswith("__")
                
                if is_allowed or is_standard:
                    # Exclude standard sklearn attributes that might be huge arrays (like labels_ or embedding_)
                    if attr in ["embedding_", "labels_", "components_", "fit_transform"]: 
                        continue
                        
                    try:
                        val = getattr(source, attr)
                        if isinstance(val, (int, float, np.number)):
                             scores[attr] = float(val)
                        # Allow explicit lists/arrays if small enough or if in allow_list (careful with graph)
                        elif isinstance(val, (list, tuple, np.ndarray)):
                             if is_allowed:
                                 scores[attr] = val
                             elif np.size(val) < 50: # Increased limit slightly
                                 scores[attr] = val
                    except Exception:
                        pass

        return scores

    def plot(self, labels: Optional[Union[np.ndarray, List]] = None, mode: str = 'embedding', **kwargs):
        """
        Visualize the result.

        Parameters
        ----------
        labels : array-like, optional
            Labels for coloring points.
        mode : {'embedding', 'shepard', 'streamlines'}, default='embedding'
            Plot type.
        **kwargs : dict
            Passed to underlying viz function (e.g. title, cmap, ax).
            
        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        # Lazy import to avoid circular dependency or heavy load
        from ..viz import dim_reduction as viz
        
        if self.embedding_ is None and mode != 'shepard': # shepard needs separate handling if not cached?
             raise RuntimeError("No embedding found. Run fit_transform() first.")

        if self.embedding_.ndim == 3:
             raise NotImplementedError("Plotting 3D spatiotemporal embeddings is not yet supported.")

        if mode == 'embedding':
            return viz.plot_embedding(self.embedding_, labels=labels, **kwargs)
        elif mode == 'shepard':
            if 'X' not in kwargs:
                raise ValueError("Plotting Shepard diagram requires 'X' (original data) in kwargs.")
            X = kwargs.pop('X')
            return viz.plot_shepard_diagram(X, self.embedding_, **kwargs)
        elif mode == 'streamlines':
            if 'V_emb' not in kwargs:
                raise ValueError("Plotting streamlines requires 'V_emb' (velocity vectors) in kwargs.")
            V_emb = kwargs.pop('V_emb')
            return viz.plot_streamlines(self.embedding_, V_emb, **kwargs)
        else:
            raise ValueError(f"Unknown plot mode '{mode}'")

    def save(self, path: Union[str, Path]):
        """Save the reducer to disk."""
        self.reducer.save(path)

    @classmethod
    def load(cls, path: Union[str, Path], method: str) -> "DimReduction":
        """
        Load a reducer from disk.
        
        Note: We need to know the 'method' to instantiate the correct class wrapper,
        or we just return the raw reducer? 
        The BaseReducer.load() returns the specific Reducer instance (e.g. UMAPReducer).
        But that instance is not a 'DimReduction' manager instance.
        
        To restore a 'DimReduction' manager, we can wrap the loaded reducer.
        """
        # Load raw reducer
        reducer_instance = BaseReducer.load(path)
        
        # Hacky reconstruction:
        manager = cls(method=method, n_components=reducer_instance.n_components)
        manager.reducer = reducer_instance
        return manager

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "DimReduction":
        """
        Initialize from configuration dictionary.
        
        Expected keys:
        - method: str
        - n_components: int
        - params: dict
        """
        if 'method' not in config:
            raise ValueError("Config must contain 'method' key.")
            
        return cls(
            method=config['method'],
            n_components=config.get('n_components', 2),
            params=config.get('params', {})
        )
