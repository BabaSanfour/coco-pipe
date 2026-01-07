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
from typing import Optional, Union, Dict, Any, List, Tuple

from .config import METHODS, METHODS_DICT
from .reducers.base import BaseReducer, ArrayLike
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

    def plot(self, 
             mode: str = 'embedding', 
             dims: Union[Tuple[int, int], Tuple[int, int, int]] = (0, 1),
             X: Optional[ArrayLike] = None, 
             V_emb: Optional[np.ndarray] = None, 
             labels: Optional[np.ndarray] = None,
             y: Optional[np.ndarray] = None,
             show_metrics: bool = False,
             **kwargs) -> Any:
        """
        Plot the results.

        Parameters
        ----------
        mode : {'embedding', 'shepard', 'streamlines', 'metrics', 'diagnostics', 'native'}
            Type of plot.
        dims : tuple
            Dimensions to plot (e.g., (0, 1) for 2D, (0, 1, 2) for 3D).
        X : ArrayLike, optional
            Original high-dimensional data (required for 'shepard', 'metrics').
        V_emb : np.ndarray, optional
            Velocity vectors (required for 'streamlines').
        labels : np.ndarray, optional
            Labels for coloring points.
        y : np.ndarray, optional
            Alias for labels.
        show_metrics : bool, default=False
            If True, overlays quality metrics on the 'embedding' plot. Requires X.
        **kwargs : dict
            Additional arguments passed to the plotting function.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from ..viz import dim_reduction as viz
        
        # Dispatch 'native' mode immediately
        if mode == 'native':
            return self.plot_native(**kwargs)

        # Alias resolution
        lbls = y if y is not None else labels

        # Decide on embedding to use
        # Diagnostics mode might not need embedding (e.g. loss history)
        if mode != 'diagnostics' and self.embedding_ is None:
             raise RuntimeError("Model is not fitted. Use fit() or fit_transform() first.")
        
        X_emb = self.embedding_

        # Calculate metrics if requested and X provided
        metrics_dict = None
        if show_metrics:
            if X is None:
                raise ValueError("show_metrics=True requires 'X' to compute scores.")
            metrics_dict = self.score(X)

        if mode == 'embedding':
            return viz.plot_embedding(X_emb, labels=lbls, dims=dims, 
                                      metrics=metrics_dict, **kwargs)
        
        elif mode == 'metrics':
            if X is None:
                 raise ValueError("mode='metrics' requires 'X' to compute scores.")
            scores = self.score(X)
            return viz.plot_metrics(scores, **kwargs)

        elif mode == 'diagnostics':
            # Check for loss history (Neural)
            if hasattr(self.reducer, 'loss_history_') and self.reducer.loss_history_:
                return viz.plot_loss_history(self.reducer.loss_history_, title=f"Loss History ({self.method})", **kwargs)
            
            # Check for eigenvalues/explained variance (Linear)
            if hasattr(self.reducer, 'explained_variance_ratio_'):
                return viz.plot_eigenvalues(self.reducer.explained_variance_ratio_, title=f"Explained Variance ({self.method})", **kwargs)
            
            if hasattr(self.reducer, 'eigs_'):
                # eigs_ can be complex for DMD, take magnitude
                eigs = np.abs(self.reducer.eigs_)
                # Sort descending
                eigs = -np.sort(-eigs) # numpy sort is ascending
                return viz.plot_eigenvalues(eigs, title=f"Eigenvalues ({self.method})", ylabel="Magnitude", **kwargs)

            # Check for singular values
            if hasattr(self.reducer, 'singular_values_'):
                 return viz.plot_eigenvalues(self.reducer.singular_values_, title=f"Singular Values ({self.method})", ylabel="Value", **kwargs)
                 
            # Fallback: Shepard Diagram (Manifold)
            if X is None:
                raise ValueError("mode='diagnostics' fallback to Shepard diagram requires 'X'.")
            return viz.plot_shepard_diagram(np.array(X), X_emb, **kwargs)

        elif mode == 'shepard':
            if X is None:
                raise ValueError("mode='shepard' requires original data 'X'")
            # Handle MNE object 
            X_arr = np.array(X) if not hasattr(X, 'get_data') else X.get_data()
            if X_arr.ndim > 2:
                # Shepard needs 2D inputs usually (scipy pdist)
                X_arr = X_arr.reshape(len(X_arr), -1)
                
            return viz.plot_shepard_diagram(X_arr, X_emb, **kwargs)

        elif mode == 'streamlines':
            if V_emb is None:
                raise ValueError("mode='streamlines' requires velocity vectors 'V_emb'")
            return viz.plot_streamlines(X_emb, V_emb, **kwargs)
        
        else:
            raise ValueError(f"Unknown plot mode: {mode}")

    def plot_native(self, **kwargs) -> Any:
        """
        Attempt to use the underlying library's native plotting function.
        
        Supported:
        - PHATE (phate.plot.*)
        - UMAP (umap.plot.*)
        - DMD (pydmd.plot_eigs / plot_modes_2D)

        Returns
        -------
        result : Any
             The result of the native plot call (usually axis or figure).
        """
        # PHATE
        if self.method == "PHATE":
             import phate
             return phate.plot.scatter(self.reducer.model, **kwargs)
        
        # UMAP
        if self.method == "UMAP":
             try:
                 import umap.plot
                 return umap.plot.points(self.reducer.model, **kwargs)
             except ImportError:
                 raise ImportError("umap.plot requires 'umap-learn[plot]' or manually installed dependencies.")

        # DMD (PyDMD)
        if self.method == "DMD":
             # Try plot_eigs first if not specified in kwargs?
             # Or check kwargs? Simple default: plot_eigs
             if hasattr(self.reducer.model, 'plot_eigs'):
                 return self.reducer.model.plot_eigs(**kwargs)
        
        raise NotImplementedError(f"Native plotting not supported or implemented for {self.method}.")

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
