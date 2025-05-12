#!/usr/bin/env python3
"""
coco_pipe/dim_reduction/reducer.py

DimReducer: a simple, method-agnostic dimensionality reducer with optional persistence.
Supported methods: PCA, TSNE, UMAP (extendable via METHODS_DICT).
"""
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np

from coco_pipe.dim_reduction.config import METHODS, METHODS_DICT
from coco_pipe.dim_reduction.reducers.base import BaseReducer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DimReducer:
    """
    Wraps any BaseReducer to provide a unified API for fit, transform,
    fit_transform, and optional save/load of the reducer state.
    """

    def __init__(
        self,
        method: str,
        n_components: int = 2,
        save_path: Optional[Union[str, Path]] = None,
        **reducer_kwargs
    ):
        method = method.upper()
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")
        self.method = method
        # instantiate the appropriate reducer
        ReducerCls = METHODS_DICT[method]
        self.reducer: BaseReducer = ReducerCls(
            n_components=n_components,
            **reducer_kwargs
        )
        self.save_path: Optional[Path] = Path(save_path) if save_path is not None else None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the underlying reducer on X (and optional labels y).
        If save_path is set, persist the fitted reducer to disk.
        """
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        self.reducer.fit(X, y=y)
        if self.save_path:
            logger.info(f"Saving fitted reducer to {self.save_path}")
            self.reducer.save(str(self.save_path))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X using the fitted reducer. If save_path exists and the
        reducer isn't already loaded, load it first.
        """
        # lazy-load if a saved reducer is present
        if (
            self.save_path
            and self.save_path.exists()
            and not hasattr(self.reducer, 'model')
            and not hasattr(self.reducer, 'embedding_')
        ):
            logger.info(f"Loading reducer from {self.save_path}")
            self.reducer = BaseReducer.load(str(self.save_path))
        return self.reducer.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convenience: fit then transform. Respects reducers that implement
        fit_transform directly (e.g. TSNE).
        """
        # if reducer provides a direct fit_transform and no model attr yet
        if hasattr(self.reducer, 'fit_transform') and not hasattr(self.reducer, 'model'):
            return self.reducer.fit_transform(X)
        self.fit(X, y=y)
        return self.transform(X)

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> 'DimReducer':
        """
        Load a previously saved reducer from disk and wrap it in DimReducer.
        The method name is inferred from the reducer class name.
        """
        loaded: BaseReducer = BaseReducer.load(str(load_path))
        inst = cls.__new__(cls)
        inst.method = loaded.__class__.__name__.replace('Reducer','').upper()
        inst.reducer = loaded
        inst.save_path = Path(load_path)
        return inst
