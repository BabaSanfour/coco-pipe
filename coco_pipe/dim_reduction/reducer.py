#!/usr/bin/env python3
"""
coco_pipe/dim_reduction/processor.py

DimReducer class that takes any (n_samples, n_features) array
and applies PCA/TSNE/UMAP, with save/load support.
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
    Core reducer class. You simply pass it your data array and it
    handles fit, transform, fit_transform, and optional persistence.
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
        self.reducer: BaseReducer = METHODS_DICT[method](n_components=n_components, **reducer_kwargs)
        self.save_path = Path(save_path) if save_path is not None else None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Fit the reducer on X (and optional labels y).
        """
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        self.reducer.fit(X, y=y)
        if self.save_path:
            logger.info(f"Saving fitted reducer to {self.save_path}")
            self.reducer.save(str(self.save_path))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the fitted reducer to X.
        """
        # if a save_path exists and reducer isn’t loaded yet, load it
        if self.save_path and self.save_path.exists() and not hasattr(self.reducer, "model") and not hasattr(self.reducer, "embedding_"):
            logger.info(f"Loading reducer from {self.save_path}")
            self.reducer = BaseReducer.load(str(self.save_path))
        return self.reducer.transform(X)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit then transform. 
        """
        # Some reducers (TSNE) implement fit_transform directly
        if hasattr(self.reducer, "fit_transform") and not hasattr(self.reducer, "model"):
            return self.reducer.fit_transform(X)
        self.fit(X, y=y)
        return self.transform(X)

    @classmethod
    def load(cls, load_path: Union[str, Path]) -> "DimReducer":
        """
        Load a previously saved reducer and wrap it.
        """
        loaded = BaseReducer.load(str(load_path))
        inst = cls.__new__(cls)
        inst.method   = loaded.__class__.__name__.replace("Reducer","").upper()
        inst.reducer  = loaded
        inst.save_path= Path(load_path)
        return inst
