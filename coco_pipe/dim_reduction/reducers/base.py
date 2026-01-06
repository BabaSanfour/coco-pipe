"""
Base Reducer Module
===================

This module defines the abstract base class for all dimensionality reducers
in the coco_pipe package. It enforces a consistent API for fitting, transforming,
and persisting models.

Classes
-------
BaseReducer
    Abstract base class for dimensionality reduction.

"""

import os
import joblib
from abc import ABC, abstractmethod
from typing import Optional, Union, Any
import numpy as np

# Type alias for array-like objects
ArrayLike = Union[np.ndarray, list]


class BaseReducer(ABC):
    """
    Abstract base class for all dimensionality reduction implementations.
    
    This class defines the standard interface that all reducers must implement.
    It provides built-in support for model persistence (save/load) using joblib.
    
    Attributes
    ----------
    model : Any
        The underlying model object (e.g., sklearn estimator, pydmd object).
        This attribute should be populated by the `fit` method.
    
    """
    
    def __init__(self, n_components: int = 2, **kwargs):
        """
        Initialize the reducer.
        
        Parameters
        ----------
        n_components : int, default=2
            The target number of dimensions.
        **kwargs : dict
            Additional keyword arguments for the underlying model.
        """
        self.n_components = n_components
        self.params = kwargs
        self.model = None

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "BaseReducer":
        """
        Fit the model to the data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,), optional
            Target values (labels) for supervised reduction.

        Returns
        -------
        self : BaseReducer
            The fitted reducer instance.
        """
        pass

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        pass

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit the model to data and return the transformed data.
        
        This method usually just calls fit() then transform(), but can be overridden
        for efficiency if the underlying algorithm supports it.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike of shape (n_samples,), optional
            Target values.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        self.fit(X, y=y)
        return self.transform(X)

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        """
        Persist the reducer to a file.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.
        """
        filepath = str(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike]) -> "BaseReducer":
        """
        Load a reducer from a file.

        Parameters
        ----------
        filepath : str or Path
            Path to the file to load.

        Returns
        -------
        reducer : BaseReducer
            The loaded reducer instance.
        """
        return joblib.load(str(filepath))