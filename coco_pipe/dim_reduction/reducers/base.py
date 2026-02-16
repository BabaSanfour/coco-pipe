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
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import joblib
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

    def _filter_params(self, fn_or_class: Any, params: dict) -> dict:
        """
        Filter parameters to match the signature of a function or class.

        Parameters
        ----------
        fn_or_class : Any
            The function or class to inspect.
        params : dict
            The parameters to filter.

        Returns
        -------
        filtered_params : dict
            Parameters present in the signature.
        """
        import inspect

        try:
            if inspect.isclass(fn_or_class):
                target = fn_or_class.__init__
            else:
                target = fn_or_class

            sig = inspect.signature(target)
            allowed_params = sig.parameters.keys()

            # If the target accepts **kwargs, don't filter
            if any(
                p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
            ):
                return params

            return {k: v for k, v in params.items() if k in allowed_params}
        except (ValueError, TypeError):
            # Fallback if signature extraction fails (e.g. C extensions)
            return params

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
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self, filepath)

    @property
    def capabilities(self) -> dict:
        """
        Returns a dictionary of capabilities for this reducer.
        """
        # Default capabilities for most sklearn-like 2D reducers
        return {
            "input_ndim": 2,
            "input_layout": "standard",  # (samples, features)
            "has_transform": True,
            "has_inverse_transform": hasattr(self.model, "inverse_transform")
            if self.model
            else False,
            "has_components": hasattr(self.model, "components_")
            if self.model
            else False,
            "supported_diagnostics": self._get_supported_diagnostics(),
            "has_native_plot": False,
        }

    def _get_supported_diagnostics(self) -> list:
        """
        Safely determine supported diagnostics without triggering property errors.
        """
        candidates = [
            "explained_variance_ratio_",
            "singular_values_",
            "loss_history_",
            "eigs_",
        ]
        supported = []
        for k in candidates:
            # 1. Check if it is a property on the class (static capability)
            cls_attr = getattr(type(self), k, None)
            if isinstance(cls_attr, property):
                supported.append(k)
                continue
            
            # 2. Check if instance attribute (safely)
            if k in self.__dict__:
                supported.append(k)
                continue
            
            # 3. Check model (only if fitted)
            if self.model is not None and hasattr(self.model, k):
                supported.append(k)
        
        return supported

    def get_diagnostics(self) -> dict:
        """
        Extract model-specific diagnostic attributes (e.g., kl_divergence, stress).

        Returns
        -------
        diagnostics : dict
            Dictionary of diagnostic names and values.
        """
        if self.model is None:
            return {}

        # Common diagnostic attributes in sklearn/neighbor methods
        diag_attrs = [
            "kl_divergence_",
            "stress_",
            "reconstruction_error_",
            "n_iter_",
            "loss_history_",
            "explained_variance_ratio_",
            "singular_values_",
            "eigs",
            "diff_potential",
        ]

        results = {}
        for attr in diag_attrs:
            # Check top-level wrapper first (some might be @property)
            if hasattr(self, attr):
                try:
                    results[attr] = getattr(self, attr)
                except Exception:
                    pass
            # Check internal model directly
            elif hasattr(self.model, attr):
                results[attr] = getattr(self.model, attr)

        return results

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
