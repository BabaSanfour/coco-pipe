"""
Base interfaces for dimensionality reduction backends.

This module defines the reducer contract shared by built-in reducers and
user-defined reducers. A reducer is any object derived from `BaseReducer`
implementing `fit` and `transform`, optionally exposing diagnostics and scalar
quality metadata through helper methods.

The surrounding dim-reduction stack uses these interfaces to provide:

- input validation through the reducer `capabilities` mapping
- standardized persistence with `save` and `load`
- reducer-aware reporting and visualization hooks
- optional dependency loading through `import_optional_dependency`

Notes
-----
`BaseReducer` is the intended extension point for custom reducers. Third-party
reducers can participate in `DimReduction` workflows without extra wrappers as
long as they respect the method contract documented here.
"""

import os
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    Iterable,
    Optional,
    Union,
)

import joblib
import numpy as np

# Type alias for array-like objects
ArrayLike = Union[np.ndarray, list]

__all__ = ["ArrayLike", "BaseReducer", "import_optional_dependency"]


def import_optional_dependency(
    loader: Any,
    feature: str,
    dependency: str,
    install_hint: Optional[str] = None,
) -> Any:
    """
    Lazily import an optional dependency with clearer failure modes.

    This helper is primarily reducer infrastructure. It keeps heavy optional
    imports inside `fit` and `transform` paths and normalizes both
    missing-package errors and runtime initialization failures.

    Parameters
    ----------
    loader : callable
        Zero-argument callable returning the imported dependency.
    feature : str
        Feature or reducer name using the dependency.
    dependency : str
        Human-readable dependency name.
    install_hint : str, optional
        Installation hint shown on ImportError.

    Returns
    -------
    Any
        The imported dependency object returned by `loader`.

    Raises
    ------
    ImportError
        If the dependency is not installed.
    RuntimeError
        If the dependency is installed but fails during initialization.

    Notes
    -----
    Reducers should call this helper inside execution paths instead of module
    import time so that optional scientific dependencies do not break
    lightweight package imports.
    """
    try:
        return loader()
    except ImportError as exc:
        msg = f"{dependency} is required for {feature}."
        if install_hint:
            msg += f" Install it with '{install_hint}'."
        raise ImportError(msg) from exc
    except Exception as exc:
        raise RuntimeError(
            f"{dependency} failed to initialize for {feature}: {exc}"
        ) from exc


class BaseReducer(ABC):
    """
    Abstract base class for all dimensionality reduction implementations.

    This class defines the standard interface that all reducers must implement
    and is safe to subclass for custom reducers. It provides built-in support
    for model persistence (save/load) using joblib.

    For custom reducers operating on nonstandard data layouts, override
    `capabilities` so the manager layer can route validation, scoring,
    plotting, and reporting correctly.

    Parameters
    ----------
    n_components : int, default=2
        Target dimensionality of the reduced representation.
    **kwargs : dict
        Additional keyword arguments stored on `params` and typically forwarded
        to the wrapped estimator or backend implementation.

    Attributes
    ----------
    n_components : int
        Target dimensionality of the reduced representation.
    params : dict
        Additional reducer parameters captured at initialization time.
    model : Any
        Underlying fitted model object, such as a scikit-learn estimator or a
        scientific computing backend. This attribute should be populated by
        `fit`.

    Notes
    -----
    The `capabilities` property returns a plain dictionary consumed by the
    manager and evaluation layers. Custom reducers should declare supported
    diagnostics and scalar metadata explicitly through this mapping. Common
    keys include:

    - `input_ndim` : expected dimensionality of the input container
    - `input_layout` : semantic layout name such as `"standard"`
    - `has_transform` : whether `transform` is supported
    - `has_inverse_transform` : whether inverse transforms are available
    - `has_components` : whether PCA-like components are exposed
    - `supported_diagnostics` : names returned by `get_diagnostics`
    - `has_native_plot` : whether the reducer exposes its own plotting path
    - `is_linear` : whether the reducer is linear
    - `is_stochastic` : whether repeated runs can vary without a fixed seed

    Examples
    --------
    >>> from sklearn.decomposition import PCA
    >>> from coco_pipe.dim_reduction import BaseReducer
    >>>
    >>> class CustomPCAReducer(BaseReducer):
    ...     @property
    ...     def capabilities(self):
    ...         return self._merge_capabilities(
    ...             super().capabilities,
    ...             is_linear=True,
    ...             has_components=True,
    ...             supported_diagnostics=("explained_variance_ratio_",),
    ...         )
    ...
    ...     def fit(self, X, y=None):
    ...         self.model = PCA(n_components=self.n_components, **self.params)
    ...         self.model.fit(X)
    ...         return self
    ...
    ...     def transform(self, X):
    ...         return self.model.transform(X)
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
            Parameters present in the signature. If the target accepts
            ``**kwargs`` or its signature cannot be inspected, the original
            parameter dictionary is returned unchanged.

        Notes
        -----
        This is a convenience helper for reducer implementations that wrap
        third-party estimators with partially overlapping constructor
        signatures.
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

    def _build_estimator(
        self,
        estimator_cls: Any,
        params: Optional[dict] = None,
        component_param: Optional[str] = "n_components",
        **fixed_kwargs: Any,
    ) -> Any:
        """
        Instantiate an estimator with filtered reducer parameters.

        Parameters
        ----------
        estimator_cls : Any
            Estimator class to instantiate.
        params : dict, optional
            Explicit parameter dictionary to filter instead of `self.params`.
        component_param : str or None, default=\"n_components\"
            Name of the constructor argument receiving `self.n_components`.
            Set to ``None`` to skip injecting the component count.
        **fixed_kwargs : dict
            Keyword arguments always forwarded to the estimator constructor.

        Returns
        -------
        Any
            Instantiated estimator.

        Notes
        -----
        This helper assumes the wrapped backend is constructor-driven and can
        be configured from keyword arguments.
        """
        raw_params = self.params if params is None else params
        filtered_params = self._filter_params(estimator_cls, raw_params)
        constructor_kwargs = dict(fixed_kwargs)
        if component_param is not None:
            constructor_kwargs[component_param] = self.n_components
        return estimator_cls(**constructor_kwargs, **filtered_params)

    def _require_fitted(self, method_name: str = "transform", model: Any = None) -> Any:
        """
        Validate that a reducer backend has been fitted before access.

        Parameters
        ----------
        method_name : str, default=\"transform\"
            Operation requiring a fitted model.
        model : Any, optional
            Backend model to check. Defaults to `self.model`.

        Returns
        -------
        Any
            The validated model instance.

        Raises
        ------
        RuntimeError
            If no fitted model is available.
        """
        resolved_model = self.model if model is None else model
        if resolved_model is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before calling {method_name}()."
            )
        return resolved_model

    def _merge_capabilities(
        self, base_caps: Dict[str, Any], **overrides: Any
    ) -> Dict[str, Any]:
        """
        Return a capability mapping updated with reducer-specific overrides.

        Parameters
        ----------
        base_caps : dict
            Base capability mapping, typically `super().capabilities`.
        **overrides : dict
            Reducer-specific capability values to apply.

        Returns
        -------
        dict
            Capability mapping with overrides applied.
        """
        caps = dict(base_caps)
        caps.update(overrides)
        for key in ("supported_diagnostics", "supported_metadata"):
            if key in caps:
                caps[key] = list(caps[key])
        return caps

    @abstractmethod
    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "BaseReducer":
        """
        Fit the model to the data.

        Parameters
        ----------
        X : ArrayLike
            Training data. Most reducers expect `(n_samples, n_features)`, but
            reducers with custom `capabilities["input_layout"]` may accept other
            layouts such as snapshot matrices or grouped trajectory tensors.
        y : ArrayLike, optional
            Optional supervision aligned with the sample axis used by the
            reducer's declared input layout.

        Returns
        -------
        self : BaseReducer
            The fitted reducer instance.

        Notes
        -----
        Most reducers expect `X` to have shape `(n_samples, n_features)`. Some
        reducers operate on alternative layouts and should document those
        layouts through `capabilities`.
        """
        pass

    @abstractmethod
    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X : ArrayLike
            New data to transform. Its layout should match the reducer's
            declared `capabilities`.

        Returns
        -------
        X_new : np.ndarray
            Reduced representation. The exact output shape depends on the
            reducer, but the last dimension usually matches `n_components`.

        Raises
        ------
        RuntimeError
            Raised by concrete implementations when `transform` is called
            before fitting or when the reducer does not support out-of-sample
            transforms.
        """
        pass

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Fit the model to data and return the transformed data.

        This method usually calls `fit` and then `transform`, but reducers may
        override it for efficiency if the underlying algorithm supports a
        native combined path.

        Parameters
        ----------
        X : ArrayLike
            Training data following the reducer's declared layout.
        y : ArrayLike, optional
            Optional supervision aligned with the reducer's input layout.

        Returns
        -------
        X_new : np.ndarray
            Reduced representation returned by `transform`.
        """
        self.fit(X, y=y)
        return self.transform(X)

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        """
        Persist the reducer to a file.

        The default implementation serializes the reducer instance with joblib.
        Custom reducers should either remain joblib-serializable or override
        this method and `load()` with a custom persistence strategy.

        Parameters
        ----------
        filepath : str or Path
            Path to the output file.

        Notes
        -----
        The default implementation serializes the reducer instance with
        `joblib.dump`. Custom reducers should either remain joblib-serializable
        or override this method and `load` with a custom persistence strategy.
        """
        filepath = str(filepath)
        out_dir = os.path.dirname(filepath)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        joblib.dump(self, filepath)

    @property
    def capabilities(self) -> Dict[str, Any]:
        """
        Return reducer capability flags consumed by the manager layer.

        Custom reducers with nonstandard inputs should override at least
        `input_ndim` and `input_layout`. Reducers exposing diagnostics or
        scalar quality metadata should declare them explicitly through
        `supported_diagnostics` and `supported_metadata`.

        Returns
        -------
        dict
            Mapping of reducer capability flags.

        Notes
        -----
        The default capabilities describe a typical estimator consuming
        `(samples, features)` input and exposing `transform`.
        """
        return {
            "input_ndim": 2,
            "input_layout": "standard",
            "has_transform": True,
            "has_inverse_transform": hasattr(self.model, "inverse_transform")
            if self.model
            else False,
            "has_components": hasattr(self.model, "components_")
            if self.model
            else False,
            "supported_diagnostics": [],
            "supported_metadata": [],
            "has_native_plot": False,
            "is_linear": False,
            "is_stochastic": False,
        }

    def _attribute_dict(self, obj: Any, attrs: Iterable[str]) -> Dict[str, Any]:
        """
        Extract requested attributes from a target object into a dictionary.

        This helper filters missing attributes and swallows common access
        errors (such as deferred scikit-learn properties) to return only what
        is currently available on the target.

        Parameters
        ----------
        obj : Any
            Target object to inspect.
        attrs : iterable of str
            Attribute names to attempt to extract.

        Returns
        -------
        dict
            Mapping of available attribute names to their values.
        """
        if obj is None:
            return {}

        out = {}
        for attr in attrs:
            try:
                out[attr] = getattr(obj, attr)
            except (AttributeError, RuntimeError, ValueError):
                continue
        return out

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Return diagnostic arrays or structured artifacts.

        Diagnostics are intended for non-scalar outputs such as explained
        variance curves, eigenvalues, modes, graphs, or training histories.
        Only names declared in `capabilities["supported_diagnostics"]` are
        queried.

        Returns
        -------
        diagnostics : dict
            Dictionary of diagnostic attributes declared in
            `capabilities["supported_diagnostics"]`.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        attrs = self.capabilities.get("supported_diagnostics", [])
        diag = self._attribute_dict(self.model, attrs)
        diag.update(self._attribute_dict(self, attrs))
        return diag

    def get_quality_metadata(self) -> Dict[str, Any]:
        """
        Return scalar metadata about the reduction process or quality.

        Typical examples include iteration counts, optimization stress, final
        loss values, or backend-specific convergence flags. Only names
        declared in `capabilities["supported_metadata"]` are queried.

        Returns
        -------
        metadata : dict
            Dictionary containing only scalar values corresponding to keys
            declared in `capabilities["supported_metadata"]`.

        Raises
        ------
        RuntimeError
            If the reducer has not been fitted.
        """
        self._require_fitted()
        attrs = self.capabilities.get("supported_metadata", [])
        meta = self._attribute_dict(self.model, attrs)
        meta.update(self._attribute_dict(self, attrs))
        return meta

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

        Notes
        -----
        This method assumes the reducer was serialized with `save` or a
        compatible `joblib.dump` call.
        """
        return joblib.load(str(filepath))
