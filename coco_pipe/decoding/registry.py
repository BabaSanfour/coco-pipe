"""
Decoding Registry Engine
========================

Central engine for resolving, registering, and lazy-loading decoding
estimators. This allows configurations to refer to models by name
without triggering eager imports of heavyweight dependencies.
"""

from __future__ import annotations

import difflib
import importlib
import pkgutil
import threading
import warnings
from dataclasses import replace
from typing import Any, Callable, Dict, Type

from ._specs import (
    ESTIMATOR_SPECS,
    SELECTOR_CAPABILITIES,
    EstimatorCapabilities,
    EstimatorSpec,
    SelectorCapabilities,
    canonical_estimator_name,
)

# Runtime class cache
_ESTIMATOR_REGISTRY: Dict[str, Type] = {}
_INTERNAL_SCANNED = False
_REGISTRY_LOCK = threading.Lock()


class EstimatorNotFoundError(KeyError, ValueError):
    """Raised when an estimator is not found in the registry."""

    pass


def _discover_entry_points():  # pragma: no cover
    """Import 'coco_pipe.estimators' entry points."""
    try:
        from importlib.metadata import entry_points

        try:
            eps = entry_points(group="coco_pipe.estimators")
        except TypeError:
            eps = entry_points().get("coco_pipe.estimators", [])
    except Exception:
        return

    for ep in eps:
        try:
            ep.load()
        except Exception:
            warnings.warn(f"Could not load estimator entry point '{ep.name}'")


def _discover_internal_modules():  # pragma: no cover
    """Import all internal decoding submodules to trigger decorators."""
    package = importlib.import_module("coco_pipe.decoding")
    if not hasattr(package, "__path__"):
        return

    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(name)
        except ImportError:
            pass


# Lazy entry point discovery
_discover_entry_points()


def register_estimator(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register a custom estimator class under a specific name.

    Parameters
    ----------
    name : str
        The unique name to register the estimator under.

    Returns
    -------
    Callable[[Type], Type]
        A decorator that adds the class to the internal registry.
    """

    def decorator(cls: Type) -> Type:
        if name in _ESTIMATOR_REGISTRY:
            warnings.warn(f"Overwriting existing estimator registry for '{name}'")
        _ESTIMATOR_REGISTRY[name] = cls
        return cls

    return decorator


def register_estimator_spec(spec: EstimatorSpec) -> EstimatorSpec:
    """
    Register or replace an estimator spec in the global specs registry.

    This allows the execution engine to support new model types by name,
    defining their capabilities, required input formats, and importance
    extraction logic.

    Parameters
    ----------
    spec : EstimatorSpec
        The typed specification object for the estimator.

    Returns
    -------
    EstimatorSpec
        The registered specification object.

    See Also
    --------
    get_estimator_spec : Retrieve a registered specification.
    """
    if spec.name in ESTIMATOR_SPECS:
        warnings.warn(f"Overwriting existing estimator spec for '{spec.name}'")
    ESTIMATOR_SPECS[spec.name] = spec
    return spec


def get_estimator_cls(name: str) -> Type:
    """
    Retrieve an estimator class by name, triggering lazy loading if needed.

    Parameters
    ----------
    name : str
        The canonical name of the estimator (e.g., 'LogisticRegression').

    Returns
    -------
    Type
        The uninstantiated estimator class.

    Raises
    ------
    EstimatorNotFoundError
        If the estimator is unknown.
    ImportError
        If the underlying module cannot be imported.

    Examples
    --------
    >>> from coco_pipe.decoding.registry import get_estimator_cls
    >>> cls = get_estimator_cls('LogisticRegression')

    See Also
    --------
    get_estimator_spec : Retrieve the metadata for an estimator.
    """
    if name in _ESTIMATOR_REGISTRY:
        return _ESTIMATOR_REGISTRY[name]

    # Try lazy import from spec
    spec = ESTIMATOR_SPECS.get(name)
    if spec is not None:
        try:
            module = importlib.import_module(spec.module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not load estimator '{name}' from '{spec.import_path}'. "
                f"Ensure optional dependencies are installed."
            ) from e

        if hasattr(module, spec.class_name):
            cls = getattr(module, spec.class_name)
            _ESTIMATOR_REGISTRY[name] = cls
            return cls

        if name in _ESTIMATOR_REGISTRY:  # pragma: no cover
            return _ESTIMATOR_REGISTRY[name]

    # Try internal discovery
    global _INTERNAL_SCANNED
    if not _INTERNAL_SCANNED:
        with _REGISTRY_LOCK:
            if not _INTERNAL_SCANNED:
                _discover_internal_modules()  # pragma: no cover
                _INTERNAL_SCANNED = True  # pragma: no cover
        if name in _ESTIMATOR_REGISTRY:  # pragma: no cover
            return _ESTIMATOR_REGISTRY[name]

    if name not in _ESTIMATOR_REGISTRY:
        available = sorted(set(_ESTIMATOR_REGISTRY) | set(ESTIMATOR_SPECS))
        matches = difflib.get_close_matches(name, available, n=3, cutoff=0.6)
        msg = f"Estimator '{name}' not found in registry."
        if matches:
            msg += f" Did you mean: {matches}?"
        msg += f"\nAvailable estimators: {available[:10]}... (Total: {len(available)})"
        raise EstimatorNotFoundError(msg)

    return _ESTIMATOR_REGISTRY[name]  # pragma: no cover


def get_estimator_spec(name: str) -> EstimatorSpec:
    """
    Return the typed estimator spec for a given name.

    Parameters
    ----------
    name : str
        The canonical name of the estimator (e.g., 'LogisticRegression').

    Returns
    -------
    EstimatorSpec
        The registered specification object.

    Raises
    ------
    ValueError
        If no specification is registered for the given name.

    See Also
    --------
    get_capabilities : Retrieve derived lightweight capabilities.
    """
    if name not in ESTIMATOR_SPECS:
        raise ValueError(f"No decoding estimator spec registered for '{name}'.")
    return ESTIMATOR_SPECS[name]


def get_capabilities(name: str) -> EstimatorCapabilities:
    """
    Return machine-readable capability metadata for a given estimator.

    Parameters
    ----------
    name : str
        The canonical name of the estimator.

    Returns
    -------
    EstimatorCapabilities
        Lightweight metadata summary for validation.
    """
    return get_estimator_spec(name).to_capabilities()


def list_capabilities() -> Dict[str, EstimatorCapabilities]:
    """
    Return capability metadata for all registered estimators.

    Returns
    -------
    Dict[str, EstimatorCapabilities]
        A dictionary mapping estimator names to their capability objects.

    See Also
    --------
    get_capabilities : Retrieve capabilities for a single estimator.
    """
    return {name: spec.to_capabilities() for name, spec in ESTIMATOR_SPECS.items()}


def list_estimator_specs() -> Dict[str, EstimatorSpec]:
    """
    Return all registered estimator specs.

    Returns
    -------
    Dict[str, EstimatorSpec]
        A dictionary mapping estimator names to their full specification objects.

    See Also
    --------
    get_estimator_spec : Retrieve a single estimator specification.
    """
    return dict(ESTIMATOR_SPECS)


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """
    Retrieve a value from a configuration object or dictionary.

    This helper provides a unified interface for accessing parameters from
    both Pydantic models and raw dictionaries, which is common when
    handling polymorphic or nested estimator configurations.

    Parameters
    ----------
    obj : Any
        The configuration source (dict or object).
    key : str
        The parameter name to retrieve.
    default : Any, optional
        The default value if not found.

    Returns
    -------
    Any
        The retrieved value.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def resolve_estimator_spec(config: Any) -> EstimatorSpec:
    """
    Resolve a hydrated EstimatorSpec from a model configuration.

    This function handles polymorphic model types (Foundation, Temporal,
    Neural) and applies runtime parameter fixups based on the user's
    specific configuration (e.g., handling SVC probability flags).

    Parameters
    ----------
    config : Any
        A configuration object (typically a Pydantic model) containing
        the 'kind' and specific estimator parameters.

    Returns
    -------
    EstimatorSpec
        A hydrated spec object containing accurate flags for the training engine.

    Examples
    --------
    >>> from coco_pipe.decoding.configs import LogisticRegressionConfig
    >>> from coco_pipe.decoding.registry import resolve_estimator_spec
    >>> config = LogisticRegressionConfig()
    >>> spec = resolve_estimator_spec(config)

    See Also
    --------
    resolve_estimator_capabilities : Lightweight capability summary.
    """
    # 1. Temporal Wrapper logic (Needs special handling for base spec)
    kind = _get_val(config, "kind", "classical")
    if kind == "temporal":
        base_spec = resolve_estimator_spec(_get_val(config, "base"))
        wrapper = _get_val(config, "wrapper", "sliding")
        wrapper_name = (
            "SlidingEstimator" if wrapper == "sliding" else "GeneralizingEstimator"
        )
        return replace(
            get_estimator_spec(wrapper_name),
            task=base_spec.task,
            supports_proba=base_spec.supports_proba,
            supports_decision_function=base_spec.supports_decision_function,
            supports_calibration=base_spec.supports_calibration,
            supports_feature_names=False,
        )

    if kind == "classical":
        name_val = _get_val(config, "method")
        if name_val == "ClassicalModel":
            name_val = _get_val(config, "estimator")
        spec_name = canonical_estimator_name(name_val or str(config))
    elif kind == "foundation_embedding":
        spec_name = _get_val(config, "provider", kind)
    else:
        spec_name = kind

    spec = get_estimator_spec(spec_name or str(config))

    # Runtime Fixups (Original logic)
    if spec.name == "SVC" and not _get_val(config, "probability", True):
        spec = replace(spec, supports_proba=False, supports_decision_function=True)

    if spec.name == "SGDClassifier" and _get_val(config, "loss") in {
        "log_loss",
        "modified_huber",
    }:
        spec = replace(spec, supports_proba=True, supports_decision_function=True)

    return spec


def resolve_estimator_capabilities(config: Any) -> EstimatorCapabilities:
    """
    Resolve lightweight capabilities derived from ``resolve_estimator_spec``.

    This is a convenience wrapper that first resolves the full spec (handling
    polymorphism and runtime fixups) and then converts it to the capability
    summary used by the engine for validation.

    Parameters
    ----------
    config : Any
        The model configuration object.

    Returns
    -------
    EstimatorCapabilities
        The resolved capability metadata.

    See Also
    --------
    resolve_estimator_spec : The underlying spec resolution logic.
    """
    return resolve_estimator_spec(config).to_capabilities()


def get_selector_capabilities(method: str) -> SelectorCapabilities:
    """
    Return feature-selector capabilities for a given name.

    Parameters
    ----------
    method : str
        The name of the feature selection method (e.g., 'k_best', 'sfs').

    Returns
    -------
    SelectorCapabilities
        The registered capability metadata for the selector.

    Raises
    ------
    ValueError
        If the method name is not found in the SELECTOR_CAPABILITIES registry.
    """
    if method not in SELECTOR_CAPABILITIES:
        raise ValueError(
            f"No decoding capabilities registered for selector '{method}'."
        )
    return SELECTOR_CAPABILITIES[method]
