"""
Decoding Registry
=================

Central registry for decoding estimators (classifiers, regressors, and FMs).
This allows instantiating models from string names in configuration files,
avoiding circular imports and simplifying the config layer.

Usage
-----
>>> from coco_pipe.decoding.registry import register_estimator, get_estimator_cls
>>> 
>>> @register_estimator("MyModel")
>>> class MyModel: ...
>>> 
>>> cls = get_estimator_cls("MyModel")
"""

import importlib
import warnings
from typing import Callable, Dict, Type

# Registry Storage
# Maps string alias -> class object
_ESTIMATOR_REGISTRY: Dict[str, Type] = {}


# Lazy Loading Map
# specific_class_name -> module_path
_LAZY_MODULES = {
    "SkorchWrapper": "coco_pipe.decoding.deep",
    "FoundationEstimator": "coco_pipe.decoding.deep",
    "LPFTClassifier": "coco_pipe.decoding.fm.adapters",
    "SlidingEstimator": "mne.decoding",
    "GeneralizingEstimator": "mne.decoding",
}


def register_estimator(name: str) -> Callable[[Type], Type]:
    """
    Decorator to register an estimator class under a specific name.

    Parameters
    ----------
    name : str
        The unique alias for the estimator (e.g., "RandomForestClassifier").
    """

    def decorator(cls: Type) -> Type:
        if name in _ESTIMATOR_REGISTRY:
            warnings.warn(f"Overwriting existing estimator registry for '{name}'")
        _ESTIMATOR_REGISTRY[name] = cls
        return cls

    return decorator


def get_estimator_cls(name: str) -> Type:
    """
    Retrieve an estimator class by name.

    Parameters
    ----------
    name : str
        Name of the estimator.

    Returns
    -------
    Type
        The class object.

    Raises
    ------
    ValueError
        If name is not found.
    """
    # 1. Check if already loaded
    if name in _ESTIMATOR_REGISTRY:
        return _ESTIMATOR_REGISTRY[name]

    # 2. Try Lazy Loading
    if name in _LAZY_MODULES:
        try:
            importlib.import_module(_LAZY_MODULES[name])
        except ImportError as e:
            # If the module is missing dependencies (e.g. torch), re-raise with clear context
            raise ImportError(
                f"Could not load estimator '{name}' from '{_LAZY_MODULES[name]}'. "
                f"Ensure optional dependencies are installed."
            ) from e

    # 3. Last Ditch: Ensure core is loaded
    if "RandomForestClassifier" not in _ESTIMATOR_REGISTRY:
        try:
            importlib.import_module("coco_pipe.decoding.core")
        except ImportError:
            pass

    if name not in _ESTIMATOR_REGISTRY:
        # Generate helpful error
        available = sorted(list(_ESTIMATOR_REGISTRY.keys()))
        raise ValueError(
            f"Estimator '{name}' not found in registry.\n"
            f"Available estimators: {available}\n"
            f"Tip: Ensure the containing module is imported."
        )

    return _ESTIMATOR_REGISTRY[name]


def list_estimators() -> Dict[str, Type]:
    """Return a copy of the current registry."""
    return dict(_ESTIMATOR_REGISTRY)
