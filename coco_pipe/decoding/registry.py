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
from importlib.metadata import entry_points

import pkgutil
import warnings
from typing import Callable, Dict, Type

# Registry Storage
# Maps string alias -> class object
_ESTIMATOR_REGISTRY: Dict[str, Type] = {}


_LAZY_MODULES = {
    # MNE
    "SlidingEstimator": "mne.decoding",
    "GeneralizingEstimator": "mne.decoding",
    # Classifiers
    "LogisticRegression": "sklearn.linear_model",
    "RandomForestClassifier": "sklearn.ensemble",
    "SVC": "sklearn.svm",
    "KNeighborsClassifier": "sklearn.neighbors",
    "GradientBoostingClassifier": "sklearn.ensemble",
    "SGDClassifier": "sklearn.linear_model",
    "MLPClassifier": "sklearn.neural_network",
    "GaussianNB": "sklearn.naive_bayes",
    "LDA": "sklearn.discriminant_analysis",
    "AdaBoostClassifier": "sklearn.ensemble",
    "DummyClassifier": "sklearn.dummy",
    # Regressors
    "LinearRegression": "sklearn.linear_model",
    "Ridge": "sklearn.linear_model",
    "Lasso": "sklearn.linear_model",
    "ElasticNet": "sklearn.linear_model",
    "RandomForestRegressor": "sklearn.ensemble",
    "SVR": "sklearn.svm",
    "ARDRegression": "sklearn.linear_model",
}

def _discover_entry_points():
    """
    Populate _LAZY_MODULES from 'coco_pipe.estimators' entry points.
    This allows plugins to register estimators without modifying code.
    """
    eps = entry_points(group="coco_pipe.estimators")
    for ep in eps:
        if ep.name not in _LAZY_MODULES:
            _LAZY_MODULES[ep.name] = ep.value


def _discover_internal_modules():
    """
    Walk through the 'coco_pipe.decoding' subpackage and import all modules.
    This triggers the @register_estimator decorators.
    """
    package = importlib.import_module("coco_pipe.decoding")
    if not hasattr(package, "__path__"):
        return

    for _, name, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            importlib.import_module(name)
        except ImportError:
            # warn but continue - we don't want to crash if deep learning libs are missing
            pass


# 1. Load Entry Points on startup (lazy map update only)
_discover_entry_points()


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

    # 2. Try Lazy Loading Map
    if name in _LAZY_MODULES:
        try:
            mod_path = _LAZY_MODULES[name]
            if ":" in mod_path:
                 mod_path = mod_path.split(":")[0]
            
            module = importlib.import_module(mod_path)
        except ImportError as e:
            raise ImportError(
                f"Could not load estimator '{name}' from '{_LAZY_MODULES[name]}'. "
                f"Ensure optional dependencies are installed."
            ) from e
        
        if hasattr(module, name):
             cls = getattr(module, name)
             _ESTIMATOR_REGISTRY[name] = cls
             return cls

        # Check if the import triggered a decorator registration
        if name in _ESTIMATOR_REGISTRY:
             return _ESTIMATOR_REGISTRY[name]

    # 3. Last Ditch: Internal Discovery
    if not getattr(get_estimator_cls, "_internal_scanned", False):
         _discover_internal_modules()
         setattr(get_estimator_cls, "_internal_scanned", True)
         if name in _ESTIMATOR_REGISTRY:
             return _ESTIMATOR_REGISTRY[name]

    if name not in _ESTIMATOR_REGISTRY:
        # Generate helpful error
        available = sorted(list(_ESTIMATOR_REGISTRY.keys()))
        raise ValueError(
            f"Estimator '{name}' not found in registry.\n"
            f"Available estimators: {available}\n"
            f"Tip: Ensure the containing module is imported or registered via entry points."
        )

    return _ESTIMATOR_REGISTRY[name]


def list_estimators() -> Dict[str, Type]:
    """Return a copy of the current registry."""
    # Ensure everything is discovered before listing
    if not getattr(get_estimator_cls, "_internal_scanned", False):
         _discover_internal_modules()
         setattr(get_estimator_cls, "_internal_scanned", True)
    return dict(_ESTIMATOR_REGISTRY)
