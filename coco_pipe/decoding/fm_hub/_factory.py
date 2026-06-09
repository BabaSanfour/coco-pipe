"""
Foundation Model Factory
========================
Handles instantiation of foundation models with lazy loading of providers.
"""

from typing import Any

_PROVIDER_MAP = {
    "reve": (".reve", "REVEModel"),
    "custom": (".custom", "CustomNeuralModel"),
}


def build_foundation_model(config: Any) -> Any:
    """
    Instantiate a foundation model from a config object.

    This function uses lazy loading to avoid importing heavy dependencies
    (torch, transformers) until a model is actually requested.
    """
    provider = getattr(config, "provider", None)
    if provider not in _PROVIDER_MAP:
        supported = list(_PROVIDER_MAP.keys())
        raise ValueError(
            f"Unknown foundation model provider '{provider}'. "
            f"Supported providers: {supported}"
        )

    # Lazy import of the provider class
    from importlib import import_module

    module_path, class_name = _PROVIDER_MAP[provider]

    # Resolve relative import
    module = import_module(module_path, package="coco_pipe.decoding.fm_hub")
    model_cls = getattr(module, class_name)

    # Extract common parameters from config
    params = {
        "model_name": getattr(config, "model_name", None),
        "electrode_names": getattr(config, "electrode_names", None),
        "sfreq": getattr(config, "sfreq", 200.0),
        "train_mode": getattr(config, "train_mode", "frozen"),
        "pooling": getattr(config, "pooling", "mean"),
        "device": getattr(config, "device", None),
        "token": getattr(config, "token", None),
        "task": getattr(config, "task", "classification"),
    }

    # Filter out None values to allow class defaults to take over
    params = {k: v for k, v in params.items() if v is not None}

    return model_cls(**params)
