"""
Foundation Model Loader
=======================

Public entry point for loading pretrained EEG/MEG foundation models.
Backend selection, device resolution, and dependency checking are
centralised here so callers interact with a single function.
"""

from __future__ import annotations

from ..registry import get_foundation_model_spec
from ._base import BackendBase
from ._braindecode import BrainDecodeBackend
from ._hugging_face import HuggingFaceBackend

_BACKEND_MAP: dict[str, type[BackendBase]] = {
    "hugging_face": HuggingFaceBackend,
    "braindecode": BrainDecodeBackend,
}


def load(
    model_key: str,
    *,
    backend: str = "auto",
    n_outputs: int | None = None,
    device: str = "auto",
    train_mode: str = "frozen",
    **backend_kwargs,
) -> BackendBase:
    """Load a pretrained EEG foundation model.

    Parameters
    ----------
    model_key : str
        Canonical model name. Use ``list_foundation_models()`` to see available keys.
    backend : str
        ``"auto"`` selects the preferred backend from the registry.
        Explicit: ``"hugging_face"``, ``"braindecode"``.
    n_outputs : int or None
        Classification outputs. ``None`` = headless / pure feature extractor.
    device : str
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, or ``"mps"``.
    train_mode : str
        ``"frozen"`` (default) | ``"full"`` | ``"lora"`` | ``"qlora"``.
    **backend_kwargs
        Passed through to the backend's ``load()`` method.

    Returns
    -------
    backend : BackendBase
        A fully initialised backend ready to ``fit`` / ``transform`` / ``predict``.

    Raises
    ------
    KeyError
        If ``model_key`` is not found in the registry.
    ValueError
        If the resolved backend name is not recognised.
    ImportError
        If the backend's required packages are not installed.
    """
    spec = get_foundation_model_spec(model_key)
    backend_name = spec.preferred_backend if backend == "auto" else backend
    if backend_name not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown backend '{backend_name}'. Available: {sorted(_BACKEND_MAP)}"
        )
    backend_cls = _BACKEND_MAP[backend_name]

    if not backend_cls.is_available():
        raise ImportError(
            f"Model '{model_key}' requires the '{backend_name}' backend "
            f"but its dependencies are not installed.\n"
            f'To install: pip install "coco-pipe[{spec.dependency_extra}]"'
        )

    return backend_cls.load(
        model_key,
        spec,
        n_outputs=n_outputs,
        device=_resolve_device(device),
        train_mode=train_mode,
        **backend_kwargs,
    )


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
