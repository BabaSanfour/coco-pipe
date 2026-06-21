"""Shared foundation-backend loading and signal adaptation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._specs import FoundationModelSpec, SignalMetadata
from ..registry import get_foundation_model_spec
from ._loader import load

_CHANNEL_ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize_channel_names(ch_names: Sequence[str]) -> list[str]:
    """Normalize legacy 10-20 temporal labels."""
    return [_CHANNEL_ALIASES.get(str(name), str(name)) for name in ch_names]


@dataclass
class PreparedBackend:
    """A loaded backend plus deterministic input adaptation metadata."""

    backend: Any
    spec: FoundationModelSpec
    source_sfreq: float
    target_sfreq: float
    ch_names: list[str]
    model_n_times: int

    def adapt(self, X: np.ndarray) -> np.ndarray:
        """Convert values to float32 and resample to the model rate."""
        values = np.asarray(X, dtype=np.float32)
        if self.source_sfreq != self.target_sfreq:
            from scipy.signal import resample

            n_times = round(values.shape[-1] * self.target_sfreq / self.source_sfreq)
            values = resample(values, n_times, axis=-1).astype(np.float32)
        if (
            self.spec.pretrained_n_times is not None
            and values.shape[-1] != self.spec.pretrained_n_times
        ):
            raise ValueError(
                f"{self.spec.display_name} checkpoint requires "
                f"{self.spec.pretrained_n_times} samples per input at "
                f"{self.target_sfreq:g} Hz, got {values.shape[-1]}. "
                "Adjust upstream epoch duration; temporal padding/cropping is "
                "not performed implicitly."
            )
        return values

    def channel_adaptation(self) -> dict[str, Any]:
        """Return backend-observed channel adaptation details."""
        if hasattr(self.backend, "get_channel_adaptation"):
            return dict(self.backend.get_channel_adaptation())
        return {
            "source_channels": list(self.ch_names),
            "target_channels": list(self.ch_names),
            "interpolated_channels": [],
            "zero_filled_channels": [],
            "dropped_channels": [],
        }


def prepare_backend(
    model_key: str,
    X: np.ndarray,
    *,
    backend: str = "auto",
    device: str = "auto",
    train_mode: str = "frozen",
    n_outputs: int | None = None,
    sfreq: float | None = None,
    ch_names: Sequence[str] | None = None,
    pooling: str = "mean",
    backend_kwargs: Mapping[str, Any] | None = None,
    model: Any | None = None,
) -> PreparedBackend:
    """Load and bind one backend using a common adaptation path."""
    values = np.asarray(X)
    if values.ndim != 3:
        raise ValueError("Foundation model input must be (sample, channel, time).")
    spec = get_foundation_model_spec(model_key)
    normalized = normalize_channel_names(
        ch_names or [f"ch{index}" for index in range(values.shape[1])]
    )
    source_sfreq = float(sfreq or spec.pretrained_sfreq)
    target_sfreq = float(spec.pretrained_sfreq)
    model_n_times = spec.pretrained_n_times or round(
        values.shape[-1] * target_sfreq / source_sfreq
    )
    kwargs = dict(backend_kwargs or {})
    kwargs.setdefault("sfreq", target_sfreq)
    kwargs.setdefault("n_times", model_n_times)
    kwargs.setdefault("revision", spec.checkpoint_revision)
    if spec.checkpoint_filename is not None:
        kwargs.setdefault("filename", spec.checkpoint_filename)
    metadata = SignalMetadata(sfreq=target_sfreq, ch_names=normalized)
    if model is None:
        # Declared at load time — no separate binding step.
        model = load(
            model_key,
            backend=backend,
            n_outputs=n_outputs,
            device=device,
            train_mode=train_mode,
            pooling=pooling,
            electrode_names=normalized,
            signal_metadata=metadata,
            **kwargs,
        )
    elif hasattr(model, "bind_signal_metadata"):
        # Pre-built/injected model (e.g. test doubles): bind directly.
        model.bind_signal_metadata(metadata)
    else:
        model.signal_metadata_ = metadata
    return PreparedBackend(
        backend=model,
        spec=spec,
        source_sfreq=source_sfreq,
        target_sfreq=target_sfreq,
        ch_names=normalized,
        model_n_times=model_n_times,
    )
