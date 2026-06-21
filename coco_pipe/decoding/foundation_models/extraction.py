"""Reusable extraction and preflight utilities for EEG foundation models."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._cache import make_feature_cache_key
from .._specs import SignalMetadata
from ..registry import get_foundation_model_spec
from ._loader import _BACKEND_MAP
from ._prepare import normalize_channel_names, prepare_backend

if TYPE_CHECKING:
    from ...io import DataContainer


@dataclass(frozen=True)
class CapabilityResult:
    """Preflight result for one model/training-mode/input combination."""

    model_key: str
    train_mode: str
    status: str
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def check_capability(
    model_key: str,
    train_mode: str,
    sfreq: float | None = None,
    ch_names: Sequence[str] | None = None,
    n_times: int | None = None,
    backend: str = "auto",
    backend_kwargs: Mapping[str, Any] | None = None,
) -> CapabilityResult:
    """Check registry support and obvious signal incompatibilities without loading."""
    try:
        spec = get_foundation_model_spec(model_key)
    except KeyError as exc:
        return CapabilityResult(
            model_key, train_mode, "invalid_configuration", str(exc)
        )
    registry_mode = "frozen" if train_mode == "linear_probe" else train_mode
    if registry_mode not in spec.supported_train_modes:
        return CapabilityResult(
            model_key,
            train_mode,
            "unsupported",
            f"{spec.display_name} does not advertise train_mode={train_mode!r}.",
        )
    try:
        resolved_backend = spec.preferred_backend if backend == "auto" else backend
        backend_cls = _BACKEND_MAP.get(resolved_backend)
        if backend_cls is None:
            return CapabilityResult(
                model_key,
                train_mode,
                "invalid_configuration",
                f"Unknown foundation backend {resolved_backend!r}.",
            )
        if not backend_cls.is_available():
            return CapabilityResult(
                model_key,
                train_mode,
                "missing_dependency",
                (f"Backend dependencies for {resolved_backend!r} are not installed."),
            )
    except Exception as exc:
        return CapabilityResult(
            model_key,
            train_mode,
            "missing_dependency",
            str(exc),
        )
    if spec.requires_auth and resolved_backend == spec.preferred_backend:
        token = (backend_kwargs or {}).get("token")
        if token is None:
            try:
                from huggingface_hub import get_token

                token = get_token()
            except Exception:
                token = None
        if not token:
            return CapabilityResult(
                model_key,
                train_mode,
                "authentication_required",
                (
                    f"{spec.display_name} uses a gated checkpoint. Authenticate "
                    "with Hugging Face or pass backend_kwargs.token."
                ),
                {
                    "checkpoint": spec.hub_repo,
                    "checkpoint_revision": spec.checkpoint_revision,
                },
            )
    normalized = normalize_channel_names(ch_names or [])
    requires_adaptation = (
        spec.pretrained_n_chans is not None
        and len(normalized) != spec.pretrained_n_chans
    )
    interpolation_requested = bool(
        (backend_kwargs or {}).get("interpolate_channels", False)
    )
    if requires_adaptation and (
        not spec.supports_channel_interpolation or not interpolation_requested
    ):
        return CapabilityResult(
            model_key,
            train_mode,
            "incompatible_channels",
            (
                f"Expected {spec.pretrained_n_chans} channels, got "
                f"{len(normalized)}. Enable and validate channel interpolation."
            ),
            {"normalized_channels": normalized},
        )
    details = {
        "display_name": spec.display_name,
        "pretrained_sfreq": spec.pretrained_sfreq,
        "input_sfreq": sfreq,
        "normalized_channels": normalized,
        "requires_resampling": sfreq is not None and sfreq != spec.pretrained_sfreq,
        "requires_channel_adaptation": requires_adaptation,
        "interpolation_requested": interpolation_requested,
        "checkpoint_revision": spec.checkpoint_revision,
        "checkpoint_filename": spec.checkpoint_filename,
        "requires_auth": spec.requires_auth,
    }
    if n_times is not None and sfreq is not None and spec.pretrained_n_times:
        model_n_times = round(n_times * spec.pretrained_sfreq / sfreq)
        if model_n_times != spec.pretrained_n_times:
            return CapabilityResult(
                model_key,
                train_mode,
                "incompatible_window",
                (
                    f"{spec.display_name} requires {spec.pretrained_n_times} "
                    f"samples at {spec.pretrained_sfreq:g} Hz; this input maps "
                    f"to {model_n_times}."
                ),
                {**details, "model_n_times": model_n_times},
            )
    return CapabilityResult(model_key, train_mode, "available", details=details)


def normalize_inclusive_endpoint(
    container: DataContainer,
    *,
    segment_duration: float,
    expected_sfreq: float,
    model_key: str = "model",
    on_mismatch: str = "error",
) -> tuple[DataContainer | None, str | None]:
    """Drop MNE's inclusive-endpoint extra sample from epoched windows.

    MNE epochs span ``[tmin, tmax]`` inclusively, yielding ``expected_n_times +
    1`` samples; foundation backends want the half-open count. This trims the
    trailing sample (recording it in ``meta``) when the off-by-one is observed,
    so ``prepare`` does not raise on the mismatch.

    Returns ``(container, None)`` when the window is already correct or was
    normalized. On any other length mismatch, returns ``(None, reason)`` if
    ``on_mismatch == "skip"`` and otherwise raises ``ValueError``.
    """
    sfreq = float(container.meta.get("sfreq", expected_sfreq))
    expected_n_times = round(segment_duration * sfreq)
    observed_n_times = int(container.X.shape[-1])
    if observed_n_times == expected_n_times:
        return container, None
    if observed_n_times == expected_n_times + 1:
        normalized = container.isel(time=np.arange(expected_n_times, dtype=int))
        normalized.meta = {
            **dict(normalized.meta),
            "inclusive_endpoint_removed": True,
            "original_n_times": observed_n_times,
            "normalized_n_times": expected_n_times,
        }
        return normalized, None

    reason = (
        f"{model_key} expected {expected_n_times} samples for a "
        f"{segment_duration:g} s window at {sfreq:g} Hz, but loaded "
        f"{observed_n_times}."
    )
    if on_mismatch == "skip":
        return None, reason
    raise ValueError(reason)


@dataclass
class FoundationEmbeddingResult:
    """Window- and recording-level embeddings plus extraction provenance."""

    window_embeddings: np.ndarray
    recording_embedding: np.ndarray
    window_start: np.ndarray
    window_stop: np.ndarray
    window_index: np.ndarray
    metadata: dict[str, Any]


class FoundationEmbeddingExtractor:
    """Compose model loading, metadata binding, extraction, and pooling."""

    def __init__(
        self,
        model_key: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        pooling: str = "mean",
        recording_pooling: str = "mean",
        normalize_embeddings: bool = True,
        resample: bool = True,
        cache_embeddings: bool = False,
        backend_kwargs: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        if recording_pooling not in {"mean", "median", "max"}:
            raise ValueError("recording_pooling must be mean, median, or max.")
        self.model_key = model_key
        self.backend = backend
        self.device = device
        self.pooling = pooling
        self.recording_pooling = recording_pooling
        self.normalize_embeddings = normalize_embeddings
        self.resample = resample
        self.cache_embeddings = cache_embeddings
        self.backend_kwargs = dict(backend_kwargs or {})
        self.model = model
        self._embedding_cache: dict[str, np.ndarray] = {}

    def clear_cache(self) -> None:
        """Drop all memoized window embeddings."""
        self._embedding_cache.clear()

    def _embed_windows(self, model: Any, model_input: np.ndarray) -> np.ndarray:
        """Run the backbone forward pass and pool/normalize to 2-D rows."""
        embeddings = np.asarray(model.transform(model_input), dtype=np.float32)
        if embeddings.ndim > 2:
            if self.pooling == "flatten":
                embeddings = embeddings.reshape(len(embeddings), -1)
            else:
                embeddings = embeddings.mean(axis=tuple(range(1, embeddings.ndim - 1)))
        if self.normalize_embeddings and embeddings.ndim == 2:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = np.divide(
                embeddings,
                norms,
                out=np.zeros_like(embeddings),
                where=norms > 0,
            )
        return embeddings

    def _backbone_fingerprint(self, prepared: Any) -> str:
        """Stable identity of the deterministic window->embedding mapping."""
        return "|".join(
            str(part)
            for part in (
                self.model_key,
                getattr(prepared, "backend_name", self.backend),
                self.pooling,
                self.normalize_embeddings,
                float(getattr(prepared, "target_sfreq", 0.0)),
                tuple(getattr(prepared, "ch_names", ()) or ()),
            )
        )

    def _embed_windows_cached(
        self, model: Any, model_input: np.ndarray, prepared: Any
    ) -> np.ndarray:
        """Embed windows, reusing memoized rows for previously seen content."""
        backbone_fp = self._backbone_fingerprint(prepared)
        keys = [
            make_feature_cache_key(
                [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()],
                [],
                "embed",
                backbone_fp,
                sort_ids=False,
            )
            for row in model_input
        ]
        missing = [i for i, key in enumerate(keys) if key not in self._embedding_cache]
        if missing:
            computed = self._embed_windows(model, model_input[missing])
            if computed.ndim != 2 or len(computed) != len(missing):
                raise ValueError(
                    "Foundation backend must return one 2-D embedding row per window."
                )
            for offset, index in enumerate(missing):
                self._embedding_cache[keys[index]] = computed[offset]
        return np.stack([self._embedding_cache[key] for key in keys])

    def extract(
        self,
        epochs: np.ndarray,
        *,
        signal_metadata: SignalMetadata,
        window_start: Sequence[int] | None = None,
        window_stop: Sequence[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FoundationEmbeddingResult:
        """Extract one embedding per input window and one pooled recording vector."""
        X = np.asarray(epochs, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("epochs must have shape (window, channel, time).")
        prepared = prepare_backend(
            self.model_key,
            X,
            backend=self.backend,
            device=self.device,
            train_mode="frozen",
            n_outputs=None,
            sfreq=float(signal_metadata.sfreq),
            ch_names=signal_metadata.ch_names,
            pooling=self.pooling,
            backend_kwargs=self.backend_kwargs,
            model=self.model,
        )
        self.model = prepared.backend
        model = prepared.backend
        spec = prepared.spec
        if not self.resample and prepared.source_sfreq != prepared.target_sfreq:
            warnings.warn(
                f"resample=False but input sfreq ({prepared.source_sfreq} Hz) != "
                f"model sfreq ({prepared.target_sfreq} Hz); the model receives "
                "mismatched-rate data and the embeddings will be unreliable.",
                stacklevel=2,
            )
        model_input = prepared.adapt(X) if self.resample else X
        resampled = self.resample and prepared.source_sfreq != prepared.target_sfreq
        if self.cache_embeddings:
            embeddings = self._embed_windows_cached(model, model_input, prepared)
        else:
            embeddings = self._embed_windows(model, model_input)
        if embeddings.ndim != 2 or len(embeddings) != len(X):
            raise ValueError(
                "Foundation backend must return one 2-D embedding row per window."
            )
        reducer = {
            "mean": np.mean,
            "median": np.median,
            "max": np.max,
        }[self.recording_pooling]
        recording = reducer(embeddings, axis=0)
        starts = np.asarray(
            window_start if window_start is not None else np.arange(len(X)),
            dtype=int,
        )
        stops = np.asarray(
            window_stop if window_stop is not None else starts + X.shape[-1],
            dtype=int,
        )
        payload = {
            **dict(metadata or {}),
            "model_name": spec.display_name or self.model_key,
            "model_key": self.model_key,
            "model_checkpoint": spec.hub_repo,
            "checkpoint_revision": spec.checkpoint_revision,
            "checkpoint_filename": spec.checkpoint_filename,
            "backend": (
                self.backend if self.backend != "auto" else spec.preferred_backend
            ),
            "input_sfreq": float(prepared.source_sfreq),
            "model_sfreq": float(prepared.target_sfreq),
            "pretrained_sfreq": float(spec.pretrained_sfreq),
            "original_sfreq": float(signal_metadata.sfreq),
            "resampled": resampled,
            "requires_resampling": prepared.source_sfreq != prepared.target_sfreq,
            "original_channels": list(signal_metadata.ch_names),
            "model_channels": prepared.ch_names,
            "channel_mapping": dict(
                zip(signal_metadata.ch_names, prepared.ch_names, strict=False)
            ),
            "channel_adaptation": {
                "legacy_1020_renamed": [
                    {
                        "from": original,
                        "to": normalized,
                    }
                    for original, normalized in zip(
                        signal_metadata.ch_names, prepared.ch_names, strict=False
                    )
                    if original != normalized
                ],
                "interpolation_requested": bool(
                    self.backend_kwargs.get("interpolate_channels", False)
                ),
                "pretrained_n_chans": spec.pretrained_n_chans,
                "input_n_chans": len(prepared.ch_names),
                **prepared.channel_adaptation(),
            },
            "within_window_pooling": self.pooling,
            "recording_pooling": self.recording_pooling,
            "normalize_embeddings": self.normalize_embeddings,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "window_count": len(X),
            "window_n_times_original": int(X.shape[-1]),
            "window_n_times_model": int(model_input.shape[-1]),
            "window_duration_seconds": float(
                X.shape[-1] / float(signal_metadata.sfreq)
            ),
            "windowing_source": "upstream_epochs",
            "remainder_policy": "defined_by_upstream_epoching",
            "padding_policy": "defined_by_upstream_epoching",
            "model_revision": (
                self.backend_kwargs.get("revision") or spec.checkpoint_revision
            ),
        }
        return FoundationEmbeddingResult(
            window_embeddings=embeddings,
            recording_embedding=np.asarray(recording),
            window_start=starts,
            window_stop=stops,
            window_index=np.arange(len(X), dtype=int),
            metadata=payload,
        )
