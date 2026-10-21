"""Reusable extraction and preflight utilities for EEG foundation models."""

from __future__ import annotations

import contextlib
import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .._cache import make_feature_cache_key
from .._specs import SignalMetadata
from ..registry import get_foundation_model_spec
from . import _montages
from ._base import check_finite
from ._channels import normalize_channel_names
from ._loader import _BACKEND_MAP
from ._prepare import prepare_backend

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
    # Fixed-montage models (biot/bendr/eegpt) use a faithful per-model channel
    # construction; their availability is the required-electrode / name-vocab
    # check, sharing one source of truth with the backend adapter.
    if _montages.is_special(model_key):
        fill_missing = bool((backend_kwargs or {}).get("fill_missing_channels", False))
        plan = _montages.plan_channels(
            model_key, normalized, fill_missing=fill_missing, sfreq=sfreq
        )
        if plan.status != "available":
            return CapabilityResult(
                model_key,
                train_mode,
                "incompatible_channels",
                plan.reason,
                {"normalized_channels": normalized},
            )
    requires_adaptation = (
        spec.pretrained_n_chans is not None
        and len(normalized) != spec.pretrained_n_chans
    )
    interpolation_requested = bool(
        (backend_kwargs or {}).get("interpolate_channels", False)
    )
    if requires_adaptation and not interpolation_requested:
        # LaBraM takes any subset of its pretraining channels, found by name.
        from braindecode.models.labram import LABRAM_CHANNEL_ORDER

        known = {name.upper() for name in LABRAM_CHANNEL_ORDER}
        unknown = [name for name in normalized if name.upper() not in known]
        if unknown:
            return CapabilityResult(
                model_key,
                train_mode,
                "incompatible_channels",
                (
                    f"{unknown} are not among the {spec.pretrained_n_chans} "
                    f"pretraining channels of {spec.display_name}. Drop them or "
                    "enable channel interpolation."
                ),
                {"normalized_channels": normalized},
            )
        requires_adaptation = False
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
        "expects_microvolts": spec.expects_microvolts,
        "input_scale": spec.input_scale,
        "input_normalization": spec.input_normalization,
        "input_reference": spec.input_reference,
        "input_clip": spec.input_clip,
        "finetune_input_scale": spec.finetune_input_scale,
        "expected_passband": spec.expected_passband,
        "expected_reference": spec.expected_reference,
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


#: The token axis each pooled view keeps; it averages the other observation axes.
_VIEW_AXES = {"channel": "channel", "patch": "time_patch"}


def _pooled_views(token_metadata: Mapping[str, Any]) -> list[str]:
    """Return the pooled views the token axes allow.

    A view keeps one observation axis and averages the others, so it needs
    that axis and at least one more (LUNA's tokens, one per time patch, are
    already its patch view), with the feature axis last.
    """
    axes = token_metadata["token_axes"]
    if axes[-1] != token_metadata["token_feature_axis"]:
        return []
    observation = axes[1:-1]
    return [
        view
        for view, axis in _VIEW_AXES.items()
        if axis in observation and len(observation) > 1
    ]


def _pool_tokens(
    tokens: np.ndarray, token_metadata: Mapping[str, Any], view: str
) -> np.ndarray:
    """Average tokens over the observation axes a view does not keep.

    Returns ``(window, <the view's axis>, feature)``.
    """
    axes = token_metadata["token_axes"]
    kept = axes.index(_VIEW_AXES[view])
    return tokens.mean(
        axis=tuple(index for index in range(1, len(axes) - 1) if index != kept)
    )


@dataclass
class FoundationEmbeddingResult:
    """Window embeddings plus extraction provenance.

    ``token_embeddings`` holds the native backbone feature tensor, without
    reshaping or normalization, and is populated only when the extractor is
    built with ``store_tokens=True``.

    ``channel_embeddings`` (window, channel, feature) holds each channel's
    tokens averaged over time patches, and ``patch_embeddings`` (window,
    time_patch, feature) each time patch's tokens averaged over channels. Each
    is populated when the backend's tokens have its axis and another one to
    average over (CBraMod, REVE and LaBraM without channel interpolation have
    both), unless the extractor caches window embeddings. They are not
    normalized, whatever ``normalize_embeddings`` says.
    """

    window_embeddings: np.ndarray
    window_start: np.ndarray
    window_stop: np.ndarray
    window_index: np.ndarray
    metadata: dict[str, Any]
    token_embeddings: np.ndarray | None = None
    channel_embeddings: np.ndarray | None = None
    patch_embeddings: np.ndarray | None = None


class FoundationEmbeddingExtractor:
    """Compose model loading, metadata binding, extraction, and pooling."""

    def __init__(
        self,
        model_key: str,
        *,
        backend: str = "auto",
        device: str = "auto",
        pooling: str = "mean",
        normalize_embeddings: bool = True,
        resample: bool = True,
        cache_embeddings: bool = False,
        store_tokens: bool = False,
        batch_size: int | None = None,
        backend_kwargs: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be a positive integer or None.")
        if store_tokens and cache_embeddings:
            raise ValueError("store_tokens is incompatible with cache_embeddings.")
        self.model_key = model_key
        self.backend = backend
        self.device = device
        self.pooling = pooling
        self.normalize_embeddings = normalize_embeddings
        self.resample = resample
        self.cache_embeddings = cache_embeddings
        self.store_tokens = store_tokens
        self.batch_size = batch_size
        self.backend_kwargs = dict(backend_kwargs or {})
        self.model = model
        # The input a model loaded here was built for; None for a given model.
        self._loaded_for: tuple | None = None
        self._embedding_cache: dict[str, np.ndarray] = {}

    def clear_cache(self) -> None:
        """Drop all memoized window embeddings."""
        self._embedding_cache.clear()

    def _transform_batched(
        self,
        model: Any,
        model_input: np.ndarray,
        *,
        token_metadata: Mapping[str, Any] | None = None,
        views: Sequence[str] = (),
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Forward windows through the backbone, mini-batched along axis 0.

        Transformer backbones (e.g. LaBraM) build an attention map per window, so
        forwarding every window of a long recording at once can exhaust GPU
        memory. When ``batch_size`` is set, windows are forwarded in chunks --
        numerically identical to a single pass, since each window is embedded
        independently -- capping peak memory at one chunk.

        Each chunk is written straight into a pre-allocated destination rather
        than collected and concatenated, so peak host memory is one output array
        plus one chunk instead of two full copies of the output.

        ``views`` are returned with the pooled rows, from each chunk's tokens
        (described by ``token_metadata``): ``"token"`` the tokens as they are,
        ``"channel"`` and ``"patch"`` the tokens averaged (``_pool_tokens``).
        A chunk's tokens are dropped once its views are taken.
        """
        batch_size = self.batch_size
        n_windows = len(model_input)
        batches = (
            [model_input]
            if not batch_size or n_windows <= batch_size
            else [
                model_input[start : start + batch_size]
                for start in range(0, n_windows, batch_size)
            ]
        )

        embeddings: np.ndarray | None = None
        outputs: dict[str, np.ndarray] = {}
        filled = 0
        for batch in batches:
            if views:
                embedding_part, tokens = model.transform(batch, return_tokens=True)
                tokens = np.asarray(tokens)
                token_axes = token_metadata["token_axes"]
                if len(token_axes) != tokens.ndim:
                    raise ValueError(
                        "Backend token metadata must name every native tensor axis: "
                        f"axes={token_axes!r}, shape={tokens.shape}."
                    )
                parts = {
                    view: (
                        tokens
                        if view == "token"
                        else _pool_tokens(tokens, token_metadata, view)
                    )
                    for view in views
                }
            else:
                embedding_part, parts = model.transform(batch), {}
            embedding_part = np.asarray(embedding_part)

            if embeddings is None:
                embeddings = np.empty(
                    (n_windows, *embedding_part.shape[1:]), dtype=embedding_part.dtype
                )
            embeddings[filled : filled + len(embedding_part)] = embedding_part

            for view, part in parts.items():
                if view not in outputs:
                    outputs[view] = np.empty(
                        (n_windows, *part.shape[1:]), dtype=part.dtype
                    )
                outputs[view][filled : filled + len(part)] = part

            filled += len(embedding_part)

        if filled != n_windows:
            embeddings = embeddings[:filled]
            outputs = {view: array[:filled] for view, array in outputs.items()}
        return embeddings, outputs

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        """L2-normalize along the last axis, mapping zero-norm rows to zero.

        A NaN row stays NaN.
        """
        norms = np.linalg.norm(x, axis=-1, keepdims=True)
        return np.divide(x, norms, out=np.zeros_like(x), where=norms != 0)

    def _embed_windows(
        self,
        model: Any,
        model_input: np.ndarray,
        *,
        token_metadata: Mapping[str, Any] | None = None,
        views: Sequence[str] = (),
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Extract backend-pooled rows, normalized if asked, and token views."""
        embeddings, outputs = self._transform_batched(
            model, model_input, token_metadata=token_metadata, views=views
        )
        if self.normalize_embeddings:
            embeddings = self._l2_normalize(embeddings)
        return embeddings, outputs

    def _backbone_fingerprint(self, prepared: Any) -> str:
        """Stable identity of the deterministic window->embedding mapping."""
        return "|".join(
            str(part)
            for part in (
                self.model_key,
                self.backend,
                self.pooling,
                self.normalize_embeddings,
                prepared.target_sfreq,
                tuple(prepared.ch_names),
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
            computed, _ = self._embed_windows(model, model_input[missing])
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
        window_index: Sequence[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> FoundationEmbeddingResult:
        """Extract one embedding per input window.

        ``window_index`` is the number of each window in its recording; by
        default the windows are numbered in the order given.
        """
        X = np.asarray(epochs, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("epochs must have shape (window, channel, time).")
        check_finite(X, self.model_key)
        ch_names = normalize_channel_names(signal_metadata.ch_names)
        # A model is built for one montage, rate and window length (channel
        # positions and adapters, n_times): another recording loads its own.
        loaded_for = (float(signal_metadata.sfreq), tuple(ch_names), X.shape[-1])
        if self._loaded_for not in (None, loaded_for):
            self.model = None
        prepared = prepare_backend(
            self.model_key,
            X,
            backend=self.backend,
            device=self.device,
            train_mode="frozen",
            n_outputs=None,
            sfreq=float(signal_metadata.sfreq),
            ch_names=ch_names,
            pooling=self.pooling,
            backend_kwargs=self.backend_kwargs,
            model=self.model,
        )
        if self.model is None:
            self._loaded_for = loaded_for
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
        if spec.expected_passband is not None:
            band = spec.expected_passband
            warnings.warn(
                f"{spec.display_name or spec.name} expects data filtered between "
                f"{band[0]} and {band[1]} Hz. Ensure your input is bandpassed, "
                "as extraction does not filter automatically.",
                stacklevel=2,
            )
        token_metadata = None
        if self.store_tokens:
            token_metadata = model.get_token_output_metadata()
        elif not self.cache_embeddings and hasattr(model, "get_token_output_metadata"):
            # without documented tokens, window embeddings only
            with contextlib.suppress(NotImplementedError):
                token_metadata = model.get_token_output_metadata()
        views = []
        if token_metadata is not None:
            if not isinstance(token_metadata.get("token_axes"), list):
                raise ValueError(
                    "Backend token metadata must list the token axes, got "
                    f"{token_metadata.get('token_axes')!r}."
                )
            views = (["token"] if self.store_tokens else []) + _pooled_views(
                token_metadata
            )
        outputs = {}
        if self.cache_embeddings:
            embeddings = self._embed_windows_cached(model, model_input, prepared)
        else:
            embeddings, outputs = self._embed_windows(
                model, model_input, token_metadata=token_metadata, views=views
            )
        tokens = outputs.get("token")
        if embeddings.ndim != 2 or len(embeddings) != len(X):
            raise ValueError(
                "Foundation backend must return one 2-D embedding row per window."
            )
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
            "checkpoint_revisions": dict(getattr(model, "checkpoint_revisions", {})),
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
            "normalize_embeddings": self.normalize_embeddings,
            "expects_microvolts": bool(spec.expects_microvolts),
            "input_scale": float(spec.input_scale),
            "input_normalization": spec.input_normalization,
            "input_reference": spec.input_reference,
            "input_clip": spec.input_clip,
            "finetune_input_scale": spec.finetune_input_scale,
            "expected_passband": (
                list(spec.expected_passband)
                if spec.expected_passband is not None
                else None
            ),
            "expected_reference": spec.expected_reference,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype),
            "window_count": len(X),
            "window_n_times_original": int(X.shape[-1]),
            "window_n_times_model": int(model_input.shape[-1]),
            "window_duration_seconds": float(
                X.shape[-1] / float(signal_metadata.sfreq)
            ),
        }
        if tokens is not None:
            payload.update(token_metadata)
            payload.update(
                {
                    "token_layout": "native",
                    "token_layout_version": 1,
                    "token_shape": list(tokens.shape),
                    "token_dtype": str(tokens.dtype),
                }
            )
        return FoundationEmbeddingResult(
            window_embeddings=embeddings,
            window_start=starts,
            window_stop=stops,
            window_index=(
                np.arange(len(X), dtype=int)
                if window_index is None
                else np.asarray(window_index, dtype=int)
            ),
            metadata=payload,
            token_embeddings=tokens,
            channel_embeddings=outputs.get("channel"),
            patch_embeddings=outputs.get("patch"),
        )
