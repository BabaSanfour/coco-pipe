"""Faithful channel construction for fixed-montage braindecode foundation models.

Single source of truth shared by the capability gate (``extraction.py``) and the
backend channel adapters (``_braindecode.py``). Each model that was pretrained on
a fixed channel layout is served by an exact, physically-faithful construction
from a standard unipolar 10-20 montage — never a midpoint/bipolar spatial
interpolation (which is physically wrong; see
``docs/foundation_model_channel_adaptation.md``):

- **BIOT**: 16-channel TCP bipolar derivation by exact subtraction ``V(A) - V(B)``.
- **BENDR**: the 19 standard EEG channels reordered into BENDR's layout, dn3
  Deep1010 min-max normalization, plus the computed ``SCALE`` amplitude channel.
- **EEGPT**: name-indexed channel-embedding subset — drop channels outside the
  62-name pretraining vocabulary; the remaining channels keep their pretrained
  per-channel embeddings (looked up by name).

The braindecode-side constants (``BIOT_CHANNEL_ORDER`` / ``_BIOT_TARGET_CHS_INFO``,
``BENDR_CHANNEL_ORDER`` / ``_BENDR_TARGET_CHS_INFO``, ``CHANNEL_DICT``) remain the
ground truth; this module derives its name-level constants from them lazily so a
braindecode upgrade cannot silently desync the two (``tests`` assert the match).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._channels import normalize_channel_names

# BENDR SCALE reference, from the published dn3 pretraining config (TUEG):
# ``max_scale = data_max - data_min``. The SCALE channel is
# ``2 * (clamp(ptp / max_scale, 1.0) - 0.5)`` computed from the raw (microvolt)
# window. Source: SPOClab-ca/BENDR configs/pretraining.yml.
BENDR_DATA_MAX = 3276.7
BENDR_DATA_MIN = -1583.9258304722666
BENDR_SCALE_MAX = BENDR_DATA_MAX - BENDR_DATA_MIN

_SPECIAL_MODELS = frozenset({"biot", "bendr", "eegpt"})


# --------------------------------------------------------------------------- #
# dn3 Deep1010 normalization (BENDR)
# --------------------------------------------------------------------------- #
def min_max_normalize(
    x: np.ndarray, low: float = -1.0, high: float = 1.0
) -> np.ndarray:
    """dn3 ``min_max_normalize`` for a single 2-D ``(channel, time)`` window.

    Uses the global min/max across all channels and time (matching
    ``dn3.utils.min_max_normalize`` on a 2-D tensor), so the whole EEG block is
    scaled jointly into ``[low, high]``. A constant window maps to all zeros.
    """
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax - xmin == 0:
        return np.zeros_like(x)
    scaled = (x - xmin) / (xmax - xmin)
    scaled = scaled - 0.5 + (high + low) / 2.0
    return (high - low) * scaled


# --------------------------------------------------------------------------- #
# Channel-adaptation plan (capability + adapter, one source of truth)
# --------------------------------------------------------------------------- #
@dataclass
class ChannelPlan:
    """Resolved channel adaptation for one model + source montage.

    ``status`` is ``"available"`` (``adapter``/``target_chs_info`` populated) or
    ``"unsupported"`` (``reason`` set, everything else ``None``). ``adapter``
    maps ``X`` of shape ``(n_windows, n_source, n_times)`` to the model's native
    channel layout ``(n_windows, model_n_chans, n_times)``.
    """

    model_key: str
    status: str
    reason: str = ""
    model_n_chans: int | None = None
    model_ch_names: list[str] | None = None
    adapter: Callable[[np.ndarray], np.ndarray] | None = None
    target_chs_info: list[dict] | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def _index_map(source_names: Sequence[str]) -> dict[str, int]:
    """First-occurrence, case-insensitive name->index map over source channels.

    Keyed by upper-cased name so the model's canonical all-caps electrode labels
    (e.g. ``FP1``/``FZ``) resolve against standard mixed-case input (``Fp1``/``Fz``).
    """
    index: dict[str, int] = {}
    for i, name in enumerate(source_names):
        index.setdefault(name.upper(), i)
    return index


def plan_channels(
    model_key: str,
    source_names: Sequence[str],
    *,
    fill_missing: bool = False,
    sfreq: float | None = None,
) -> ChannelPlan:
    """Resolve the faithful channel construction for ``model_key``.

    Parameters
    ----------
    model_key : str
        One of ``"biot"``, ``"bendr"``, ``"eegpt"``.
    source_names : sequence of str
        Input channel names, already normalized (legacy 10-20 renamed).
    fill_missing : bool, default False
        For biot/bendr only: when a required unipolar electrode is absent,
        spatially interpolate it (a well-posed unipolar interpolation) before
        the exact derivation, instead of returning ``unsupported``. Never
        interpolates a bipolar/derived channel.
    sfreq : float, optional
        Sampling rate, required when ``fill_missing`` interpolates.
    """
    names = list(source_names)
    if model_key == "biot":
        return _plan_biot(names, fill_missing=fill_missing, sfreq=sfreq)
    if model_key == "bendr":
        return _plan_bendr(names, fill_missing=fill_missing, sfreq=sfreq)
    if model_key == "eegpt":
        return _plan_eegpt(names)
    raise ValueError(
        f"plan_channels() has no faithful construction for {model_key!r}; "
        f"expected one of {sorted(_SPECIAL_MODELS)}."
    )


def is_special(model_key: str) -> bool:
    """Whether ``model_key`` uses a faithful channel construction here."""
    return model_key in _SPECIAL_MODELS


# --------------------------------------------------------------------------- #
# Per-model plans
# --------------------------------------------------------------------------- #
def _plan_biot(
    names: list[str], *, fill_missing: bool, sfreq: float | None
) -> ChannelPlan:
    from braindecode.models.biot import BIOT_CHANNEL_ORDER

    # The 16 TCP bipolar pairs (prest-16chs layout); modern names already.
    pairs = [tuple(label.split("-")) for label in BIOT_CHANNEL_ORDER[:16]]
    required = list(dict.fromkeys(e for pair in pairs for e in pair))
    present = {n.upper() for n in names}
    missing = [e for e in required if e.upper() not in present]
    filled: list[str] = []
    fill_fn: Callable[[np.ndarray], np.ndarray] | None = None
    if missing:
        if not fill_missing:
            return ChannelPlan(
                "biot",
                "unsupported",
                reason=(
                    "BIOT needs the 16-channel TCP bipolar electrodes; missing "
                    f"{missing}. Enable fill_missing_channels to interpolate them."
                ),
            )
        fill_fn = _make_fill(names, missing, sfreq)
        filled = list(missing)
        names = names + missing

    index = _index_map(names)
    pair_idx = [(index[a.upper()], index[b.upper()]) for a, b in pairs]

    def adapter(X: np.ndarray) -> np.ndarray:
        if fill_fn is not None:
            X = fill_fn(X)
        derived = [X[:, a, :] - X[:, b, :] for a, b in pair_idx]
        return np.stack(derived, axis=1).astype(np.float32)

    from braindecode.models.biot import _BIOT_TARGET_CHS_INFO

    bipolar_names = [f"{a}-{b}" for a, b in pairs]
    provenance = {
        "channel_adaptation": "bipolar_derive",
        "derived_channels": bipolar_names,
        "filled_channels": filled,
        "dropped_channels": [],
    }
    return ChannelPlan(
        "biot",
        "available",
        model_n_chans=16,
        model_ch_names=bipolar_names,
        adapter=adapter,
        target_chs_info=[dict(ch) for ch in _BIOT_TARGET_CHS_INFO[:16]],
        provenance=provenance,
    )


def _plan_bendr(
    names: list[str], *, fill_missing: bool, sfreq: float | None
) -> ChannelPlan:
    from braindecode.models.bendr import BENDR_CHANNEL_ORDER

    # 19 standard EEG channels in BENDR order (drop the trailing SCALE entry);
    # normalize legacy T5/T6 -> P7/P8 to match our montage naming.
    order = normalize_channel_names(BENDR_CHANNEL_ORDER[:-1])
    present = {n.upper() for n in names}
    missing = [e for e in order if e.upper() not in present]
    filled: list[str] = []
    fill_fn: Callable[[np.ndarray], np.ndarray] | None = None
    if missing:
        if not fill_missing:
            return ChannelPlan(
                "bendr",
                "unsupported",
                reason=(
                    "BENDR needs the 19 standard 10-20 EEG channels; missing "
                    f"{missing}. Enable fill_missing_channels to interpolate them."
                ),
            )
        fill_fn = _make_fill(names, missing, sfreq)
        filled = list(missing)
        names = names + missing

    index = _index_map(names)
    order_idx = [index[name.upper()] for name in order]

    def adapter(X: np.ndarray) -> np.ndarray:
        if fill_fn is not None:
            X = fill_fn(X)
        reordered = X[:, order_idx, :]
        n_windows, _, n_times = reordered.shape
        out = np.empty((n_windows, 20, n_times), dtype=np.float32)
        for w in range(n_windows):
            window = reordered[w]
            ptp = float(window.max() - window.min())
            scale = 2.0 * (min(ptp / BENDR_SCALE_MAX, 1.0) - 0.5)
            out[w, :19] = min_max_normalize(window)
            out[w, 19] = scale
        return out

    provenance = {
        "channel_adaptation": "reorder_augment",
        "derived_channels": [],
        "filled_channels": filled,
        "dropped_channels": [],
        "scale_channel": "computed",
        "bendr_order": order,
    }
    return ChannelPlan(
        "bendr",
        "available",
        model_n_chans=20,
        model_ch_names=[*order, "SCALE"],
        adapter=adapter,
        target_chs_info=None,  # plain BENDR loads with n_chans=20, no chs_info
        provenance=provenance,
    )


def _plan_eegpt(names: list[str]) -> ChannelPlan:
    from braindecode.models.eegpt import CHANNEL_DICT

    # EEGPT's 62-name pretraining vocabulary (channel embeddings indexed by name).
    vocab = {name.upper() for name in CHANNEL_DICT}
    keep = [(i, name) for i, name in enumerate(names) if name.upper() in vocab]
    dropped = [name for name in names if name.upper() not in vocab]
    if not keep:
        return ChannelPlan(
            "eegpt",
            "unsupported",
            reason=(
                "None of the input channels are in EEGPT's 62-name pretraining "
                f"vocabulary; cannot index any channel embedding. Got {names}."
            ),
        )
    keep_idx = [i for i, _ in keep]
    kept_names = [name for _, name in keep]

    def adapter(X: np.ndarray) -> np.ndarray:
        return X[:, keep_idx, :].astype(np.float32)

    provenance = {
        "channel_adaptation": "name_select",
        "derived_channels": [],
        "filled_channels": [],
        "dropped_channels": dropped,
    }
    return ChannelPlan(
        "eegpt",
        "available",
        model_n_chans=len(kept_names),
        model_ch_names=kept_names,
        adapter=adapter if dropped else None,
        target_chs_info=None,  # backend builds chs_info from kept_names
        provenance=provenance,
    )


# --------------------------------------------------------------------------- #
# Opt-in unipolar interpolation for missing electrodes (biot/bendr fallback)
# --------------------------------------------------------------------------- #
def _make_fill(
    present_names: list[str],
    missing: list[str],
    sfreq: float | None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build an adapter that appends spline-interpolated missing electrodes.

    The missing channels are zero-inserted, marked bad, and reconstructed by
    MNE spherical-spline interpolation from the present 10-20 electrodes — a
    well-posed unipolar interpolation. Returns ``X`` augmented with the missing
    channels appended in ``missing`` order.
    """
    augmented_names = present_names + missing

    def fill(X: np.ndarray) -> np.ndarray:
        import mne

        if sfreq is None:
            raise ValueError("fill_missing_channels requires the input sampling rate.")
        n_windows, _, n_times = X.shape
        padded = np.concatenate(
            [X, np.zeros((n_windows, len(missing), n_times), dtype=X.dtype)],
            axis=1,
        )
        info = mne.create_info(augmented_names, sfreq=float(sfreq), ch_types="eeg")
        # standard_1020 uses mixed-case labels (Fp1/Fz/Cz/...); match
        # case-insensitively so positions resolve for the spline interpolation.
        info.set_montage("standard_1020", match_case=False, on_missing="warn")
        epochs = mne.EpochsArray(padded, info, verbose="error")
        epochs.info["bads"] = list(missing)
        epochs.interpolate_bads(reset_bads=True, verbose="error")
        return epochs.get_data(copy=False).astype(np.float32)

    return fill
