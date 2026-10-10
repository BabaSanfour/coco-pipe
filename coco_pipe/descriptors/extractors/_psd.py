"""
Shared PSD computation for PSD-consuming descriptor paths.

This module holds the PSD step the descriptors planner shares between
PSD-consuming extractors. It does not define descriptor semantics. It only:

- prepare a writable runtime environment for MNE-backed PSD helpers
- lazily import the MNE PSD functions used by descriptors
- compute Welch or multitaper PSD batches on explicit NumPy inputs

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from ...utils import import_optional_dependency


def load_mne_psd_functions():
    """Lazily import MNE PSD helpers with writable runtime cache locations.

    Returns
    -------
    tuple
        ``(psd_array_welch, psd_array_multitaper)`` imported from
        `mne.time_frequency`.

    Notes
    -----
    MNE may write cache or config files during import/use. The descriptors
    module keeps those paths inside the system temp directory so PSD
    computation remains sandbox-friendly.
    """
    tmp_root = os.path.join(tempfile.gettempdir(), "coco_pipe_descriptors")
    mpl_dir = os.path.join(tmp_root, "mplconfig")
    mne_dir = os.path.join(tmp_root, "mne")
    os.makedirs(mpl_dir, exist_ok=True)
    os.makedirs(mne_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
    os.environ.setdefault("MNE_HOME", mne_dir)
    os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

    return import_optional_dependency(
        lambda: (
            __import__(
                "mne.time_frequency",
                fromlist=["psd_array_welch", "psd_array_multitaper"],
            ).psd_array_welch,
            __import__(
                "mne.time_frequency",
                fromlist=["psd_array_welch", "psd_array_multitaper"],
            ).psd_array_multitaper,
        ),
        feature="descriptor spectral extraction",
        dependency="mne",
        install_hint="pip install coco-pipe[descriptors,eeg]",
    )


def compute_psd(
    X: np.ndarray,
    sfreq: float,
    method: str,
    fmin: float,
    fmax: float,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD values for one batch of segmented signals.

    Parameters
    ----------
    X : np.ndarray
        Input array with shape ``(n_obs, n_channels, n_times)``.
    sfreq : float
        Sampling frequency in Hertz.
    method : {"welch", "multitaper"}
        PSD estimator to use.
    fmin : float
        Lower frequency bound passed to the PSD backend.
    fmax : float
        Upper frequency bound passed to the PSD backend.
    n_jobs : int, optional
        Parallel worker count forwarded to the MNE PSD backend when the caller
        enables PSD-level parallelism. `None` leaves the backend default in
        place.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        PSD values with shape ``(n_obs, n_channels, n_freqs)`` and the aligned
        frequency grid with shape ``(n_freqs,)``.

    Notes
    -----
    For Welch PSDs, the descriptors module uses:

    - ``n_fft = min(n_times, 256)``
    - ``n_per_seg = n_fft``

    while enforcing a minimum of `8` for both values. This keeps Welch
    behavior bounded and deterministic across the current descriptor tests and
    examples.
    """
    psd_array_welch, psd_array_multitaper = load_mne_psd_functions()

    if method == "welch":
        n_fft = min(int(X.shape[-1]), 256)
        psd, freqs = psd_array_welch(
            X,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=max(n_fft, 8),
            n_per_seg=max(n_fft, 8),
            average="mean",
            n_jobs=n_jobs,
            verbose=False,
        )
        return np.asarray(psd, dtype=float), np.asarray(freqs, dtype=float)

    if method == "multitaper":
        psd, freqs = psd_array_multitaper(
            X,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_jobs=n_jobs,
            verbose=False,
        )
        return np.asarray(psd, dtype=float), np.asarray(freqs, dtype=float)

    raise ValueError(f"Unknown PSD method: {method}")
