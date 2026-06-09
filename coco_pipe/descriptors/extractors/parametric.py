"""
Parametric spectral descriptor extraction backend.

This module implements the built-in parametric spectral family for
`coco_pipe.descriptors`. The extractor operates on already segmented NumPy
inputs with shape ``(n_obs, n_channels, n_times)`` and computes one or more
specparam-derived summary descriptors per sensor, per observation.

Notes
-----
The parametric family is a PSD consumer. When used through
`DescriptorPipeline.extract()`, it can share one batch-scoped PSD computation
with other compatible PSD consumers such as the spectral band family. The
actual descriptor outputs are then derived from that shared `psds, freqs` pair.

Model fitting itself still happens one spectrum at a time. When runtime
parallelism is enabled and the planner allows it, those per-spectrum fits can
run in parallel across observation-channel units.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._constants import KNOWN_FAMILY_TOKENS
from ..configs import ParametricDescriptorConfig
from ._parametric_fit import _ParametricFitBatch, fit_parametric_batch
from ._psd import compute_psd
from .base import BasePSDDescriptorExtractor, _DescriptorBlock, make_failure_record


class ParametricDescriptorExtractor(BasePSDDescriptorExtractor):
    """
    Parametric spectral descriptor extractor.

    This extractor fits one specparam model per observation and sensor in a
    validated descriptor input array, then exposes scalar summaries such as
    aperiodic parameters, fit quality, and dominant peak statistics.

    Parameters
    ----------
    config : ParametricDescriptorConfig
        Parsed family configuration controlling the PSD method, fit range,
        specparam settings, and requested output groups.

    Attributes
    ----------
    config : ParametricDescriptorConfig
        Stored typed configuration for the parametric family.
    family_name : str
        Stable family identifier used in metadata and failure records.

    Notes
    -----
    The extractor always computes descriptor values per sensor first. Public
    sensor-level naming is applied afterward through
    :meth:`BaseDescriptorExtractor._finalize_descriptor`.

    When the pipeline provides a precomputed PSD batch through
    :meth:`extract_psd`, the extractor reuses that shared spectral input
    and expects an explicit shared `fit_batch`. Standalone :meth:`extract`
    remains available for family-local PSD and fit computation.
    """

    family_name = "parametric"

    def __init__(self, config: ParametricDescriptorConfig):
        super().__init__(config)
        self.config = config

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return static parametric extractor capability metadata.

        Returns
        -------
        dict[str, Any]
            Capability metadata describing sampling-rate requirements and the
            optional backends used by the parametric family.
        """
        return {
            **super().capabilities,
            "requires_sfreq": True,
            "optional_dependencies": ["specparam", "mne"],
        }

    def psd_request(self) -> dict[str, Any]:
        """Describe the PSD requirements for the shared planner."""
        return {
            "method": self.config.psd_method,
            "fmin": self.config.freq_range[0],
            "fmax": self.config.freq_range[1],
        }

    def parametric_fit_requirements(self) -> dict[str, Any]:
        """Describe whether this family needs shared parametric-fit outputs."""
        return {
            "needed": True,
            "metrics": True,
            "periodic_psds": False,
            "config": self.config,
        }

    def extract_psd(
        self,
        psds: np.ndarray,
        freqs: np.ndarray,
        channel_names: list[str] | None,
        ids: np.ndarray | None,
        runtime,
        obs_offset: int = 0,
        fit_batch: _ParametricFitBatch | None = None,
    ) -> _DescriptorBlock:
        """Extract parametric descriptors from a precomputed PSD batch.

        Parameters
        ----------
        psds : np.ndarray
            Power spectral density array with shape
            ``(n_obs, n_channels, n_freqs)``.
        freqs : np.ndarray
            Frequency grid aligned with the last axis of ``psds``.
        channel_names : list of str, optional
            Explicit channel labels aligned with axis 1 of ``psds``. If
            omitted, fallback names ``"ch-0"``, ``"ch-1"``, ... are used
            internally.
        ids : np.ndarray, optional
            Observation identifiers aligned with axis 0 of ``psds``.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across descriptor families.
        obs_offset : int, default=0
            Global observation offset added to any collected failure records
            when this extractor is called on one observation batch.

        Returns
        -------
        _DescriptorBlock
            Parametric-family descriptor block aligned with the input
            observation axis.

        Raises
        ------
        ValueError
            If ``fit_batch`` is not supplied.

        Notes
        -----
        This method consumes explicit shared intermediates. It does not compute
        PSDs or fit models on its own.
        """
        channel_names = channel_names or [f"ch-{idx}" for idx in range(psds.shape[1])]
        if fit_batch is None:
            raise ValueError("Parametric extract_psd() requires a supplied fit_batch.")

        chunk_metric_arrays = fit_batch.metrics
        metrics: list[str] = []
        if "aperiodic" in self.config.outputs:
            metrics.extend(["offset", "exponent"])
            if "knee" in chunk_metric_arrays:
                metrics.append("knee")
        if "fit_quality" in self.config.outputs:
            metrics.extend(["fit_error", "r_squared"])
        if "peak_summary" in self.config.outputs:
            metrics.extend(
                [
                    "peak_count",
                    "peak_freq_dom",
                    "peak_power_dom",
                    "peak_bandwidth_dom",
                    "alpha_peak_freq",
                    "alpha_peak_power",
                ]
            )
        failures: list[dict[str, Any]] = []
        for obs_rel, unit_idx, exception_type, message in fit_batch.errors:
            if runtime.on_error == "raise":
                raise RuntimeError(message)
            failures.append(
                make_failure_record(
                    family=self.family_name,
                    obs_index=obs_offset + obs_rel,
                    obs_id=None if ids is None else ids[obs_rel],
                    channel_index=unit_idx,
                    channel_name=channel_names[unit_idx],
                    exception_type=exception_type,
                    message=message,
                )
            )

        chunk_features: list[np.ndarray] = []
        descriptor_names: list[str] = []
        for metric_name in metrics:
            feature, names = self._finalize_descriptor(
                chunk_metric_arrays[metric_name],
                family_prefix=KNOWN_FAMILY_TOKENS[1],
                metric_name=metric_name,
                channel_names=channel_names,
            )
            chunk_features.append(feature)
            descriptor_names.extend(names)

        return _DescriptorBlock(
            family=self.family_name,
            X=np.concatenate(chunk_features, axis=1)
            if chunk_features
            else np.empty((psds.shape[0], 0)),
            descriptor_names=descriptor_names,
            meta={
                **fit_batch.meta,
                "psd_method": self.config.psd_method,
            },
            failures=failures,
        )

    def extract(
        self,
        X: np.ndarray,
        sfreq: float | None,
        channel_names: list[str] | None,
        ids: np.ndarray | None,
        runtime,
        obs_offset: int = 0,
    ) -> _DescriptorBlock:
        """Extract parametric descriptors from segmented multi-channel data.

        Parameters
        ----------
        X : np.ndarray
            Input array with shape ``(n_obs, n_channels, n_times)``. Each row
            already represents one observation segment produced upstream.
        sfreq : float, optional
            Sampling frequency in Hertz.
        channel_names : list of str, optional
            Explicit channel labels aligned with axis 1 of ``X``.
        ids : np.ndarray, optional
            Observation identifiers aligned with axis 0 of ``X``.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across descriptor families.
        obs_offset : int, default=0
            Global observation offset added to any collected failure records.

        Returns
        -------
        _DescriptorBlock
            Parametric-family descriptor block aligned with the input
            observation axis.

        Raises
        ------
        ImportError
            If the optional `mne` or `specparam` backend is unavailable.
        ValueError
            If PSD computation encounters an invalid runtime condition.
        RuntimeError
            If shared fit materialization encounters a runtime failure and
            ``runtime.on_error == "raise"``.

        Notes
        -----
        This standalone path computes a PSD for the current batch, fits one
        explicit parametric batch payload for this family, and then delegates
        to :meth:`extract_psd`. When the family is executed through
        `DescriptorPipeline`, the shared planner supplies the PSD and fit
        payload instead.
        """
        psds, freqs = compute_psd(
            X,
            sfreq=sfreq,
            method=self.config.psd_method,
            fmin=self.config.freq_range[0],
            fmax=self.config.freq_range[1],
            n_jobs=None,
        )
        fit_batch = fit_parametric_batch(
            psds,
            freqs,
            self.config,
            runtime,
            need_periodic_psd=False,
            include_metrics=True,
        )
        return self.extract_psd(
            psds,
            freqs,
            channel_names=channel_names,
            ids=ids,
            runtime=runtime,
            obs_offset=obs_offset,
            fit_batch=fit_batch,
        )
