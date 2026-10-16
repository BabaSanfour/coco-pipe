"""
Band summary descriptor extraction backend.

This module implements the built-in spectral band family for
`coco_pipe.descriptors`. The extractor operates on already segmented NumPy
inputs with shape ``(n_obs, n_channels, n_times)`` and computes PSD-derived
band summaries per sensor, per observation.

Notes
-----
The spectral family is a PSD consumer. When used through
`DescriptorPipeline.extract()`, it can share one batch-scoped PSD computation
with other compatible PSD consumers such as the parametric family. The actual
descriptor outputs are then derived from that shared `psds, freqs` pair.

Within one extracted PSD batch, the family computes band integrals once and
reuses them for all requested outputs:

- absolute power
- log absolute power
- relative power
- band ratios
- corrected absolute power
- corrected log absolute power
- corrected relative power

Corrected outputs are derived from the periodic-only PSDs of the parametric
fit batch that `DescriptorPipeline` shares between families. A band counts as
periodic only where the fit placed a peak centre inside it; elsewhere every
corrected output of that band is ``NaN`` (not measured), not zero.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .._constants import KNOWN_FAMILY_TOKENS
from ..configs import BandDescriptorConfig
from ._parametric_fit import _ParametricFitBatch
from .base import BasePSDDescriptorExtractor, _DescriptorBlock, make_failure_record


class BandDescriptorExtractor(BasePSDDescriptorExtractor):
    """
    Spectral band descriptor extractor.

    This extractor computes PSD-derived band summaries for each observation and
    sensor in a validated descriptor input array. It is intended for signals
    that are already segmented upstream, such as epochs, windows, or trial
    blocks.

    Parameters
    ----------
    config : BandDescriptorConfig
        Parsed family configuration controlling the PSD method, frequency
        window, band definitions, and requested spectral outputs.

    Attributes
    ----------
    config : BandDescriptorConfig
        Stored typed configuration for the spectral band family.
    family_name : str
        Stable family identifier used in metadata and failure records.

    Notes
    -----
    The extractor always computes descriptor values per sensor first. Public
    sensor-level naming is applied afterward through
    :meth:`BaseDescriptorExtractor._finalize_descriptor`.

    `DescriptorPipeline` passes the shared PSD batch to :meth:`extract_psd`,
    and the shared parametric fit batch when corrected outputs are requested.
    """

    family_name = "bands"

    def __init__(self, config: BandDescriptorConfig, fit_config):
        super().__init__(config)
        self.fit_config = fit_config

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return static spectral extractor capability metadata.

        Returns
        -------
        dict[str, Any]
            Capability metadata describing sampling-rate requirements and the
            optional backend used by the spectral family.

        Notes
        -----
        Spectral band extraction always requires an explicit sampling rate
        because the PSD frequency axis depends on it.
        """
        return {
            **super().capabilities,
            "requires_sfreq": True,
            "optional_dependencies": ["mne"],
        }

    def psd_request(self) -> dict[str, Any]:
        """Describe the PSD requirements for the shared planner.

        Returns
        -------
        dict[str, Any]
            Minimal PSD request containing the PSD method, the Welch segment
            settings and the required frequency range for this family.

        Notes
        -----
        `DescriptorPipeline` uses this request to group compatible PSD
        consumers and decide when one batch-scoped PSD can be reused across
        families.
        """
        if self.needs_parametric_fit():
            fit_low, fit_high = self.fit_config.freq_range
            return {
                "method": self.config.psd_method,
                "welch_segment_length": self.config.welch_segment_length,
                "welch_overlap": self.config.welch_overlap,
                "fmin": min(self.config.fmin, fit_low),
                "fmax": max(self.config.fmax, fit_high),
            }
        return {
            "method": self.config.psd_method,
            "welch_segment_length": self.config.welch_segment_length,
            "welch_overlap": self.config.welch_overlap,
            "fmin": self.config.fmin,
            "fmax": self.config.fmax,
        }

    def needs_parametric_fit(self) -> bool:
        """Whether corrected spectral outputs require a shared parametric fit."""
        return any(
            output
            in {
                "corrected_absolute_power",
                "corrected_log_absolute_power",
                "corrected_relative_power",
            }
            for output in self.config.outputs
        )

    def parametric_fit_requirements(self) -> dict[str, Any]:
        """Describe whether this family needs shared parametric-fit outputs."""
        return {
            "needed": self.needs_parametric_fit(),
            "metrics": False,
            "periodic_psds": self.needs_parametric_fit(),
            "config": self.fit_config,
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
        """Extract band descriptors from a precomputed PSD batch.

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
            Spectral-family descriptor block aligned with the input
            observation axis.

        Raises
        ------
        ValueError
            If a configured band has no overlap with the computed PSD range and
            runtime error handling is configured to raise.

        Notes
        -----
        The extractor first restricts the incoming PSD to the configured
        frequency window ``[fmin, fmax)``, then integrates one power value per
        configured band and sensor: the sum of the PSD bins in ``[low, high)``
        times the frequency step. The total power of relative outputs is the
        same sum over the window, so contiguous bands that cover it have
        relative powers summing to 1. Those band integrals are reused for all
        enabled outputs, such as absolute power, log absolute power, relative
        power, and ratios. Ratios are always derived from absolute band
        powers, not from relative or log-transformed outputs.

        Example
        -------
        With ``channel_names=["Fz", "Cz"]``, an absolute alpha-band request
        yields names such as ``band_abs_alpha_ch-Fz`` and
        ``band_abs_alpha_ch-Cz``.
        """
        channel_names = channel_names or [f"ch-{idx}" for idx in range(psds.shape[1])]
        eps = np.finfo(float).eps

        # a bin belongs to [low, high) and stands for density x frequency step,
        # so contiguous bands add up to the power of [fmin, fmax)
        freq_step = freqs[1] - freqs[0] if freqs.size > 1 else np.nan
        freq_mask = (freqs >= self.config.fmin) & (freqs < self.config.fmax)
        local_freqs = freqs[freq_mask]
        local_psds = psds[..., freq_mask]

        total_power = None
        if "relative_power" in self.config.outputs:
            if local_freqs.size == 0:
                total_power = np.full(psds.shape[:-1], np.nan, dtype=float)
            else:
                total_power = local_psds.sum(axis=-1) * freq_step

        failures: list[dict[str, Any]] = []
        descriptor_names: list[str] = []
        chunk_features: list[np.ndarray] = []

        def integrate_band_power(
            spectra: np.ndarray,
            band_freqs: np.ndarray,
            band_label_prefix: str,
        ) -> dict[str, np.ndarray]:
            computed_band_power: dict[str, np.ndarray] = {}
            range_label = (
                "computed PSD range"
                if band_label_prefix == "Raw"
                else "parametric fit range"
            )
            for band_name, (low, high) in self.config.bands.items():
                mask = (band_freqs >= low) & (band_freqs < high)
                if not np.any(mask):
                    message = (
                        f"{band_label_prefix} band '{band_name}' does not overlap "
                        f"the {range_label}."
                    )
                    if runtime.on_error == "raise":
                        raise ValueError(message)
                    computed_band_power[band_name] = np.full(
                        spectra.shape[:-1],
                        np.nan,
                        dtype=float,
                    )
                    for obs_rel, ch_idx in np.argwhere(
                        ~np.isfinite(computed_band_power[band_name])
                    ):
                        failures.append(
                            make_failure_record(
                                family=self.family_name,
                                obs_index=obs_offset + int(obs_rel),
                                obs_id=None if ids is None else ids[obs_rel],
                                channel_index=int(ch_idx),
                                channel_name=channel_names[ch_idx],
                                exception_type="BandResolutionError",
                                message=message,
                            )
                        )
                    continue
                computed_band_power[band_name] = (
                    spectra[..., mask].sum(axis=-1) * freq_step
                )
            return computed_band_power

        def append_band_outputs(
            band_power_dict: dict[str, np.ndarray],
            total_power_array: np.ndarray | None,
            output_prefix: str | None,
            enabled_absolute_output: str,
            enabled_log_output: str,
            enabled_relative_output: str,
            relative_message_prefix: str,
        ) -> None:
            metric_prefix = [] if output_prefix is None else [output_prefix]
            denominator_floor = self.config.min_denominator_power

            if enabled_absolute_output in self.config.outputs:
                for band_name, values in band_power_dict.items():
                    feature, names = self._finalize_descriptor(
                        values,
                        family_prefix=KNOWN_FAMILY_TOKENS[0],
                        metric_name="_".join([*metric_prefix, "abs", band_name]),
                        channel_names=channel_names,
                    )
                    chunk_features.append(feature)
                    descriptor_names.extend(names)

            if enabled_log_output in self.config.outputs:
                for band_name, values in band_power_dict.items():
                    log_values = np.log10(np.clip(values, eps, None))
                    feature, names = self._finalize_descriptor(
                        log_values,
                        family_prefix=KNOWN_FAMILY_TOKENS[0],
                        metric_name="_".join([*metric_prefix, "log", "abs", band_name]),
                        channel_names=channel_names,
                    )
                    chunk_features.append(feature)
                    descriptor_names.extend(names)

            if enabled_relative_output in self.config.outputs:
                for band_name, values in band_power_dict.items():
                    relative = np.divide(
                        values,
                        total_power_array,
                        out=np.full_like(values, np.nan, dtype=float),
                        where=total_power_array > denominator_floor,
                    )
                    undefined = (
                        np.isfinite(values)
                        & np.isfinite(total_power_array)
                        & ~np.isfinite(relative)
                    )
                    for obs_rel, ch_idx in np.argwhere(undefined):
                        failures.append(
                            make_failure_record(
                                family=self.family_name,
                                obs_index=obs_offset + int(obs_rel),
                                obs_id=None if ids is None else ids[obs_rel],
                                channel_index=int(ch_idx),
                                channel_name=channel_names[ch_idx],
                                exception_type="NumericalIssue",
                                message=(
                                    f"{relative_message_prefix} for band "
                                    f"'{band_name}' became non-finite."
                                ),
                            )
                        )
                    feature, names = self._finalize_descriptor(
                        relative,
                        family_prefix=KNOWN_FAMILY_TOKENS[0],
                        metric_name="_".join([*metric_prefix, "rel", band_name]),
                        channel_names=channel_names,
                    )
                    chunk_features.append(feature)
                    descriptor_names.extend(names)

            # a ratio of periodic powers is not offered: it mixes two peaks
            if output_prefix is None and "ratios" in self.config.outputs:
                for numerator, denominator in self.config.ratio_pairs:
                    ratio = np.divide(
                        band_power_dict[numerator],
                        band_power_dict[denominator],
                        out=np.full_like(
                            band_power_dict[numerator],
                            np.nan,
                            dtype=float,
                        ),
                        where=band_power_dict[denominator] > denominator_floor,
                    )
                    undefined = (
                        np.isfinite(band_power_dict[numerator])
                        & np.isfinite(band_power_dict[denominator])
                        & ~np.isfinite(ratio)
                    )
                    for obs_rel, ch_idx in np.argwhere(undefined):
                        failures.append(
                            make_failure_record(
                                family=self.family_name,
                                obs_index=obs_offset + int(obs_rel),
                                obs_id=None if ids is None else ids[obs_rel],
                                channel_index=int(ch_idx),
                                channel_name=channel_names[ch_idx],
                                exception_type="NumericalIssue",
                                message=(
                                    "Band ratio "
                                    f"'{numerator}/{denominator}' "
                                    "became non-finite."
                                ),
                            )
                        )
                    feature, names = self._finalize_descriptor(
                        ratio,
                        family_prefix=KNOWN_FAMILY_TOKENS[0],
                        metric_name="_".join(
                            [*metric_prefix, "ratio", numerator, denominator]
                        ),
                        channel_names=channel_names,
                    )
                    chunk_features.append(feature)
                    descriptor_names.extend(names)

        band_power = integrate_band_power(local_psds, local_freqs, "Raw")

        corrected_band_power: dict[str, np.ndarray] = {}
        corrected_total_power = None
        corrected_outputs_requested = self.needs_parametric_fit()
        if corrected_outputs_requested:
            for obs_rel, ch_idx, exception_type, message in fit_batch.errors:
                failures.append(
                    make_failure_record(
                        family=self.family_name,
                        obs_index=obs_offset + obs_rel,
                        obs_id=None if ids is None else ids[obs_rel],
                        channel_index=ch_idx,
                        channel_name=channel_names[ch_idx],
                        exception_type=exception_type,
                        message=f"Corrected band estimation unavailable: {message}",
                    )
                )

            corrected_freq_mask = (fit_batch.freqs >= self.config.fmin) & (
                fit_batch.freqs < self.config.fmax
            )
            corrected_freqs = fit_batch.freqs[corrected_freq_mask]
            corrected_psds = fit_batch.periodic_psds[..., corrected_freq_mask]

            if "corrected_relative_power" in self.config.outputs:
                if corrected_freqs.size == 0:
                    corrected_total_power = np.full(
                        psds.shape[:-1],
                        np.nan,
                        dtype=float,
                    )
                else:
                    corrected_total_power = corrected_psds.sum(axis=-1) * freq_step

            corrected_band_power = integrate_band_power(
                corrected_psds,
                corrected_freqs,
                "Corrected",
            )
            # without a fitted peak in a band, its periodic power is not measured
            for band_name, (low, high) in self.config.bands.items():
                has_peak = np.array(
                    [
                        [np.any((peaks >= low) & (peaks < high)) for peaks in row]
                        for row in fit_batch.peak_freqs
                    ],
                    dtype=bool,
                ).reshape(psds.shape[:-1])
                corrected_band_power[band_name] = np.where(
                    has_peak, corrected_band_power[band_name], np.nan
                )

        append_band_outputs(
            band_power,
            total_power,
            None,
            "absolute_power",
            "log_absolute_power",
            "relative_power",
            "Relative power",
        )
        append_band_outputs(
            corrected_band_power,
            corrected_total_power,
            "corr",
            "corrected_absolute_power",
            "corrected_log_absolute_power",
            "corrected_relative_power",
            "Corrected relative power",
        )

        if chunk_features:
            X_out = np.concatenate(chunk_features, axis=1)
        else:
            X_out = np.empty((psds.shape[0], 0), dtype=float)

        return _DescriptorBlock(
            family=self.family_name,
            X=X_out,
            descriptor_names=descriptor_names,
            meta={
                "psd_method": self.config.psd_method,
                "bands": self.config.bands,
                "freq_range": [self.config.fmin, self.config.fmax],
                "n_freqs": int(local_freqs.size),
                "corrected_outputs": [
                    output
                    for output in self.config.outputs
                    if output.startswith("corrected_")
                ],
            },
            failures=failures,
        )
