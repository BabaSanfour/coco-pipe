"""
Complexity descriptor extraction backend.

This module implements the built-in complexity family for
`coco_pipe.descriptors`. The extractor operates on already segmented NumPy
inputs with shape ``(n_obs, n_channels, n_times)`` and computes one or more
complexity measures per sensor, per observation.

Notes
-----
The complexity family prefers batched backend calls when the selected library
supports them. In the current implementation:

- `spectral_entropy`, `hjorth_mobility`, and `hjorth_complexity` use batched
  `antropy` calls over flattened observation-channel units
- `sample_entropy`, `perm_entropy`, `approx_entropy`, `svd_entropy`,
  `petrosian_fd`, `katz_fd`, `higuchi_fd`, and `lziv_complexity` are still
  evaluated one 1D signal at a time
- `shannon_entropy`, `fuzzy_entropy`, `dispersion_entropy`, and
  `hurst_exponent` use scalar `neurokit2` calls
- `zero_crossings`, `kurtosis`, and `rms` are computed as simple scalar
  channelwise signal descriptors

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis

from ...utils import import_optional_dependency
from .._constants import KNOWN_FAMILY_TOKENS
from ..configs import ComplexityDescriptorConfig
from .base import BaseDescriptorExtractor, _DescriptorBlock, make_failure_record

_ANTROPY_BATCHED_MEASURES = frozenset(
    {"spectral_entropy", "hjorth_mobility", "hjorth_complexity"}
)
_ANTROPY_SCALAR_MEASURES = frozenset(
    {
        "sample_entropy",
        "perm_entropy",
        "approx_entropy",
        "svd_entropy",
        "petrosian_fd",
        "katz_fd",
        "higuchi_fd",
        "lziv_complexity",
    }
)
_NEUROKIT_SCALAR_MEASURES = frozenset(
    {
        "sample_entropy",
        "perm_entropy",
        "spectral_entropy",
        "shannon_entropy",
        "fuzzy_entropy",
        "dispersion_entropy",
        "hurst_exponent",
    }
)
_CUSTOM_SCALAR_MEASURES = frozenset({"zero_crossings", "kurtosis", "rms"})


def _normalize_scalar_output(value: Any) -> float:
    """Normalize backend scalar outputs to one plain float."""
    if isinstance(value, tuple):
        value = value[0]
    array = np.asarray(value, dtype=float)
    if array.size != 1:
        raise ValueError("Complexity backend returned a non-scalar result.")
    return float(array.reshape(-1)[0])


class ComplexityDescriptorExtractor(BaseDescriptorExtractor):
    """
    Complexity descriptor extractor.

    This extractor computes scalar complexity measures for each observation and
    sensor in a validated descriptor input array. It is intended for signals
    that are already segmented upstream, such as epochs, windows, or trial
    blocks.

    Parameters
    ----------
    config : ComplexityDescriptorConfig
        Parsed family configuration controlling the selected measures, backend,
        and any per-measure keyword arguments.

    Attributes
    ----------
    config : ComplexityDescriptorConfig
        Stored typed configuration for the complexity family.
    family_name : str
        Stable family identifier used in metadata and failure records.

    Notes
    -----
    The extractor always computes descriptor values per sensor first. Public
    deterministic sensor-level naming is applied afterward through
    :meth:`BaseDescriptorExtractor._finalize_descriptor`.

    When `backend="auto"` is selected, the extractor resolves each measure to
    the preferred available implementation:

    - `antropy` for the existing antropy-backed measures
    - `neurokit2` for measures that are only supported there
    - built-in NumPy/SciPy implementations for simple scalar signal summaries
    """

    family_name = "complexity"

    def __init__(self, config: ComplexityDescriptorConfig):
        super().__init__(config)
        self.config = config

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return static complexity extractor capability metadata.

        Returns
        -------
        dict[str, Any]
            Capability metadata describing sampling-rate requirements and the
            optional backends used by the complexity family.

        Notes
        -----
        `spectral_entropy` requires an explicit sampling rate, while the other
        currently supported measures do not.
        """
        return {
            **super().capabilities,
            "requires_sfreq": "spectral_entropy" in self.config.measures,
            "optional_dependencies": ["antropy", "neurokit2"],
        }

    def _load_antropy(self):
        """Import `antropy` lazily when the configured backend needs it.

        Returns
        -------
        module
            Imported `antropy` module.

        Raises
        ------
        ImportError
            If `antropy` is not installed.
        """
        return import_optional_dependency(
            lambda: __import__("antropy"),
            feature="complexity descriptor extraction",
            dependency="antropy",
            install_hint="pip install coco-pipe[descriptors]",
        )

    def _load_neurokit(self):
        """Import `neurokit2` lazily when the configured backend needs it.

        Returns
        -------
        module
            Imported `neurokit2` module.

        Raises
        ------
        ImportError
            If `neurokit2` is not installed.
        """
        return import_optional_dependency(
            lambda: __import__("neurokit2"),
            feature="neurokit complexity descriptor extraction",
            dependency="neurokit2",
            install_hint="pip install coco-pipe[descriptors]",
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
        """Extract complexity descriptors from segmented multi-channel data.

        Parameters
        ----------
        X : np.ndarray
            Input array with shape ``(n_obs, n_channels, n_times)``. Each row
            already represents one observation segment produced upstream.
        sfreq : float, optional
            Sampling frequency in Hertz. Required when
            `spectral_entropy` is requested.
        channel_names : list of str, optional
            Explicit channel labels aligned with axis 1 of ``X``. If omitted,
            fallback names ``"ch-0"``, ``"ch-1"``, ... are used internally.
        ids : np.ndarray, optional
            Observation identifiers aligned with axis 0 of ``X``.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across descriptor families.
        obs_offset : int, default=0
            Global observation offset added to any collected failure records
            when this extractor is called on one observation batch.

        Returns
        -------
        _DescriptorBlock
            Complexity-family descriptor block aligned with the input
            observation axis.

        Raises
        ------
        ImportError
            If the configured optional backend is unavailable.
        ValueError
            If a requested measure is unsupported by the selected backend, or
            if runtime error handling is configured to raise on a numerical or
            backend failure.

        Notes
        -----
        The extractor uses a mixed execution strategy:

        - batched `antropy` calls for `spectral_entropy`,
          `hjorth_mobility`, and `hjorth_complexity`
        - scalar `antropy` calls for the remaining antropy-backed measures
        - scalar `neurokit2` calls for `shannon_entropy`, `fuzzy_entropy`,
          `dispersion_entropy`, and `hurst_exponent`

        Non-finite outputs are converted to `NaN` and recorded under
        ``failures`` unless `runtime.on_error == "raise"`, in which case the
        extractor fails immediately.

        Example
        -------
        With ``channel_names=["Fz", "Cz"]``, a requested measure such as
        ``perm_entropy`` yields channel-resolved names like
        ``complexity_perm_entropy_ch-Fz`` and
        ``complexity_perm_entropy_ch-Cz``.
        """
        channel_names = channel_names or [f"ch-{idx}" for idx in range(X.shape[1])]

        descriptor_names: list[str] | None = None
        failures: list[dict[str, Any]] = []
        metric_arrays = {
            measure: np.full((X.shape[0], X.shape[1]), np.nan, dtype=float)
            for measure in self.config.measures
        }
        measure_kwargs = {
            measure: dict(self.config.measure_kwargs.get(measure, {}))
            for measure in self.config.measures
        }
        flat_signals = X.reshape(-1, X.shape[-1])
        batched_outputs: dict[str, np.ndarray] = {}
        scalar_dispatch: dict[str, Any] = {}
        measure_backends: dict[str, str] = {}
        custom_scalar_dispatch = {
            "zero_crossings": lambda signal, kwargs, sfreq: float(
                np.count_nonzero(np.diff(np.signbit(np.asarray(signal, dtype=float))))
            ),
            "kurtosis": lambda signal, kwargs, sfreq: float(
                scipy_kurtosis(
                    np.asarray(signal, dtype=float),
                    fisher=kwargs.get("fisher", True),
                    bias=kwargs.get("bias", False),
                )
            ),
            "rms": lambda signal, kwargs, sfreq: float(
                np.sqrt(np.mean(np.square(np.asarray(signal, dtype=float))))
            ),
        }

        unsupported: list[str] = []
        for measure in self.config.measures:
            if measure in _CUSTOM_SCALAR_MEASURES:
                measure_backends[measure] = "custom"
                continue
            if self.config.backend == "antropy":
                if (
                    measure in _ANTROPY_BATCHED_MEASURES
                    or measure in _ANTROPY_SCALAR_MEASURES
                ):
                    measure_backends[measure] = "antropy"
                else:
                    unsupported.append(measure)
                continue
            if self.config.backend == "neurokit2":
                if measure in _NEUROKIT_SCALAR_MEASURES:
                    measure_backends[measure] = "neurokit2"
                else:
                    unsupported.append(measure)
                continue
            if (
                measure in _ANTROPY_BATCHED_MEASURES
                or measure in _ANTROPY_SCALAR_MEASURES
            ):
                measure_backends[measure] = "antropy"
            elif measure in _NEUROKIT_SCALAR_MEASURES:
                measure_backends[measure] = "neurokit2"
            else:
                unsupported.append(measure)

        if unsupported:
            raise ValueError(
                f"Measures {sorted(unsupported)} are not supported by backend "
                f"'{self.config.backend}'."
            )

        ant = None
        if "antropy" in measure_backends.values():
            ant = self._load_antropy()

            if measure_backends.get("spectral_entropy") == "antropy":
                batched_outputs["spectral_entropy"] = np.asarray(
                    ant.spectral_entropy(
                        flat_signals,
                        sf=sfreq,
                        axis=-1,
                        **measure_kwargs["spectral_entropy"],
                    ),
                    dtype=float,
                )

            if (
                measure_backends.get("hjorth_mobility") == "antropy"
                or measure_backends.get("hjorth_complexity") == "antropy"
            ):
                mobility, complexity = ant.hjorth_params(
                    flat_signals,
                    axis=-1,
                )
                if measure_backends.get("hjorth_mobility") == "antropy":
                    batched_outputs["hjorth_mobility"] = np.asarray(
                        mobility,
                        dtype=float,
                    )
                if measure_backends.get("hjorth_complexity") == "antropy":
                    batched_outputs["hjorth_complexity"] = np.asarray(
                        complexity,
                        dtype=float,
                    )

            antropy_scalar_dispatch = {
                "sample_entropy": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(ant.sample_entropy(signal, **kwargs))
                ),
                "perm_entropy": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    ant.perm_entropy(signal, **kwargs)
                ),
                "approx_entropy": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(ant.app_entropy(signal, **kwargs))
                ),
                "svd_entropy": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    ant.svd_entropy(signal, **kwargs)
                ),
                "petrosian_fd": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    ant.petrosian_fd(signal, **kwargs)
                ),
                "katz_fd": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    ant.katz_fd(signal, **kwargs)
                ),
                "higuchi_fd": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    ant.higuchi_fd(signal, **kwargs)
                ),
                "lziv_complexity": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(
                        ant.lziv_complexity(
                            (signal > np.median(signal)).astype(int),
                            **kwargs,
                        )
                    )
                ),
            }
            for measure, func in antropy_scalar_dispatch.items():
                if measure_backends.get(measure) == "antropy":
                    scalar_dispatch[measure] = func

        nk = None
        if "neurokit2" in measure_backends.values():
            nk = self._load_neurokit()
            neurokit_scalar_dispatch = {
                "sample_entropy": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(nk.entropy_sample(signal, **kwargs))
                ),
                "perm_entropy": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    nk.entropy_permutation(signal, **kwargs)
                ),
                "spectral_entropy": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(
                        nk.entropy_spectral(
                            signal,
                            sampling_rate=sfreq,
                            **kwargs,
                        )
                    )
                ),
                "shannon_entropy": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(nk.entropy_shannon(signal, **kwargs))
                ),
                "fuzzy_entropy": lambda signal, kwargs, sfreq: _normalize_scalar_output(
                    nk.entropy_fuzzy(signal, **kwargs)
                ),
                "dispersion_entropy": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(nk.entropy_dispersion(signal, **kwargs))
                ),
                "hurst_exponent": lambda signal, kwargs, sfreq: (
                    _normalize_scalar_output(nk.fractal_hurst(signal, **kwargs))
                ),
            }
            for measure, func in neurokit_scalar_dispatch.items():
                if measure_backends.get(measure) == "neurokit2":
                    scalar_dispatch[measure] = func

        for measure, func in custom_scalar_dispatch.items():
            if measure_backends.get(measure) == "custom":
                scalar_dispatch[measure] = func

        for measure, flat_values in batched_outputs.items():
            values = np.asarray(flat_values, dtype=float).reshape(
                X.shape[0],
                X.shape[1],
            )
            metric_arrays[measure][:] = np.where(np.isfinite(values), values, np.nan)
            bad_positions = np.argwhere(~np.isfinite(values))
            if bad_positions.size == 0:
                continue

            message = f"Complexity measure '{measure}' produced a non-finite result."
            if runtime.on_error == "raise":
                raise ValueError(message)

            for obs_rel, unit_idx in bad_positions:
                failures.append(
                    make_failure_record(
                        family=self.family_name,
                        obs_index=obs_offset + int(obs_rel),
                        obs_id=None if ids is None else ids[int(obs_rel)],
                        channel_index=int(unit_idx),
                        channel_name=channel_names[int(unit_idx)],
                        exception_type="NumericalIssue",
                        message=message,
                    )
                )

        scalar_measures = [
            measure
            for measure in self.config.measures
            if measure not in batched_outputs
        ]

        for obs_rel in range(X.shape[0]):
            unit_signals = X[obs_rel]
            obs_id = None if ids is None else ids[obs_rel]

            for unit_idx, signal in enumerate(unit_signals):
                for measure in scalar_measures:
                    try:
                        value = scalar_dispatch[measure](
                            signal,
                            measure_kwargs[measure],
                            sfreq,
                        )
                        if np.isfinite(value):
                            metric_arrays[measure][obs_rel, unit_idx] = float(value)
                        else:
                            if runtime.on_error == "raise":
                                raise ValueError(
                                    "Complexity measure produced a non-finite result."
                                )
                            failures.append(
                                make_failure_record(
                                    family=self.family_name,
                                    obs_index=obs_offset + obs_rel,
                                    obs_id=obs_id,
                                    channel_index=unit_idx,
                                    channel_name=channel_names[unit_idx],
                                    exception_type="NumericalIssue",
                                    message=(
                                        "Complexity measure "
                                        f"'{measure}' produced a non-finite result."
                                    ),
                                )
                            )
                    except Exception as exc:  # pragma: no cover - hit via failure tests
                        if isinstance(exc, ImportError):
                            raise
                        if runtime.on_error == "raise":
                            raise
                        failures.append(
                            make_failure_record(
                                family=self.family_name,
                                obs_index=obs_offset + obs_rel,
                                obs_id=obs_id,
                                channel_index=unit_idx,
                                channel_name=channel_names[unit_idx],
                                exception_type=type(exc).__name__,
                                message=str(exc),
                            )
                        )

        chunk_features: list[np.ndarray] = []
        chunk_names: list[str] = []
        for measure in self.config.measures:
            feature, names = self._finalize_descriptor(
                metric_arrays[measure],
                family_prefix=KNOWN_FAMILY_TOKENS[2],
                metric_name=measure,
                channel_names=channel_names,
            )
            chunk_features.append(feature)
            chunk_names.extend(names)

        descriptor_names = chunk_names

        return _DescriptorBlock(
            family=self.family_name,
            X=np.concatenate(chunk_features, axis=1)
            if chunk_features
            else np.empty((X.shape[0], 0)),
            descriptor_names=descriptor_names or [],
            meta={
                "backend": self.config.backend,
                "measures": list(self.config.measures),
                "batched_measures": sorted(batched_outputs),
                "measure_backends": dict(measure_backends),
            },
            failures=failures,
        )
