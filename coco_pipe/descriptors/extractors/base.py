"""
Base interfaces for descriptor extraction backends.

This module defines the internal contracts shared by built-in descriptor
extractors. The module exposes:

- `BaseDescriptorExtractor`, shared by all families; families that read the
  raw signal (complexity) add `extract()`
- `BasePSDDescriptorExtractor` for families that consume shared PSD batches
- `_DescriptorBlock` as the private family output payload
- `make_failure_record` as the shared normalized failure-record helper

The surrounding descriptors stack uses these interfaces to provide:

- explicit runtime dispatch from `DescriptorPipeline`
- deterministic sensor-level descriptor naming
- family-wise metadata and failure collection
- safe merging of family outputs into one stable result dictionary

Notes
-----
`BaseDescriptorExtractor` is an internal extension point for descriptor
families. Unlike dim-reduction reducers, descriptor extractors are stateless at
runtime and do not expose `fit`, persistence, or model objects.

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..configs import DescriptorRuntimeConfig

__all__ = [
    "BaseDescriptorExtractor",
    "BasePSDDescriptorExtractor",
    "make_failure_record",
]


@dataclass(slots=True)
class _DescriptorBlock:
    """Private in-memory descriptor payload for one family.

    Attributes
    ----------
    family : str
        Canonical family name that produced the block.
    X : np.ndarray
        Family-specific descriptor matrix aligned on the observation axis.
    descriptor_names : list of str
        Deterministic column names aligned with the columns of ``X``.
    meta : dict
        Family-specific metadata to preserve under the merged result.
    failures : list of dict
        Normalized failure records collected during extraction.

    Notes
    -----
    ``X.shape[0]`` must always match the input observation count seen by the
    extractor, and ``len(descriptor_names)`` must always match ``X.shape[1]``.
    The pipeline depends on these alignment guarantees when merging family
    outputs.
    """

    family: str
    X: np.ndarray
    descriptor_names: list[str]
    meta: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)


def make_failure_record(
    family: str,
    obs_index: int,
    obs_id: Any = None,
    channel_index: int | None = None,
    channel_name: str | None = None,
    exception_type: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """Create one normalized extractor failure record."""
    return {
        "family": family,
        "obs_index": obs_index,
        "obs_id": obs_id,
        "channel_index": channel_index,
        "channel_name": channel_name,
        "exception_type": exception_type,
        "message": message,
    }


class BaseDescriptorExtractor:
    """
    Base class for descriptor extraction families.

    Subclasses receive already validated NumPy inputs and return one
    `_DescriptorBlock` aligned on the observation axis. Families that read
    the raw signal define ``extract()``; PSD families derive from
    `BasePSDDescriptorExtractor` instead. The base class keeps
    the extractor API narrow and provides a shared helper for sensor-level
    finalization and deterministic descriptor naming.

    Parameters
    ----------
    config : Any
        Typed family configuration parsed by `DescriptorConfig`.

    Attributes
    ----------
    config : Any
        Stored family-specific configuration object.
    family_name : str
        Stable family identifier used in failure records and merged metadata.

    Notes
    -----
    Extractors are stateless at runtime. They do not learn parameters across
    calls; all runtime state is passed explicitly.

    Concrete extractors are expected to:

    1. compute family-specific values with shape ``(n_obs, n_channels)`` for
       each metric
    2. pass those values through :meth:`_finalize_descriptor`
    3. return one `_DescriptorBlock` with aligned names, metadata, and failures
    """

    family_name = "base"

    def __init__(self, config: Any):
        """Store the typed family configuration."""
        self.config = config

    @property
    def capabilities(self) -> dict[str, Any]:
        """Return static extractor capability metadata.

        Returns
        -------
        dict[str, Any]
            Static metadata describing optional dependencies and general
            execution properties for the extractor.

        Notes
        -----
        The descriptors pipeline currently uses this mapping only as lightweight
        backend metadata. It is intentionally much smaller than the reducer
        capability surface in `dim_reduction`.
        """
        return {
            "requires_sfreq": False,
            "supports_batching": True,
            "supports_channelwise": True,
            "deterministic": True,
            "optional_dependencies": [],
        }

    def _finalize_descriptor(
        self,
        values: np.ndarray,
        family_prefix: str,
        metric_name: str,
        channel_names: list[str] | None,
    ) -> tuple[np.ndarray, list[str]]:
        """Build deterministic sensor-level descriptor names.

        Parameters
        ----------
        values : np.ndarray
            Family metric values with shape ``(n_obs, n_channels)`` or
            ``(n_obs,)``.
        family_prefix : str
            Stable family prefix, for example ``"band"`` or ``"param"``.
        metric_name : str
            Family-local metric identifier used in the descriptor name.
        channel_names : list of str, optional
            Channel labels used when building channel-resolved descriptor names.

        Returns
        -------
        tuple
            ``(X_metric, names)`` where ``X_metric`` is the finalized metric
            matrix and ``names`` is the aligned list of descriptor names.

        Notes
        -----
        This helper assumes ``values`` already represents descriptor values, not
        raw signals. It therefore only handles the stable sensor-level naming
        convention used by the public extract result.

        Examples
        --------
        Given ``channel_names=["Fz", "Cz", "Pz"]`` and
        ``metric_name="abs_alpha"``:

        - yields
          ``["band_abs_alpha_ch-Fz", "band_abs_alpha_ch-Cz", "band_abs_alpha_ch-Pz"]``
        """
        if values.ndim == 1:
            values = values[:, None]
        channel_names = channel_names or [f"ch-{idx}" for idx in range(values.shape[1])]
        scopes = [
            channel_name if channel_name.startswith("ch-") else f"ch-{channel_name}"
            for channel_name in channel_names
        ]
        names = ["_".join((family_prefix, metric_name, scope)) for scope in scopes]
        return values, names


class BasePSDDescriptorExtractor(BaseDescriptorExtractor, ABC):
    """
    Abstract base class for descriptor families that consume PSD batches.

    PSD-consuming families still participate in the shared descriptor contract,
    but they expose one additional explicit entry point:

    - `extract_psd(...)` consumes precomputed `psds, freqs`
    - `psd_request()` tells the planner which PSD range and method is needed

    This keeps the generic raw-signal interface narrow while still giving the
    planner one formal PSD-consumer contract shared by spectral and parametric
    families.

    """

    @abstractmethod
    def psd_request(self) -> dict[str, Any]:
        """Describe the PSD requirements for the shared planner.

        Returns
        -------
        dict[str, Any]
            Minimal request payload containing the PSD method and the required
            frequency range for this family.
        """

    def parametric_fit_requirements(self) -> dict[str, Any]:
        """Describe whether this PSD consumer needs a shared parametric fit.

        Returns
        -------
        dict[str, Any]
            Shared-fit requirements with the keys:

            - `needed`
            - `metrics`
            - `periodic_psds`
            - `config`
        """
        return {
            "needed": False,
            "metrics": False,
            "periodic_psds": False,
            "config": None,
        }

    @abstractmethod
    def extract_psd(
        self,
        psds: np.ndarray,
        freqs: np.ndarray,
        channel_names: list[str] | None,
        ids: np.ndarray | None,
        runtime: DescriptorRuntimeConfig,
        obs_offset: int = 0,
        fit_batch: Any | None = None,
    ) -> _DescriptorBlock:
        """Extract descriptors from explicit PSD intermediates.

        Parameters
        ----------
        psds : np.ndarray
            PSD batch with shape ``(n_obs, n_channels, n_freqs)``.
        freqs : np.ndarray
            Frequency grid aligned with the last axis of ``psds``.
        channel_names : list of str, optional
            Explicit channel labels aligned with the channel axis.
        ids : np.ndarray, optional
            Observation identifiers aligned with the observation axis.
        runtime : DescriptorRuntimeConfig
            Runtime execution controls shared across extractors.
        obs_offset : int, default=0
            Global observation offset applied to collected failure records.
        fit_batch : Any, optional
            Additional shared fit payload required by some PSD consumers.

        Returns
        -------
        _DescriptorBlock
            Family-specific descriptor block aligned with the input PSD batch.
        """
