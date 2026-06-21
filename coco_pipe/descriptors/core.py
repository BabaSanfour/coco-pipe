"""
Descriptor extraction planner and execution pipeline.

This module owns the config-bound runtime orchestration for descriptor
extraction. It does not implement family-specific descriptor math; instead it:

- validates the explicit runtime inputs accepted by the module
- instantiates enabled descriptor families from typed config
- plans shared PSD computation for compatible PSD consumers
- executes one observation batch at a time with controlled parallelism
- merges aligned family outputs into one flat descriptor matrix

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from coco_pipe.io.structures import DataContainer

from ._constants import FAILURE_FAMILY_ALIASES
from .configs import DescriptorConfig, ParametricDescriptorConfig
from .extractors._parametric_fit import fit_parametric_batch
from .extractors._psd import compute_psd
from .extractors.base import (
    BaseDescriptorExtractor,
    BasePSDDescriptorExtractor,
    _DescriptorBlock,
)
from .extractors.complexity import ComplexityDescriptorExtractor
from .extractors.parametric import ParametricDescriptorExtractor
from .extractors.spectral import BandDescriptorExtractor
from .validation import (
    validate_runtime_inputs,
)

__all__ = ["DescriptorPipeline"]


@dataclass(slots=True)
class _PSDGroup:
    """Plan one shared PSD compute for a compatible set of consumers.

    Attributes
    ----------
    method : str
        Shared PSD estimator name.
    fmin, fmax : float
        Union frequency window required by all consumers in the group.
    consumers : list of BasePSDDescriptorExtractor
        PSD-consuming extractors that will reuse the shared PSD output.
    needs_parametric_fit : bool, default=False
        Whether the group also needs one shared parametric fit.
    need_parametric_metrics : bool, default=False
        Whether the shared fit must expose scalar parametric metrics.
    need_periodic_psds : bool, default=False
        Whether the shared fit must reconstruct periodic-only PSDs.
    fit_config : ParametricDescriptorConfig or None, default=None
        Shared parametric fit configuration when a fit is required.
    """

    method: str
    fmin: float
    fmax: float
    consumers: list[BasePSDDescriptorExtractor]
    needs_parametric_fit: bool = False
    need_parametric_metrics: bool = False
    need_periodic_psds: bool = False
    fit_config: ParametricDescriptorConfig | None = None


def _parallel_jobs(n_jobs: int, limit: int) -> int:
    """Clamp worker count to the amount of useful parallel work.

    Parameters
    ----------
    n_jobs : int
        Requested worker count from runtime config. ``-1`` means "use as much
        parallelism as this stage can use".
    limit : int
        Number of parallel tasks available in the current stage.

    Returns
    -------
    int
        Worker count capped at ``limit``.
    """
    return limit if n_jobs == -1 else min(n_jobs, limit)


def _cast_precision(values: np.ndarray, precision: str) -> np.ndarray:
    """Cast the final descriptor matrix to the configured floating precision.

    Parameters
    ----------
    values : np.ndarray
        Floating descriptor matrix to cast.
    precision : {"float32", "float64"}
        Requested output precision.

    Returns
    -------
    np.ndarray
        ``values`` cast in-place when possible.
    """
    return values.astype(
        np.float32 if precision == "float32" else np.float64, copy=False
    )


def _merge_descriptor_blocks(
    blocks: list[_DescriptorBlock],
    n_obs: int,
    precision: str,
) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    """Merge family descriptor blocks column-wise on the descriptor axis.

    Parameters
    ----------
    blocks : list of _DescriptorBlock
        Family-specific descriptor block objects to concatenate column-wise.
    n_obs : int
        Expected number of observations in each block.
    precision : {"float32", "float64"}
        Precision applied to the merged descriptor matrix.

    Returns
    -------
    tuple
        ``(X, descriptor_names, failures)`` where ``X`` is the merged
        descriptor matrix, ``descriptor_names`` is the deterministic merged
        column order, and ``failures`` concatenates all family failure records.

    Raises
    ------
    ValueError
        If any block is misaligned on the observation axis.
    """
    if not blocks:
        empty = np.empty(
            (n_obs, 0),
            dtype=np.float32 if precision == "float32" else np.float64,
        )
        return empty, [], []

    matrices = []
    names: list[str] = []
    failures: list[dict[str, Any]] = []

    for block in blocks:
        if block.X.shape[0] != n_obs:
            raise ValueError(
                "Descriptor block "
                f"'{block.family}' is misaligned: expected {n_obs} rows, "
                f"got {block.X.shape[0]}."
            )
        matrices.append(block.X)
        names.extend(block.descriptor_names)
        failures.extend(block.failures)

    if len(matrices) == 1:
        X = _cast_precision(matrices[0], precision)
    else:
        X = _cast_precision(np.concatenate(matrices, axis=1), precision)

    return (
        X,
        names,
        failures,
    )


def _sequential_runtime(runtime):
    """Return a sequential runtime copy for nested work.

    Parameters
    ----------
    runtime : DescriptorRuntimeConfig
        Runtime configuration for the current extraction stage.

    Returns
    -------
    DescriptorRuntimeConfig
        Copy with nested parallelism disabled.
    """
    return runtime.model_copy(update={"execution_backend": "sequential", "n_jobs": 1})


def _build_psd_groups(
    extractors: list[BaseDescriptorExtractor],
) -> list[_PSDGroup]:
    """Plan shared PSD groups for the enabled PSD-consuming extractors.

    Parameters
    ----------
    extractors : list of BaseDescriptorExtractor
        Config-bound extractors in deterministic family order.

    Returns
    -------
    list of _PSDGroup
        Shared PSD execution groups keyed by compatible PSD method and merged
        fit requirements.

    Raises
    ------
    ValueError
        If consumers that would share one parametric fit disagree on the fit
        configuration.
    """
    groups_by_method: dict[str, _PSDGroup] = {}
    for extractor in extractors:
        if not isinstance(extractor, BasePSDDescriptorExtractor):
            continue
        request = extractor.psd_request()
        method = str(request["method"])
        if method not in groups_by_method:
            groups_by_method[method] = _PSDGroup(
                method=method,
                fmin=float(request["fmin"]),
                fmax=float(request["fmax"]),
                consumers=[extractor],
            )
        else:
            current = groups_by_method[method]
            current.fmin = min(current.fmin, float(request["fmin"]))
            current.fmax = max(current.fmax, float(request["fmax"]))
            current.consumers.append(extractor)
        current = groups_by_method[method]
        fit_req = extractor.parametric_fit_requirements()
        if fit_req["needed"]:
            current.needs_parametric_fit = True
            current.need_parametric_metrics = current.need_parametric_metrics or bool(
                fit_req["metrics"]
            )
            current.need_periodic_psds = current.need_periodic_psds or bool(
                fit_req["periodic_psds"]
            )
            if current.fit_config is None:
                current.fit_config = fit_req["config"]
            elif current.fit_config.model_dump() != fit_req["config"].model_dump():
                raise ValueError(
                    "PSD consumers sharing one parametric fit must use the same "
                    "parametric fit configuration."
                )
    return list(groups_by_method.values())


def _merge_family_blocks(
    batch_results: list[dict[str, _DescriptorBlock]],
    family_order: list[str],
) -> list[_DescriptorBlock]:
    """Merge per-batch results row-wise within each descriptor family.

    Parameters
    ----------
    batch_results : list of dict
        One family-block mapping per processed observation batch.
    family_order : list of str
        Deterministic family order used for the final merged output.

    Returns
    -------
    list of _DescriptorBlock
        One merged block per family, still separated by family but aligned
        across all processed batches.

    Raises
    ------
    ValueError
        If descriptor names drift across batches for the same family.
    """
    merged_blocks: list[_DescriptorBlock] = []
    for family_name in family_order:
        family_blocks = [
            batch_result[family_name]
            for batch_result in batch_results
            if family_name in batch_result
        ]
        if not family_blocks:
            continue

        reference_names = family_blocks[0].descriptor_names
        for block in family_blocks[1:]:
            if block.descriptor_names != reference_names:
                raise ValueError(
                    "Descriptor names changed across batches for family "
                    f"'{family_name}'."
                )

        merged_blocks.append(
            _DescriptorBlock(
                family=family_name,
                X=np.concatenate([block.X for block in family_blocks], axis=0),
                descriptor_names=list(reference_names),
                meta={},
                failures=[
                    failure for block in family_blocks for failure in block.failures
                ],
            )
        )
    return merged_blocks


def _process_psd_group(
    group: _PSDGroup,
    X_batch: np.ndarray,
    sfreq: float,
    channel_names: list[str] | None,
    ids_batch: np.ndarray | None,
    runtime,
    obs_offset: int,
    joblib=None,
    consumer_parallel: bool = False,
    psd_n_jobs: int | None = None,
) -> dict[str, _DescriptorBlock]:
    """Execute one shared PSD group for a single observation batch.

    Parameters
    ----------
    group : _PSDGroup
        Planned PSD reuse group for the current batch.
    X_batch : np.ndarray
        Observation batch with shape ``(n_obs_batch, n_channels, n_times)``.
    sfreq : float
        Sampling frequency in Hertz.
    channel_names : list of str or None
        Runtime channel labels.
    ids_batch : np.ndarray or None
        Observation identifiers aligned with ``X_batch``.
    runtime : DescriptorRuntimeConfig
        Runtime policy used for this stage.
    obs_offset : int
        Absolute observation offset of the batch in the full input array.
    joblib : module, optional
        Imported ``joblib`` module when the selected strategy uses it.
    consumer_parallel : bool, default=False
        Whether compatible PSD consumers should run in parallel after the PSD
        has been computed.
    psd_n_jobs : int or None, default=None
        Worker count forwarded to the PSD backend when the selected strategy is
        PSD-level parallelism.

    Returns
    -------
    dict[str, _DescriptorBlock]
        Family-name mapping for all consumers in the PSD group.
    """
    if psd_n_jobs is not None and joblib is not None:
        with joblib.parallel_backend("threading", n_jobs=psd_n_jobs):
            psds, freqs = compute_psd(
                X_batch,
                sfreq=sfreq,
                method=group.method,
                fmin=group.fmin,
                fmax=group.fmax,
                n_jobs=psd_n_jobs,
            )
    else:
        psds, freqs = compute_psd(
            X_batch,
            sfreq=sfreq,
            method=group.method,
            fmin=group.fmin,
            fmax=group.fmax,
            n_jobs=psd_n_jobs,
        )

    fit_batch = None
    if group.needs_parametric_fit:
        fit_batch = fit_parametric_batch(
            psds,
            freqs,
            group.fit_config,
            runtime,
            need_periodic_psd=group.need_periodic_psds,
            include_metrics=group.need_parametric_metrics,
        )

    consumer_runtime = _sequential_runtime(runtime) if consumer_parallel else runtime
    if consumer_parallel and joblib is not None and len(group.consumers) > 1:
        blocks = joblib.Parallel(
            n_jobs=_parallel_jobs(runtime.n_jobs, len(group.consumers)),
            prefer="threads",
        )(
            joblib.delayed(consumer.extract_psd)(
                psds,
                freqs,
                channel_names=channel_names,
                ids=ids_batch,
                runtime=consumer_runtime,
                obs_offset=obs_offset,
                fit_batch=fit_batch,
            )
            for consumer in group.consumers
        )
    else:
        blocks = [
            consumer.extract_psd(
                psds,
                freqs,
                channel_names=channel_names,
                ids=ids_batch,
                runtime=consumer_runtime,
                obs_offset=obs_offset,
                fit_batch=fit_batch,
            )
            for consumer in group.consumers
        ]
    return {block.family: block for block in blocks}


def _process_batch(
    obs_slice: slice,
    X: np.ndarray,
    sfreq: float | None,
    channel_names: list[str] | None,
    ids: np.ndarray | None,
    signal_extractors: list[BaseDescriptorExtractor],
    psd_groups: list[_PSDGroup],
    runtime,
    strategy: str,
    joblib=None,
) -> dict[str, _DescriptorBlock]:
    """Execute one observation batch under the selected planner strategy.

    Parameters
    ----------
    obs_slice : slice
        Observation slice for the current batch.
    X : np.ndarray
        Full validated input array with shape ``(n_obs, n_channels, n_times)``.
    sfreq : float or None
        Sampling frequency in Hertz.
    channel_names : list of str or None
        Runtime channel labels.
    ids : np.ndarray or None
        Observation identifiers aligned with ``X``.
    signal_extractors : list of BaseDescriptorExtractor
        Non-PSD families that consume raw signal batches directly.
    psd_groups : list of _PSDGroup
        Planned PSD reuse groups for this pipeline instance.
    runtime : DescriptorRuntimeConfig
        Runtime policy for the current extraction call.
    strategy : str
        Selected parallelization strategy for this execution path.
    joblib : module, optional
        Imported ``joblib`` module when the selected strategy uses it.

    Returns
    -------
    dict[str, _DescriptorBlock]
        Family-name mapping for all blocks produced from the batch.
    """
    X_batch = X[obs_slice]
    ids_batch = None if ids is None else ids[obs_slice]
    obs_offset = obs_slice.start or 0
    family_blocks: dict[str, _DescriptorBlock] = {}

    if strategy == "work-unit" and joblib is not None:

        def _signal_unit(extractor):
            block = extractor.extract(
                X_batch,
                sfreq=sfreq,
                channel_names=channel_names,
                ids=ids_batch,
                runtime=_sequential_runtime(runtime),
                obs_offset=obs_offset,
            )
            return {block.family: block}

        def _psd_unit(group):
            return _process_psd_group(
                group,
                X_batch,
                sfreq=sfreq,
                channel_names=channel_names,
                ids_batch=ids_batch,
                runtime=_sequential_runtime(runtime),
                obs_offset=obs_offset,
                joblib=None,
                consumer_parallel=False,
            )

        work_units = [
            joblib.delayed(_signal_unit)(extractor) for extractor in signal_extractors
        ] + [joblib.delayed(_psd_unit)(group) for group in psd_groups]
        for unit_result in joblib.Parallel(
            n_jobs=_parallel_jobs(
                runtime.n_jobs,
                len(signal_extractors) + len(psd_groups),
            ),
            prefer="threads",
        )(work_units):
            family_blocks.update(unit_result)
    else:
        signal_runtime = _sequential_runtime(runtime)
        for extractor in signal_extractors:
            block = extractor.extract(
                X_batch,
                sfreq=sfreq,
                channel_names=channel_names,
                ids=ids_batch,
                runtime=signal_runtime,
                obs_offset=obs_offset,
            )
            family_blocks[block.family] = block

        for group in psd_groups:
            consumer_parallel = (
                strategy == "psd-consumer"
                and joblib is not None
                and len(group.consumers) > 1
            )
            psd_n_jobs = None
            if strategy == "psd-n_jobs":
                psd_n_jobs = runtime.n_jobs
            family_blocks.update(
                _process_psd_group(
                    group,
                    X_batch,
                    sfreq=sfreq,
                    channel_names=channel_names,
                    ids_batch=ids_batch,
                    runtime=runtime
                    if strategy == "parametric-inner" and group.needs_parametric_fit
                    else signal_runtime,
                    obs_offset=obs_offset,
                    joblib=joblib
                    if consumer_parallel or strategy == "psd-n_jobs"
                    else None,
                    consumer_parallel=consumer_parallel,
                    psd_n_jobs=psd_n_jobs,
                )
            )

    return family_blocks


class DescriptorPipeline:
    """Run config-driven descriptor extraction on explicit arrays.

    Parameters
    ----------
    config : DescriptorConfig or Mapping[str, Any]
        Typed descriptors configuration or a mapping accepted by
        :class:`DescriptorConfig`.

    Attributes
    ----------
    config : DescriptorConfig
        Parsed descriptors configuration.
    extractors : list of BaseDescriptorExtractor
        Enabled family extractors in deterministic family order.
    signal_extractors : list of BaseDescriptorExtractor
        Enabled non-PSD extractors that consume raw signal batches directly.
    psd_groups : list of _PSDGroup
        Planned PSD reuse groups derived once from the enabled extractors.
    family_order : list of str
        Deterministic family order used when merging batch-local outputs.

    Notes
    -----
    The pipeline is config-bound but runtime-stateless. Construction performs
    config parsing, corrected-band compatibility checks, and planner setup once.
    Each call to :meth:`extract` then validates the explicit runtime inputs,
    executes the planned families, and returns one flat descriptor matrix plus
    any collected failures.
    """

    def __init__(self, config: DescriptorConfig | Mapping[str, Any]):
        """Create a config-bound descriptor extraction pipeline.

        Parameters
        ----------
        config : DescriptorConfig or Mapping[str, Any]
            Typed descriptors configuration or a mapping accepted by
            :class:`DescriptorConfig`.

        Raises
        ------
        ValueError
            If corrected band outputs are enabled but the parametric fit range
            does not cover the requested band PSD window.
        """
        self.config = (
            config
            if isinstance(config, DescriptorConfig)
            else DescriptorConfig.model_validate(config)
        )
        corrected_outputs = {
            "corrected_absolute_power",
            "corrected_log_absolute_power",
            "corrected_relative_power",
            "corrected_ratios",
        }
        if any(
            output in corrected_outputs for output in self.config.families.bands.outputs
        ):
            fit_low, fit_high = self.config.families.parametric.freq_range
            band_low = self.config.families.bands.fmin
            band_high = self.config.families.bands.fmax
            if fit_low > band_low or fit_high < band_high:
                raise ValueError(
                    "Corrected band outputs require families.parametric.freq_range "
                    f"to cover the band PSD window [{band_low}, {band_high}]."
                )
        self.extractors: list[BaseDescriptorExtractor] = []
        if self.config.families.bands.enabled:
            self.extractors.append(
                BandDescriptorExtractor(
                    self.config.families.bands,
                    fit_config=self.config.families.parametric,
                )
            )
        if self.config.families.parametric.enabled:
            self.extractors.append(
                ParametricDescriptorExtractor(self.config.families.parametric)
            )
        if self.config.families.complexity.enabled:
            self.extractors.append(
                ComplexityDescriptorExtractor(self.config.families.complexity)
            )
        self.signal_extractors = [
            extractor
            for extractor in self.extractors
            if not isinstance(extractor, BasePSDDescriptorExtractor)
        ]
        self.psd_groups = _build_psd_groups(self.extractors)
        self.family_order = [extractor.family_name for extractor in self.extractors]

    def extract(
        self,
        X: np.ndarray,
        ids: Sequence[Any] | np.ndarray | None = None,
        sfreq: float | None = None,
        channel_names: Sequence[str] | np.ndarray | None = None,
    ) -> DataContainer:
        """Extract descriptors from explicit NumPy inputs.

        Parameters
        ----------
        X : np.ndarray
            Signal array with shape ``(n_obs, n_channels, n_times)``.
        ids : sequence or np.ndarray, optional
            Observation identifiers aligned with ``X``.
        sfreq : float, optional
            Sampling frequency in Hertz. Required when enabled families depend
            on spectral estimates or spectral entropy.
        channel_names : sequence of str or np.ndarray, optional
            Channel labels. Required for channel-resolved outputs.

        Returns
        -------
        DataContainer
            Flat ``("obs", "feature")`` container: ``X`` is the descriptor
            matrix, ``coords["feature"]`` the column names, and
            ``coords["feature_family"]`` the per-column family token (carried
            from the extractors, not parsed). ``ids`` are the observation ids and
            ``meta`` holds ``failures`` and ``sfreq``.

        Raises
        ------
        ValueError
            If the explicit input contract is not satisfied.
        ImportError
            If an optional backend required by the enabled families is missing.

        Notes
        -----
        When ``runtime.on_error="warn"``, extraction still completes and stores
        the failures in ``meta["failures"]`` before emitting one aggregate
        warning at the pipeline level.

        The returned row order always matches the input observation order.
        """
        inputs = validate_runtime_inputs(
            self.config,
            X=X,
            ids=ids,
            channel_names=channel_names,
            sfreq=sfreq,
        )

        planner_runtime = self.config.runtime
        n_obs = inputs["X"].shape[0]
        obs_chunk = planner_runtime.obs_chunk
        if not obs_chunk or obs_chunk >= n_obs:
            batch_slices = [slice(0, n_obs)]
        else:
            batch_slices = [
                slice(start, min(start + obs_chunk, n_obs))
                for start in range(0, n_obs, obs_chunk)
            ]

        if (
            planner_runtime.execution_backend == "sequential"
            or planner_runtime.n_jobs == 1
        ):
            parallel_strategy = "sequential"
        elif len(batch_slices) > 1:
            parallel_strategy = "obs-batch"
        else:
            work_units = len(self.signal_extractors) + len(self.psd_groups)
            if work_units > 1:
                parallel_strategy = "work-unit"
            elif len(self.psd_groups) == 1 and len(self.psd_groups[0].consumers) > 1:
                parallel_strategy = "psd-consumer"
            elif (
                len(self.psd_groups) == 1
                and len(self.psd_groups[0].consumers) == 1
                and self.psd_groups[0].needs_parametric_fit
            ):
                parallel_strategy = "parametric-inner"
            elif len(self.psd_groups) == 1 and len(self.psd_groups[0].consumers) == 1:
                parallel_strategy = "psd-n_jobs"
            else:
                parallel_strategy = "sequential"

        if parallel_strategy == "obs-batch":
            import joblib

            batch_results = joblib.Parallel(
                n_jobs=_parallel_jobs(planner_runtime.n_jobs, len(batch_slices)),
                prefer="threads",
            )(
                joblib.delayed(_process_batch)(
                    obs_slice,
                    X=inputs["X"],
                    sfreq=inputs["sfreq"],
                    channel_names=inputs["channel_names"],
                    ids=inputs["ids"],
                    signal_extractors=self.signal_extractors,
                    psd_groups=self.psd_groups,
                    runtime=_sequential_runtime(planner_runtime),
                    strategy="sequential",
                    joblib=None,
                )
                for obs_slice in batch_slices
            )
        else:
            if parallel_strategy != "sequential":
                import joblib
            else:
                joblib = None
            batch_results = [
                _process_batch(
                    obs_slice,
                    X=inputs["X"],
                    sfreq=inputs["sfreq"],
                    channel_names=inputs["channel_names"],
                    ids=inputs["ids"],
                    signal_extractors=self.signal_extractors,
                    psd_groups=self.psd_groups,
                    runtime=planner_runtime,
                    strategy=parallel_strategy,
                    joblib=joblib,
                )
                for obs_slice in batch_slices
            ]

        blocks = _merge_family_blocks(
            batch_results,
            family_order=self.family_order,
        )

        X_desc, descriptor_names, failures = _merge_descriptor_blocks(
            blocks,
            n_obs=inputs["X"].shape[0],
            precision=self.config.precision,
        )

        # ``feature_family`` is carried straight from each block's family token
        # (normalised to the canonical column form), never inferred from the
        # generated column strings. It aligns with ``descriptor_names`` because
        # ``_merge_descriptor_blocks`` concatenates columns in ``blocks`` order.
        feature_family = [
            FAILURE_FAMILY_ALIASES.get(block.family, block.family)
            for block in blocks
            for _ in block.descriptor_names
        ]

        if self.config.runtime.on_error == "warn" and failures:
            warnings.warn(
                f"Collected {len(failures)} descriptor failures during extract().",
                stacklevel=2,
            )

        return DataContainer(
            X=X_desc,
            dims=("obs", "feature"),
            ids=inputs["ids"],
            coords={
                "feature": np.asarray(descriptor_names, dtype=object),
                "feature_family": np.asarray(feature_family, dtype=object),
            },
            meta={"failures": failures, "sfreq": inputs["sfreq"]},
        )

    def pool_channels(
        self,
        container: DataContainer,
        channel_groups: Mapping[str, Sequence[str]],
    ) -> DataContainer:
        """Pool sensor-level descriptor columns into grouped channel outputs.

        Parameters
        ----------
        container : DataContainer
            Flat descriptor container produced by :meth:`extract`.
        channel_groups : mapping of str to sequence of str
            Channel groups used to replace sensor-level descriptor columns with
            grouped ``"chgrp-..."`` outputs.

        Returns
        -------
        DataContainer
            New container with grouped channel features. ``ids`` and ``meta``
            (including ``failures``) are preserved; ``coords["feature"]`` and
            ``coords["feature_family"]`` reflect the pooled columns.

        Raises
        ------
        ValueError
            If the container is malformed or if any requested group cannot be
            formed from the sensor-level descriptor columns.
        """
        X_desc = np.asarray(container.X, dtype=float)
        descriptor_names = [str(name) for name in container.coords.get("feature", [])]
        input_family = [
            str(name) for name in container.coords.get("feature_family", [])
        ]
        if X_desc.ndim != 2:
            raise ValueError(
                "pool_channels() expects a 2D ('obs', 'feature') container."
            )
        if X_desc.shape[1] != len(descriptor_names):
            raise ValueError(
                "pool_channels() requires coords['feature'] to align with X columns."
            )
        if not channel_groups:
            raise ValueError("channel_groups must define at least one group.")

        base_to_channel_cols: dict[str, dict[str, int]] = {}
        known_channels: set[str] = set()
        for col_idx, descriptor_name in enumerate(descriptor_names):
            if "_ch-" not in descriptor_name:
                continue
            base_name, channel_name = descriptor_name.rsplit("_ch-", 1)
            base_to_channel_cols.setdefault(base_name, {})[channel_name] = col_idx
            known_channels.add(channel_name)

        if not base_to_channel_cols:
            raise ValueError(
                "pool_channels() requires sensor-level descriptor names "
                "ending in '_ch-<sensor>'."
            )

        normalized_groups: dict[str, list[str]] = {}
        assigned: dict[str, str] = {}
        for raw_group_name, raw_members in channel_groups.items():
            group_name = str(raw_group_name)
            if not group_name:
                raise ValueError("channel_groups keys must be non-empty strings.")
            members = [str(member) for member in raw_members]
            if not members:
                raise ValueError(
                    f"channel_groups['{group_name}'] must define at least one channel."
                )
            if len(set(members)) != len(members):
                raise ValueError(
                    f"channel_groups['{group_name}'] must not contain duplicates."
                )
            for member in members:
                if member not in known_channels:
                    raise ValueError(
                        f"channel_groups['{group_name}'] references unknown channel "
                        f"'{member}'."
                    )
                if member in assigned:
                    raise ValueError(
                        f"Channel '{member}' is assigned to multiple channel_groups: "
                        f"'{assigned[member]}' and '{group_name}'."
                    )
                assigned[member] = group_name
            normalized_groups[group_name] = members

        # Per-column family is carried, not re-parsed: a pooled column keeps the
        # family of its source sensor columns.
        family_at = (
            (lambda idx: input_family[idx])
            if len(input_family) == len(descriptor_names)
            else (lambda idx: "")
        )
        output_columns: list[np.ndarray] = []
        pooled_names: list[str] = []
        pooled_family: list[str] = []
        seen_bases: set[str] = set()
        for col_idx, descriptor_name in enumerate(descriptor_names):
            if "_ch-" not in descriptor_name:
                output_columns.append(X_desc[:, col_idx][:, None])
                pooled_names.append(descriptor_name)
                pooled_family.append(family_at(col_idx))
                continue

            base_name, _ = descriptor_name.rsplit("_ch-", 1)
            if base_name in seen_bases:
                continue
            seen_bases.add(base_name)

            channel_to_col = base_to_channel_cols[base_name]
            for group_name, members in normalized_groups.items():
                missing = [member for member in members if member not in channel_to_col]
                if missing:
                    raise ValueError(
                        "pool_channels() could not form group "
                        f"'{group_name}' for descriptor base '{base_name}'. "
                        f"Missing channels: {missing}."
                    )
                member_indices = [channel_to_col[member] for member in members]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    grouped = np.nanmean(X_desc[:, member_indices], axis=1)
                output_columns.append(grouped[:, None])
                pooled_names.append(f"{base_name}_chgrp-{group_name}")
                pooled_family.append(family_at(member_indices[0]))

        X_pooled = _cast_precision(
            np.concatenate(output_columns, axis=1)
            if output_columns
            else np.empty((X_desc.shape[0], 0), dtype=float),
            self.config.precision,
        )
        coords: dict[str, np.ndarray] = {
            "feature": np.asarray(pooled_names, dtype=object)
        }
        if len(input_family) == len(descriptor_names):
            coords["feature_family"] = np.asarray(pooled_family, dtype=object)
        return DataContainer(
            X=X_pooled,
            dims=("obs", "feature"),
            ids=container.ids,
            coords=coords,
            meta=dict(container.meta),
        )
