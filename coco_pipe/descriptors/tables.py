"""Assemble descriptor extraction containers into epoch/subject feature tables.

The descriptor lifecycle is **extract → reject → aggregate → merge**:

* **extract** — :meth:`coco_pipe.descriptors.core.DescriptorPipeline.extract`
  returns a flat ``("obs", "feature")`` :class:`~coco_pipe.io.structures.DataContainer`.
* **reject** — epoch MAD-outlier rejection is just
  :func:`coco_pipe.io.quality.drop_epoch_outliers` on that container;
  :func:`mad_failures_from_qc` turns the dropped epochs into failure records.
* **aggregate** — :func:`build_descriptor_tables` builds the per-epoch table and
  the group-aggregated subject table (mean + extra grouped stats + optional band
  ratios) via :meth:`~coco_pipe.io.structures.DataContainer.aggregate` /
  :meth:`~coco_pipe.io.structures.DataContainer.aggregate_groups`.
* **merge** — cross-shard concatenation lives in
  :func:`coco_pipe.descriptors.io.merge_descriptor_tables`.

Project-specific concerns (BIDS grouping-key derivation, channel-group pooling,
shard file layout, QC reports) stay with the caller.

Author: Hamza Abdelhedi <hamza.abdelhedi@umontreal.ca>
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from coco_pipe.io.quality import QCResult
from coco_pipe.io.structures import DataContainer

from ._constants import DEFAULT_RATIO_PREFIXES

__all__ = [
    "add_aggregated_band_ratios",
    "build_descriptor_tables",
    "mad_failures_from_qc",
]


def mad_failures_from_qc(qc_result: QCResult | None) -> list[dict[str, Any]]:
    """Turn epochs dropped by :func:`drop_epoch_outliers` into failure records.

    Mirrors the extractor failure schema (``obs_id``, ``obs_index``,
    ``channel_index``, ``channel_name``, ``family``, ``exception_type``,
    ``message``) so MAD drops flow into the same failure log as extraction
    failures. When rejection was made per descriptor group, one record is
    emitted per (group, epoch); otherwise one per dropped epoch.

    Parameters
    ----------
    qc_result
        The :class:`~coco_pipe.io.quality.QCResult` returned by
        :func:`coco_pipe.io.quality.drop_epoch_outliers` (``None`` yields ``[]``).
    """
    if qc_result is None:
        return []
    thresholds = qc_result.thresholds or {}
    z = thresholds.get("epoch_z_threshold")
    fraction = thresholds.get("epoch_outlier_fraction_threshold")
    pct = f"{fraction * 100:.1f}%" if fraction is not None else "the allowed fraction"
    mads = f"{z} MADs" if z is not None else "the MAD threshold"

    records_by_family = qc_result.per_family_dropped or {
        "global": qc_result.epochs_dropped
    }
    out: list[dict[str, Any]] = []
    for family, records in records_by_family.items():
        label = "MAD_Rejection" if family == "global" else f"MAD_Rejection:{family}"
        for record in records:
            out.append(
                {
                    "obs_id": record.obs_id,
                    "obs_index": record.obs_index,
                    "channel_index": -1,
                    "channel_name": "ALL",
                    "family": label,
                    "exception_type": "MADOutlierError",
                    "message": (
                        f"Epoch dropped for {family}: > {pct} of features "
                        f"exceeded {mads}."
                    ),
                }
            )
    return out


def add_aggregated_band_ratios(
    base_features_df: pd.DataFrame,
    ratio_pairs: Sequence[tuple[str, str]],
    floor: float = 0.0,
    prefixes: Sequence[tuple[str, str]] = DEFAULT_RATIO_PREFIXES,
) -> pd.DataFrame:
    """Compute band-ratio columns from aggregated mean band-power features.

    For each ``(numerator, denominator)`` band pair and each ``(input_prefix,
    output_prefix)`` in *prefixes*, divide every matching
    ``{input_prefix}{numerator}_{suffix}`` column by its
    ``{input_prefix}{denominator}_{suffix}`` counterpart. Denominators at or
    below *floor* yield ``NaN`` instead of an unstable division.

    Returns
    -------
    pandas.DataFrame
        One column per emitted ratio (empty when nothing matched). Aligned to
        *base_features_df*'s row index.
    """
    columns: dict[str, np.ndarray] = {}
    for numerator, denominator in ratio_pairs:
        for in_prefix, out_prefix in prefixes:
            for column in base_features_df.columns:
                head = f"{in_prefix}{numerator}_"
                if not str(column).startswith(head):
                    continue
                suffix = str(column)[len(head) :]
                den_col = f"{in_prefix}{denominator}_{suffix}"
                if den_col not in base_features_df.columns:
                    continue
                num_vals = base_features_df[column].to_numpy(dtype=float)
                den_vals = base_features_df[den_col].to_numpy(dtype=float)
                columns[f"{out_prefix}{numerator}_{denominator}_{suffix}"] = np.divide(
                    num_vals,
                    den_vals,
                    out=np.full_like(num_vals, np.nan),
                    where=den_vals > floor,
                )
    return pd.DataFrame(columns, index=base_features_df.index)


def build_descriptor_tables(
    container: DataContainer,
    metadata_df: pd.DataFrame,
    group_by: str,
    id_col: str = "obs_id",
    target_col: str | None = None,
    aggregation_groups: Sequence[Mapping[str, Any]] | None = None,
    ratio_pairs: Sequence[tuple[str, str]] | None = None,
    ratio_floor: float = 0.0,
    ratio_prefixes: Sequence[tuple[str, str]] = DEFAULT_RATIO_PREFIXES,
    min_count: int = 1,
    on_insufficient: str = "raise",
) -> dict[str, Any]:
    """Build epoch- and group-aggregated subject-level descriptor tables.

    Parameters
    ----------
    container
        Flat ``("obs", "feature")`` descriptor container from
        :meth:`~coco_pipe.descriptors.core.DescriptorPipeline.extract` (typically
        after epoch MAD rejection).
    metadata_df
        One row per epoch, aligned with ``container.X``. Must contain *id_col*;
        every other column is carried as an observation coordinate and, when
        constant within a group, into the subject table.
    group_by
        Metadata column defining the aggregation groups (e.g. a recording id).
    id_col
        Observation-id column in *metadata_df* (default ``"obs_id"``).
    target_col
        Optional target column carried onto the subject table.
    aggregation_groups
        ``aggregate_groups`` specs producing the subject feature columns
        (each ``{"stats": ..., <selectors>}``). Defaults to ``[{"stats":
        "mean"}]`` (mean of every feature). This is where median / IQR / etc.
        subject-level stats are requested.
    ratio_pairs, ratio_floor, ratio_prefixes
        When *ratio_pairs* is given, band ratios from the aggregated mean
        features are appended via :func:`add_aggregated_band_ratios`.
    min_count, on_insufficient
        Forwarded to :meth:`~coco_pipe.io.structures.DataContainer.aggregate`
        and ``aggregate_groups``. With ``on_insufficient="warn"`` a group (or a
        single descriptor family within ``aggregate_groups``) whose surviving
        rows are all-NaN emits NaN features instead of raising, so the subject
        is retained with whatever else is computable.

    Returns
    -------
    dict
        ``epoch_df``, ``subject_df``, ``epoch_feature_columns``, and
        ``subject_feature_columns``.

    Raises
    ------
    ValueError
        If *id_col* or *group_by* is missing from *metadata_df*.
    """
    metadata_df = metadata_df.reset_index(drop=True)
    if id_col not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain the id column '{id_col}'.")
    if group_by not in metadata_df.columns:
        raise ValueError(f"metadata_df must contain the group column '{group_by}'.")

    names = [str(name) for name in container.coords["feature"]]
    X = np.asarray(container.X)

    epoch_df = pd.concat(
        [metadata_df, pd.DataFrame(X, columns=names)],
        axis=1,
    )

    coords: dict[str, np.ndarray] = {
        column: metadata_df[column].to_numpy(dtype=object)
        for column in metadata_df.columns
        if column != id_col
    }
    coords["feature"] = np.asarray(names, dtype=object)
    y = (
        metadata_df[target_col].to_numpy()
        if target_col and target_col in metadata_df.columns
        else None
    )
    work = DataContainer(
        X=X,
        y=y,
        ids=metadata_df[id_col].to_numpy(dtype=object),
        dims=("obs", "feature"),
        coords=coords,
    )

    grouped_mean = work.aggregate(
        by=group_by,
        stats="mean",
        min_count=min_count,
        on_insufficient=on_insufficient,
    )
    agg_df = grouped_mean.obs_table(include_y=bool(target_col), y_col=target_col or "y")
    base_agg_features = pd.DataFrame(
        grouped_mean.X,
        columns=[str(column) for column in grouped_mean.coords["feature"]],
    )

    groups = list(aggregation_groups) if aggregation_groups else [{"stats": "mean"}]
    grouped_features = work.aggregate_groups(
        by=group_by,
        groups=groups,
        min_count=min_count,
        on_insufficient=on_insufficient,
    )
    agg_features = pd.DataFrame(
        grouped_features.X,
        columns=[str(column) for column in grouped_features.coords["feature"]],
    )

    if ratio_pairs:
        ratio_df = add_aggregated_band_ratios(
            base_agg_features,
            ratio_pairs,
            floor=ratio_floor,
            prefixes=ratio_prefixes,
        )
        if not ratio_df.empty:
            agg_features = pd.concat([agg_features, ratio_df], axis=1)

    subject_df = pd.concat([agg_df.reset_index(drop=True), agg_features], axis=1)
    return {
        "epoch_df": epoch_df,
        "subject_df": subject_df,
        "epoch_feature_columns": list(names),
        "subject_feature_columns": list(agg_features.columns),
    }
