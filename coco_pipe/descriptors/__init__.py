"""Descriptor extraction and the table lifecycle.

The descriptor lifecycle has four stages:

1. **extract** — :class:`DescriptorPipeline` turns
   ``(n_obs, n_channels, n_times)`` arrays into a flat
   :class:`~coco_pipe.io.structures.DataContainer` of descriptors.
2. **reject** — epoch MAD-outlier rejection via
   :func:`coco_pipe.io.quality.drop_epoch_outliers` (+
   :func:`mad_failures_from_qc` for failure logging).
3. **aggregate** — :func:`build_descriptor_tables` builds per-epoch and
   group-aggregated subject tables (+ :func:`add_aggregated_band_ratios`).
4. **merge** — :func:`merge_descriptor_tables` concatenates per-shard tables of
   one table kind; :func:`save_descriptor_table` / :func:`load_descriptor_table`
   handle on-disk descriptor tables.
"""

from ._constants import KNOWN_FAMILY_TOKENS
from .configs import DescriptorConfig
from .core import DescriptorPipeline
from .io import (
    check_feature_column_consistency,
    load_descriptor_table,
    merge_descriptor_tables,
    save_descriptor_table,
)
from .naming import parse_descriptor_feature_column, split_family_token
from .tables import (
    add_aggregated_band_ratios,
    build_descriptor_tables,
    mad_failures_from_qc,
)

__all__ = [
    "KNOWN_FAMILY_TOKENS",
    "DescriptorConfig",
    "DescriptorPipeline",
    "add_aggregated_band_ratios",
    "build_descriptor_tables",
    "check_feature_column_consistency",
    "load_descriptor_table",
    "mad_failures_from_qc",
    "merge_descriptor_tables",
    "parse_descriptor_feature_column",
    "save_descriptor_table",
    "split_family_token",
]
