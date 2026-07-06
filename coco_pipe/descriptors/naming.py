"""Descriptor column-name contract.

The descriptor extractors emit column names of the form
``{family}_{feature}_{ch|chgrp}-{sensor}`` (see
:class:`~coco_pipe.descriptors.core.DescriptorPipeline`). Parsing that contract
is descriptor-domain knowledge, so it lives here next to the family QC that
consumes it (:mod:`coco_pipe.descriptors.qc`) and shares the single
:data:`~coco_pipe.descriptors._constants.DESCRIPTOR_SCOPE_RE`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ._constants import DESCRIPTOR_SCOPE_RE, KNOWN_FAMILY_TOKENS

__all__ = [
    "build_descriptor_feature_metadata",
    "parse_descriptor_feature_column",
    "split_family_token",
]


def split_family_token(
    text: str,
    known_families: tuple[str, ...],
) -> tuple[str | None, str]:
    """Split ``text`` into ``(family, remainder)`` on the first known family token.

    The family token may appear as a leading prefix (``"band_..."``) or embedded
    (``"mean_band_..."``). Returns ``(None, text)`` when no known family token is
    present. Shared by :func:`parse_descriptor_feature_column` and the family
    classifier so the token-matching rule lives in one place.
    """
    for family_name in known_families:
        if text.startswith(f"{family_name}_"):
            return family_name, text[len(f"{family_name}_") :]
        token = f"_{family_name}_"
        if token in text:
            prefix, remainder = text.split(token, 1)
            return family_name, f"{prefix}_{remainder}"
    return None, text


def parse_descriptor_feature_column(
    column: str,
    known_families: tuple[str, ...],
) -> dict[str, str]:
    """Strictly parse one descriptor column into its constituent parts.

    Parameters
    ----------
    column
        Descriptor column name, e.g. ``"band_abs_alpha_ch-Fz"``.
    known_families
        Recognised family prefix tokens (e.g.
        :data:`~coco_pipe.descriptors._constants.KNOWN_FAMILY_TOKENS`).

    Returns
    -------
    dict[str, str]
        Keys ``column``, ``family``, ``feature``, ``scope``
        (``"sensor"`` / ``"sensor_group"``), and ``sensor``.

    Raises
    ------
    ValueError
        If the column does not match the scope contract or contains no known
        family token.
    """
    match = DESCRIPTOR_SCOPE_RE.match(str(column))
    if match is None:
        raise ValueError(
            f"Could not parse descriptor column '{column}'. "
            "Expected format: '{family}_{feature}_{chgrp|ch}-{sensor}'."
        )

    body, scope, sensor = match.group(1), match.group(2), match.group(3)
    family, feature = split_family_token(body, known_families)
    if family is None:
        raise ValueError(
            f"Column '{column}' does not contain a known family token. "
            f"Known: {known_families}."
        )

    return {
        "column": str(column),
        "family": family,
        "feature": feature,
        "scope": "sensor_group" if scope == "chgrp" else "sensor",
        "sensor": sensor,
    }


def build_descriptor_feature_metadata(
    columns: Sequence[str],
    *,
    known_families: tuple[str, ...] = KNOWN_FAMILY_TOKENS,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build decoding feature metadata from descriptor feature-column names."""
    from .qc import descriptor_subfamily

    rows = []
    for column in columns:
        item = parse_descriptor_feature_column(str(column), known_families)
        rows.append(
            {
                "FeatureName": f"{item['sensor']}_{item['feature']}",
                "Sensor": item["sensor"],
                "Feature": item["feature"],
                "FeatureFamily": descriptor_subfamily(
                    item["family"],
                    item["feature"],
                ),
            }
        )
    metadata = pd.DataFrame(
        rows,
        columns=("FeatureName", "Sensor", "Feature", "FeatureFamily"),
    ).drop_duplicates("FeatureName")
    if feature_names is not None:
        requested = {str(value) for value in feature_names}
        metadata = metadata[metadata["FeatureName"].isin(requested)]
    return metadata.reset_index(drop=True)
