"""
Analysis-unit enumeration for multi-mode EEG pipelines.

An *analysis unit* is a (container-slice, metadata) pair that feeds one
independent analysis run (dim-reduction, decoding, connectivity, …).
The four supported modes are:

``flat``
    One unit per scope — the entire container is used as-is.
``sensor``
    One unit per sensor/channel.  For raw inputs the channel dimension is
    sliced; for descriptor inputs each sensor's columns are selected and
    flattened.
``family``
    One unit per descriptor feature family (descriptor inputs only).
``sensor_within_family``
    One unit per (sensor, family) combination (descriptor inputs only).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from coco_pipe.io.structures import (
    DataContainer,  # same package — absolute import kept for clarity
)

__all__ = ["iter_analysis_units"]


def iter_analysis_units(
    container: DataContainer,
    analysis_mode: str,
    input_mode: str,
    descriptor_families: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate analysis units from *container* according to *analysis_mode*.

    Each returned dict has the keys:

    - ``unit_type`` — ``"global"`` | ``"sensor"`` | ``"family"``
    - ``unit_name`` — human-readable identifier (e.g. ``"Fz"``, ``"band"``)
    - ``unit_key``  — filesystem-safe unique key (e.g. ``"band_Fz"``)
    - ``family``    — descriptor family name, or ``None``
    - ``container`` — the sliced/flattened :class:`DataContainer` for this unit

    The container's ``meta`` dict is also updated in-place with the same
    four fields so that downstream functions can read the unit context from
    the container without needing to carry the dict separately.

    Parameters
    ----------
    container:
        Full data container for one analysis scope/condition.
    analysis_mode:
        One of ``"flat"``, ``"sensor"``, ``"family"``,
        ``"sensor_within_family"``.
    input_mode:
        ``"raw"`` or ``"descriptors"``.  Controls which container dimension is
        sliced for the ``"sensor"`` mode.
    descriptor_families:
        Ordered list of feature families to include.  ``None`` means all
        families present in the container.

    Returns
    -------
    list of dict
        One entry per analysis unit in the order they should be processed.

    Raises
    ------
    ValueError
        On unsupported *analysis_mode* / *input_mode* combinations, or when
        no features match the requested families.
    RuntimeError
        When no features match in sensor mode.
    """
    units: list[dict[str, Any]] = []

    def _add_unit(
        u_type: str,
        u_name: str,
        u_key: str,
        u_family: str | None,
        u_container: DataContainer,
    ) -> None:
        u_container.meta = {
            **dict(u_container.meta),
            "unit_type": u_type,
            "unit_name": u_name,
            "unit_key": u_key,
            "family": u_family,
        }
        units.append(
            {
                "unit_type": u_type,
                "unit_name": u_name,
                "unit_key": u_key,
                "family": u_family,
                "container": u_container,
            }
        )

    if analysis_mode == "flat":
        _add_unit("global", "all", "all", None, container)
        return units

    if analysis_mode == "sensor":
        if input_mode == "raw":
            for idx, channel_name in enumerate(
                np.asarray(container.coords["channel"], dtype=object)
            ):
                _add_unit(
                    "sensor",
                    str(channel_name),
                    str(channel_name),
                    None,
                    container.isel(channel=idx).flatten(preserve="obs"),
                )
            return units

        # descriptor sensor mode: select allowed families then iterate sensors
        feature_families = np.asarray(container.coords["feature_family"], dtype=object)
        allowed_families = set(descriptor_families or feature_families.tolist())
        feature_mask = np.isin(feature_families.astype(str), list(allowed_families))
        if not feature_mask.any():
            raise RuntimeError("No descriptor features matched the requested families.")
        feature_indices = np.flatnonzero(feature_mask).tolist()
        for idx, sensor_name in enumerate(
            np.asarray(container.coords["sensor"], dtype=object)
        ):
            _add_unit(
                "sensor",
                str(sensor_name),
                str(sensor_name),
                None,
                container.isel(sensor=idx, feature=feature_indices).flatten(
                    preserve="obs"
                ),
            )
        return units

    if input_mode != "descriptors":
        raise ValueError(
            f"analysis_mode='{analysis_mode}' is only supported for descriptor inputs."
        )

    sensor_names = np.asarray(container.coords["sensor"], dtype=object).astype(str)
    feature_families = np.asarray(
        container.coords["feature_family"], dtype=object
    ).astype(str)
    wanted_families = list(
        dict.fromkeys(descriptor_families or feature_families.tolist())
    )

    if analysis_mode == "family":
        for family in wanted_families:
            feature_indices = np.flatnonzero(feature_families == family).tolist()
            if not feature_indices:
                continue
            _add_unit(
                "family",
                family,
                family,
                family,
                container.isel(feature=feature_indices).flatten(preserve="obs"),
            )
        return units

    if analysis_mode == "sensor_within_family":
        for family in wanted_families:
            feature_indices = np.flatnonzero(feature_families == family).tolist()
            if not feature_indices:
                continue
            for idx, sensor_name in enumerate(sensor_names):
                _add_unit(
                    "sensor",
                    str(sensor_name),
                    f"{family}_{sensor_name}",
                    family,
                    container.isel(sensor=idx, feature=feature_indices).flatten(
                        preserve="obs"
                    ),
                )
        return units

    raise ValueError(f"Unsupported analysis_mode '{analysis_mode}'.")
