"""
Analysis-unit enumeration for multi-mode EEG pipelines.

An *analysis unit* is a (container-slice, metadata) pair that feeds one
independent analysis run (dim-reduction, decoding, connectivity, …).
The supported modes are:

``flat``
    One unit per scope — the entire container is used as-is.
``sensor``
    One unit per sensor/channel.  For raw inputs the channel dimension is
    sliced; for descriptor inputs each sensor's columns are selected and
    flattened.
``family``
    One unit per descriptor feature family (descriptor inputs only).
``subfamily``
    One unit per descriptor sub-family — the output type within a family, e.g.
    band ``log_abs`` / ``rel`` or param ``aperiodic`` / ``peaks`` (descriptor
    inputs only; requires a ``feature_subfamily`` coord).
``sensor_within_family``
    One unit per (sensor, family) combination (descriptor inputs only).
``sensor_within_subfamily``
    One unit per (sensor, sub-family) combination (descriptor inputs only;
    requires a ``feature_subfamily`` coord).
``feature``
    One unit per descriptor feature across all sensors.
``feature_within_family``
    One unit per (feature, family) combination across all sensors.
``descriptor``
    One unit per descriptor across all sensors — all aggregation-stat columns of
    a descriptor (e.g. mean + iqr of alpha) kept together (requires a
    ``feature_descriptor`` coord).
``descriptor_sensor``
    One unit per (descriptor, sensor) — one descriptor's stats at a single
    sensor (requires a ``feature_descriptor`` coord).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .structures import DataContainer

__all__ = [
    "iter_analysis_units",
    "split_unit_sensor",
]


def split_unit_sensor(unit_name: str, unit_key: str) -> str | None:
    """Return the sensor suffix from a ``{unit_name}_{sensor}`` unit key.

    ``descriptor_sensor`` units use ``unit_name`` for the descriptor and
    ``unit_key`` for the descriptor/sensor pair. Returns ``None`` when the key
    does not follow that explicit convention.
    """
    prefix = f"{unit_name}_"
    key = str(unit_key)
    if not key.startswith(prefix):
        return None
    sensor = key[len(prefix) :]
    return sensor or None


def iter_analysis_units(
    container: DataContainer,
    analysis_mode: str,
    input_mode: str,
    descriptor_families: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate analysis units from *container* according to *analysis_mode*.

    Each returned dict has the keys:

    - ``unit_type`` — one of ``"global"``, ``"sensor"``, ``"family"``,
      ``"subfamily"``, ``"feature"``, or ``"descriptor"``
    - ``unit_name`` — human-readable identifier (e.g. ``"Fz"``, ``"band"``)
    - ``unit_key``  — filesystem-safe unique key (e.g. ``"band_Fz"``)
    - ``family``    — descriptor family name, or ``None``
    - ``subfamily`` — descriptor sub-family name, or ``None``
    - ``container`` — the sliced/flattened
      :class:`~coco_pipe.io.DataContainer` for this unit

    The container's ``meta`` dict is also updated in-place with the
    ``unit_type``/``unit_name``/``unit_key``/``family``/``subfamily`` fields so
    that downstream functions can read the unit context from the container
    without needing to carry the dict separately.

    Parameters
    ----------
    container:
        Full data container for one analysis scope/condition.
    analysis_mode:
        Descriptor inputs may use either the canonical flat
        ``("obs", "feature")`` layout, where sensor and descriptor identity
        come from ``feature_schema()``, or the legacy rectangular
        ``("obs", "sensor", "feature")`` layout.
        The unit granularity. ``"flat"`` and ``"sensor"`` work for both input
        modes; the remaining modes require descriptor inputs:

        - ``"flat"`` — one unit for the whole container.
        - ``"sensor"`` — one unit per sensor/channel.
        - ``"family"`` — one unit per descriptor feature family.
        - ``"subfamily"`` — one unit per sub-family (needs a
          ``feature_subfamily`` coord).
        - ``"sensor_within_family"`` — one unit per (sensor, family).
        - ``"sensor_within_subfamily"`` — one unit per (sensor, sub-family)
          (needs a ``feature_subfamily`` coord).
        - ``"feature"`` — one unit per feature across all sensors.
        - ``"feature_within_family"`` — one unit per (feature, family).
        - ``"descriptor"`` — one unit per descriptor across all sensors
          (needs a ``feature_descriptor`` coord).
        - ``"descriptor_sensor"`` — one unit per (descriptor, sensor)
          (needs a ``feature_descriptor`` coord).
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
        u_subfamily: str | None = None,
    ) -> None:
        u_container.meta = {
            **dict(u_container.meta),
            "unit_type": u_type,
            "unit_name": u_name,
            "unit_key": u_key,
            "family": u_family,
            "subfamily": u_subfamily,
        }
        units.append(
            {
                "unit_type": u_type,
                "unit_name": u_name,
                "unit_key": u_key,
                "family": u_family,
                "subfamily": u_subfamily,
                "container": u_container,
            }
        )

    if analysis_mode == "flat":
        _add_unit("global", "all", "all", None, container)
        return units

    if input_mode == "descriptors" and container.dims == ("obs", "feature"):
        schema = container.feature_schema()
        if schema is None or len(schema) != container.X.shape[1]:
            raise ValueError(
                "Flat descriptor analysis requires feature metadata aligned "
                "with the feature axis."
            )

        def _schema_values(column: str) -> np.ndarray:
            if column not in schema:
                raise ValueError(
                    f"analysis_mode={analysis_mode!r} requires {column!r} "
                    "in the descriptor feature schema."
                )
            return schema[column].fillna("unknown").astype(str).to_numpy()

        families = _schema_values("family")
        wanted_families = list(dict.fromkeys(descriptor_families or families.tolist()))
        family_mask = np.isin(families, [str(value) for value in wanted_families])

        def _add_masked_unit(
            unit_type: str,
            unit_name: str,
            unit_key: str,
            family: str | None,
            mask: np.ndarray,
            *,
            subfamily: str | None = None,
        ) -> None:
            indices = np.flatnonzero(mask).tolist()
            if indices:
                _add_unit(
                    unit_type,
                    unit_name,
                    unit_key,
                    family,
                    container.isel(feature=indices),
                    u_subfamily=subfamily,
                )

        if analysis_mode == "sensor":
            channels = _schema_values("channel")
            for sensor in dict.fromkeys(channels[family_mask].tolist()):
                _add_masked_unit(
                    "sensor",
                    sensor,
                    sensor,
                    None,
                    family_mask & (channels == sensor),
                )
            if not units:
                raise RuntimeError(
                    "No descriptor features matched the requested families."
                )
            return units

        if analysis_mode == "family":
            for family in wanted_families:
                _add_masked_unit(
                    "family",
                    str(family),
                    str(family),
                    str(family),
                    families == str(family),
                )
            return units

        if analysis_mode in {"subfamily", "sensor_within_subfamily"}:
            subfamilies = _schema_values("subfamily")
            channels = (
                _schema_values("channel")
                if analysis_mode == "sensor_within_subfamily"
                else None
            )
            for subfamily in dict.fromkeys(subfamilies[family_mask].tolist()):
                subfamily_mask = family_mask & (subfamilies == subfamily)
                family = str(families[np.flatnonzero(subfamily_mask)[0]])
                if channels is None:
                    _add_masked_unit(
                        "subfamily",
                        subfamily,
                        subfamily,
                        family,
                        subfamily_mask,
                        subfamily=subfamily,
                    )
                else:
                    for sensor in dict.fromkeys(channels[subfamily_mask].tolist()):
                        _add_masked_unit(
                            "sensor",
                            sensor,
                            f"{subfamily}_{sensor}",
                            family,
                            subfamily_mask & (channels == sensor),
                            subfamily=subfamily,
                        )
            return units

        if analysis_mode in {"descriptor", "descriptor_sensor"}:
            descriptors = _schema_values("descriptor")
            channels = (
                _schema_values("channel")
                if analysis_mode == "descriptor_sensor"
                else None
            )
            for descriptor in dict.fromkeys(descriptors[family_mask].tolist()):
                descriptor_mask = family_mask & (descriptors == descriptor)
                family = str(families[np.flatnonzero(descriptor_mask)[0]])
                if channels is None:
                    _add_masked_unit(
                        "descriptor",
                        descriptor,
                        descriptor,
                        family,
                        descriptor_mask,
                    )
                else:
                    for sensor in dict.fromkeys(channels[descriptor_mask].tolist()):
                        _add_masked_unit(
                            "descriptor",
                            descriptor,
                            f"{descriptor}_{sensor}",
                            family,
                            descriptor_mask & (channels == sensor),
                        )
            return units

        if analysis_mode in {"feature", "feature_within_family"}:
            measures = _schema_values("measure")
            if analysis_mode == "feature":
                for measure in dict.fromkeys(measures[family_mask].tolist()):
                    _add_masked_unit(
                        "feature",
                        measure,
                        measure,
                        None,
                        family_mask & (measures == measure),
                    )
            else:
                for family in wanted_families:
                    current_family = families == str(family)
                    for measure in dict.fromkeys(measures[current_family].tolist()):
                        _add_masked_unit(
                            "feature",
                            measure,
                            f"{family}_{measure}",
                            str(family),
                            current_family & (measures == measure),
                        )
            return units

        if analysis_mode == "sensor_within_family":
            channels = _schema_values("channel")
            for family in wanted_families:
                current_family = families == str(family)
                for sensor in dict.fromkeys(channels[current_family].tolist()):
                    _add_masked_unit(
                        "sensor",
                        sensor,
                        f"{family}_{sensor}",
                        str(family),
                        current_family & (channels == sensor),
                    )
            return units

        raise ValueError(f"Unknown analysis_mode '{analysis_mode}'.")

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
    feature_names = np.asarray(container.coords["feature"], dtype=object).astype(str)
    wanted_families = list(
        dict.fromkeys(descriptor_families or feature_families.tolist())
    )

    if analysis_mode in {"subfamily", "sensor_within_subfamily"}:
        if "feature_subfamily" not in container.coords:
            raise ValueError(
                f"analysis_mode='{analysis_mode}' requires a 'feature_subfamily' coord."
            )
        feature_subfamilies = np.asarray(
            container.coords["feature_subfamily"], dtype=object
        ).astype(str)
        family_mask = (
            np.isin(feature_families, [str(f) for f in descriptor_families])
            if descriptor_families
            else np.ones(feature_families.shape, dtype=bool)
        )
        for subfamily in dict.fromkeys(feature_subfamilies[family_mask].tolist()):
            feature_indices = np.flatnonzero(
                family_mask & (feature_subfamilies == subfamily)
            ).tolist()
            if not feature_indices:
                continue
            family = str(feature_families[feature_indices[0]])
            if analysis_mode == "subfamily":
                _add_unit(
                    "subfamily",
                    subfamily,
                    subfamily,
                    family,
                    container.isel(feature=feature_indices).flatten(preserve="obs"),
                    u_subfamily=subfamily,
                )
            else:  # sensor_within_subfamily
                for idx, sensor_name in enumerate(sensor_names):
                    _add_unit(
                        "sensor",
                        str(sensor_name),
                        f"{subfamily}_{sensor_name}",
                        family,
                        container.isel(sensor=idx, feature=feature_indices).flatten(
                            preserve="obs"
                        ),
                        u_subfamily=subfamily,
                    )
        return units

    if analysis_mode in {"descriptor", "descriptor_sensor"}:
        if "feature_descriptor" not in container.coords:
            raise ValueError(
                f"analysis_mode='{analysis_mode}' requires a "
                "'feature_descriptor' coord."
            )
        feature_descriptors = np.asarray(
            container.coords["feature_descriptor"], dtype=object
        ).astype(str)
        for descriptor in dict.fromkeys(feature_descriptors.tolist()):
            feature_indices = np.flatnonzero(feature_descriptors == descriptor).tolist()
            if not feature_indices:
                continue
            family = str(feature_families[feature_indices[0]])
            if analysis_mode == "descriptor":
                _add_unit(
                    "descriptor",
                    descriptor,
                    descriptor,
                    family,
                    container.isel(feature=feature_indices).flatten(preserve="obs"),
                )
            else:  # descriptor_sensor — one descriptor (all its stats) at one sensor
                for idx, sensor_name in enumerate(sensor_names):
                    _add_unit(
                        "descriptor",
                        descriptor,
                        f"{descriptor}_{sensor_name}",
                        family,
                        container.isel(sensor=idx, feature=feature_indices).flatten(
                            preserve="obs"
                        ),
                    )
        return units

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

    if analysis_mode == "feature":
        for feature_name in dict.fromkeys(feature_names.tolist()):
            feature_indices = np.flatnonzero(feature_names == feature_name).tolist()
            if not feature_indices:
                continue
            _add_unit(
                "feature",
                feature_name,
                feature_name,
                None,
                container.isel(feature=feature_indices).flatten(preserve="obs"),
            )
        return units

    if analysis_mode == "feature_within_family":
        for family in wanted_families:
            family_indices = np.flatnonzero(feature_families == family)
            for feature_name in dict.fromkeys(feature_names[family_indices].tolist()):
                feature_indices = np.flatnonzero(
                    (feature_families == family) & (feature_names == feature_name)
                ).tolist()
                if not feature_indices:
                    continue
                _add_unit(
                    "feature",
                    feature_name,
                    f"{family}_{feature_name}",
                    family,
                    container.isel(feature=feature_indices).flatten(preserve="obs"),
                )
        return units

    raise ValueError(f"Unsupported analysis_mode '{analysis_mode}'.")
