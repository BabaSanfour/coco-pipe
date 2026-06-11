import numpy as np
import pytest

from coco_pipe.io import DataContainer, iter_analysis_units


def _descriptor_container():
    return DataContainer(
        X=np.arange(5 * 2 * 3).reshape(5, 2, 3),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz", "Cz"],
            "feature": ["theta", "beta", "exponent"],
            "feature_family": ["band", "band", "parametric"],
        },
    )


def _modes_container():
    return DataContainer(
        np.zeros((2, 2, 3)),
        dims=["obs", "sensor", "feature"],
        coords={
            "sensor": ["S1", "S2"],
            "feature_family": ["famA", "famA", "famB"],
            "feature": ["f1", "f2", "f3"],
        },
    )


def test_feature_units_span_all_sensors():
    units = iter_analysis_units(_descriptor_container(), "feature", "descriptors")
    assert [unit["unit_name"] for unit in units] == ["theta", "beta", "exponent"]
    assert all(unit["container"].X.shape == (5, 2) for unit in units)


def test_feature_within_family_units_include_family_key():
    units = iter_analysis_units(
        _descriptor_container(),
        "feature_within_family",
        "descriptors",
    )
    assert [unit["unit_key"] for unit in units] == [
        "band_theta",
        "band_beta",
        "parametric_exponent",
    ]


@pytest.mark.parametrize(
    "mode, expected_count",
    [
        ("sensor", 2),
        ("family", 2),
        ("sensor_within_family", 4),
        ("feature", 3),
        ("feature_within_family", 3),
    ],
)
def test_iter_analysis_units_descriptor_modes(mode, expected_count):
    units = iter_analysis_units(_modes_container(), mode, "descriptors")
    assert len(units) == expected_count


def test_iter_analysis_units_raw_sensor():
    dc = DataContainer(
        np.zeros((2, 3, 4)),
        dims=["obs", "channel", "time"],
        coords={"channel": ["C1", "C2", "C3"]},
    )
    units = iter_analysis_units(dc, "sensor", "raw")
    assert len(units) == 3
    assert units[0]["unit_name"] == "C1"


def test_iter_analysis_units_flat():
    dc = DataContainer(np.zeros((2,)), dims=["obs"])
    units = iter_analysis_units(dc, "flat", "raw")
    assert len(units) == 1
    assert units[0]["unit_name"] == "all"


@pytest.mark.parametrize(
    "container, mode, mod, kwargs, exc",
    [
        pytest.param(
            DataContainer(np.zeros((2,)), dims=["obs"]),
            "family",
            "raw",
            {},
            ValueError,
            id="family_mode_on_raw",
        ),
        pytest.param(
            _modes_container(),
            "unknown_mode",
            "descriptors",
            {},
            ValueError,
            id="unknown_mode",
        ),
        pytest.param(
            _modes_container(),
            "sensor",
            "descriptors",
            {"descriptor_families": ["famC"]},
            RuntimeError,
            id="missing_descriptor_family",
        ),
    ],
)
def test_iter_analysis_units_errors(container, mode, mod, kwargs, exc):
    with pytest.raises(exc):
        iter_analysis_units(container, mode, mod, **kwargs)


@pytest.mark.parametrize(
    "mode, expected_count",
    [
        ("family", 1),
        ("sensor_within_family", 2),
        ("feature_within_family", 2),
    ],
)
def test_iter_analysis_units_skips_missing_families(mode, expected_count):
    # "famC" does not exist and is silently skipped (the `continue` branch).
    units = iter_analysis_units(
        _modes_container(), mode, "descriptors", descriptor_families=["famA", "famC"]
    )
    assert len(units) == expected_count
