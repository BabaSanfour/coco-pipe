import numpy as np
import pytest

from coco_pipe.io import DataContainer, iter_analysis_units, split_unit_sensor


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


def _subfamily_container():
    return DataContainer(
        X=np.arange(5 * 2 * 3).reshape(5, 2, 3),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz", "Cz"],
            "feature": ["log_abs_alpha", "rel_alpha", "sample_entropy"],
            "feature_family": ["band", "band", "complexity"],
            "feature_subfamily": ["log_abs", "rel", "entropy"],
        },
    )


def test_iter_analysis_units_subfamily_mode():
    units = iter_analysis_units(_subfamily_container(), "subfamily", "descriptors")
    # band splits into log_abs + rel; complexity contributes entropy
    assert [unit["unit_name"] for unit in units] == ["log_abs", "rel", "entropy"]
    assert [unit["family"] for unit in units] == ["band", "band", "complexity"]
    assert [unit["unit_type"] for unit in units] == ["subfamily"] * 3
    assert all(unit["container"].X.shape == (5, 2) for unit in units)


def test_iter_analysis_units_subfamily_requires_coord():
    container = _descriptor_container()  # no feature_subfamily coord
    with pytest.raises(ValueError, match="feature_subfamily"):
        iter_analysis_units(container, "subfamily", "descriptors")


def test_iter_analysis_units_sensor_within_subfamily():
    units = iter_analysis_units(
        _subfamily_container(), "sensor_within_subfamily", "descriptors"
    )
    # 3 sub-families x 2 sensors
    assert len(units) == 6
    assert {unit["subfamily"] for unit in units} == {"log_abs", "rel", "entropy"}
    assert [unit["unit_type"] for unit in units] == ["sensor"] * 6
    assert "log_abs_Fz" in {unit["unit_key"] for unit in units}
    assert all(unit["container"].X.shape == (5, 1) for unit in units)


def _descriptor_stat_container():
    # Two stat columns of one descriptor (mean + iqr of log_abs_alpha) plus one
    # other descriptor, at a single sensor.
    return DataContainer(
        X=np.arange(5 * 1 * 3).reshape(5, 1, 3),
        dims=("obs", "sensor", "feature"),
        coords={
            "sensor": ["Fz"],
            "feature": [
                "mean_log_abs_alpha",
                "iqr_log_abs_alpha",
                "mean_sample_entropy",
            ],
            "feature_family": ["band", "band", "complexity"],
            "feature_descriptor": ["log_abs_alpha", "log_abs_alpha", "sample_entropy"],
        },
    )


def test_iter_analysis_units_descriptor_combines_stats():
    units = iter_analysis_units(
        _descriptor_stat_container(), "descriptor", "descriptors"
    )
    # 2 descriptors; log_abs_alpha keeps mean+iqr together (2 columns)
    assert [unit["unit_name"] for unit in units] == ["log_abs_alpha", "sample_entropy"]
    assert [unit["unit_type"] for unit in units] == ["descriptor", "descriptor"]
    assert units[0]["container"].X.shape == (5, 2)  # mean + iqr together
    assert units[1]["container"].X.shape == (5, 1)


def test_iter_analysis_units_descriptor_sensor():
    units = iter_analysis_units(
        _descriptor_stat_container(), "descriptor_sensor", "descriptors"
    )
    assert {unit["unit_key"] for unit in units} == {
        "log_abs_alpha_Fz",
        "sample_entropy_Fz",
    }
    assert units[0]["container"].X.shape == (5, 2)  # alpha mean+iqr at Fz, one unit


def test_split_unit_sensor_uses_descriptor_sensor_key_contract():
    assert split_unit_sensor("log_abs_alpha", "log_abs_alpha_Fz") == "Fz"
    assert split_unit_sensor("log_abs_alpha", "other_Fz") is None
    assert split_unit_sensor("log_abs_alpha", "log_abs_alpha_") is None


def test_iter_analysis_units_descriptor_requires_coord():
    with pytest.raises(ValueError, match="feature_descriptor"):
        iter_analysis_units(_descriptor_container(), "descriptor", "descriptors")
