import numpy as np
import pytest

from coco_pipe.io import DataContainer, iter_analysis_units, split_unit_sensor


def _sparse_flat_descriptor_container():
    """Two sensors with deliberately non-rectangular descriptor coverage."""
    return DataContainer(
        X=np.arange(5 * 3, dtype=float).reshape(5, 3),
        dims=("obs", "feature"),
        coords={
            "feature": [
                "band_mean_alpha_ch-Fz",
                "band_iqr_alpha_ch-Fz",
                "complexity_mean_entropy_ch-Cz",
            ],
            "feature_family": ["band", "band", "complexity"],
            "feature_measure": ["mean_alpha", "iqr_alpha", "mean_entropy"],
            "feature_channel": ["Fz", "Fz", "Cz"],
            "feature_scope": ["sensor", "sensor", "sensor"],
            "feature_subfamily": ["abs", "abs", "entropy"],
            "feature_descriptor": ["alpha", "alpha", "entropy"],
        },
    )


@pytest.mark.parametrize(
    ("mode", "expected_keys"),
    [
        ("sensor", {"Fz", "Cz"}),
        ("family", {"band", "complexity"}),
        ("subfamily", {"abs", "entropy"}),
        ("sensor_within_family", {"band_Fz", "complexity_Cz"}),
        ("sensor_within_subfamily", {"abs_Fz", "entropy_Cz"}),
        ("feature", {"mean_alpha", "iqr_alpha", "mean_entropy"}),
        (
            "feature_within_family",
            {"band_mean_alpha", "band_iqr_alpha", "complexity_mean_entropy"},
        ),
        ("descriptor", {"alpha", "entropy"}),
        ("descriptor_sensor", {"alpha_Fz", "entropy_Cz"}),
    ],
)
def test_flat_descriptor_units_use_only_existing_atomic_columns(mode, expected_keys):
    units = iter_analysis_units(
        _sparse_flat_descriptor_container(), mode, "descriptors"
    )

    assert {unit["unit_key"] for unit in units} == expected_keys
    assert all(unit["container"].dims == ("obs", "feature") for unit in units)
    assert all(np.isfinite(unit["container"].X).all() for unit in units)


def test_flat_descriptor_units_respect_family_filter():
    units = iter_analysis_units(
        _sparse_flat_descriptor_container(),
        "sensor",
        "descriptors",
        descriptor_families=["band"],
    )

    assert [unit["unit_key"] for unit in units] == ["Fz"]
    assert units[0]["container"].X.shape == (5, 2)


def test_flat_descriptor_sensor_does_not_create_cartesian_units():
    units = iter_analysis_units(
        _sparse_flat_descriptor_container(), "descriptor_sensor", "descriptors"
    )

    keys = {unit["unit_key"] for unit in units}
    assert keys == {"alpha_Fz", "entropy_Cz"}
    assert "alpha_Cz" not in keys
    assert "entropy_Fz" not in keys


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
            _sparse_flat_descriptor_container(),
            "unknown_mode",
            "descriptors",
            {},
            ValueError,
            id="unknown_mode",
        ),
        pytest.param(
            _sparse_flat_descriptor_container(),
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


def test_split_unit_sensor_uses_descriptor_sensor_key_contract():
    assert split_unit_sensor("log_abs_alpha", "log_abs_alpha_Fz") == "Fz"
    assert split_unit_sensor("log_abs_alpha", "other_Fz") is None
    assert split_unit_sensor("log_abs_alpha", "log_abs_alpha_") is None


@pytest.mark.parametrize("analysis_mode", ["flat", "sensor"])
def test_iter_analysis_units_rejects_descriptor_tensor_layout(analysis_mode):
    container = DataContainer(
        X=np.zeros((2, 1, 1)),
        dims=("obs", "sensor", "feature"),
        coords={"sensor": ["Fz"], "feature": ["alpha"]},
    )
    with pytest.raises(ValueError, match="requires a flat"):
        iter_analysis_units(container, analysis_mode, "descriptors")
