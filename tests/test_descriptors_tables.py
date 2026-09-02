"""Tests for the descriptor table-assembly core (descriptors.tables)."""

import numpy as np
import pandas as pd
import pytest

from coco_pipe.descriptors.tables import (
    add_aggregated_band_ratios,
    build_descriptor_tables,
    mad_failures_from_qc,
)
from coco_pipe.io import DataContainer
from coco_pipe.io.quality import drop_epoch_outliers


def _container(X, names, ids=None):
    X = np.asarray(X, dtype=float)
    return DataContainer(
        X=X,
        dims=("obs", "feature"),
        coords={"feature": np.asarray(names, dtype=object)},
        ids=np.asarray(ids if ids is not None else [f"o{i}" for i in range(len(X))]),
    )


def _metadata(n, recordings):
    return pd.DataFrame(
        {
            "obs_id": [f"o{i}" for i in range(n)],
            "recording_id": recordings,
            "subject": ["s1"] * n,
            "target": ["case"] * n,
        }
    )


# --------------------------------------------------------------------------- #
# mad_failures_from_qc  (reject = drop_epoch_outliers + this helper)
# --------------------------------------------------------------------------- #
def test_mad_failures_from_qc_records_dropped_epochs():
    X = np.ones((6, 2))
    X[0, 0] = 1000.0  # strong outlier
    container = _container(X, ["band_abs_alpha_ch-Fz", "band_abs_beta_ch-Fz"])

    clean, qc = drop_epoch_outliers(container, outlier_fraction_threshold=0.0)
    failures = mad_failures_from_qc(qc)

    assert clean.X.shape[0] == 5
    assert len(failures) == 1
    assert failures[0]["obs_id"] == "o0"
    assert failures[0]["family"] == "MAD_Rejection"
    assert failures[0]["exception_type"] == "MADOutlierError"


def test_mad_failures_from_qc_none_and_no_drops():
    assert mad_failures_from_qc(None) == []
    clean_container = _container(np.ones((4, 2)), ["a", "b"])
    _, qc = drop_epoch_outliers(clean_container, outlier_fraction_threshold=0.5)
    assert mad_failures_from_qc(qc) == []


# --------------------------------------------------------------------------- #
# add_aggregated_band_ratios
# --------------------------------------------------------------------------- #
def test_add_aggregated_band_ratios():
    base = pd.DataFrame(
        {
            "band_abs_alpha_ch-Fz": [2.0, 4.0],
            "band_abs_beta_ch-Fz": [1.0, 2.0],
        }
    )
    ratios = add_aggregated_band_ratios(base, [("alpha", "beta")])
    assert ratios["agg_band_ratio_alpha_beta_ch-Fz"].tolist() == [2.0, 2.0]


def test_add_aggregated_band_ratios_floor_and_no_match():
    base = pd.DataFrame({"band_abs_alpha_ch-Fz": [2.0], "band_abs_beta_ch-Fz": [0.0]})
    ratios = add_aggregated_band_ratios(base, [("alpha", "beta")], floor=0.0)
    assert np.isnan(ratios["agg_band_ratio_alpha_beta_ch-Fz"].iloc[0])
    # No matching numerator -> empty frame.
    assert add_aggregated_band_ratios(base, [("gamma", "beta")]).empty
    # Numerator present but denominator column absent -> skipped (empty frame).
    only_num = pd.DataFrame({"band_abs_alpha_ch-Fz": [1.0]})
    assert add_aggregated_band_ratios(only_num, [("alpha", "beta")]).empty


# --------------------------------------------------------------------------- #
# build_descriptor_tables (container-in)
# --------------------------------------------------------------------------- #
def test_build_descriptor_tables_epoch_and_subject():
    names = ["band_abs_alpha_ch-Fz", "band_abs_beta_ch-Fz"]
    X = [[2.0, 1.0], [4.0, 3.0], [10.0, 5.0], [20.0, 5.0]]
    container = _container(X, names)
    metadata = _metadata(4, ["A", "A", "B", "B"])

    tables = build_descriptor_tables(
        container,
        metadata,
        group_by="recording_id",
        target_col="target",
        ratio_pairs=[("alpha", "beta")],
    )

    # Epoch table = metadata + descriptor matrix.
    assert len(tables["epoch_df"]) == 4
    assert tables["epoch_feature_columns"] == names
    assert names[0] in tables["epoch_df"].columns

    # Subject table = one row per recording, mean-prefixed features + ratio.
    subject = tables["subject_df"]
    assert len(subject) == 2
    assert "mean_band_abs_alpha_ch-Fz" in subject.columns
    assert "agg_band_ratio_alpha_beta_ch-Fz" in subject.columns
    assert "agg_band_ratio_alpha_beta_ch-Fz" in tables["subject_feature_columns"]
    # Recording A alpha mean = (2+4)/2 = 3; recording B = (10+20)/2 = 15.
    np.testing.assert_allclose(
        sorted(subject["mean_band_abs_alpha_ch-Fz"]), [3.0, 15.0]
    )


def test_build_descriptor_tables_validates_columns():
    container = _container([[1.0]], ["band_abs_alpha_ch-Fz"], ids=["o0"])
    meta = pd.DataFrame({"obs_id": ["o0"], "recording_id": ["A"]})
    with pytest.raises(ValueError, match="id column"):
        build_descriptor_tables(
            container, meta, group_by="recording_id", id_col="missing"
        )
    with pytest.raises(ValueError, match="group column"):
        build_descriptor_tables(container, meta, group_by="nope")
