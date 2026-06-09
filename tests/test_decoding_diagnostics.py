import numpy as np
import pandas as pd
import pytest

from coco_pipe.decoding._diagnostics import (
    confusion_matrix_frame,
    curve_score_groups,
    optional_values,
    paired_unit_indices,
    prediction_rows,
    proba_matrix,
    row_value,
    scalar_prediction_frame,
    score_frame,
    score_rows,
    time_value,
    unit_indices,
)


def test_time_value():
    assert time_value(0, [10, 20]) == 10
    assert time_value(5, [10, 20]) == 5
    assert time_value(0, None) == 0


def test_score_rows():
    assert len(score_rows("m", 0, "a", 0.5)) == 1
    assert len(score_rows("m", 0, "a", [0.5, 0.6])) == 2
    assert len(score_rows("m", 0, "a", [[0.5, 0.6], [0.7, 0.8]])) == 4
    assert len(score_rows("m", 0, "a", np.zeros((2, 2, 2)))) == 1


def test_prediction_rows_all_paths():
    # 1. Standard Binary Proba (1D)
    p_bin = {"y_true": [0, 1], "y_pred": [0, 1], "y_proba": np.array([0.2, 0.8])}
    r_bin = prediction_rows("m", 0, p_bin)
    assert r_bin[0]["y_proba_0"] == 0.8
    assert r_bin[0]["y_proba_1"] == 0.2

    # 2. Standard Multiclass Proba (2D)
    p_multi = {
        "y_true": [0, 1],
        "y_pred": [0, 1],
        "y_proba": np.array([[0.8, 0.2], [0.1, 0.9]]),
    }
    r_multi = prediction_rows("m", 0, p_multi)
    assert r_multi[0]["y_proba_0"] == 0.8

    # 3. Sliding Proba (3D)
    p_sl = {
        "y_true": [0, 1],
        "y_pred": np.zeros((2, 2)),
        "y_proba": np.zeros((2, 2, 2)),
    }
    r_sl = prediction_rows("m", 0, p_sl)
    assert "y_proba_0" in r_sl[0]

    # 4. Generalizing Proba (4D)
    p_gen = {
        "y_true": [0, 1],
        "y_pred": np.zeros((2, 2, 2)),
        "y_proba": np.zeros((2, 2, 2, 2)),
    }
    r_gen = prediction_rows("m", 0, p_gen)
    assert "y_proba_0" in r_gen[0]

    # 5. Standard Binary Score (1D)
    p_s1 = {"y_true": [0, 1], "y_pred": [0, 1], "y_score": np.array([0.5, 0.8])}
    r_s1 = prediction_rows("m", 0, p_s1)
    assert r_s1[0]["y_score"] == 0.5

    # 6. Standard Multiclass Score (2D)
    p_sm = {
        "y_true": [0, 1],
        "y_pred": [0, 1],
        "y_score": np.array([[1.0, -1.0], [-1.0, 1.0]]),
    }
    r_sm = prediction_rows("m", 0, p_sm)
    assert r_sm[0]["y_score_0"] == 1.0

    # 7. Metadata and Groups
    p_meta = {
        "y_true": [0],
        "y_pred": [0],
        "group": [1],
        "sample_metadata": {"m": [10]},
    }
    r_meta = prediction_rows("m", 0, p_meta)
    assert r_meta[0]["m"] == 10
    assert r_meta[0]["Group"] == 1


def test_row_value_ndarray():
    assert row_value(np.array([np.array([1])], dtype=object), 0) == [1]


def test_optional_values():
    assert optional_values(None, 2).tolist() == [None, None]
    assert optional_values([1, 2], 2).tolist() == [1, 2]


def test_proba_matrix():
    assert proba_matrix(pd.DataFrame({"y_proba_0": [0.8]}), 2) is None
    assert (
        proba_matrix(
            pd.DataFrame({"y_proba_0": [0.8, np.nan], "y_proba_1": [0.2, 0.8]}), 2
        )
        is None
    )
    assert (
        proba_matrix(pd.DataFrame({"y_proba_0": [0.8], "y_proba_1": [0.2]}), 2)
        is not None
    )


def test_unit_indices():
    df = pd.DataFrame(
        {
            "SampleID": [1, 2],
            "Group": [1, 2],
            "Subject": [1, 2],
            "Session": [1, 2],
            "Site": [1, 2],
        }
    )
    for u in ["sample", "epoch", "group", "subject", "session", "site"]:
        assert len(unit_indices(df, u)) == 2
    with pytest.raises(ValueError):
        unit_indices(df, "unknown")
    df_err = pd.DataFrame({"SampleID": [1], "Site": [np.nan]})
    with pytest.raises(ValueError):
        unit_indices(df_err, "site")


def test_paired_unit_indices():
    df = pd.DataFrame(
        {
            "SampleID": [1, 1],
            "Group_A": [1, 2],
            "Group_B": [1, 2],
            "Subject_A": [1, 2],
            "Subject_B": [1, 2],
        }
    )
    assert len(paired_unit_indices(df, "group")) == 2
    assert len(paired_unit_indices(df, "subject")) == 2


def test_score_frame_all():
    df_l = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]})
    assert score_frame(df_l, "accuracy") == 1.0
    df_p = pd.DataFrame(
        {"y_true": [0, 1], "y_proba_0": [0.8, 0.2], "y_proba_1": [0.2, 0.8]}
    )
    assert score_frame(df_p, "roc_auc") == 1.0
    assert score_frame(df_p, "brier_score") < 1.0
    df_s = pd.DataFrame({"y_true": [0, 1], "y_score": [0.5, 0.8]})
    assert score_frame(df_s, "roc_auc") == 1.0
    df_m = pd.DataFrame(
        {
            "y_true": [0, 1, 2],
            "y_proba_0": [0.8, 0.1, 0.1],
            "y_proba_1": [0.1, 0.8, 0.1],
            "y_proba_2": [0.1, 0.1, 0.8],
        }
    )
    assert score_frame(df_m, "roc_auc") == 1.0
    assert score_frame(df_m, "average_precision") == 1.0
    with pytest.raises(ValueError):
        score_frame(df_l, "roc_auc")


def test_scalar_prediction_frame():
    df = pd.DataFrame({"Time": [0.1, np.nan]})
    assert len(scalar_prediction_frame(df)) == 1
    assert scalar_prediction_frame(pd.DataFrame()).empty


def test_confusion_matrix_frame():
    df = pd.DataFrame(
        {"Model": ["m", "m"], "Fold": [0, 0], "y_true": [0, 1], "y_pred": [0, 1]}
    )
    assert not confusion_matrix_frame(df, [0, 1]).empty
    assert confusion_matrix_frame(pd.DataFrame(columns=["Model", "Fold"]), [0, 1]).empty
    assert "Model" in confusion_matrix_frame(df, [0, 1], group_cols=["Model"]).columns


def test_curve_score_groups():
    df = pd.DataFrame(
        {
            "Model": ["m", "m"],
            "Fold": [0, 0],
            "y_true": [0, 1],
            "y_proba_0": [0.5, 0.5],
            "y_proba_1": [0.5, 0.5],
        }
    )
    assert len(list(curve_score_groups(df))) == 1
    assert len(list(curve_score_groups(df, model="m2"))) == 0
    df_s0 = pd.DataFrame(
        {"Model": ["m", "m"], "Fold": [0, 0], "y_true": [0, 1], "y_score": [0.5, 0.8]}
    )
    assert len(list(curve_score_groups(df_s0, pos_label=0))) == 1
    df_m = pd.DataFrame(
        {
            "Model": ["m"] * 3,
            "Fold": [0] * 3,
            "y_true": [0, 1, 2],
            "y_proba_0": [0.8, 0.1, 0.1],
            "y_proba_1": [0.1, 0.8, 0.1],
            "y_proba_2": [0.1, 0.1, 0.8],
        }
    )
    assert len(list(curve_score_groups(df_m))) == 3


def test_paired_unit_indices_exhaustive():
    df = pd.DataFrame(
        {
            "SampleID": [1],
            "Group_A": [1],
            "Subject_A": [1],
            "Session_A": [1],
            "Site_A": [1],
            "Group_B": [1],
            "Subject_B": [1],
            "Session_B": [1],
            "Site_B": [1],
        }
    )
    for u in ["sample", "epoch", "group", "subject", "session", "site"]:
        assert len(paired_unit_indices(df, u)) == 1


def test_score_frame_error_paths():
    df = pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1]})
    with pytest.raises(ValueError, match="cannot be scored"):
        score_frame(df, "roc_auc")

    # Brier score multiclass error
    df_m = pd.DataFrame(
        {
            "y_true": [0, 1, 2],
            "y_proba_0": [0.3] * 3,
            "y_proba_1": [0.3] * 3,
            "y_proba_2": [0.4] * 3,
        }
    )
    with pytest.raises(ValueError, match="binary classification only"):
        score_frame(df_m, "brier_score")


def test_prediction_rows_temporal_multiclass():
    # Sliding Multiclass
    p_sl = {
        "y_true": [0],
        "y_pred": np.zeros((1, 2)),
        "y_proba": np.zeros((1, 2, 3)),  # (samples, times, classes)
    }
    r_sl = prediction_rows("m", 0, p_sl)
    assert len(r_sl) == 2
    assert all(f"y_proba_{c}" in r_sl[0] for c in range(3))

    # Generalizing Multiclass
    p_gen = {
        "y_true": [0],
        "y_pred": np.zeros((1, 2, 2)),
        "y_proba": np.zeros((1, 2, 2, 3)),  # (samples, tr, te, classes)
    }
    r_gen = prediction_rows("m", 0, p_gen)
    assert len(r_gen) == 4
    assert all(f"y_proba_{c}" in r_gen[0] for c in range(3))
