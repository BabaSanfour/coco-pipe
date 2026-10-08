import numpy as np
import pandas as pd
import pytest

from coco_pipe.io import utils as utils_mod
from coco_pipe.io.utils import (
    compute_constant_feature_summary,
    compute_feature_missingness,
    load_participants_tsv,
    make_strata,
    row_quality_score,
    sample_indices,
)


def test_row_quality_score():
    """Test badness score calculation."""
    df = pd.DataFrame({"a": [1, np.nan, 0, np.inf], "b": [1, 1, 1, 1]})
    # Row 0: a=1, b=1 -> score 0
    # Row 1: a=NaN -> score 1
    # Row 2: a=0 -> score 1 (if count_zero=True)
    # Row 3: a=Inf -> score 1

    scores = row_quality_score(df, count_zero=True)
    assert np.array_equal(scores, [0, 1, 1, 1])

    scores_nz = row_quality_score(df, count_zero=False)
    assert np.array_equal(scores_nz, [0, 1, 0, 1])


def test_row_quality_score_normalized():
    df = pd.DataFrame(
        {
            "a": [0.0, np.nan, 1.0],
            "b": [1.0, np.inf, 2.0],
            "label": ["x", "y", "z"],
        },
        index=[10, 20, 30],
    )

    scores = row_quality_score(df, normalize=True)

    assert scores.index.tolist() == [10, 20, 30]
    assert scores.tolist() == [0.5, 1.0, 0.0]

    no_numeric = row_quality_score(
        df[["label"]],
        normalize=True,
    )
    assert no_numeric.tolist() == [0.0, 0.0, 0.0]
    assert no_numeric.dtype == float


def test_make_strata():
    """Test stratification label generation."""
    df = pd.DataFrame({"num": [1, 10, 100], "cat": ["a", "b", "a"]})
    # Numeric binning (3 bins) + cat
    # 1 -> bin0, 10 -> bin1, 100 -> bin2 (roughly)

    strata = utils_mod.make_strata(df, covariates=["num", "cat"], n_bins=3)
    assert len(strata) == 3
    assert len(strata.unique()) == 3  # All distinct combos


def test_sample_indices():
    """Test index sampling logic."""
    df = pd.DataFrame({"target": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
    rng = np.random.default_rng(42)
    size_map = {"A": 1, "B": 2}  # Downsample A, Keep B

    idx = utils_mod.sample_indices(
        df, "target", size_map, rng, replace=False, prefer_clean=True, exclude=[]
    )
    assert len(idx) == 3
    # Check coverage
    sampled_rows = df.loc[idx]
    assert sampled_rows["target"].value_counts()["A"] == 1
    assert sampled_rows["target"].value_counts()["B"] == 2


def test_split_column():
    """Test column splitting."""
    # Normal
    res = utils_mod.split_column("unit_feat", "_", False)
    assert res == ("unit", "feat")

    # Reverse
    res_rev = utils_mod.split_column("unit_feat", "_", True)
    assert res_rev == ("feat", "unit")

    # No sep
    res_none = utils_mod.split_column("unit", "_", False)
    assert res_none == ("", "unit")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("42", "0042"),
        ("sub-42", "0042"),
        (42, "0042"),
        ("P001", "P001"),
    ],
)
def test_normalize_subject_value(value, expected):
    assert utils_mod.normalize_subject_value(value) == expected


def test_participants_tsv(tmp_path):
    """Test TSV parsing."""
    p_tsv = tmp_path / "participants.tsv"

    # Valid
    df = pd.DataFrame({"participant_id": ["sub-01", "sub-02"], "age": [20, 30]})
    df.to_csv(p_tsv, sep="\t", index=False)

    lookup = utils_mod.load_participants_tsv(tmp_path)
    assert lookup["0001"] == {"age": 20}

    # Missing
    assert utils_mod.load_participants_tsv(tmp_path / "nowhere") == {}


def test_make_strata_extra():
    df = pd.DataFrame({"num": [1, 2, 3, 4, 5], "num_dup": [1, 1, 1, 1, 1]})

    # uniform
    res = make_strata(df, covariates=["num"], n_bins=2, binning="uniform")
    assert len(res) == 5

    # qcut exception fallback to cut
    res2 = make_strata(df, covariates=["num_dup"], n_bins=2, binning="quantile")
    assert len(res2) == 5


def test_sample_indices_extra():
    df = pd.DataFrame({"target": ["A", "A", "B"], "feat": [1, 2, 3]})
    rng = np.random.default_rng(42)

    # n <= 0
    res = sample_indices(
        df, "target", {"A": 0}, rng, replace=False, prefer_clean=False, exclude=[]
    )
    assert len(res) == 0

    # prefer_clean with replace
    res2 = sample_indices(
        df, "target", {"A": 3}, rng, replace=True, prefer_clean=True, exclude=[]
    )
    assert len(res2) == 3


def test_sample_indices_not_clean():
    df = pd.DataFrame({"target": ["A", "A", "B"], "feat": [1, 2, 3]})
    rng = np.random.default_rng(42)

    # n <= len and not replace
    res = sample_indices(
        df, "target", {"A": 2}, rng, replace=False, prefer_clean=False, exclude=[]
    )
    assert len(res) == 2

    # n > len or replace
    res2 = sample_indices(
        df, "target", {"A": 3}, rng, replace=True, prefer_clean=False, exclude=[]
    )
    assert len(res2) == 3


def test_io_init_getattr():
    from coco_pipe import io as coco_io

    # 1. Valid attributes
    assert coco_io.BIDSDataset is not None
    assert coco_io.EmbeddingDataset is not None
    assert coco_io.TabularDataset is not None

    # 2. Invalid attribute raises AttributeError
    with pytest.raises(AttributeError, match="has no attribute 'invalid_attr'"):
        _ = coco_io.invalid_attr


def test_load_participants_tsv_without_participant_id(tmp_path):
    """participants.tsv lacking a participant_id column yields an empty lookup."""
    (tmp_path / "participants.tsv").write_text("age\t30\n", encoding="utf-8")
    assert load_participants_tsv(tmp_path) == {}


def test_compute_feature_missingness_separates_nan_and_infinity():
    df = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.inf, -np.inf],
            "b": [np.nan, 2.0, 3.0, 4.0],
            "metadata": ["x", "y", "z", "w"],
        }
    )

    summary = compute_feature_missingness(df, ["a", "b"])

    assert summary.to_dict(orient="records") == [
        {
            "column": "a",
            "missing_count": 1,
            "missing_rate": 0.25,
            "nonfinite_count": 2,
            "nonfinite_rate": 0.5,
        },
        {
            "column": "b",
            "missing_count": 1,
            "missing_rate": 0.25,
            "nonfinite_count": 0,
            "nonfinite_rate": 0.0,
        },
    ]


def test_compute_feature_missingness_empty_rows_has_stable_schema():
    summary = compute_feature_missingness(
        pd.DataFrame(columns=["a", "b"]),
        ["a", "b"],
    )

    assert summary.columns.tolist() == [
        "column",
        "missing_count",
        "missing_rate",
        "nonfinite_count",
        "nonfinite_rate",
    ]
    assert summary["missing_rate"].tolist() == [0.0, 0.0]


def test_compute_constant_feature_summary():
    df = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan],
            "constant": [2.0, 2.0, 2.0],
            "near_constant": [1.0, 1.0, 1.0 + 1e-13],
            "variable": [1.0, 2.0, 3.0],
        }
    )

    summary = compute_constant_feature_summary(
        df,
        ["all_nan", "constant", "near_constant", "variable"],
    ).set_index("column")

    assert bool(summary.loc["all_nan", "is_all_nan"])
    assert not bool(summary.loc["all_nan", "is_constant"])
    assert summary.loc["constant", "std"] == 0.0
    assert bool(summary.loc["constant", "is_constant"])
    assert bool(summary.loc["near_constant", "is_constant"])
    assert not bool(summary.loc["variable", "is_constant"])

    with pytest.raises(ValueError, match="tol must be non-negative"):
        compute_constant_feature_summary(df, ["constant"], tol=-1)
