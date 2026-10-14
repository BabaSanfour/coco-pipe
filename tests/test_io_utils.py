import numpy as np
import pandas as pd
import pytest

from coco_pipe.io.utils import (
    compute_constant_feature_summary,
    compute_feature_missingness,
    load_participants_tsv,
    make_strata,
    normalize_subject_value,
    row_quality_score,
    sample_indices,
    split_column,
)


def test_row_quality_score():
    frame = pd.DataFrame({"a": [1, np.nan, 0, np.inf], "b": [1, 1, 1, 1]})
    assert row_quality_score(frame, count_zero=True).tolist() == [0, 1, 1, 1]
    assert row_quality_score(frame, count_zero=False).tolist() == [0, 1, 0, 1]

    mixed = pd.DataFrame(
        {"a": [0.0, np.nan, 1.0], "b": [1.0, np.inf, 2.0], "label": ["x", "y", "z"]},
        index=[10, 20, 30],
    )
    scores = row_quality_score(mixed, normalize=True)
    assert scores.index.tolist() == [10, 20, 30]
    assert scores.tolist() == [0.5, 1.0, 0.0]
    text = row_quality_score(mixed[["label"]], normalize=True)
    assert text.tolist() == [0.0, 0.0, 0.0]
    assert text.dtype == float


def test_make_strata():
    frame = pd.DataFrame({"num": [1, 10, 100], "cat": ["a", "b", "a"]})
    strata = make_strata(frame, covariates=["num", "cat"], n_bins=3)
    assert len(strata.unique()) == 3

    values = pd.DataFrame({"num": [1, 2, 3, 4, 5], "flat": [1] * 5})
    assert len(make_strata(values, ["num"], n_bins=2, binning="uniform")) == 5
    # quantile bins that cannot be formed fall back to equal-width bins
    assert len(make_strata(values, ["flat"], n_bins=2, binning="quantile")) == 5


def test_sample_indices():
    frame = pd.DataFrame({"target": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
    rng = np.random.default_rng(42)

    def draw(sizes, *, replace=False, prefer_clean=True):
        return sample_indices(
            frame,
            "target",
            sizes,
            rng,
            replace=replace,
            prefer_clean=prefer_clean,
            exclude=[],
        )

    picked = frame.loc[draw({"A": 1, "B": 2})]
    assert picked["target"].value_counts().to_dict() == {"B": 2, "A": 1}
    assert len(draw({"A": 0})) == 0
    for prefer_clean in (True, False):
        assert len(draw({"A": 2}, prefer_clean=prefer_clean)) == 2
        assert len(draw({"A": 3}, replace=True, prefer_clean=prefer_clean)) == 3


def test_split_column():
    assert split_column("unit_feat", "_", False) == ("unit", "feat")
    assert split_column("unit_feat", "_", True) == ("feat", "unit")
    assert split_column("unit", "_", False) == ("", "unit")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("42", "0042"), ("sub-42", "0042"), (42, "0042"), ("P001", "P001")],
)
def test_normalize_subject_value(value, expected):
    assert normalize_subject_value(value) == expected


def test_load_participants_tsv(tmp_path):
    pd.DataFrame({"participant_id": ["sub-01", "sub-02"], "age": [20, 30]}).to_csv(
        tmp_path / "participants.tsv", sep="\t", index=False
    )
    assert load_participants_tsv(tmp_path)["0001"] == {"age": 20}
    assert load_participants_tsv(tmp_path / "nowhere") == {}

    derivative = tmp_path / "derivatives" / "preproc"
    derivative.mkdir(parents=True)
    (derivative / "dataset_description.json").write_text(
        '{"DatasetType": "derivative", "SourceDatasets": [{"URL": "../../"}]}'
    )
    assert load_participants_tsv(derivative)["0001"] == {"age": 20}

    (tmp_path / "participants.tsv").write_text("age\t30\n", encoding="utf-8")
    assert load_participants_tsv(tmp_path) == {}


def test_io_exposes_dataset_classes_lazily():
    from coco_pipe import io as coco_io

    assert coco_io.BIDSDataset is not None
    assert coco_io.EmbeddingDataset is not None
    with pytest.raises(AttributeError, match="has no attribute 'invalid_attr'"):
        _ = coco_io.invalid_attr


def test_compute_feature_missingness():
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, np.inf, -np.inf],
            "b": [np.nan, 2.0, 3.0, 4.0],
            "metadata": ["x", "y", "z", "w"],
        }
    )
    # NaN and infinity are counted apart
    assert compute_feature_missingness(frame, ["a", "b"]).to_dict(orient="records") == [
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
    empty = compute_feature_missingness(pd.DataFrame(columns=["a", "b"]), ["a", "b"])
    assert empty.columns.tolist() == [
        "column",
        "missing_count",
        "missing_rate",
        "nonfinite_count",
        "nonfinite_rate",
    ]
    assert empty["missing_rate"].tolist() == [0.0, 0.0]


def test_compute_constant_feature_summary():
    frame = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan],
            "constant": [2.0, 2.0, 2.0],
            "near_constant": [1.0, 1.0, 1.0 + 1e-13],
            "variable": [1.0, 2.0, 3.0],
        }
    )
    summary = compute_constant_feature_summary(frame, list(frame)).set_index("column")
    assert bool(summary.loc["all_nan", "is_all_nan"])
    assert not bool(summary.loc["all_nan", "is_constant"])
    assert summary.loc["constant", "std"] == 0.0
    assert bool(summary.loc["near_constant", "is_constant"])
    assert not bool(summary.loc["variable", "is_constant"])
    with pytest.raises(ValueError, match="tol must be non-negative"):
        compute_constant_feature_summary(frame, ["constant"], tol=-1)
