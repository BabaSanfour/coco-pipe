import numpy as np
import pytest

from coco_pipe.decoding.targets import prepare_target, safe_group_n_splits
from coco_pipe.io.structures import DataContainer


def _container(coords):
    """Build a DataContainer whose obs count matches the coord columns."""
    n = len(next(iter(coords.values())))
    return DataContainer(X=np.zeros((n, 2)), dims=("obs", "chan"), coords=coords)


def test_prepare_target_binary_success():
    container = _container(
        {
            "target_col": ["a", "b", "a", "b", "c", None],
            "patient_group_id": ["g1", "g2", "g3", "g4", "g5", "g6"],
        }
    )
    spec = {"target_col": "target_col", "label_map": {"c": "b"}, "positive_class": "b"}

    selected, y, groups, frame = prepare_target(container, spec)
    assert len(y) == 5
    assert list(y) == [0, 1, 0, 1, 1]
    assert list(groups) == ["g1", "g2", "g3", "g4", "g5"]
    assert "target_label" in frame.columns
    assert "target_encoded" in frame.columns
    assert frame.attrs["positive_class"] == "b"


def test_prepare_target_multiclass_success():
    container = _container(
        {
            "target_col": ["c", "b", "a"],
            "patient_group_id": ["g1", "g2", "g3"],
        }
    )
    spec = {"target_col": "target_col", "class_order": ["a", "b", "c"]}
    _, y, groups, _ = prepare_target(container, spec)
    assert list(y) == [2, 1, 0]


@pytest.mark.parametrize(
    "coords, spec, exc, match",
    [
        pytest.param(
            {"other_col": ["a"]},
            {"target_col": "target_col"},
            KeyError,
            "Target column 'target_col' is not available",
            id="missing_target_col",
        ),
        pytest.param(
            {"target_col": ["a", "a"], "patient_group_id": ["g1", "g2"]},
            {"target_col": "target_col"},
            ValueError,
            "Decoding requires at least two target classes.",
            id="less_than_two_classes",
        ),
        pytest.param(
            {"target_col": ["a", "b"], "patient_group_id": ["g1", "g2"]},
            {"target_col": "target_col"},
            ValueError,
            "Binary target specifications must define positive_class explicitly",
            id="binary_missing_positive_class",
        ),
        pytest.param(
            {"target_col": ["a", "b"], "patient_group_id": ["g1", "g2"]},
            {"target_col": "target_col", "positive_class": "c"},
            ValueError,
            "positive_class='c' is absent",
            id="binary_wrong_positive_class",
        ),
        pytest.param(
            {"target_col": ["a", "b", "c"], "patient_group_id": ["g1", "g2", "g3"]},
            {"target_col": "target_col"},
            ValueError,
            "Multiclass target specifications must define class_order explicitly.",
            id="multiclass_missing_class_order",
        ),
        pytest.param(
            {"target_col": ["a", "b", "c"], "patient_group_id": ["g1", "g2", "g3"]},
            {"target_col": "target_col", "class_order": ["a", "b"]},
            ValueError,
            r"class_order=\['a', 'b'\] does not match target classes",
            id="multiclass_wrong_class_order",
        ),
        pytest.param(
            {"target_col": ["a", "b", "a"]},
            {
                "target_col": "target_col",
                "positive_class": "a",
                "group_col": "missing_group",
            },
            KeyError,
            "Leakage-safe group column 'missing_group' is not available.",
            id="missing_group_col",
        ),
    ],
)
def test_prepare_target_errors(coords, spec, exc, match):
    container = _container(coords)
    with pytest.raises(exc, match=match):
        prepare_target(container, spec)


@pytest.mark.parametrize(
    "y, groups, requested, stratified, expected",
    [
        # y=0 -> groups {g1, g2}, y=1 -> groups {g3, g4}; min unique = 2
        pytest.param(
            [0, 0, 0, 1, 1], ["g1", "g1", "g2", "g3", "g4"], 5, True, 2, id="stratified"
        ),
        # total unique groups = 4, capped at requested=3
        pytest.param(
            [0, 0, 0, 1, 1],
            ["g1", "g1", "g2", "g3", "g4"],
            3,
            False,
            3,
            id="unstratified",
        ),
    ],
)
def test_safe_group_n_splits(y, groups, requested, stratified, expected):
    assert (
        safe_group_n_splits(y, groups, requested=requested, stratified=stratified)
        == expected
    )


@pytest.mark.parametrize(
    "y, groups, kwargs, match",
    [
        pytest.param(
            [],
            [],
            {},
            "Cannot determine grouped CV folds from an empty target",
            id="empty",
        ),
        pytest.param(
            [0, 0, 1, 1],
            ["g1", "g1", "g2", "g3"],
            {"requested": 5, "stratified": True},
            "requires at least two independent groups",
            id="insufficient",
        ),
    ],
)
def test_safe_group_n_splits_errors(y, groups, kwargs, match):
    with pytest.raises(ValueError, match=match):
        safe_group_n_splits(y, groups, **kwargs)
