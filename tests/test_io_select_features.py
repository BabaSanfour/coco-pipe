import pytest
import pandas as pd

from coco_pipe.io.select_features import select_features

@pytest.fixture
def example_df():
    # Build a small DataFrame with:
    # - two target columns: "target1", "target2"
    # - two covariates: "age", "sex"
    # - sensors: "sensorA", "sensorB"
    # - features: "alpha", "beta"
    data = {
        "subject": ["s1", "s2", "s3", "s4"],
        "target": [0, 1, 0, 1],
        "target2": [1.0, 2.0, 3.0, 4.0],
        "age": [10, 20, 30, 40],
        "sex": ["M", "F", "M", "F"],
        "sensorA_alpha": [0.1, 0.2, 0.3, 0.4],
        "sensorA_beta": [1.1, 1.2, 1.3, 1.4],
        "sensorB_alpha": [0.5, 0.6, 0.7, 0.8],
        "sensorB_beta": [1.5, 1.6, 1.7, 1.8],
    }
    return pd.DataFrame(data)

def test_only_covariates(example_df):
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=["age", "sex"],
        spatial_units=None,
        feature_names="all",
        row_filter=None
    )
    # X should have only age & sex
    assert list(X.columns) == ["age", "sex"]
    # y should be a Series of length 4
    pd.testing.assert_series_equal(y, example_df["target"], check_names=False)

def test_only_spatial_all_features(example_df):
    # no covariates, select both sensors, all features
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=None,
        spatial_units=["sensorA", "sensorB"],
        feature_names="all",
        row_filter=None
    )
    # expect four columns: sensorA_alpha, sensorA_beta, sensorB_alpha, sensorB_beta
    expected = ["sensorA_alpha","sensorA_beta","sensorB_alpha","sensorB_beta"]
    assert set(X.columns) == set(expected)
    assert X.shape == (4, 4)

def test_only_features_all_sensors(example_df):
    # no covariates, all sensors, only alpha
    X, y = select_features(
        df=example_df,
        target_columns=["target","target2"],
        covariates=None,
        spatial_units="all",
        feature_names="alpha",
        row_filter=None
    )
    # "all" sensors picks sensorA and sensorB; only alpha feature => 2 columns
    expected = ["sensorA_alpha","sensorB_alpha"]
    assert set(X.columns) == set(expected)
    # y should be DataFrame with two columns
    assert list(y.columns) == ["target","target2"]
    assert y.shape == (4,2)

def test_covariates_plus_custom_selection(example_df):
    # covariates + sensorA only + beta only
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=["age"],
        spatial_units=["sensorA"],
        feature_names=["beta"],
        row_filter=None
    )
    # columns: age, sensorA_beta
    assert list(X.columns) == ["age","sensorA_beta"]
    # types preserved
    assert X["age"].dtype == example_df["age"].dtype

def test_row_filtering(example_df):
    # filter to only subject s1 & s3
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=["age"],
        spatial_units=["sensorA"],
        feature_names=["alpha"],
        row_filter={"column":"subject","values":["s1","s3"]}
    )
    # only two rows remain
    assert X.shape[0] == 2
    assert list(X.index) == [0,2]
    # y matches filtered rows
    assert list(y.values) == [0,0]

def test_no_selection_raises(example_df):
    # neither covariates nor spatial_units => error
    with pytest.raises(ValueError):
        select_features(
            df=example_df,
            target_columns="target",
            covariates=None,
            spatial_units=None,
            feature_names="alpha",
            row_filter=None
        )

def test_mismatch_feature_or_sensor_ignored(example_df):
    # request a sensor that doesn't exist plus one valid
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=None,
        spatial_units=["sensorA", "sensorX"],
        feature_names=["alpha","gamma"],
        row_filter=None
    )
    # sensorA_alpha exists; sensorA_gamma and sensorX_* ignored
    assert list(X.columns) == ["sensorA_alpha"]
    assert X.shape == (4,1)

def test_multi_target_list(example_df):
    # target_columns as list
    X, y = select_features(
        df=example_df,
        target_columns=["target","target2"],
        covariates=["sex"],
        spatial_units=None,
        feature_names="all",
        row_filter=None
    )
    assert list(y.columns) == ["target","target2"]
    assert X.shape == (4,1)
