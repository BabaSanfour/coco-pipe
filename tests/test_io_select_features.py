import pytest
import pandas as pd
import logging

from coco_pipe.io.select_features import select_features

@pytest.fixture
def example_df():
    # Build a small DataFrame with:
    # - two target columns: "target", "target2"
    # - two covariates: "age", "sex"
    # - sensors: "sensorA", "sensorB"
    # - features: "alpha", "beta"
    data = {
        "subject": [1, 2, 3, 4],
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

def assert_groups(result_df, original_df, filter_index):
    # Groups should match the subject values after filtering
    expected_groups = original_df.loc[filter_index, "subject"]
    pd.testing.assert_series_equal(result_df, expected_groups, check_names=False)

def test_only_covariates(example_df):
    # Without groups_column
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=["AGE", "SEX"],
        spatial_units=None,
        feature_names="all",
        row_filter=None
    )
    assert set(X.columns) == {"age", "sex"}
    pd.testing.assert_series_equal(y, example_df["target"], check_names=False)

    # With groups_column
    Xg, yg, groups = select_features(
        df=example_df,
        target_columns="target",
        covariates=["AGE", "SEX"],
        spatial_units=None,
        feature_names="all",
        row_filter=None,
        groups_column="subject"
    )
    assert set(Xg.columns) == {"age", "sex"}
    pd.testing.assert_series_equal(yg, example_df["target"], check_names=False)
    # groups should be the same as the subject column (unfiltered)
    assert_groups(groups, example_df, Xg.index)

def test_only_spatial_all_features(example_df):
    X, y = select_features(
        df=example_df,
        target_columns="TARGET",
        covariates=None,
        spatial_units=["SENSORA", "sensorB"],
        feature_names="all",
        row_filter=None
    )
    expected = [
        "sensorA_alpha", "sensorA_beta",
        "sensorB_alpha", "sensorB_beta"
    ]
    assert set(X.columns) == set(expected)
    assert X.shape == (4, 4)

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns="TARGET",
        covariates=None,
        spatial_units=["SENSORA", "sensorB"],
        feature_names="all",
        row_filter=None,
        groups_column="subject"
    )
    assert set(Xg.columns) == set(expected)
    assert Xg.shape == (4, 4)
    assert_groups(groups, example_df, Xg.index)

def test_only_features_all_sensors(example_df):
    X, y = select_features(
        df=example_df,
        target_columns=["TARGET", "TARGET2"],
        covariates=None,
        spatial_units="all",
        feature_names="ALPHA",
        row_filter=None
    )
    expected = ["sensorA_alpha", "sensorB_alpha"]
    assert set(X.columns) == set(expected)
    assert list(y.columns) == ["target", "target2"]
    assert y.shape == (4, 2)

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns=["TARGET", "TARGET2"],
        covariates=None,
        spatial_units="all",
        feature_names="ALPHA",
        row_filter=None,
        groups_column="subject"
    )
    assert set(Xg.columns) == set(expected)
    assert list(yg.columns) == ["target", "target2"]
    assert yg.shape == (4, 2)
    assert_groups(groups, example_df, Xg.index)

def test_covariates_plus_custom_selection(example_df):
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=["AGE"],
        spatial_units=["SENSORA"],
        feature_names=["BETA"],
        row_filter=None
    )
    assert set(X.columns) == {"age", "sensorA_beta"}
    assert X["age"].dtype == example_df["age"].dtype

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns="target",
        covariates=["AGE"],
        spatial_units=["SENSORA"],
        feature_names=["BETA"],
        row_filter=None,
        groups_column="subject"
    )
    assert set(Xg.columns) == {"age", "sensorA_beta"}
    assert Xg["age"].dtype == example_df["age"].dtype
    assert_groups(groups, example_df, Xg.index)

def test_row_filtering(example_df):
    X, y = select_features(
        df=example_df,
        target_columns="target",
        covariates=["age"],
        spatial_units=["sensorA"],
        feature_names=["ALPHA"],
        row_filter={"column": "SUBJECT", "values": [1, 2]}
    )
    assert X.shape[0] == 2
    assert list(X.index) == [0, 1]
    assert list(y.values) == [0, 1]

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns="target",
        covariates=["age"],
        spatial_units=["sensorA"],
        feature_names=["ALPHA"],
        row_filter={"column": "SUBJECT", "values": [1, 3]},
        groups_column="subject"
    )
    assert Xg.shape[0] == 2
    assert list(Xg.index) == [0, 2]
    assert list(yg.values) == [0, 0]
    assert_groups(groups, example_df, Xg.index)

def test_no_selection_raises(example_df):
    with pytest.raises(ValueError, match="No features selected"):
        select_features(
            df=example_df,
            target_columns="target",
            covariates=None,
            spatial_units=None,
            feature_names="alpha",
            row_filter=None
        )


def test_missing_features_raises(example_df):
    with pytest.raises(ValueError, match="No features selected"):
        select_features(
            df=example_df,
            target_columns="target",
            covariates=None,
            spatial_units=["sensorA"],
            feature_names=["gamma"],
            row_filter=None
        )


def test_missing_spatial_units_raises(example_df):
    with pytest.raises(ValueError, match="No features selected"):
        select_features(
            df=example_df,
            target_columns="target",
            covariates=None,
            spatial_units=["sensorX"],
            feature_names=["alpha"],
            row_filter=None
        )


def test_missing_covariates_raises(example_df):
    with pytest.raises(ValueError, match="Covariate 'height' not found"):
        select_features(
            df=example_df,
            target_columns="target",
            covariates=["height"],
            spatial_units=None,
            feature_names="all",
            row_filter=None
        )


def test_missing_target_raises(example_df):
    with pytest.raises(ValueError, match="Target 'nonexistent' not found"):
        select_features(
            df=example_df,
            target_columns="nonexistent",
            covariates=["age"],
            spatial_units=None,
            feature_names="all",
            row_filter=None
        )


def test_multi_target_list(example_df):
    X, y = select_features(
        df=example_df,
        target_columns=["TARGET", "TARGET2"],
        covariates=["SEX"],
        spatial_units=None,
        feature_names="all",
        row_filter=None
    )
    assert list(y.columns) == ["target", "target2"]
    assert X.shape == (4, 1)

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns=["TARGET", "TARGET2"],
        covariates=["SEX"],
        spatial_units=None,
        feature_names="all",
        row_filter=None,
        groups_column="subject"
    )
    assert list(yg.columns) == ["target", "target2"]
    assert Xg.shape == (4, 1)
    assert_groups(groups, example_df, Xg.index)

def test_reverse_naming(example_df):
    df = example_df.rename(columns={
        "sensorA_alpha": "alpha_sensorA",
        "sensorA_beta": "beta_sensorA",
        "sensorB_alpha": "alpha_sensorB",
        "sensorB_beta": "beta_sensorB",
    })
    X, y = select_features(
        df=df,
        target_columns="target",
        covariates=None,
        spatial_units=["sensorA", "sensorB"],
        feature_names=["alpha"],
        sep="_",
        reverse=True
    )
    assert set(X.columns) == {"alpha_sensorA", "alpha_sensorB"}

    Xg, yg, groups = select_features(
        df=df,
        target_columns="target",
        covariates=None,
        spatial_units=["sensorA", "sensorB"],
        feature_names=["alpha"],
        sep="_",
        reverse=True,
        groups_column="subject"
    )
    assert set(Xg.columns) == {"alpha_sensorA", "alpha_sensorB"}
    assert_groups(groups, df, Xg.index)

def test_custom_sep(example_df):
    df = example_df.rename(columns={
        "sensorA_alpha": "sensorA-alpha",
        "sensorB_alpha": "sensorB-alpha"
    })
    X, y = select_features(
        df=df,
        target_columns="target",
        covariates=None,
        spatial_units=["sensorA", "sensorB"],
        feature_names=["alpha"],
        sep='-'
    )
    assert set(X.columns) == {"sensorA-alpha", "sensorB-alpha"}

    Xg, yg, groups = select_features(
        df=df,
        target_columns="target",
        covariates=None,
        spatial_units=["sensorA", "sensorB"],
        feature_names=["alpha"],
        sep='-',
        groups_column="subject"
    )
    assert set(Xg.columns) == {"sensorA-alpha", "sensorB-alpha"}
    assert_groups(groups, df, Xg.index)

def test_spatial_units_dict(example_df):
    # Provide mapping keys as actual spatial units
    groups_dict = {'sensorA': ['alpha', 'beta']}
    X, y = select_features(
        df=example_df,
        target_columns='target',
        covariates=None,
        spatial_units=groups_dict,
        feature_names='all',
        row_filter=None
    )
    # Should select all features under sensorA
    assert set(X.columns) == {'sensorA_alpha', 'sensorA_beta'}

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns='target',
        covariates=None,
        spatial_units=groups_dict,
        feature_names='all',
        row_filter=None,
        groups_column="subject"
    )
    assert set(Xg.columns) == {'sensorA_alpha', 'sensorA_beta'}
    assert_groups(groups, example_df, Xg.index)

def test_row_filter_operator_gt(example_df):
    X, y = select_features(
        df=example_df,
        target_columns='target',
        covariates=['age'],
        spatial_units=['sensorA'],
        feature_names=['alpha'],
        row_filter={'column': 'AGE', 'operator': '>', 'values': 20}
    )
    assert list(X.index) == [2, 3]
    assert list(y.values) == [0, 1]

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns='target',
        covariates=['age'],
        spatial_units=['sensorA'],
        feature_names=['alpha'],
        row_filter={'column': 'AGE', 'operator': '>', 'values': 20},
        groups_column="subject"
    )
    assert list(Xg.index) == [2, 3]
    assert list(yg.values) == [0, 1]
    assert_groups(groups, example_df, Xg.index)

def test_multiple_row_filters(example_df):
    filters = [
        {'column': 'SUBJECT', 'values': [1, 2]},
        {'column': 'age', 'operator': '<', 'values': 20}
    ]
    X, y = select_features(
        df=example_df,
        target_columns='target',
        covariates=['age'],
        spatial_units=['sensorB'],
        feature_names=['beta'],
        row_filter=filters
    )
    assert list(X.index) == [0]
    assert list(y.values) == [0]

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns='target',
        covariates=['age'],
        spatial_units=['sensorB'],
        feature_names=['beta'],
        row_filter=filters,
        groups_column="subject"
    )
    assert list(Xg.index) == [0]
    assert list(yg.values) == [0]
    assert_groups(groups, example_df, Xg.index)

def test_invalid_operator_defaults_to_isin(example_df):
    X, y = select_features(
        df=example_df,
        target_columns='target',
        covariates=['age'],
        spatial_units=['sensorB'],
        feature_names=['beta'],
        row_filter={'column': 'age', 'operator': 'invalid', 'values': [10, 30]}
    )
    assert list(X.index) == [0, 2]
    assert list(y.values) == [0, 0]

    Xg, yg, groups = select_features(
        df=example_df,
        target_columns='target',
        covariates=['age'],
        spatial_units=['sensorB'],
        feature_names=['beta'],
        row_filter={'column': 'age', 'operator': 'invalid', 'values': [10, 30]},
        groups_column="subject"
    )
    assert list(Xg.index) == [0, 2]
    assert list(yg.values) == [0, 0]
    assert_groups(groups, example_df, Xg.index)

def test_no_features_selected_when_reverse_and_none(example_df):
    with pytest.raises(ValueError, match="No features selected"):
        select_features(
            df=example_df,
            target_columns='target',
            covariates=None,
            spatial_units=None,
            feature_names='alpha',
            reverse=True
        )


def test_row_filter_missing_values_key(example_df):
    with pytest.raises(ValueError, match="Row filter for column 'age' must include 'values'."):
        select_features(
            df=example_df,
            target_columns='target',
            covariates=None,
            spatial_units=None,
            feature_names='alpha',
            row_filter={'column': 'age'}
        )


def test_row_filter_missing_column_key(example_df):
    with pytest.raises(ValueError, match="Row filter missing 'column' key"):
        select_features(
            df=example_df,
            target_columns='target',
            covariates=None,
            spatial_units=None,
            feature_names='alpha',
            row_filter={'values': ['s1']}
        )


def test_row_filter_column_not_found(example_df):
    with pytest.raises(ValueError, match="Row filter column 'foo' not found. Did you mean"):
        select_features(
            df=example_df,
            target_columns='target',
            covariates=None,
            spatial_units=None,
            feature_names='alpha',
            row_filter={'column': 'foo', 'values': ['s1']}
        )

def test_verbose_logging(example_df, caplog):
    # Test if verbose=True produces log output.
    # This assumes select_features uses the logging module when verbose is True.
    logger = logging.getLogger('coco_pipe.io.select_features')
    logger.setLevel(logging.INFO)
    logger.propagate = True
    with caplog.at_level(logging.INFO):
        X, y, groups = select_features(
            df=example_df,
            target_columns="target",
            groups_column="subject",
            covariates=["AGE", "SEX"],
            spatial_units=None,
            feature_names="all",
            row_filter={'column': 'SUBJECT', 'values': [1, 2]},
            verbose=True
        )
    # Check that some log messages were produced.
    assert len(caplog.records) > 0, "No log records produced with verbose=True"
    assert any("Selected features:" in record.message for record in caplog.records), "Expected 'Selected features' in log messages"
    assert any("Selected covariates:" in record.message for record in caplog.records), "Expected 'Selected covariates' in log messages"
    assert any("Selected spatial units:" in record.message for record in caplog.records), "Expected 'Selected spatial units' in log messages"
    assert any("Actual features:" in record.message for record in caplog.records), "Expected 'Actual features' in log messages"
    assert any("Target columns selected:" in record.message for record in caplog.records), "Expected 'Target columns selected' in log messages"
    assert any("Feature matrix shape:" in record.message for record in caplog.records), "Expected 'Feature matrix shape' in log messages"
    assert any("Applied row filters." in record.message for record in caplog.records), "Expected 'Applied row filters' in log messages"
    assert any("Groups selected with shape" in record.message for record in caplog.records), "Expected 'Groups selected with shape' in log messages"