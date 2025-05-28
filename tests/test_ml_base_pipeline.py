import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error

from coco_pipe.ml.base import BasePipeline
from coco_pipe.ml.config import CLASSIFICATION_METRICS, REGRESSION_METRICS


class DummyPipeline(BasePipeline):
    """Concrete subclass for testing BasePipeline."""
    pass


def test_validate_input_errors():
    # X not DataFrame/ndarray
    with pytest.raises(ValueError):
        DummyPipeline(
            X="not array",
            y=np.zeros(3),
            metric_funcs=CLASSIFICATION_METRICS,
            model_configs={},
            default_metrics=["accuracy"]
        )
    # X and y length mismatch
    with pytest.raises(ValueError):
        DummyPipeline(
            X=np.zeros((3, 2)),
            y=np.zeros(4),
            metric_funcs=CLASSIFICATION_METRICS,
            model_configs={},
            default_metrics=["accuracy"]
        )


def test_validate_metrics_error():
    X = np.zeros((5, 2)); y = np.zeros(5)
    with pytest.raises(ValueError):
        DummyPipeline(
            X, y,
            metric_funcs={'acc': lambda a, b: 0},
            model_configs={},
            default_metrics=['bad']
        )


def test_feature_names_and_importances():
    # Prepare balanced classes
    X = np.random.randn(20, 3)
    y = np.concatenate([np.zeros(10), np.ones(10)])
    np.random.shuffle(y)
    # Test feature importances extraction
    lr = LogisticRegression(solver='liblinear').fit(X, y)
    imp = DummyPipeline._extract_feature_importances(lr)
    assert isinstance(imp, np.ndarray) and imp.shape == (3,)
    # Estimator without attributes
    dc = DummyClassifier().fit(X, y)
    assert DummyPipeline._extract_feature_importances(dc) is None
    # Test feature names
    df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    assert DummyPipeline._get_feature_names(df) == ['a', 'b', 'c']
    arr_names = DummyPipeline._get_feature_names(X)
    assert arr_names == ['feature_0', 'feature_1', 'feature_2']


def test_cross_val_and_baseline_evaluation_classification():
    # Balanced binary classification
    X = np.arange(40).reshape(-1, 2)
    y = np.array([0]*10 + [1]*10)
    model_configs = {'dummy': {
        'estimator': DummyClassifier(strategy='most_frequent'),
        'params': {}
    }}
    # Custom accuracy metric
    def acc(y_t, y_p):
        return float(np.mean(y_t == y_p))

    pipe = DummyPipeline(
        X, y,
        metric_funcs={'accuracy': acc},
        model_configs=model_configs,
        default_metrics=['accuracy'],
        cv_kwargs={
            'cv_strategy': 'stratified',
            'n_splits': 5,
            'shuffle': True,
            'random_state': 0
        },
        n_jobs=1
    )
    # Test cross_val directly
    cv_res = pipe.cross_val(DummyClassifier(strategy='most_frequent'), X, y)
    # Check returned keys
    expected_keys = [
        'cv_fold_predictions', 'cv_fold_scores', 'cv_fold_estimators',
        'cv_fold_importances', 'final_estimator', 'final_importances',
    ]
    for key in expected_keys:
        assert key in cv_res
    # All fold accuracies should be 0.5
    assert np.all(cv_res['cv_fold_scores']['accuracy'] == pytest.approx(0.5))

    # Test baseline_evaluation (formerly baseline)
    eval_res = pipe.baseline_evaluation('dummy')
    # check model_name and params
    assert eval_res['model_name'] == 'dummy'
    assert 'params' in eval_res and isinstance(eval_res['params'], dict)
    # consistency between cross_val and baseline_evaluation
    assert np.all(eval_res['cv_fold_scores']['accuracy'] == cv_res['cv_fold_scores']['accuracy'])


def test_cross_val_requires_groups_for_group_kfold():
    X = np.zeros((6, 1)); y = np.zeros(6)
    pipe = DummyPipeline(
        X, y,
        metric_funcs=CLASSIFICATION_METRICS,
        model_configs={'dummy': {'estimator': DummyClassifier(), 'params': {}}},
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'group_kfold', 'n_splits': 2, 'shuffle': True, 'random_state': 0},
        n_jobs=1
    )
    with pytest.raises(ValueError):
        pipe.cross_val(DummyClassifier(), X, y)


def test_baseline_evaluation_errors_and_params():
    X = np.random.randn(10, 2); y = np.random.randint(0, 2, 10)
    pipe = DummyPipeline(
        X, y,
        metric_funcs=CLASSIFICATION_METRICS,
        model_configs={'clf': {'estimator': LogisticRegression(), 'params': {'C': [1]}}},
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'stratified', 'n_splits': 2, 'shuffle': True, 'random_state': 0},
    )
    # invalid model name
    with pytest.raises(KeyError):
        pipe.baseline_evaluation('bad_model')
    # passing best_params overrides default estimator params
    out = pipe.baseline_evaluation('clf', best_params={'C': 0.5})
    assert out['params']['C'] == 0.5


def test_baseline_evaluation_regression():
    X = np.arange(30).reshape(-1, 3); y = np.arange(10)
    model_configs = {'dummy': {'estimator': DummyRegressor(), 'params': {'strategy': ['mean']}}}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'mse': mean_squared_error},
        model_configs=model_configs,
        default_metrics=['mse'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 3, 'shuffle': True, 'random_state': 0},
    )
    out = pipe.baseline_evaluation('dummy')
    # MSE values non-negative
    assert np.all(out['cv_fold_scores']['mse'] >= 0)
    assert 'final_estimator' in out and 'final_importances' in out