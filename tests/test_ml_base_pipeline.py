import pytest
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from coco_pipe.ml.base import BasePipeline
from coco_pipe.ml.config import CLASSIFICATION_METRICS, REGRESSION_METRICS

class DummyPipeline(BasePipeline):
    pass 

def test_validate_input_errors():
    # X not DataFrame/ndarray
    with pytest.raises(ValueError):
        DummyPipeline(X="not array", y=np.zeros(3), 
                      metric_funcs=CLASSIFICATION_METRICS, model_configs={}, default_metrics=["accuracy"] )
    # X and y length mismatch
    with pytest.raises(ValueError):
        DummyPipeline(X=np.zeros((3,2)), y=np.zeros(4), 
                      metric_funcs=CLASSIFICATION_METRICS, model_configs={}, default_metrics=["accuracy"] )

def test_validate_metrics_error():
    X = np.zeros((5,2)); y = np.zeros(5)
    # metric_funcs missing 'bad'
    with pytest.raises(ValueError):
        DummyPipeline(X, y,
                      metric_funcs={'acc': lambda a,b:0},
                      model_configs={},
                      default_metrics=['bad'])

def test_get_feature_importances_and_names():
    # Test get_feature_importances
    n_samples = 20  # Ensure enough samples per class
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    # Create balanced classes with enough samples
    y = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    np.random.shuffle(y)
    
    lr = LogisticRegression().fit(X, y)
    fi = DummyPipeline.get_feature_importances(lr)
    assert isinstance(fi, np.ndarray) and fi.shape == (3,)

    # Estimator without importances
    dc = DummyClassifier().fit(X, y)
    assert DummyPipeline.get_feature_importances(dc) is None

    # Test _get_feature_names
    df = pd.DataFrame(X, columns=['a','b','c'])
    names_df = DummyPipeline._get_feature_names(df)
    assert names_df == ['a','b','c']
    names_arr = DummyPipeline._get_feature_names(X)
    assert names_arr == ['feature_0','feature_1','feature_2']

def test_compute_metrics():
    # Two folds for accuracy with y_proba
    fold_preds = [
        {'y_true': np.array([1,0]), 'y_pred': np.array([1,1]), 'y_proba': np.array([[0.8, 0.2], [0.1, 0.9]])},
        {'y_true': np.array([0,1]), 'y_pred': np.array([0,1]), 'y_proba': np.array([[0.6, 0.4], [0.7, 0.3]])},
    ]
    funcs = {'acc': lambda y_true, y_pred: float(np.mean(y_true==y_pred))}
    out = BasePipeline.compute_metrics(fold_preds, ['acc'], funcs)
    assert out['metrics']['acc']['mean'] == pytest.approx((0.5+1.0)/2)
    assert out['predictions']['y_true'].tolist() == [1,0,0,1]

def test_cross_validate_and_baseline_classification():
    # Simple balanced binary with guaranteed class sizes
    X = np.arange(40).reshape(-1,2)  # 20 samples, 2 features
    y = np.array([0]*10 + [1]*10)  # 20 samples total
    model_configs = {'dummy':{'estimator':DummyClassifier(strategy='most_frequent'), 'params':{}}}
    pipe = DummyPipeline(X, y,
                         metric_funcs={'accuracy': lambda y_true, y_pred: float(np.mean(y_true==y_pred))},
                         model_configs=model_configs,
                         default_metrics=['accuracy'],
                         cv_kwargs={'cv_strategy':'stratified','n_splits':5,'shuffle':True,'random_state':0,'n_groups':1},
                         n_jobs=1)
    # baseline
    res = pipe.baseline('dummy')
    assert 'metrics' in res and 'predictions' in res
    assert res['metrics']['accuracy']['mean'] == pytest.approx(0.5)
    # cross_validate adjust n_splits
    with pytest.warns(UserWarning):
        pipe.cv_kwargs['n_splits'] = 11  # Must be > min class count (10) to trigger warning
        _ = pipe.cross_validate(DummyClassifier(), X, y)
        
def test_feature_selection_regression():
    # Regression with DummyRegressor
    X = np.arange(40).reshape(-1,2)  # 20 samples, 2 features
    y = np.array([0]*10 + [1]*10)  # 20 samples total
    from sklearn.dummy import DummyRegressor
    model_configs = {'dummy':{'estimator': DummyRegressor(), 'params':{}}}
    pipe = DummyPipeline(X, y,
                         metric_funcs=REGRESSION_METRICS,
                         model_configs=model_configs,
                         default_metrics=['neg_mse'],
                         cv_kwargs={'cv_strategy':'stratified','n_splits':2,'shuffle':True,'random_state':0,'n_groups':1},
                         n_jobs=1)
    out = pipe.feature_selection('dummy', n_features=1)
    assert 'selected features' in out and len(out['selected features'])==1

def test_hp_search_regression():
    # HP search grid - use more samples for CV
    X = np.arange(40).reshape(-1,2)  # 20 samples, 2 features
    y = np.array([0]*10 + [1]*10)  # 20 samples total
    from sklearn.dummy import DummyRegressor
    model_configs = {'dummy':{'estimator':DummyRegressor(), 'params':{'strategy':['mean','median']}}}
    pipe = DummyPipeline(X, y,
                         metric_funcs=REGRESSION_METRICS,
                         model_configs=model_configs,
                         default_metrics=['neg_mse'],
                         cv_kwargs={'cv_strategy':'stratified','n_splits':2,'shuffle':True,'random_state':0,'n_groups':1},
                         n_jobs=1)
    out = pipe.hp_search('dummy', search_type='grid')
    assert 'best_params' in out and 'metrics' in out

def test_execute_dispatch():
    # execute baseline and feature_selection
    X = np.arange(40).reshape(-1,2)  # 20 samples, 2 features
    y = np.array([0]*10 + [1]*10)  # 20 samples total
    from sklearn.dummy import DummyRegressor
    model_configs = {'dummy':{'estimator':DummyRegressor(), 'params':{
        'strategy':['mean','median']
    }}}
    pipe = DummyPipeline(X, y,
                         metric_funcs=REGRESSION_METRICS,
                         model_configs=model_configs,
                         default_metrics=['neg_mse'],
                         cv_kwargs={'cv_strategy':'stratified','n_splits':3,'shuffle':True,'random_state':0},
                         n_jobs=1)
    r1 = pipe.execute(type='baseline', model_name='dummy')
    r2 = pipe.baseline('dummy')
    assert r1['metrics'] == r2['metrics']
    assert np.array_equal(r1['predictions']['y_true'], r2['predictions']['y_true'])
    assert np.array_equal(r1['predictions']['y_pred'], r2['predictions']['y_pred'])
    assert np.array_equal(r1['predictions']['y_proba'], r2['predictions']['y_proba'])
    assert r1['feature_importances'] == r2['feature_importances']
    r3 = pipe.execute(type='hp_search', model_name='dummy', search_type='grid')
    assert 'best_params' in r3
    assert 'metrics' in r3
    assert r3['best_params']['strategy'] in ['mean','median']

def test_cross_validate_requires_groups_for_group_strategies():
    X = np.zeros((6, 1))
    y = np.zeros(6)
    groups = None
    clf = DummyClassifier()
    pipe = DummyPipeline(X, y,
                         metric_funcs=CLASSIFICATION_METRICS,
                         model_configs={},
                         default_metrics=['accuracy'],
                         cv_kwargs={'cv_strategy':'group_kfold', 'n_splits':2, 'shuffle':True, 'random_state':0, 'n_groups':1},
                         n_jobs=1)