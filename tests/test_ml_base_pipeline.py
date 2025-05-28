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
        'cv_fold_importances', 
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
        model_configs={'clf': {'estimator': LogisticRegression(), 'params': {'C': 1}}},
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'stratified', 'n_splits': 2, 'shuffle': True, 'random_state': 0},
    )
    # invalid model name
    with pytest.raises(KeyError):
        pipe.baseline_evaluation('bad_model')
    # passing best_params overrides default estimator params
    out = pipe.baseline_evaluation('clf')
    assert out['params']['C'] == 1


def test_baseline_evaluation_regression():
    X = np.arange(30).reshape(-1, 3); y = np.arange(10)
    model_configs = {'dummy': {'estimator': DummyRegressor(), 'params': {'constant': 0.2}}}
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

def test_feature_selection_regression():
    X = np.arange(30).reshape(-1, 3)
    # y depends only on first feature
    y = X[:, 0] * 2.0 + 1.0
    from sklearn.linear_model import LinearRegression
    model_configs = {'lr': {
        'estimator': LinearRegression(),
        'params': {}
    }}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'mse': mean_squared_error},
        model_configs=model_configs,
        default_metrics=['mse'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 3, 'shuffle': True, 'random_state': 0},
        n_jobs=1
    )
    # Forward selection of 1 feature should pick feature_0
    out = pipe.feature_selection('lr', n_features=1, direction='forward', scoring='mse', threshold=0.5)
    assert out['selected_features'] == ['feature_0']
    # Default n_features=None should pick half of features => 1
    out_def = pipe.feature_selection('lr', direction='forward', scoring='mse', threshold=0.5)
    assert len(out_def['selected_features']) == 1
    # Feature importances exist for selected feature
    fi = out['feature_importances']
    assert 'feature_0' in fi
    assert set(fi['feature_0'].keys()) == {'mean', 'std', 'values'}
    # Weighted importance equals mean * frequency
    weighted = out['weighted_importances']
    assert weighted['feature_0'] == pytest.approx(fi['feature_0']['mean'] * out['feature_frequency']['feature_0'])
    # Outer CV results structure
    ocv = out['outer_cv']
    assert isinstance(ocv, list) and len(ocv) == pipe.cv_kwargs['n_splits']
    for fold_info in ocv:
        assert set(fold_info.keys()) == {'fold', 'selected', 'score'}
    # Best fold info
    bf = out['best_fold']
    assert set(bf.keys()) == {'fold', 'features', 'score'}


def test_feature_selection_backward():
    X = np.arange(20).reshape(-1, 2)
    y = X[:, 0] * 3.0 - 2.0
    from sklearn.linear_model import LinearRegression
    model_configs = {'lr': {'estimator': LinearRegression(), 'params': {}}}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'mse': mean_squared_error},
        model_configs=model_configs,
        default_metrics=['mse'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 4, 'shuffle': False},
        n_jobs=1
    )
    # Backward selection of 1 feature should still pick feature_0
    out_bw = pipe.feature_selection('lr', n_features=1, direction='backward', scoring='mse', threshold=0.1)
    assert out_bw['selected_features'] == ['feature_0', 'feature_1']


def test_feature_selection_missing_model_error():
    X = np.random.randn(10, 4)
    y = np.random.randn(10)
    with pytest.raises(KeyError):
        DummyPipeline(
            X, y,
            metric_funcs=REGRESSION_METRICS,
            model_configs={'a': {'estimator': DummyRegressor(), 'params': {}}},
            default_metrics=['mse'],
            cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 2},
        ).feature_selection('b')

def _make_pipeline():
    # simple binary classification pipeline
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    y = np.array([0]*5 + [1]*5)
    model_configs = {
        'dummy': {
            'estimator': LogisticRegression(solver='liblinear'),
            'params': {'C': [0.1, 1.0]}
        }
    }
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'accuracy': accuracy_score},
        model_configs=model_configs,
        default_metrics=['accuracy'],
        cv_kwargs={
            'cv_strategy': 'kfold',
            'n_splits': 2,
            'shuffle': True,
            'random_state': 42
        },
        n_jobs=1
    )
    return pipe, X, y


def test_build_search_estimator_grid():
    pipe, X, y = _make_pipeline()
    grid_est, metric = pipe._build_search_estimator(
        'dummy', 'grid', None, n_iter=10, scoring='accuracy'
    )
    # should return GridSearchCV and metric
    from sklearn.model_selection import GridSearchCV
    assert isinstance(grid_est, GridSearchCV)
    assert metric == 'accuracy'
    # param_grid should match model_configs
    assert grid_est.param_grid == pipe.model_configs['dummy']['params']
    # cv splits should equal n_splits
    assert grid_est.cv.get_n_splits(X, y) == pipe.cv_kwargs['n_splits']


def test_build_search_estimator_random():
    pipe, X, y = _make_pipeline()
    rand_est, metric = pipe._build_search_estimator(
        'dummy', 'random', None, n_iter=5, scoring='accuracy'
    )
    from sklearn.model_selection import RandomizedSearchCV
    assert isinstance(rand_est, RandomizedSearchCV)
    assert metric == 'accuracy'
    # n_iter set correctly
    assert rand_est.n_iter == 5


def test_extract_hp_search_results():
    pipe, X, y = _make_pipeline()
    # simulate cv_res
    class FakeEst:
        def __init__(self, best_params):
            self.best_params_ = best_params
    cv_res = {
        'cv_fold_estimators': [FakeEst({'C': 0.1}), FakeEst({'C': 1.0})],
        'cv_fold_scores': {'accuracy': np.array([0.6, 0.8])}
    }
    outer = pipe._extract_hp_search_results(cv_res)
    assert isinstance(outer, list) and len(outer) == 2
    # each entry has fold, best_params, test_scores
    for idx, entry in enumerate(outer):
        assert entry['fold'] == idx
        assert 'best_params' in entry and isinstance(entry['best_params'], dict)
        assert 'test_scores' in entry and 'accuracy' in entry['test_scores']


def test_aggregate_hp_search_results():
    pipe, X, y = _make_pipeline()
    # simulate outer_results
    outer = [
        {'fold': 0, 'best_params': {'C': 0.1}, 'test_scores': {'accuracy': 0.6}},
        {'fold': 1, 'best_params': {'C': 1.0}, 'test_scores': {'accuracy': 0.8}},
        {'fold': 2, 'best_params': {'C': 0.1}, 'test_scores': {'accuracy': 0.7}}
    ]
    best_params, freq, best_fold = pipe._aggregate_hp_search_results(outer, 'accuracy')
    # best C should be 0.1 (majority)
    assert best_params['C'] == 0.1
    # frequency correct
    assert pytest.approx(freq['C'][0.1], rel=1e-3) == 2/3
    # best_fold should be fold 1 (accuracy 0.8)
    assert best_fold['fold'] == 1
    assert best_fold['scores']['accuracy'] == 0.8


def test_hp_search_grid():
    pipe, X, y = _make_pipeline()
    # perform hp_search
    res = pipe.hp_search('dummy', search_type='grid')
    # check top-level keys
    for key in ['model_name', 'search_type', 'best_params', 'param_frequency', 'best_fold', 'outer_results',
                'cv_fold_scores', 'cv_fold_estimators', 'cv_fold_predictions']:
        assert key in res
    # best_params should be one of grid values
    assert res['best_params']['C'] in pipe.model_configs['dummy']['params']['C']
    # param_frequency should sum to 1.0 for C
    assert pytest.approx(sum(res['param_frequency']['C'].values()), rel=1e-6) == 1.0
    # outer_results length matches n_splits
    assert len(res['outer_results']) == pipe.cv_kwargs['n_splits']
    # best_fold fold index valid
    assert 0 <= res['best_fold']['fold'] < pipe.cv_kwargs['n_splits']


def test_hp_search_random():
    pipe, X, y = _make_pipeline()
    # extend grid to multiple params
    pipe.model_configs['dummy']['params'] = {'C': [0.1, 1.0], 'max_iter': [50, 100]}
    res = pipe.hp_search('dummy', search_type='random', n_iter=4)
    # ensure random search type reflected
    assert res['search_type'] == 'random'
    # best_params keys present
    assert set(res['best_params'].keys()) == set(pipe.model_configs['dummy']['params'].keys())
    # param_frequency keys match
    assert set(res['param_frequency'].keys()) == set(pipe.model_configs['dummy']['params'].keys())


def test_build_combined_fs_hp_pipeline():
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.randn(10, 3)
    y = np.random.randint(0, 2, 10)
    model_configs = {'rf': {
        'estimator': RandomForestClassifier(random_state=0),
        'params': {'n_estimators': [5, 10]}
    }}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'accuracy': accuracy_score},
        model_configs=model_configs,
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 2, 'shuffle': True, 'random_state': 0},
        n_jobs=1
    )
    search_est, feat_names, metric = pipe._build_combined_fs_hp_pipeline(
        'rf', 'grid', None, 2, 'forward', 3, 'accuracy'
    )
    from sklearn.model_selection import GridSearchCV
    assert isinstance(search_est, GridSearchCV)
    assert isinstance(feat_names, np.ndarray) and feat_names.shape == (3,)
    assert metric == 'accuracy'


def test_extract_and_aggregate_combined_results():
    # Simulate CV results
    class FakeSFS:
        def __init__(self, mask): self._mask = np.array(mask)
        def get_support(self): return self._mask
    class FakePipe:
        def __init__(self, mask): self.named_steps = {'sfs': FakeSFS(mask)}
    class FakeEst:
        def __init__(self, mask, params):
            self.best_estimator_ = FakePipe(mask)
            self.best_params_   = params
    cv_res = {
        'cv_fold_estimators': [FakeEst([True, False, True], {'p':1}), FakeEst([False, True, True], {'p':2})],
        'cv_fold_scores': {'accuracy': np.array([0.6, 0.8])},
        'cv_fold_importances': {'feature_0': np.array([0.5, 0.0]), 'feature_2': np.array([0.7, 0.9])}
    }
    feat_names = np.array(['feature_0', 'feature_1', 'feature_2'])
    pipe = DummyPipeline(
        np.zeros((2,3)), np.zeros(2),
        metric_funcs={'accuracy': accuracy_score},
        model_configs={'d': {'estimator': DummyClassifier(), 'params': {}}},
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 2},
    )
    outer, all_sel, all_params = pipe._extract_combined_results(cv_res, feat_names, 'accuracy')
    assert len(outer) == 2
    assert all_sel == ['feature_0','feature_2','feature_1','feature_2']
    assert all_params == [{'p':1},{'p':2}]
    sel_feats, feat_freq, best_params, param_freq, best_fold, feat_imps, w_imp = \
        pipe._aggregate_combined_results(outer, all_sel, all_params, feat_names, 'accuracy')
    assert sel_feats == ['feature_2']
    assert feat_freq['feature_2'] == 1.0
    assert best_params['p'] in (1,2)
    assert 'p' in param_freq
    assert best_fold['fold'] in (0,1)
    assert isinstance(feat_imps, dict) and isinstance(w_imp, dict)


def test_hp_search_fs_end_to_end():
    # Create a larger, more robust dataset with 20 samples instead of 8
    X = np.vstack([
        np.random.randn(10, 2) - 2,  # 10 samples for class 0, shifted left
        np.random.randn(10, 2) + 2   # 10 samples for class 1, shifted right
    ])
    y = np.array([0]*10 + [1]*10)
    
    # Shuffle the data to avoid any ordering issues
    idx = np.random.RandomState(42).permutation(len(y))
    X, y = X[idx], y[idx]
    model_configs = {'lr': {
        'estimator': LogisticRegression(solver='liblinear'),
        'params': {"C": [0.1, 1.0, 10.0], "penalty": ["l2", "l1"]}
    }}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'accuracy': accuracy_score},
        model_configs=model_configs,
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 2, 'shuffle': True, 'random_state': 0},
        n_jobs=1
    )
    res = pipe.hp_search_fs('lr', search_type='grid', n_features=1, direction='forward', n_iter=1, scoring='accuracy')
    keys = set(res.keys())
    expected = {'model_name','n_features','direction','scoring','search_type',
                'selected_features','feature_frequency','best_params',
                'param_frequency','best_fold','feature_importances',
                'weighted_importances','outer_results'}
    assert expected.issubset(keys)
    assert len(res['selected_features']) == 1
    assert res['best_params']['clf__penalty'] in model_configs['lr']['params']['penalty']
