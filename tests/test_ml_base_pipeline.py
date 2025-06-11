import pytest
import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score

from coco_pipe.ml.base import BasePipeline
from coco_pipe.ml.config import CLASSIFICATION_METRICS, REGRESSION_METRICS


class DummyPipeline(BasePipeline):
    """Concrete subclass for testing BasePipeline."""
    pass


def test_validate_input_errors():
    with pytest.raises(ValueError):
        DummyPipeline(
            X="not array",
            y=np.zeros(3),
            metric_funcs=CLASSIFICATION_METRICS,
            model_configs={},
            default_metrics=["accuracy"]
        )
    with pytest.raises(ValueError):
        DummyPipeline(
            X=np.zeros((3, 2)),
            y=np.zeros(4),
            metric_funcs=CLASSIFICATION_METRICS,
            model_configs={},
            default_metrics=["accuracy"]
        )
    with pytest.raises(ValueError):
        DummyPipeline(
            X=np.zeros((3, 2)),
            y=np.zeros(3),
            metric_funcs=CLASSIFICATION_METRICS,
            model_configs={},
            default_metrics=["accuracy"],
            groups=np.zeros(10)
        )

def test_select_columns():
    df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [7, 8, 9]
    })
    selected = DummyPipeline._select_columns(df, [True, False, True])
    expected = pd.DataFrame({'a': [1, 2, 3], 'c': [7, 8, 9]})
    pd.testing.assert_frame_equal(selected, expected)

    # Test with numpy array
    arr = np.array([[1, 2], [3, 4], [5, 6]])
    selected_arr = DummyPipeline._select_columns(arr, [0])
    expected_arr = np.array([[1], [3], [5]])
    np.testing.assert_array_equal(selected_arr, expected_arr)

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
    X = np.random.randn(20, 3)
    y = np.concatenate([np.zeros(10), np.ones(10)])
    np.random.shuffle(y)

    lr = LogisticRegression(solver='liblinear').fit(X, y)
    imp = DummyPipeline._extract_feature_importances(lr)
    assert isinstance(imp, np.ndarray) and imp.shape == (3,)

    dc = DummyClassifier().fit(X, y)
    assert DummyPipeline._extract_feature_importances(dc) is None

    df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    assert DummyPipeline._get_feature_names(df) == ['a', 'b', 'c']
    arr_names = DummyPipeline._get_feature_names(X)
    assert arr_names == ['feature_0', 'feature_1', 'feature_2']


def test_cross_val_and_baseline_evaluation_classification():
    X = np.arange(40).reshape(-1, 2)
    y = np.array([0]*10 + [1]*10)
    model_configs = {'dummy': {
        'estimator': DummyClassifier(strategy='most_frequent'),
        'params': {}
    }}
    def acc(y_t, y_p): return float(np.mean(y_t == y_p))

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

    # cross_val
    cv_res = pipe.cross_val(DummyClassifier(strategy='most_frequent'), X, y)
    for key in ['cv_fold_predictions', 'cv_fold_scores', 'cv_fold_estimators', 'cv_fold_importances']:
        assert key in cv_res
    assert np.all(cv_res['cv_fold_scores']['accuracy'] == pytest.approx(0.5))

    # baseline_evaluation
    eval_res = pipe.baseline_evaluation('dummy')
    assert eval_res['model_name'] == 'dummy'
    # compare fold scores
    assert np.all(
        eval_res['metric_scores']['accuracy']['fold_scores']
        == cv_res['cv_fold_scores']['accuracy']
    )
    # predictions key
    preds = eval_res['predictions']
    assert set(preds.keys()) >= {'y_true', 'y_pred', 'fold_preds'}


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
    with pytest.raises(KeyError):
        pipe.baseline_evaluation('bad_model')

    out = pipe.baseline_evaluation('clf')
    # params comes back empty dict (no default_params in config)
    assert out['params'] == {}


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
    assert np.all(out['metric_scores']['mse']['fold_scores'] >= 0)


def test_feature_selection_regression():
    X = np.arange(30).reshape(-1, 3)
    y = X[:, 0] * 2.0 + 1.0
    model_configs = {'lr': {'estimator': LinearRegression(), 'params': {}}}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'mse': mean_squared_error},
        model_configs=model_configs,
        default_metrics=['mse'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 3, 'shuffle': True, 'random_state': 0},
        n_jobs=1
    )
    out = pipe.feature_selection('lr', n_features=1, direction='forward', scoring='mse')
    assert isinstance(out['selected_per_fold'], dict)
    # feature_importances keys
    fi = out['feature_importances']
    assert 'feature_0' in fi
    keys = set(fi['feature_0'].keys())
    assert {'mean', 'std', 'weighted_mean', 'weighted_std', 'fold_importances'} <= keys
    # weighted mean consistency
    freq = out['feature_frequency']['feature_0']
    assert fi['feature_0']['weighted_mean'] == pytest.approx(fi['feature_0']['mean'] * freq)
    # fold count
    spp = out['selected_per_fold']
    assert len(spp) == pipe.cv_kwargs['n_splits']
    best = out['best_fold']
    # best fold has metric 'mse'
    assert 'mse' in best and 'features' in best and 'fold' in best


def test_feature_selection_backward():
    X = np.arange(20).reshape(-1, 2)
    y = X[:, 0] * 3.0 - 2.0
    model_configs = {'lr': {'estimator': LinearRegression(), 'params': {}}}
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'mse': mean_squared_error},
        model_configs=model_configs,
        default_metrics=['mse'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 4, 'shuffle': False},
        n_jobs=1
    )
    out_bw = pipe.feature_selection('lr', n_features=1, direction='backward', scoring='mse')
    # backward, single feature remains or both if equal score
    assert isinstance(out_bw['selected_features'], set)
    assert out_bw['selected_features']


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
    X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
    y = np.array([0]*5 + [1]*5)
    model_configs = {
        'dummy': {
            'estimator': LogisticRegression(solver='liblinear'),
            'params': {'C': [0.1, 1.0]}
        }
    }
    return DummyPipeline(
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
    ), X, y


def test_build_search_estimator_grid():
    pipe, X, y = _make_pipeline()
    grid_est, metric = pipe._build_search_estimator('dummy', 'grid', None, n_iter=10, scoring='accuracy')
    from sklearn.model_selection import GridSearchCV
    assert isinstance(grid_est, GridSearchCV)
    assert metric == 'accuracy'
    assert grid_est.param_grid == pipe.model_configs['dummy'].param_grid
    assert grid_est.cv.get_n_splits(X, y) == pipe.cv_kwargs['n_splits']


def test_build_search_estimator_random():
    pipe, X, y = _make_pipeline()
    rand_est, metric = pipe._build_search_estimator('dummy', 'random', None, n_iter=5, scoring='accuracy')
    from sklearn.model_selection import RandomizedSearchCV
    assert isinstance(rand_est, RandomizedSearchCV)
    assert metric == 'accuracy'
    assert rand_est.n_iter == 5


def test_extract_hp_search_params():
    pipe, X, y = _make_pipeline()
    class FakeEst:
        def __init__(self, best_params): self.best_params_ = best_params
    cv_estimators = [FakeEst({'C': 0.1}), FakeEst({'C': 1.0})]
    bp_per_fold, bp, freq = pipe._extract_hp_search_params(cv_estimators)
    assert isinstance(bp_per_fold, dict) and len(bp_per_fold) == 2
    assert isinstance(bp, dict) and 'C' in bp
    assert isinstance(freq, dict) and 'C' in freq


def test_hp_search_grid():
    pipe, X, y = _make_pipeline()
    res = pipe.hp_search('dummy', search_type='grid')
    for key in ['model_name', 'hp search parameters', 'best_params', 'param_frequency',
                'best_fold', 'predictions', 'metric_scores', 'folds_estimators'
                ]:
        # 'outer_results' is renamed to 'best_params_per_fold'
        alt = 'best_params_per_fold' if key == 'outer_results' else key
        assert alt in res
    assert res['hp search parameters']['search type'] == 'grid'
    assert res['best_params']['C'] in pipe.model_configs['dummy'].param_grid['C']
    assert pytest.approx(sum(res['param_frequency']['C'].values()), rel=1e-6) == 1.0
    assert len(res['best_params_per_fold']) == pipe.cv_kwargs['n_splits']
    assert 0 <= res['best_fold']['fold'] < pipe.cv_kwargs['n_splits']


def test_hp_search_random_and_invalid_grid():
    pipe, X, y = _make_pipeline()
    # invalid grid update should fail
    with pytest.raises(ValueError):
        pipe.update_model_params('dummy', {'C': 0.5}, update_estimator=False, update_config=True, param_type='hp_search')
    # valid random search after correcting grid
    pipe.model_configs['dummy'].param_grid = {'C': [0.1, 1.0], 'max_iter': [50, 100]}
    res = pipe.hp_search('dummy', search_type='random', n_iter=4)
    assert res['hp search parameters']['search type'] == 'random'
    assert set(res['param_frequency'].keys()) == set(pipe.model_configs['dummy'].param_grid.keys())


def test_build_combined_fs_hp_pipeline():
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


def test_hp_search_fs_end_to_end():
    X = np.vstack([
        np.random.randn(10, 2) - 2,
        np.random.randn(10, 2) + 2
    ])
    y = np.array([0]*10 + [1]*10)
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
    expected = {
        'model_name', 'metric_scores', 'selected_features', 'feature_frequency',
        'feature_importances', 'best_params', 'selected_per_fold', 'best_params_per_fold',
        'best_fold', 'folds_estimators', 'hp search and fs parameters'
    }
    assert expected.issubset(res.keys())
    assert len(res['selected_features']) == 1


def test_execute_method():
    X = np.vstack([
        np.random.randn(10, 3) - 2,
        np.random.randn(10, 3) + 2
    ])
    y = np.array([0]*10 + [1]*10)

    model_configs = {
        'lr': {
            'estimator': LogisticRegression(solver='liblinear'),
            'default_params': {'C': 0.1, 'penalty': "l2"},
            'hp_search_params': {'C': [0.1, 1.0], 'penalty': ["l2", "l1"]}
        }
    }
    pipe = DummyPipeline(
        X, y,
        metric_funcs={'accuracy': accuracy_score},
        model_configs=model_configs,
        default_metrics=['accuracy'],
        cv_kwargs={'cv_strategy': 'kfold', 'n_splits': 2, 'shuffle': True, 'random_state': 0},
        n_jobs=1
    )

    # baseline
    r1 = pipe.execute(type='baseline', model_name='lr')
    assert 'metric_scores' in r1 and 'accuracy' in r1['metric_scores']
    # feature_selection
    r2 = pipe.execute(type='feature_selection', model_name='lr', n_features=2, direction='forward')
    assert 'selected_features' in r2 and 'feature_frequency' in r2
    # hp_search
    r3 = pipe.execute(type='hp_search', model_name='lr', search_type='grid')
    assert 'best_params' in r3 and 'param_frequency' in r3
    # hp_search_fs
    r4 = pipe.execute(type='hp_search_fs', model_name='lr', search_type='grid', n_features=2)
    assert 'selected_features' in r4 and 'best_params' in r4

    with pytest.raises(ValueError):
        pipe.execute(type='invalid', model_name='lr')
    with pytest.raises(TypeError):
        pipe.execute(type='baseline')
    with pytest.raises(KeyError):
        pipe.execute(type='baseline', model_name='nope')


def test_get_model_params_single_model():
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    model_configs = {
        'lr': {
            'estimator': LogisticRegression(),
            'default_params': {'C': 1.0, 'random_state': 42},
            'hp_search_params': {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
        }
    }
    pipe = DummyPipeline(X, y, metric_funcs={'accuracy': accuracy_score},
                         model_configs=model_configs,
                         default_metrics=['accuracy'], n_jobs=1)
    params = pipe.get_model_params('lr')
    assert params['estimator_type'] == 'LogisticRegression'
    assert params['init_params'] == {'C': 1.0, 'random_state': 42}
    assert params['param_grid'] == {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}


def test_get_model_params_all_models_and_update():
    X = np.random.randn(20, 4)
    y = np.random.randint(0, 2, 20)
    model_configs = {
        'lr': {
            'estimator': LogisticRegression(),
            'default_params': {'random_state': 42},
            'hp_search_params': {'C': [0.1, 1.0, 10.0]}
        }
    }
    pipe = DummyPipeline(X, y, metric_funcs={'accuracy': accuracy_score},
                         model_configs=model_configs,
                         default_metrics=['accuracy'], n_jobs=1)

    pipe.update_model_params('lr', {'C': 5.0}, param_type='default')
    p1 = pipe.get_model_params('lr')
    assert p1['init_params']['C'] == 5.0

    pipe.update_model_params('lr', {'penalty': ['l1', 'l2']}, param_type='hp_search')
    p2 = pipe.get_model_params('lr')
    assert p2['param_grid']['penalty'] == ['l1', 'l2']


def test_get_model_params_nonexistent_model():
    X = np.random.randn(20, 4); y = np.random.randint(0, 2, 20)
    pipe = DummyPipeline(X, y,
                         metric_funcs={'accuracy': accuracy_score},
                         model_configs={'lr': {'estimator': LogisticRegression(), 'params': {}}},
                         default_metrics=['accuracy'], n_jobs=1)
    with pytest.raises(KeyError):
        pipe.get_model_params('nonexistent')


def test_update_model_params_invalid_grid():
    X = np.random.randn(10, 3); y = np.random.randint(0, 2, 10)
    pipe = DummyPipeline(X, y,
                         metric_funcs={'accuracy': accuracy_score},
                         model_configs={'d': {'estimator': LogisticRegression(), 'params': {}}},
                         default_metrics=['accuracy'], n_jobs=1)
    with pytest.raises(ValueError):
        pipe.update_model_params('d', {'C': 1.0}, update_estimator=False, update_config=True, param_type='hp_search')


def test_reset_and_list_models():
    X = np.random.randn(10, 4); y = np.random.randint(0, 2, 10)
    model_configs = {
        'lr': {'estimator': LogisticRegression(C=1.0, solver='liblinear'), 'params': {'C': [1, 2]}},
        'rf': {'estimator': RandomForestClassifier(n_estimators=5), 'params': {}}
    }
    pipe = DummyPipeline(X, y,
                         metric_funcs={'accuracy': accuracy_score},
                         model_configs=model_configs,
                         default_metrics=['accuracy'], n_jobs=1)

    # list_models
    names = pipe.list_models()
    assert set(names.keys()) == {'lr', 'rf'}
    assert names['lr'] == 'LogisticRegression'

    # reset_model_params error
    with pytest.raises(KeyError):
        pipe.reset_model_params('nope')

    # update & reset roundtrip
    orig = deepcopy(pipe.get_model_params('lr'))
    pipe.update_model_params('lr', {'C': 3.0}, param_type='default')
    assert pipe.get_model_params('lr')['init_params']['C'] == 3.0
    pipe.reset_model_params('lr')
    after = pipe.get_model_params('lr')
    assert after['init_params'] == orig['init_params']
    assert after['param_grid'] == orig['param_grid']