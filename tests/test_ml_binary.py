import pytest
import numpy as np
from sklearn.datasets import make_classification

from coco_pipe.ml.binary_classification import BinaryClassificationPipeline
from coco_pipe.ml.config import BINARY_METRICS, BINARY_MODELS, DEFAULT_CV


def test_binary_metrics_correctness():
    # Create a small synthetic probability set
    fold_preds = [
        {
            'y_true': np.array([0, 1, 0]),
            'y_pred': np.array([0, 1, 1]),
            'y_proba': np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
        },
        {
            'y_true': np.array([1, 0]),
            'y_pred': np.array([1, 0]),
            'y_proba': np.array([[0.2, 0.8], [0.6, 0.4]])
        }
    ]
    # Instantiate a dummy pipeline just to access compute_metrics
    # X and y here are unused in compute_metrics
    dummy_X = np.zeros((5, 2))
    dummy_y = np.array([0,1,0,1,0])
    pipe = BinaryClassificationPipeline(
        X=dummy_X,
        y=dummy_y,
        models='all',
        metrics=list(BINARY_METRICS.keys()),
        cv_kwargs={**DEFAULT_CV, 'n_splits':2},
        n_jobs=1
    )
    # Compute metrics
    results = pipe.compute_metrics(fold_preds, pipe.metrics, pipe.metric_funcs)

    # Accuracy: fold1 = 2/3, fold2 = 2/2 => mean = (0.6667 + 1)/2
    expected_acc = (2/3 + 1) / 2
    assert pytest.approx(results['metrics']['accuracy']['mean'], rel=1e-3) == expected_acc

    # ROC AUC and average precision on concatenated samples
    # y_true = [0,1,0,1,0]
    y_true = np.concatenate([fp['y_true'] for fp in fold_preds])
    y_proba = np.concatenate([fp['y_proba'] for fp in fold_preds], axis=0)

    # Manual ROC AUC
    from sklearn.metrics import roc_auc_score, average_precision_score
    expected_roc = roc_auc_score(y_true, y_proba[:,1])
    expected_ap = average_precision_score(y_true, y_proba[:,1])

    assert pytest.approx(results['metrics']['roc_auc']['mean'], rel=1e-6) == expected_roc
    assert pytest.approx(results['metrics']['average_precision']['mean'], rel=1e-6) == expected_ap


def test_baseline_all_models_run():
    # Generate a simple binary classification dataset
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=3,
        n_redundant=0,
        n_classes=2,
        random_state=0
    )
    # Test each model through baseline
    metrics = ['accuracy', 'roc_auc', 'average_precision']
    results = []
    for model_name in BINARY_MODELS.keys():
        pipe = BinaryClassificationPipeline(
            X=X,
            y=y,
            models=[model_name],
            metrics=metrics,
            random_state=0,
            n_jobs=1,
            cv_kwargs={**DEFAULT_CV, 'n_splits':5, 'shuffle':True},
            groups=None
        )
        res = pipe.baseline(model_name)
        # Check keys
        assert 'model' in res
        assert 'feature_importances' in res
        assert 'metrics' in res
        assert 'predictions' in res
        # Predictions length matches input
        assert len(res['predictions']['y_true']) == len(y)
        # Metric values between 0 and 1
        for m in metrics:
            print(m, res['metrics'][m])
            val = res['metrics'][m]['mean']
            # For accuracy and roc_auc: between 0 and 1
            assert 0.0 <= val <= 1.0
        results.append(res)
    # Ensure we tested all models
    assert len(results) == len(BINARY_MODELS)
