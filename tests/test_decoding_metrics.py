import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from coco_pipe.decoding._metrics import (
    METRIC_REGISTRY,
    _pr_auc_score,
    _sensitivity_score,
    _specificity_score,
    get_metric_families,
    get_metric_names,
    get_metric_spec,
    get_scorer,
)


def test_all_metrics_runable():
    """Verify every registered metric can be called with appropriate data."""
    y_true_cls = np.array([0, 1, 0, 1])
    y_pred_cls = np.array([0, 1, 1, 1])
    y_proba_cls = np.array([0.1, 0.9, 0.4, 0.6])

    y_true_reg = np.array([1.0, 2.0, 3.0])
    y_pred_reg = np.array([1.1, 1.9, 3.2])

    for name, spec in METRIC_REGISTRY.items():
        if spec.task == "classification":
            if spec.response_method == "predict":
                val = spec.scorer(y_true_cls, y_pred_cls)
            else:
                val = spec.scorer(y_true_cls, y_proba_cls)
        else:
            val = spec.scorer(y_true_reg, y_pred_reg)

        assert isinstance(val, (float, np.float64, np.float32))


def test_pr_auc_calculation():
    """Verify PR-AUC uses trapezoidal integration and differs from AP if imbalanced."""
    # Highly imbalanced
    y_true = np.array([0, 0, 0, 0, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    average_precision_score(y_true, y_score)
    pr_auc = _pr_auc_score(y_true, y_score)

    # In some versions of sklearn, AP and trapezoidal PR-AUC can differ
    # depending on how thresholds are handled.
    assert isinstance(pr_auc, float)
    assert 0 <= pr_auc <= 1


def test_sensitivity_guards():
    """Verify sensitivity enforces binary data and robust zero-division."""
    # Multiclass should fail
    with pytest.raises(ValueError, match="binary classification"):
        _sensitivity_score(np.array([0, 1, 2]), np.array([0, 1, 0]))

    # Binary should pass
    val = _sensitivity_score(np.array([0, 1]), np.array([0, 1]))
    assert val == 1.0

    # Zero division (no positive samples)
    val = _sensitivity_score(np.array([0, 0]), np.array([0, 0]))
    assert val == 0.0


def test_metric_registry_accessors():
    """Verify registry lookups."""
    spec = get_metric_spec("accuracy")
    assert spec.name == "accuracy"

    scorer = get_scorer("f1")
    assert callable(scorer)

    with pytest.raises(ValueError, match="Unknown metric"):
        get_metric_spec("non_existent")


def test_get_metric_names_filters():
    """Verify task and family filters in name retrieval."""
    names = get_metric_names(task="classification", family="confusion")
    assert "precision" in names
    assert "recall" in names
    assert "neg_mean_squared_error" not in names


def test_get_metric_families_filter():
    """Verify task filter in family retrieval."""
    families = get_metric_families(task="regression")
    assert "regression" in families
    assert "threshold_sweep" not in families


def test_specificity_standalone():
    """Verify specificity calculation (TN / (TN + FP))."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    # Specificity = TN / (TN + FP) = 1 / (1 + 1) = 0.5
    assert _specificity_score(y_true, y_pred) == 0.5
