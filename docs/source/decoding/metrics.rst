.. _decoding-metrics:

================
Metric Registry
================

All metrics are registered in ``coco_pipe.decoding._metrics.METRIC_REGISTRY``.
Metric/task compatibility is enforced at config validation time — before any
model is trained — preventing silent misuse of classification metrics for
regression tasks (or vice versa).

---

1. Registry API
================

.. code-block:: python

   from coco_pipe.decoding._metrics import (
       get_metric_spec,
       get_metric_names,
       get_metric_families,
       get_scorer,
       METRIC_REGISTRY,
   )

   # Inspect a single metric
   spec = get_metric_spec("accuracy")
   print(spec.name)              # "accuracy"
   print(spec.task)              # "classification"
   print(spec.family)            # "label"
   print(spec.response_method)   # "predict"
   print(spec.greater_is_better) # True

   # List all classification metrics in the "threshold_sweep" family
   names = get_metric_names(task="classification", family="threshold_sweep")

   # Get a callable scorer
   scorer = get_scorer("f1")  # sklearn-compatible callable

Each ``MetricSpec`` contains:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``name``
     - ``str``
     - Unique key in the registry.
   * - ``task``
     - ``str``
     - ``"classification"`` or ``"regression"``.
   * - ``scorer``
     - ``Callable``
     - ``(y_true, y_pred) → float``.
   * - ``response_method``
     - ``str``
     - ``"predict"`` | ``"proba"`` | ``"score"`` | ``"proba_or_score"``.
   * - ``family``
     - ``str``
     - Grouping for reporting (see below).
   * - ``greater_is_better``
     - ``bool``
     - Directionality for permutation p-values and Max-Stat correction.

---

2. Classification Metrics
==========================

2.1 Label Metrics (``family="label"``)
---------------------------------------

Require only ``predict`` output. Work with any classifier.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``accuracy``
     - Fraction of correctly classified samples. Sensitive to class imbalance.
   * - ``balanced_accuracy``
     - Mean recall per class. Recommended over ``accuracy`` for imbalanced data.
   * - ``zero_one_loss``
     - Fraction misclassified. ``1 - accuracy``. ``greater_is_better=False``.
   * - ``hamming_loss``
     - Per-label Hamming loss (fraction of labels incorrectly predicted).

2.2 Confusion-Derived Metrics (``family="confusion"``)
--------------------------------------------------------

Derived from the confusion matrix. Require only ``predict``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``f1``
     - Binary F1 score (harmonic mean of precision and recall).
   * - ``f1_macro``
     - Unweighted macro-average F1 across classes.
   * - ``f1_micro``
     - Global precision/recall pooled across classes.
   * - ``precision``
     - Positive predictive value. TP / (TP + FP).
   * - ``recall``
     - Sensitivity / true positive rate. TP / (TP + FN).
   * - ``sensitivity``
     - Synonym for recall. Binary only; raises ``ValueError`` for multiclass.
   * - ``specificity``
     - True negative rate. TN / (TN + FP). Binary only.
   * - ``jaccard``
     - Intersection-over-union for binary labels.
   * - ``matthews_corrcoef``
     - Matthews correlation coefficient. Balanced for all class distributions.
   * - ``cohen_kappa``
     - Agreement corrected for chance. Range [-1, 1].

2.3 Threshold-Sweep Metrics (``family="threshold_sweep"``)
------------------------------------------------------------

Require probability or decision scores. Use ``predict_proba`` when available,
``decision_function`` as fallback for binary classifiers.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``roc_auc``
     - Area under the ROC curve (binary OvR). Insensitive to class threshold.
   * - ``roc_auc_ovr_weighted``
     - Macro-weighted one-vs-rest AUC for multiclass.
   * - ``average_precision``
     - Area under the PR curve using sklearn's interpolated AP (binary).
   * - ``pr_auc``
     - Trapezoidal AUC of the precision-recall curve. Preferred over AP when
       positive fraction is small.

2.4 Probability-Score Metrics (``family="score_probability"``)
---------------------------------------------------------------

Require ``predict_proba``. Enable calibration diagnostics.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``log_loss``
     - Cross-entropy loss. Lower is better (``greater_is_better=False``).
   * - ``brier_score``
     - Mean squared error of probability predictions. Lower is better.

---

3. Regression Metrics (``family="regression"``)
=================================================

Require only ``predict`` output.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``r2``
     - Coefficient of determination. 1.0 is perfect fit; can be negative.
   * - ``neg_mean_squared_error``
     - Negative MSE. Negated so higher = better for optimization consistency.
   * - ``neg_mean_absolute_error``
     - Negative MAE. More robust than MSE to outliers.
   * - ``neg_root_mean_squared_error``
     - Negative RMSE. Same units as the target variable.
   * - ``explained_variance``
     - Proportion of variance explained. Similar to R² but not penalized for bias.

---

4. Compatibility Rules
========================

The registry enforces three compatibility checks at ``ExperimentConfig``
validation time:

1. **Task mismatch**: A metric's ``task`` must match ``ExperimentConfig.task``.
2. **Proba requirement**: If ``response_method == "proba"``, the model must
   declare ``predict_proba`` **or** calibration must be enabled.
3. **Score requirement**: If ``response_method == "proba_or_score"``, the model
   must declare ``predict_proba`` **or** ``decision_function``.

These checks fire before any model is trained, producing a clear ``ValueError``
with the specific metric and model name.

---

5. Custom Metrics
==================

You can extend the registry for project-specific metrics:

.. code-block:: python

   from coco_pipe.decoding._metrics import METRIC_REGISTRY, MetricSpec
   from sklearn.metrics import top_k_accuracy_score
   from functools import partial

   top2 = partial(top_k_accuracy_score, k=2, labels=[0, 1, 2])
   METRIC_REGISTRY["top2_accuracy"] = MetricSpec(
       name="top2_accuracy",
       task="classification",
       scorer=top2,
       response_method="proba",
       family="label",
       greater_is_better=True,
   )

.. warning::

   Custom metrics are added to the in-process registry only. They are not
   persisted in saved ``ExperimentResult`` payloads and must be re-registered
   in any new Python process that loads existing results.
