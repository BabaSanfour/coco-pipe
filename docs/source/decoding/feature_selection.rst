.. _decoding-feature-selection:

============================
Feature Selection
============================

``coco_pipe.decoding`` supports two feature selection strategies that execute
**inside** each outer CV fold on the training partition only, guaranteeing
zero test-set leakage.

---

1. Filter Selection (``k_best``)
==================================

``SelectKBest`` selects the top-``k`` features based on a univariate statistical
test. It has no inner CV loop. It is fast and well-suited for high-dimensional data
(e.g., many EEG channels/frequency bins) where a quick, interpretable feature
ranking is desired.

.. code-block:: python

   from coco_pipe.decoding.configs import (
       ExperimentConfig, CVConfig, ClassicalModelConfig, FeatureSelectionConfig
   )

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       feature_selection=FeatureSelectionConfig(
           enabled=True,
           method="k_best",
           n_features=20,
           scoring="accuracy",     # optional; defaults to task-appropriate test
       ),
   )

1.1 Score Function
-------------------

For classification, the default univariate test is ``f_classif`` (ANOVA F-value).
For regression, it is ``f_regression``. Override via ``feature_selection.scoring``.

1.2 Accessing Feature Scores
------------------------------

After fitting, retrieve per-fold and per-feature scores:

.. code-block:: python

   feature_scores = result.get_feature_scores()
   # columns: FeatureName, Fold, Score, PValue

   # Mean score across folds
   mean_scores = feature_scores.groupby("FeatureName")["Score"].mean().sort_values(ascending=False)

---

2. Sequential Feature Selection (``sfs``)
==========================================

``SequentialFeatureSelector`` is a wrapper-based method. It iteratively adds
(forward SFS) or removes (backward SFS) features by evaluating the model's
cross-validated performance on each candidate feature set. Because it uses the
model's predictive performance as the selection criterion, it is more powerful
than filter methods but significantly more expensive.

.. code-block:: python

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["balanced_accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       feature_selection=FeatureSelectionConfig(
           enabled=True,
           method="sfs",
           n_features=10,
           scoring="balanced_accuracy",    # criterion for SFS inner evaluation
           cv=CVConfig(strategy="stratified_group_kfold", n_splits=3, group_key="Subject"),
           direction="forward",            # or "backward"
       ),
   )

2.1 Inner CV for SFS
----------------------

SFS requires an inner CV loop to evaluate candidate feature sets. When omitted,
``coco_pipe.decoding`` derives the inner SFS CV from:

1. ``tuning.cv`` if tuning is enabled.
2. The outer CV family (group-based if outer is group-based).

When the outer CV is group-based, the SFS inner CV is automatically group-based.
Overriding requires ``allow_nongroup_inner_cv=True``.

2.2 Group-Aware SFS
--------------------

``coco_pipe.decoding`` uses scikit-learn metadata routing to pass the
outer-fold training groups into the SFS inner CV. This requires
``scikit-learn >= 1.6``.

2.3 SFS with Tuning
--------------------

SFS combined with hyperparameter tuning evaluates feature subsets inside the
tuning inner folds. ``coco_pipe.decoding`` uses a ``sklearn.pipeline.Pipeline``
cache to avoid redundant refitting:

.. code-block:: python

   config = ExperimentConfig(
       ...,
       feature_selection=FeatureSelectionConfig(enabled=True, method="sfs", n_features=10),
       tuning=TuningConfig(enabled=True, scoring="accuracy"),
       grids={"lr": {"C": [0.1, 1.0, 10.0]}},
   )

.. warning::

   SFS + tuning is computationally intensive. Reduce the outer ``n_splits`` or
   the SFS inner ``n_splits`` for development runs.

---

3. Feature Stability Analysis
================================

For both ``k_best`` and ``sfs``, ``coco_pipe.decoding`` tracks which features
were selected in each fold. The stability score is the proportion of folds in
which a feature was selected:

.. code-block:: python

   stability = result.get_feature_stability()
   # columns: FeatureName, SelectionRate, MeanRank, StdRank

   # Most stable features
   top = stability.sort_values("SelectionRate", ascending=False).head(20)

.. note::

   Feature stability across folds is a measure of **generalizability**, not
   importance. A feature selected in all folds is a robust signal across the
   sampled subjects, regardless of its average selection score.

---

4. Selected Features per Fold
================================

.. code-block:: python

   selected = result.get_selected_features()
   # columns: FeatureName, Fold, Rank

   # Features selected in every fold
   universal = selected.groupby("FeatureName")["Fold"].count()
   universal = universal[universal == config.cv.n_splits].index.tolist()

---

5. Compatibility Notes
========================

- Feature selection is only valid for 2D tabular inputs (``input_kind in {"tabular_2d", "embedding_2d"}``).
- Feature selection is **incompatible** with temporal estimators (``SlidingEstimator``, ``GeneralizingEstimator``).
  The registry blocks this at validation time.
- ``k_best`` does not support ranked importances beyond fold scores/p-values.
  For importance-based selection, use tree ensemble importances via
  ``result.get_feature_importances()``.
