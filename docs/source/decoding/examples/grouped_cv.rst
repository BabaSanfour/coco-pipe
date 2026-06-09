.. _decoding-example-grouped:

========================
Example: Grouped CV
========================

This example demonstrates the correct use of group-based cross-validation
for multi-session EEG data, where multiple epochs per subject and multiple
sessions per subject must remain exclusive to training or test folds.

**Scientific context**: EEG data from 30 subjects, 2 sessions each, 20 epochs
per session. Goal: classify cognitive states while ensuring the model cannot
exploit within-subject patterns that would inflate test accuracy.

---

1. Prepare Multi-Session Data
================================

.. code-block:: python

   import numpy as np

   n_subjects = 30
   n_sessions_per_subject = 2
   n_epochs_per_session = 20
   n_features = 64

   rng = np.random.default_rng(42)
   total_epochs = n_subjects * n_sessions_per_subject * n_epochs_per_session

   X = rng.standard_normal((total_epochs, n_features))
   y = np.tile([0, 1], total_epochs // 2)

   # Build metadata with Subject, Session, Site columns
   subject_ids = np.repeat(np.arange(n_subjects), n_sessions_per_subject * n_epochs_per_session)
   session_ids = np.tile(
       np.repeat(np.arange(n_sessions_per_subject), n_epochs_per_session),
       n_subjects
   )
   site_ids = np.where(subject_ids < 15, "SiteA", "SiteB")

---

2. Configure Grouped CV
==========================

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, CVConfig, TuningConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "lr": ClassicalModelConfig(
               estimator="LogisticRegression",
               params={"solver": "liblinear"},
           )
       },
       metrics=["balanced_accuracy", "roc_auc"],
       cv=CVConfig(
           strategy="stratified_group_kfold",
           n_splits=5,
           group_key="Subject",    # subjects exclusive to test folds
       ),
       tuning=TuningConfig(
           enabled=True,
           scoring="balanced_accuracy",
           # inner CV automatically group-based (Subject) to prevent leakage
       ),
       grids={"lr": {"C": [0.01, 0.1, 1.0, 10.0]}},
       use_scaler=True,
       random_state=42,
   )

---

3. Run with Metadata
=====================

.. code-block:: python

   result = Experiment(config).run(
       X,
       y,
       sample_metadata={
           "Subject": subject_ids,
           "Session": session_ids,
           "Site": site_ids,
       },
       observation_level="epoch",
   )

---

4. Site-Stratified Analysis
=============================

.. code-block:: python

   # Bootstrap CI by site
   ci_site_a = result.get_bootstrap_confidence_intervals(
       metric="balanced_accuracy", unit="Subject",
       n_bootstraps=2000,
   )

   # Predictions include Subject, Session, Site columns
   preds = result.get_predictions()
   by_site = preds.groupby("Site")["y_pred"].value_counts()
   print(by_site)

---

5. Hyperparameter Diagnostics
================================

.. code-block:: python

   # Best regularization parameter per fold
   best_params = result.get_best_params()
   print(best_params[["Fold", "C"]])

   # Full grid search diagnostics
   search_results = result.get_search_results()
   print(search_results.sort_values("MeanTestScore", ascending=False).head())
