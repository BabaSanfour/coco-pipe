.. _decoding-result:

==============================
``ExperimentResult`` API
==============================

``ExperimentResult`` is the structured container returned by ``Experiment.run()``.
It provides 20+ accessor methods for tidy-data inspection, diagnostic reporting,
and statistical inference — all without rerunning the experiment.

---

1. Structure
=============

.. code-block:: python

   result.raw     # per-model dict of fold outputs
   result.meta    # environment provenance, task, model names, capabilities
   result.config  # original ExperimentConfig

---

2. Prediction Accessors
=========================

.. code-block:: python

   # All out-of-fold predictions in tidy long form
   preds = result.get_predictions()
   # columns: Model, Fold, SampleIndex, SampleID, Group, y_true, y_pred
   # + y_proba_0, y_proba_1, ... (if probabilities available)
   # + Subject, Session, Site (from sample_metadata)
   # + Time (sliding) or TrainTime, TestTime (generalizing)

---

3. Score Accessors
===================

.. code-block:: python

   # Per-fold, per-metric scores
   scores = result.get_detailed_scores()
   # columns: Model, Fold, Metric, Score, Time (if temporal)

   # Fold-level split information
   splits = result.get_splits(with_metadata=True)

   # Fit/predict/score timing and convergence warnings
   fit_diag = result.get_fit_diagnostics()

---

4. Curve Diagnostics
=====================

.. code-block:: python

   # ROC curves (binary or one-vs-rest multiclass)
   roc = result.get_roc_curve()
   # columns: Model, Fold, Class, FPR, TPR, Threshold, AUC

   # Precision-recall curves
   pr = result.get_pr_curve()
   # columns: Model, Fold, Class, Precision, Recall, Threshold

   # Calibration (reliability) curves
   cal = result.get_calibration_curve()

   # Probability quality summary (log-loss + Brier per fold)
   prob_diag = result.get_probability_diagnostics()

   # Summary statistics for ROC AUC
   roc_summary = result.get_roc_auc_summary()

   # Summary statistics for PR AUC
   pr_summary = result.get_pr_auc_summary()

---

5. Confusion Matrices
======================

.. code-block:: python

   # Per-fold confusion matrices in long form
   cm = result.get_confusion_matrices(normalize=True)
   # columns: Model, Fold, TrueLabel, PredLabel, Count

   # Pooled (over folds) confusion matrix
   pooled_cm = result.get_pooled_confusion_matrix(normalize="true")

---

6. Temporal Accessors
======================

.. code-block:: python

   # Score summary per timepoint (sliding only)
   temporal = result.get_temporal_score_summary()
   # columns: Model, Metric, Time, MeanScore, StdScore

   # Generalization matrix: shape (n_train_times, n_test_times)
   matrix = result.get_generalization_matrix("accuracy")
   # or long form:
   matrix_long = result.get_generalization_matrix("accuracy", long=True)

---

7. Statistical Inference
=========================

.. code-block:: python

   # Full-pipeline or lightweight permutation/binomial assessment
   assessment = result.get_statistical_assessment()

   # Lightweight (fixed-prediction, fast, biased)
   assessment_fast = result.get_statistical_assessment(lightweight=True, metric="accuracy")

   # Bootstrap CI over independent units
   ci = result.get_bootstrap_confidence_intervals(
       metric="accuracy",
       unit="Subject",
       n_bootstraps=2000,
       ci=0.95,
   )

   # Null distribution (if stored via store_null_distribution=True)
   nulls = result.get_statistical_nulls()

---

8. Model Comparison
====================

.. code-block:: python

   # Paired permutation test between two models (in-result)
   paired = result.compare_models_paired("lr", "svm", metric="accuracy", unit="Subject")

   # All pairwise comparisons with correction
   all_pairs = result.compare_models(metric="accuracy", correction="fdr_bh")

---

9. Feature Importances
=======================

.. code-block:: python

   # Mean ± std feature importance across folds
   importances = result.get_feature_importances()
   # columns: FeatureName, MeanImportance, StdImportance

   # Per-fold importances
   fold_imp = result.get_feature_importances(fold_level=True)

   # Ranked importances (descending by mean)
   ranked = result.get_feature_importances(rank=True)

---

10. Feature Selection Accessors
=================================

.. code-block:: python

   # Selected features per fold
   selected = result.get_selected_features(ordered=True)

   # Feature stability: selection rate across folds
   stability = result.get_feature_stability()

   # Per-fold univariate feature scores (k_best only)
   scores = result.get_feature_scores(with_pvalues=True)

---

11. Hyperparameter Tuning
===========================

.. code-block:: python

   # Best hyperparameters per fold
   best = result.get_best_params()

   # Full grid search results
   grid = result.get_search_results()

---

12. Model Artifact Metadata
=============================

.. code-block:: python

   # Neural model training history, checkpoints, etc.
   artifacts = result.get_model_artifacts()

---

13. Serialization
==================

.. code-block:: python

   # Serialize to JSON-compatible payload
   payload = result.to_payload()

   # Save to file
   path = result.save("results/my_result.json")

   # Load from file
   from coco_pipe.decoding.result import ExperimentResult
   loaded = ExperimentResult.load("results/my_result.json")
