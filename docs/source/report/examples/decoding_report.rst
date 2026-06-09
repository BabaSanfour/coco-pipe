.. _report-example-decoding:

==================================================
Example: Decoding Report From an ``ExperimentResult``
==================================================

A full decoding workflow: run the experiment, save the result, build
a paper-grade HTML report covering scores, performance curves,
temporal decoding, statistical assessment, feature importance, and
sensor topomaps.

---

1. Run the Experiment
=======================

(Same setup as :ref:`decoding-example-basic`; refer there for the
scientific rationale.)

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, CVConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
       SlidingEstimatorConfig,
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "lr": ClassicalModelConfig(estimator="LogisticRegression",
                                      params={"max_iter": 500}),
           "rf": ClassicalModelConfig(estimator="RandomForestClassifier",
                                      params={"n_estimators": 100}),
           "sliding_lr": SlidingEstimatorConfig(
               base_estimator="LogisticRegression",
               base_params={"max_iter": 500},
           ),
       },
       metrics=["accuracy", "balanced_accuracy", "roc_auc"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5,
                   group_key="Subject"),
       evaluation=StatisticalAssessmentConfig(
           enabled=True,
           chance=ChanceAssessmentConfig(method="permutation",
                                         n_permutations=1000,
                                         unit_of_inference="group_mean"),
       ),
       use_scaler=True,
       random_state=42,
   )

   result = Experiment(config).run(
       X, y,
       sample_metadata={"Subject": subject_ids, "Channel": channel_ids,
                        "Feature": feature_ids},
       observation_level="epoch",
   )
   result.save("results/cohort_a.json")

---

2. One-Shot Report
====================

The simplest path: the factory does everything.

.. code-block:: python

   from coco_pipe.report import from_experiment_result

   feature_meta = pd.DataFrame({
       "FeatureName": feature_ids,
       "Channel":     channel_ids,
       "FeatureFamily": [name.split("_")[0] for name in feature_ids],
   })

   report = from_experiment_result(
       result,
       feature_metadata=feature_meta,
       info=raw.info,                  # enables topomap section
       title="Cohort A — EEG Decoding",
       output_path="reports/cohort_a.html",
   )

The resulting report includes (every section is silently skipped if
its source data is absent):

================================================  ========================================
Section                                            Source accessor
================================================  ========================================
Overview                                           ``result.meta``
Configuration                                      ``result.config``
Provenance                                         ``result.config.provenance``
Model summary                                      ``get_detailed_scores()``
CV diagnostics                                     ``get_detailed_scores()`` + ``get_splits()``
Performance (ROC / PR / Calibration)               ``get_roc_curve()`` / ``get_pr_curve()`` /
                                                   ``get_calibration_curve()``
Statistical assessment                             ``get_statistical_assessment()``
Confusion + probability                            ``get_confusion_matrices()`` +
                                                   ``get_probability_diagnostics()``
Temporal decoding                                  ``get_temporal_score_summary()`` +
                                                   ``get_generalization_matrix()``
Features (importance + stability)                  ``get_feature_importances()`` +
                                                   ``get_feature_stability()``
Topomaps                                           ``get_feature_importances()`` +
                                                   ``info`` / ``coords``
Fit diagnostics                                    ``get_fit_diagnostics()``
Caveats                                            Derived from ``result.meta``
Export inventory                                   File list + payload sizes
================================================  ========================================

---

3. Hand-Built Variant (Narrower Scope)
========================================

When the default report is too long, build a focused one with the
fluent API:

.. code-block:: python

   from coco_pipe.report import Report

   report = (
       Report(title="Cohort A — Decoding Highlights")
       .add_decoding_overview(result)
       .add_decoding_summary(result)
       .add_decoding_performance(result)
       .add_decoding_temporal(result, metric="accuracy")
       .add_decoding_statistical_assessment(result, metric="accuracy")
       .add_decoding_features(result, feature_metadata=feature_meta)
   )
   report.save("reports/cohort_a_highlights.html")

---

4. Narrow ``sections=`` Subset Via the Factory
================================================

The factory also accepts a section subset:

.. code-block:: python

   from coco_pipe.report.decoding import make_decoding_report

   report = make_decoding_report(
       result,
       feature_metadata=feature_meta,
       info=raw.info,
       sections=[
           "overview",
           "model_summary",
           "performance",
           "statistical",
           "topomaps",
       ],
       output_path="reports/cohort_a_short.html",
   )

Unknown section names raise ``ValueError`` at the call site. See
:ref:`report-section-decoding` for the full catalog.

---

5. Cross-Cohort Comparison
============================

Build one report per cohort, then merge:

.. code-block:: python

   from coco_pipe.decoding.result import ExperimentResult
   from coco_pipe.report import from_experiment_result, merge_reports

   r1 = from_experiment_result(
       ExperimentResult.load("results/cohort_a.json"),
       title="Cohort A",
   )
   r2 = from_experiment_result(
       ExperimentResult.load("results/cohort_b.json"),
       title="Cohort B",
   )

   merged = merge_reports(r1, r2, title="Cohort A vs Cohort B")
   merged.save("reports/cross_cohort.html")

Section titles in the merged report are prefixed with the source
report's title (``"Cohort A — Overview"``, ``"Cohort B — Overview"``,
…). The TOC sidebar groups them naturally.

---

6. Loading a Result From Disk
===============================

Reports built from a loaded result are byte-identical to ones built
from the in-memory version:

.. code-block:: python

   from coco_pipe.decoding.result import ExperimentResult

   result = ExperimentResult.load("results/cohort_a.json")
   report = from_experiment_result(result, output_path="reports/cohort_a.html")

This means analysis and reporting can live in different scripts,
machines, or scheduler stages.

---

7. Tips and Common Failures
=============================

================================  =====================================================
Tip                               Detail
================================  =====================================================
Always pass ``feature_metadata``  Enables sensor topomaps and feature-family
                                  annotations. Without it the topomap section is
                                  skipped silently.
``info`` vs ``coords``            Either works; ``info`` (MNE) is preferred when
                                  available. Otherwise pass a tidy coord frame with
                                  ``x``, ``y`` columns.
``interactive=True``              Currently a no-op (raises a deprecation-style
                                  warning); all plots are already interactive.
                                  Pass ``False`` for cleanest output.
Section-level failures            Wrapped in ``try / except`` — a missing accessor
                                  logs a debug message and skips, the rest of the
                                  report still renders.
================================  =====================================================
