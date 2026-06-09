.. _viz-example-static-report:

================================================
Example: Building a Static Decoding Report Figure
================================================

This example walks through a four-panel paper figure built from an
``ExperimentResult``: aggregate scores, ROC curve with permutation null band,
confusion matrix, and top-feature importances. All panels share a single
:func:`coco_pipe.viz.theme.coco_theme` scope so they are visually consistent
when rendered as a PDF.

**Scientific context**: A grouped 5-fold logistic-regression and random-forest
comparison on EEG epochs, with subject-level groups and 1000-permutation null
assessment (see :ref:`decoding-example-basic` for the matching experiment
setup).

---

1. Run the Experiment
======================

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, CVConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "lr": ClassicalModelConfig(estimator="LogisticRegression",
                                      params={"max_iter": 500}),
           "rf": ClassicalModelConfig(estimator="RandomForestClassifier",
                                      params={"n_estimators": 100}),
       },
       metrics=["accuracy", "roc_auc"],
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
       sample_metadata={"Subject": subject_ids},
       observation_level="epoch",
   )

---

2. Compose the Figure
======================

.. code-block:: python

   import matplotlib.pyplot as plt

   from coco_pipe.viz import (
       plot_decoding_scores,
       plot_roc_curve,
       plot_confusion_matrix,
       plot_feature_importance,
   )
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           2, 2,
           figsize=figure_size(columns=2, aspect_ratio=0.8),
           constrained_layout=True,
       )

       # (a) Aggregate scores per model and metric
       plot_decoding_scores(result, kind="point", ax=axes[0, 0])
       axes[0, 0].set_title("a) Scores (mean ± SEM)")

       # (b) ROC curve, fold-averaged with ±1 SD band
       plot_roc_curve(result, mean_only=True, ax=axes[0, 1])
       axes[0, 1].set_title("b) ROC curve")

       # (c) Confusion matrix for the chosen model
       plot_confusion_matrix(result, model="lr", ax=axes[1, 0])
       axes[1, 0].set_title("c) Confusion matrix (LR)")

       # (d) Top-15 feature importances (random forest)
       plot_feature_importance(result, model="rf", top_n=15, ax=axes[1, 1])
       axes[1, 1].set_title("d) Top-15 features (RF)")

       save_figure(fig, "figures/decoding_report.pdf")

---

3. Add a Temporal Companion Figure
====================================

If the experiment included a temporal estimator
(:class:`~coco_pipe.decoding.configs.SlidingEstimatorConfig`), a single-column
companion figure summarizes time-resolved performance:

.. code-block:: python

   from coco_pipe.viz import (
       plot_temporal_score_curve,
       plot_temporal_statistical_assessment,
   )

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           2, 1,
           figsize=figure_size(columns=1, aspect_ratio=1.3),
           constrained_layout=True,
       )
       plot_temporal_score_curve(result, metric="accuracy", ax=axes[0])
       plot_temporal_statistical_assessment(
           result, metric="accuracy", ax=axes[1],
       )
       save_figure(fig, "figures/decoding_temporal.pdf")

---

4. Interactive Variant for Notebook Review
=============================================

Before committing to a static layout, the same plots can be inspected
interactively:

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz

   iviz.plot_decoding_scores(result).show()
   iviz.plot_roc_curve(result, mean_only=True).show()
   iviz.plot_confusion_matrix(result, model="lr").show()
   iviz.plot_temporal_score_curve(result, metric="accuracy").show()

---

5. Notes
=========

- Every plot accepts ``ax=`` so it can be dropped into an existing layout. Pass
  ``figsize=`` only when creating a new figure inside the plot function.
- The ``coco_theme`` context manager scopes Matplotlib rcParams so other
  notebooks and tests are unaffected.
- ``save_figure`` defaults to 300 dpi, white background, and tight bounding
  boxes — override per-call as needed (e.g., ``dpi=600`` for raster output).
