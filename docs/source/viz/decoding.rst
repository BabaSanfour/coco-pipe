.. _viz-decoding:

================================
Decoding Plots (Static Backend)
================================

This page covers the ~25 static plotting functions in
:mod:`coco_pipe.viz.decoding`. Every function accepts either an
:class:`coco_pipe.decoding.result.ExperimentResult` or the tidy DataFrame
produced by the corresponding result accessor — the wrapper detects the input
type and routes accordingly. All functions return ``(Figure, Axes)``.

The interactive Plotly mirror lives in :mod:`coco_pipe.viz.interactive.decoding`
and is described in :ref:`viz-interactive`.

---

1. Result Accessor → Plot Mapping
==================================

Each decoding plot maps to a specific :class:`ExperimentResult` accessor. If
you pass a raw DataFrame, it must match the schema that accessor produces.

==========================================  ===========================================  ===============================
``viz.decoding`` function                   ``ExperimentResult`` accessor                Required columns (raw input)
==========================================  ===========================================  ===============================
``plot_decoding_scores``                    ``get_detailed_scores()``                    ``Model``, ``Metric``, ``Value``
``plot_fold_score_dispersion``              ``get_detailed_scores()``                    ``Model``, ``Metric``, ``Fold``, ``Value``
``plot_confusion_matrix``                   ``get_confusion_matrices()`` /               ``TrueLabel``, ``PredictedLabel``, ``Value``
                                            ``get_pooled_confusion_matrix()``
``plot_roc_curve``                          ``get_roc_curve()``                          ``Model``, ``FPR``, ``TPR``
``plot_pr_curve``                           ``get_pr_curve()``                           ``Model``, ``Precision``, ``Recall``
``plot_calibration_curve``                  ``get_calibration_curve()``                  ``Model``, ``MeanPredicted``, ``Fraction``
``plot_probability_diagnostics``            ``get_probability_diagnostics()``            ``Model``, ``Metric``, ``Value``
``plot_temporal_score_curve``               ``get_temporal_score_summary()``             ``Model``, ``Metric``, ``Time``, ``Mean``
``plot_temporal_generalization_matrix``     ``get_generalization_matrix()``              ``Model``, ``TrainTime``, ``TestTime``, ``Value``
``plot_temporal_statistical_assessment``    ``get_statistical_assessment()``             ``Time``, ``Observed``, ``NullMean``, ``NullStd``
``plot_null_interval_summary``              ``get_statistical_assessment()``             ``Model``, ``Metric``, ``Observed``, ``NullLow``, ``NullHigh``
``plot_model_comparison``                   ``compare_models()`` /                       ``ModelA``, ``ModelB``, ``Difference``
                                            ``compare_models_paired()``
``plot_fit_diagnostics``                    ``get_fit_diagnostics()``                    ``Model``, ``Fold``, ``FitSeconds``, ``Score``
``plot_training_history``                   ``get_model_artifacts()``                    ``Epoch``, ``Loss`` / ``Metric``
``plot_search_results``                     ``get_search_results()``                     ``Model``, ``Param``, ``MeanScore``
``plot_subject_diagnostics``                ``get_predictions()``                        ``Subject``, ``y_true``, ``y_pred``
``plot_group_summary``                      ``get_predictions()``                        ``Subject``, ``y_true``, ``y_pred``
``plot_regression_diagnostics``             ``get_predictions()``                        ``y_true``, ``y_pred``
``plot_feature_importance``                 ``get_feature_importances()``                ``FeatureName``, ``Mean`` (or ``Importance``)
``plot_feature_stability``                  ``get_feature_stability()``                  ``FeatureName``, ``Stability``
``plot_feature_scores``                     ``get_feature_scores()``                     ``FeatureName``, ``Score``
``plot_decoding_topomap``                   ``get_feature_importances()`` +              ``FeatureName``, value column
                                            sensor layout
``plot_sensor_feature_heatmap``             ``get_feature_importances()`` +              ``FeatureName``, value column
                                            sensor layout
``plot_sensor_feature_profile``             ``get_feature_importances()`` +              ``FeatureName``, value column
                                            sensor layout
``plot_feature_sensor_profile``             ``get_feature_importances()`` +              ``FeatureName``, value column
                                            sensor layout
==========================================  ===========================================  ===============================

---

2. Score Summaries
===================

2.1 ``plot_decoding_scores`` — aggregate scalar scores
--------------------------------------------------------

Three rendering modes are exposed via ``kind``: ``"point"`` (mean ± SEM error
points, the default), ``"bar"`` (sorted bars), or ``"box"`` (fold-level
distributions). ``aggregate`` controls the centre statistic for ``point`` /
``bar``.

.. code-block:: python

   from coco_pipe.viz import plot_decoding_scores

   fig, ax = plot_decoding_scores(result, metric="accuracy", kind="point")
   fig, ax = plot_decoding_scores(result, kind="box")               # all metrics, fold dists
   fig, ax = plot_decoding_scores(result, kind="bar", aggregate="median")

2.2 ``plot_fold_score_dispersion`` — per-fold spread
------------------------------------------------------

Per-fold strip/box overlay, useful for spotting unstable models or outlier
folds before any aggregation.

.. code-block:: python

   from coco_pipe.viz import plot_fold_score_dispersion

   fig, ax = plot_fold_score_dispersion(result, metric="balanced_accuracy")

---

3. Classification Diagnostics
==============================

3.1 ``plot_confusion_matrix``
-------------------------------

Aggregated confusion matrix. Pass ``model`` and/or ``fold`` to filter; otherwise
all rows are summed.

.. code-block:: python

   from coco_pipe.viz import plot_confusion_matrix

   fig, ax = plot_confusion_matrix(result, model="logistic_regression")

3.2 ROC, PR, Calibration
--------------------------

These three plots accept either the full per-fold accessor output (each fold
draws a translucent curve, mean is highlighted) or any matching DataFrame.
Set ``mean_only=True`` to interpolate folds onto a common x-grid and draw the
mean curve with a ±1 SD band.

.. code-block:: python

   from coco_pipe.viz import plot_roc_curve, plot_pr_curve, plot_calibration_curve

   fig, ax = plot_roc_curve(result, mean_only=True)
   fig, ax = plot_pr_curve(result)
   fig, ax = plot_calibration_curve(result, model="random_forest")

3.3 ``plot_probability_diagnostics``
--------------------------------------

Bar summary of log-loss and Brier-score per model. Requires probability
estimates (either native ``predict_proba`` or a calibrated classifier — see
:ref:`decoding-concepts`).

---

4. Temporal Decoding
=====================

4.1 ``plot_temporal_score_curve``
-----------------------------------

Mean score per time-point with a ±1 SD band. Non-numeric time labels are placed
at integer positions and shown as rotated tick labels.

.. code-block:: python

   from coco_pipe.viz import plot_temporal_score_curve

   fig, ax = plot_temporal_score_curve(result, metric="accuracy")

4.2 ``plot_temporal_generalization_matrix``
---------------------------------------------

Train-time × test-time heatmap. Off-diagonal mass indicates stable neural
representations (see :ref:`decoding-concepts`).

.. code-block:: python

   from coco_pipe.viz import plot_temporal_generalization_matrix

   fig, ax = plot_temporal_generalization_matrix(result, metric="accuracy")

4.3 ``plot_temporal_statistical_assessment``
----------------------------------------------

Observed temporal curve drawn against the permutation null band. Significant
time-points (after the chosen correction) are shaded.

.. code-block:: python

   from coco_pipe.viz import plot_temporal_statistical_assessment

   fig, ax = plot_temporal_statistical_assessment(result, metric="accuracy")

---

5. Statistical Assessment
==========================

5.1 ``plot_null_interval_summary``
-----------------------------------

One row per model showing the observed scalar score against the permutation
null interval; ideal for compact scalar summaries.

5.2 ``plot_model_comparison``
-------------------------------

Pairwise model comparisons rendered as either paired-difference distributions
(``paired=True``) or independent ones. Pass ``reference`` to anchor every
comparison against a single model.

.. code-block:: python

   from coco_pipe.viz import plot_model_comparison

   fig, ax = plot_model_comparison(result, metric="accuracy",
                                   reference="logistic_regression")

---

6. Training-Time Diagnostics
=============================

- ``plot_fit_diagnostics``: per-fold fit time, score, and convergence warnings
  for catching unstable models.
- ``plot_training_history``: neural-network loss / metric curves from
  ``get_model_artifacts()``.
- ``plot_search_results``: hyperparameter grid scores, useful for sanity-checking
  ``TuningConfig`` outcomes.

---

7. Per-Subject and Regression Outputs
======================================

- ``plot_subject_diagnostics``: per-subject accuracy histogram and outliers
  derived from ``get_predictions()``.
- ``plot_group_summary``: subject-level aggregated metric summary (uses the
  same accessor).
- ``plot_regression_diagnostics``: residuals and predicted-vs-true scatter for
  ``task="regression"`` experiments.

---

8. Feature Importance, Stability, Selection
============================================

8.1 ``plot_feature_importance``
---------------------------------

Ranked horizontal bars with optional ``top_n`` cap. Pass ``signed=True`` to use
the diverging palette when importance values can be negative; ``absolute=True``
ranks by magnitude.

.. code-block:: python

   from coco_pipe.viz import plot_feature_importance

   fig, ax = plot_feature_importance(result, top_n=20)
   fig, ax = plot_feature_importance(result, signed=True, top_n=15)

If passed a non-DataFrame container (Series, mapping, ndarray) the function
delegates to the dim-reduction ``plot_feature_importance`` automatically.

8.2 ``plot_feature_stability``
--------------------------------

Selection rate across folds, as bar or heatmap. Higher is more stable.

.. code-block:: python

   from coco_pipe.viz import plot_feature_stability

   fig, ax = plot_feature_stability(result, kind="bar", top_n=25)

8.3 ``plot_feature_scores`` / ``plot_search_results``
------------------------------------------------------

Univariate selector scores and tuned hyperparameter grids, respectively.

---

9. Sensor-Level Decoding Maps
==============================

These plots accept a sensor-feature DataFrame (typically derived from
``get_feature_importances()``) and resolve sensor positions from either an
MNE :class:`~mne.Info` (``info=``) or a coordinate table (``coords=``).

==============================  ======================================
Function                        Picture
==============================  ======================================
``plot_decoding_topomap``       Topomap of one value column across sensors.
``plot_sensor_feature_heatmap`` Sensor × feature-family heatmap.
``plot_sensor_feature_profile`` Bars: feature importances within one sensor.
``plot_feature_sensor_profile`` Topomap: one feature family across sensors.
==============================  ======================================

.. code-block:: python

   from coco_pipe.viz import plot_decoding_topomap

   fig, ax = plot_decoding_topomap(
       sensor_df,
       value="Importance",
       info=raw.info,
       center=0.0,
   )

When ``center`` is provided, the colormap is diverging and anchored at that
value; otherwise the sequential palette is used.

---

10. Common Patterns
====================

10.1 Combining a result and a custom axes
-------------------------------------------

Every plot accepts an existing ``ax=`` so you can build composite figures with
:func:`coco_pipe.viz.theme.figure_size` for paper layout.

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import (
       plot_decoding_scores, plot_confusion_matrix, plot_roc_curve,
   )
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           1, 3, figsize=figure_size(columns=2, aspect_ratio=0.35),
           constrained_layout=True,
       )
       plot_decoding_scores(result, metric="accuracy", ax=axes[0])
       plot_confusion_matrix(result, model="logistic_regression", ax=axes[1])
       plot_roc_curve(result, mean_only=True, ax=axes[2])
       save_figure(fig, "figures/decoding_summary.pdf")

10.2 Filtering before plotting
--------------------------------

Many functions accept ``model`` / ``metric`` / ``fold`` filters that map to the
same column names in the accessor output. They short-circuit before rendering,
so passing a filter is cheaper than slicing the DataFrame yourself.

10.3 Backend mirror
---------------------

To get the Plotly equivalent of any of these plots, swap the import:

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz
   ifig = iviz.plot_roc_curve(result)
   ifig.write_html("roc.html")

A handful of static-only plots (``plot_decoding_topomap``,
``plot_sensor_feature_*``) are matplotlib-only because they rely on MNE; see
:ref:`viz-interactive` for the current parity table.
