.. _viz:

================================================
Visualization Module — Scientific User Guide
================================================

The ``coco_pipe.viz`` module turns tidy result tables (from
:ref:`decoding <decoding>` and ``coco_pipe.dim_reduction``) into publication-grade
figures. It is organized around three principles: **one theme for both backends**,
**a primitive layer beneath every domain plot**, and **mirrored
matplotlib / Plotly APIs** so the same plot is available for a paper or a
notebook with no rewrite.

.. admonition:: Design Philosophy

   Every domain plot is a thin wrapper that (1) extracts a tidy DataFrame from
   either an experiment result or a raw frame, then (2) routes it through a
   small set of generic primitives (``plot_bar``, ``plot_line``, ``plot_heatmap``,
   ``plot_scatter2d``, ``plot_topomap``…). The same prep helpers feed both the
   matplotlib and Plotly implementations, so the two backends cannot drift apart.

.. rubric:: Key Features

- Static (Matplotlib) and interactive (Plotly) backends with mirrored function
  names — pick a backend without relearning the API.
- A single ``coco`` theme applied to both backends from one source of truth in
  :mod:`coco_pipe.viz.theme` (palettes, colorblind-safe defaults, rcParams,
  Plotly template).
- 12 generic plotting primitives in :mod:`coco_pipe.viz.base`, reused by all
  domain plots and available directly for custom figures.
- ~25 static decoding plots and ~20 static dim-reduction plots, each accepting
  either an ``ExperimentResult``/reducer or the corresponding tidy DataFrame.
- A prep/render seam (``prepare_*`` helpers in ``viz._utils``) that makes
  every figure trivially testable in isolation from rendering.

---

.. rubric:: Quickstart

The most common static workflow: pass an ``ExperimentResult`` straight into a
plot function and get back a ``(Figure, Axes)`` pair.

.. code-block:: python

   from coco_pipe.viz import (
       plot_decoding_scores,
       plot_confusion_matrix,
       plot_roc_curve,
       save_figure,
   )

   fig, ax = plot_decoding_scores(result, metric="accuracy", kind="point")
   save_figure(fig, "scores.pdf")

   fig, ax = plot_confusion_matrix(result, model="logistic_regression")
   fig, ax = plot_roc_curve(result, mean_only=True)

The Plotly mirror lives under :mod:`coco_pipe.viz.interactive`:

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz

   fig = iviz.plot_roc_curve(result)
   fig.write_html("roc.html")

Theme scoping for paper figures:

.. code-block:: python

   from coco_pipe.viz.theme import coco_theme, figure_size

   with coco_theme("paper", colorblind=True):
       fig, ax = plot_decoding_scores(result)
       fig.set_size_inches(*figure_size(columns=1, aspect_ratio=0.7))
       save_figure(fig, "fig1.pdf")

---

.. rubric:: How to Choose a Plot

================================================  ============================================================
You have…                                          Reach for…
================================================  ============================================================
Per-fold scalar scores                             ``plot_decoding_scores``, ``plot_fold_score_dispersion``
Confusion / ROC / PR / calibration outputs         ``plot_confusion_matrix``, ``plot_roc_curve``,
                                                   ``plot_pr_curve``, ``plot_calibration_curve``
Sliding/generalizing temporal results              ``plot_temporal_score_curve``,
                                                   ``plot_temporal_generalization_matrix``,
                                                   ``plot_temporal_statistical_assessment``
Permutation/null distributions                     ``plot_null_interval_summary``
Pairwise model comparison                          ``plot_model_comparison``
Feature importance, stability, search results     ``plot_feature_importance``, ``plot_feature_stability``,
                                                   ``plot_feature_scores``, ``plot_search_results``
Sensor-level decoding values                       ``plot_decoding_topomap``,
                                                   ``plot_sensor_feature_heatmap``,
                                                   ``plot_sensor_feature_profile``,
                                                   ``plot_feature_sensor_profile``
A 2D/3D embedding                                  ``plot_embedding``
Reducer quality metrics                            ``plot_metrics``, ``plot_eigenvalues``,
                                                   ``plot_coranking_matrix``, ``plot_shepard_diagram``
Trajectories in embedding space                    ``plot_trajectory``, ``plot_trajectory_separation``,
                                                   ``plot_trajectory_metric_series``, ``plot_streamlines``
================================================  ============================================================

---

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   theme
   primitives

.. toctree::
   :maxdepth: 2
   :caption: Plot Reference

   decoding
   dim_reduction
   interactive

.. toctree::
   :maxdepth: 2
   :caption: Worked Examples

   examples/static_decoding_report
