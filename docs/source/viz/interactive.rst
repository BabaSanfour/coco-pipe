.. _viz-interactive:

==================================
Interactive (Plotly) Visualization
==================================

The :mod:`coco_pipe.viz.interactive` subpackage mirrors the static plotting
surface using Plotly. The same prep helpers feed both backends, so for any plot
that exists on both sides the same input produces equivalent figures.

When to use the Plotly backend:

- Notebooks and dashboards where hover-tooltips, zoom, and selection matter.
- Embedding plots in HTML reports (``fig.write_html``).
- Quick exploration before committing to a static figure layout.

When to prefer the matplotlib backend:

- Paper figures (``coco_theme("paper")`` + ``save_figure``).
- MNE-backed topomaps and sensor maps (interactive equivalents do not exist —
  see the parity table below).
- Complex composite figures with shared axes.

---

1. Common Patterns
===================

Each interactive plot returns a :class:`plotly.graph_objects.Figure`. Save or
embed it with:

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz

   fig = iviz.plot_roc_curve(result, mean_only=True)
   fig.write_html("roc.html")             # standalone interactive HTML
   fig.write_image("roc.png", scale=2)    # static export (requires `kaleido`)
   fig.show()                             # inline in a notebook

The ``coco`` Plotly template is registered eagerly on import and applied to
every wrapper. To apply it manually outside the wrappers:

.. code-block:: python

   import plotly.graph_objects as go
   import coco_pipe.viz.interactive   # registers the template

   fig = go.Figure(...)
   fig.update_layout(template="coco")

---

2. Decoding Mirror
===================

All plots in :mod:`coco_pipe.viz.interactive.decoding` accept the same
``ExperimentResult`` or tidy DataFrame as their static twins.

==============================================  ============================================
Function                                        Static counterpart
==============================================  ============================================
``plot_confusion_matrix``                       ``viz.decoding.plot_confusion_matrix``
``plot_roc_curve``                              ``viz.decoding.plot_roc_curve``
``plot_pr_curve``                               ``viz.decoding.plot_pr_curve``
``plot_calibration_curve``                      ``viz.decoding.plot_calibration_curve``
``plot_fold_score_dispersion``                  ``viz.decoding.plot_fold_score_dispersion``
``plot_temporal_score_curve``                   ``viz.decoding.plot_temporal_score_curve``
``plot_temporal_generalization_matrix``         ``viz.decoding.plot_temporal_generalization_matrix``
``plot_temporal_statistical_assessment``        ``viz.decoding.plot_temporal_statistical_assessment``
``plot_null_interval_summary``                  ``viz.decoding.plot_null_interval_summary``
``plot_training_history``                       ``viz.decoding.plot_training_history``
``plot_decoding_scores``                        ``viz.decoding.plot_decoding_scores``
``plot_model_comparison``                       ``viz.decoding.plot_model_comparison``
``plot_fit_diagnostics``                        ``viz.decoding.plot_fit_diagnostics``
``plot_probability_diagnostics``                ``viz.decoding.plot_probability_diagnostics``
``plot_subject_diagnostics``                    ``viz.decoding.plot_subject_diagnostics``
``plot_group_summary``                          ``viz.decoding.plot_group_summary``
``plot_regression_diagnostics``                 ``viz.decoding.plot_regression_diagnostics``
``plot_search_results``                         ``viz.decoding.plot_search_results``
``plot_feature_stability``                      ``viz.decoding.plot_feature_stability``
``plot_feature_scores``                         ``viz.decoding.plot_feature_scores``
==============================================  ============================================

Static-only (MNE-backed, no interactive twin yet):

- ``plot_decoding_topomap``
- ``plot_sensor_feature_heatmap``
- ``plot_sensor_feature_profile``
- ``plot_feature_sensor_profile``
- ``plot_feature_importance`` (interactive version exists under
  ``viz.interactive.dim_reduction``; the decoding wrapper delegates internally)

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz

   fig = iviz.plot_decoding_scores(result, metric="accuracy", kind="point")
   fig = iviz.plot_temporal_score_curve(result, metric="accuracy")
   fig = iviz.plot_confusion_matrix(result, model="random_forest")

---

3. Dimensionality Reduction Mirror
====================================

==============================================  ============================================
Function                                        Static counterpart
==============================================  ============================================
``plot_embedding``                              ``viz.dim_reduction.plot_embedding``
``plot_metrics``                                ``viz.dim_reduction.plot_metrics``
``plot_loss_history``                           ``viz.dim_reduction.plot_loss_history``
``plot_eigenvalues``                            ``viz.dim_reduction.plot_eigenvalues``
``plot_shepard_diagram``                        ``viz.dim_reduction.plot_shepard_diagram``
``plot_coranking_matrix``                       ``viz.dim_reduction.plot_coranking_matrix``
``plot_component_loadings``                     ``viz.dim_reduction.plot_component_loadings``
``plot_feature_importance``                     ``viz.dim_reduction.plot_feature_importance``
``plot_feature_correlation_heatmap``            ``viz.dim_reduction.plot_feature_correlation_heatmap``
``plot_streamlines``                            ``viz.dim_reduction.plot_streamlines``
``plot_trajectory``                             ``viz.dim_reduction.plot_trajectory``
``plot_trajectory_metric_series``               ``viz.dim_reduction.plot_trajectory_metric_series``
``plot_trajectory_separation``                  ``viz.dim_reduction.plot_trajectory_separation``
==============================================  ============================================

Interactive-only additions (no static twin yet):

- ``plot_channel_traces`` — per-channel time series viewer for raw signals.
- ``plot_raw_preview`` — quick preview of a raw data matrix.
- ``plot_radar_comparison`` — radar/spider chart for multi-metric model comparison.

---

4. Notebook Workflow
=====================

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz, dim_reduction as ivizdr

   # Decoding diagnostics
   iviz.plot_decoding_scores(result, metric="accuracy").show()
   iviz.plot_confusion_matrix(result).show()
   iviz.plot_roc_curve(result, mean_only=True).show()

   # Reducer diagnostics
   ivizdr.plot_embedding(X_emb, labels=class_ids).show()
   ivizdr.plot_metrics(reducer).show()

---

5. Embedding in HTML Reports
=============================

Use ``fig.to_html(include_plotlyjs="cdn", full_html=False)`` to get an embedded
``<div>`` suitable for inclusion in a report (see :mod:`coco_pipe.report`).

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz

   html_fragment = iviz.plot_roc_curve(result).to_html(
       include_plotlyjs="cdn", full_html=False
   )

---

6. Backend Compatibility Notes
================================

- Interactive plots accept the **same input types** as their static
  counterparts (``ExperimentResult`` or matching DataFrame); the prep layer
  (``viz._utils``) is shared.
- Default colors come from the same ``_COLORBLIND_COLORS`` palette used by
  the matplotlib theme, so static and interactive renderings of the same data
  use matching colors.
- The Plotly template name is :data:`coco_pipe.viz.interactive._utils.COCO_TEMPLATE`.
- If you build figures outside the wrappers, calling
  ``fig.update_layout(template="coco")`` is enough to match the look.
