.. _dim-reduction-visualization:

============================================
Visualization and Reports
============================================

The dim-reduction module **does not own** any plotting code. Visualization
lives in :mod:`coco_pipe.viz.dim_reduction` (matplotlib) and
:mod:`coco_pipe.viz.interactive.dim_reduction` (Plotly). All plotting helpers
are **data-first**: they consume embeddings, metric records, diagnostics, and
interpretation payloads — never manager state implicitly.

For the full plot reference, see :ref:`viz-dim-reduction`. This page summarizes
the bridge between ``DimReduction`` outputs and the corresponding plots.

---

1. Source → Plot Cheat-Sheet
==============================

================================  ===========================================
What you have on ``DimReduction``  Plot
================================  ===========================================
Explicit embedding array            :func:`coco_pipe.viz.plot_embedding`
``metric_records_``                 :func:`coco_pipe.viz.plot_metrics`
``diagnostics_["coranking_matrix"]``  :func:`coco_pipe.viz.plot_coranking_matrix`
``diagnostics_["shepard_distances"]`` :func:`coco_pipe.viz.plot_shepard_diagram`
``quality_metadata_["loss_history"]``  :func:`coco_pipe.viz.plot_loss_history`
``quality_metadata_["explained_variance_ratio_"]``  :func:`coco_pipe.viz.plot_eigenvalues`
``interpretation_records_``         :func:`coco_pipe.viz.plot_reduction_feature_importance`
``interpretation_["correlation"]``  :func:`coco_pipe.viz.plot_feature_correlation_heatmap`
``diagnostics_["trajectory_separation"]``  :func:`coco_pipe.viz.plot_trajectory_separation`
``diagnostics_["trajectory_<kinematic>"]``  :func:`coco_pipe.viz.plot_trajectory_metric_series`
Native 3D trajectory tensor         :func:`coco_pipe.viz.plot_trajectory`
================================  ===========================================

---

2. Plotting an Embedding
==========================

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.viz import plot_embedding

   reducer = DimReduction("UMAP", n_components=2)
   embedding = reducer.fit_transform(X)

   plot_embedding(embedding, labels=class_ids, title="UMAP (2D)")
   plot_embedding(reducer.fit_transform(X), dims=(0, 1, 2))   # 3D variant

The plot function never touches ``reducer`` state — you pass the explicit
``embedding`` array. This stays consistent with the rest of the module: cached
state on the manager is a convenience, not a requirement.

---

3. Quality Metric Plots
==========================

:func:`coco_pipe.viz.plot_metrics` accepts any of: a scored
:class:`~coco_pipe.dim_reduction.DimReduction`, a tidy DataFrame, a
``{metric: value}`` mapping, or a :class:`~coco_pipe.dim_reduction.evaluation.MethodSelector`.

.. code-block:: python

   from coco_pipe.viz import plot_metrics

   plot_metrics(reducer)                                     # scalar bars
   plot_metrics(reducer.metric_records_, plot_type="line",   # k sweep
                metric="trustworthiness", scope="k")
   plot_metrics(selector, plot_type="heatmap")               # method × metric

See :ref:`viz-dim-reduction` for the full set of ``plot_type`` options.

---

4. Linear Reducer Diagnostics
================================

When the reducer is linear, two extra diagnostics are available:

.. code-block:: python

   from coco_pipe.viz import plot_eigenvalues, plot_component_loadings

   reducer = DimReduction("PCA", n_components=10)
   reducer.fit(X)

   plot_eigenvalues({"PCA": reducer.reducer.explained_variance_ratio_},
                    max_components=10)
   plot_component_loadings(reducer.get_components(), feature_names=feature_names)

---

5. Iterative Reducer Loss Curves
==================================

Reducers with ``has_loss_history=True`` expose
``quality_metadata_["loss_history"]`` after fitting:

.. code-block:: python

   from coco_pipe.viz import plot_loss_history

   plot_loss_history(reducer.quality_metadata_["loss_history"])

Currently this covers ``UMAP``, ``ParametricUMAP``, ``PaCMAP``, ``TriMap``,
``IVIS``, and ``TopologicalAE``.

---

6. Interpretation Plots
=========================

After :meth:`DimReduction.interpret`, the tidy records and correlation
matrices flow straight to the viz layer:

.. code-block:: python

   from coco_pipe.viz import (
       plot_reduction_feature_importance,
       plot_feature_correlation_heatmap,
   )

   plot_reduction_feature_importance(
       reducer.interpretation_records_,
       analysis="perturbation",
       method=reducer.method,
       top_n=20,
   )
   plot_feature_correlation_heatmap(
       reducer.interpretation_["correlation"],
       method=reducer.method,
   )

---

7. Trajectory Plots
=====================

For native 3D embeddings, the trajectory plots consume the tensor directly:

.. code-block:: python

   from coco_pipe.viz import (
       plot_trajectory,
       plot_trajectory_separation,
       plot_trajectory_metric_series,
   )

   plot_trajectory(traj, times=times, labels=condition,
                   values=speeds, speed_mode="linecollection")
   plot_trajectory_separation(reducer.diagnostics_["trajectory_separation"])
   plot_trajectory_metric_series(reducer.diagnostics_["trajectory_speed"], times=times)

---

8. Interactive Mirror
=======================

Every static plot above has a Plotly mirror at the same name in
:mod:`coco_pipe.viz.interactive.dim_reduction`. Same inputs, same defaults,
returns ``plotly.graph_objects.Figure``.

.. code-block:: python

   from coco_pipe.viz.interactive import dim_reduction as iviz

   iviz.plot_embedding(embedding, labels=class_ids).write_html("embed.html")
   iviz.plot_metrics(reducer).show()

The interactive side also exposes ``plot_channel_traces``, ``plot_raw_preview``,
and ``plot_radar_comparison`` not present in the static module — see
:ref:`viz-interactive` for the parity table.

---

9. Reports
============

:meth:`coco_pipe.report.Report.add_reduction` consumes ``get_summary()`` plus
an explicit embedding payload. :meth:`Report.add_comparison` consumes tidy
metric frames or :class:`~coco_pipe.dim_reduction.evaluation.MethodSelector`
instances directly.

.. code-block:: python

   from coco_pipe.report import Report

   report = Report(title="UMAP run on subject S01")
   report.add_reduction(reducer.get_summary(), embedding=embedding)
   report.add_comparison(selector)
   report.save("reports/s01.html")

Shepard plots and comparison views reuse cached diagnostics
(``shepard_distances_``, ``coranking_matrix_``) when present, so passing the
embedding again is purely for embedding-level plots.

---

10. Composing a Paper Figure
==============================

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import plot_embedding, plot_eigenvalues, plot_shepard_diagram
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   reducer = DimReduction("PCA", n_components=10)
   embedding = reducer.fit_transform(X)
   reducer.score(embedding, X=X, k_values=[5, 10, 20])

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           1, 3, figsize=figure_size(columns=2, aspect_ratio=0.33),
           constrained_layout=True,
       )
       plot_eigenvalues({"PCA": reducer.reducer.explained_variance_ratio_},
                        ax=axes[0])
       plot_embedding(embedding, labels=class_ids, ax=axes[1])
       plot_shepard_diagram(X, embedding, ax=axes[2])
       save_figure(fig, "figures/pca_summary.pdf")
