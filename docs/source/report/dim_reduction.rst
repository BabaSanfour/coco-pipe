.. _report-section-dim-reduction:

==========================================
Dimensionality-Reduction Sections
==========================================

:mod:`coco_pipe.report.dim_reduction` provides eleven section-adder
functions and a one-shot factory for building report sections from
:class:`~coco_pipe.dim_reduction.DimReduction` objects, embeddings,
and ``MethodSelector`` comparison frames.

---

1. Section Catalog
====================

==========================================  =====================================================
Function / method                            Backing source
==========================================  =====================================================
:func:`~coco_pipe.report.dim_reduction.add_reduction_overview`           ``reduction.get_summary()``
:func:`~coco_pipe.report.dim_reduction.add_reduction_embedding`          Explicit ``X_emb`` (2-D or 3-D)
:func:`~coco_pipe.report.dim_reduction.add_reduction_metrics`            ``reduction.get_scores()`` /
                                                                          ``get_summary()["metric_records"]``
:func:`~coco_pipe.report.dim_reduction.add_reduction_diagnostics`        Shepard diagram (X_orig, X_emb)
:func:`~coco_pipe.report.dim_reduction.add_reduction_interpretation`     ``get_summary()["interpretation"]``
:func:`~coco_pipe.report.dim_reduction.add_reduction_coranking`          ``diagnostics["coranking_matrix_"]``
:func:`~coco_pipe.report.dim_reduction.add_reduction_components`         ``reduction.get_components()``
:func:`~coco_pipe.report.dim_reduction.add_reduction_trajectory`         3-D trajectory tensor
:func:`~coco_pipe.report.dim_reduction.add_reduction_trajectory_separation`  ``trajectory_separation`` output
:func:`~coco_pipe.report.dim_reduction.add_reduction`                    Single fully-assembled section
:func:`~coco_pipe.report.dim_reduction.add_comparison`                   Wide/tidy metric frame or
                                                                          ``MethodSelector``
==========================================  =====================================================

---

2. Two Levels of Granularity
==============================

Two levels of granularity are supported depending on how much
control the caller needs:

- **Coarse** — :meth:`Report.add_reduction` packages an overview,
  embedding/trajectory plot, scalar metrics, metric records,
  diagnostics (loss history, scree, co-ranking matrix, trajectory
  kinematics, separation timecourses), and interpretation into one
  section. One call per reduction.
- **Fine** — call individual ``add_reduction_*`` adders for full
  layout control. Use this when assembling a custom report.

.. code-block:: python

   # Coarse
   report.add_reduction(pca, X_emb=embedding, labels=y)

   # Fine
   (
       report
       .add_reduction_overview(pca, name="PCA Overview")
       .add_reduction_embedding(embedding, labels=y)
       .add_reduction_metrics(pca)
       .add_reduction_coranking(coranking_matrix)
       .add_reduction_components(pca.get_components())
   )

---

3. Each Adder in Detail
=========================

3.1 ``add_reduction_overview``
--------------------------------

Key/value summary from
:meth:`DimReduction.get_summary <coco_pipe.dim_reduction.core.DimReduction.get_summary>`
— method name, n_components, random_state, capability flags, scalar
quality metadata. Skips ``metrics`` / ``metric_records`` /
``diagnostics`` / ``interpretation`` blocks (they have dedicated
adders).

.. code-block:: python

   report.add_reduction_overview(pca, name="PCA Summary")

3.2 ``add_reduction_embedding``
---------------------------------

2-D scatter for ``X_emb.shape == (n, 2)``, 3-D scatter for
``(n, 3)``. Optional class labels color the points; optional metadata
dict adds hover annotations.

.. code-block:: python

   report.add_reduction_embedding(
       X_emb, labels=class_ids,
       metadata={"group": groups, "score": scores},
       name="UMAP embedding",
   )

3.3 ``add_reduction_metrics``
-------------------------------

Quality metrics table (downloadable as CSV) plus a metric bar chart
using :func:`coco_pipe.viz.dim_reduction.plot_metrics`. Accepts either
the reducer (sourced from ``get_scores()`` or the cached metric
records) or a raw scores list.

.. code-block:: python

   report.add_reduction_metrics(pca, metric="trustworthiness")

3.4 ``add_reduction_diagnostics``
-----------------------------------

Shepard diagram: original pairwise distances vs. embedded pairwise
distances on a sampled subset. Requires the original ``X_orig`` and
the embedding ``X_emb``.

.. code-block:: python

   report.add_reduction_diagnostics(X_orig, X_emb)

3.5 ``add_reduction_interpretation``
--------------------------------------

Renders ``correlation`` / ``perturbation`` / ``gradient`` results from
:meth:`DimReduction.interpret`. Accepts either the raw
``get_summary()["interpretation"]`` payload or a list of records.

.. code-block:: python

   summary = pca.get_summary()
   report.add_reduction_interpretation(summary["interpretation"])

3.6 ``add_reduction_coranking``
---------------------------------

Heatmap of a co-ranking matrix (typically
``DimReduction.diagnostics_["coranking_matrix_"]``).

.. code-block:: python

   report.add_reduction_coranking(diagnostics["coranking_matrix_"])

3.7 ``add_reduction_components``
----------------------------------

Linear reducer components heatmap (PCA, IncrementalPCA, DaskPCA,
DaskTruncatedSVD, DMD, TRCA). Pass ``feature_names`` for axis labels.

.. code-block:: python

   report.add_reduction_components(
       pca.get_components(),
       feature_names=channel_names,
   )

3.8 ``add_reduction_trajectory``
----------------------------------

Renders a 3-D trajectory tensor of shape
``(n_trajectories, n_times, n_dims)``. Pair with ``times=`` for
labelled x-axis, ``labels=`` for per-trajectory coloring.

.. code-block:: python

   report.add_reduction_trajectory(
       X_3d, times=time_axis, labels=condition_ids,
   )

3.9 ``add_reduction_trajectory_separation``
---------------------------------------------

Per-pair separation timecourses (e.g., distance between condition
centroids over time) as a multi-line plot. Accepts the output of
:func:`~coco_pipe.dim_reduction.trajectory_separation` directly.

.. code-block:: python

   sep = trajectory_separation(traj, condition_ids, method="centroid")
   report.add_reduction_trajectory_separation(sep, times=time_axis, top_n=5)

3.10 ``add_reduction``
------------------------

Full section in one call: overview + embedding/trajectory plot +
scalar metrics + metric records + every applicable diagnostic. Pass
``X_emb`` to enable embedding/trajectory plots; without it the
section still renders scalar metrics, loss curves, scree, and
co-ranking.

.. code-block:: python

   report.add_reduction(
       pca, name="PCA",
       X_emb=embedding,
       labels=condition_ids,
       metadata={"subject": subject_ids},
       times=time_axis,             # only when X_emb is 3-D
   )

3.11 ``add_comparison``
-------------------------

Multi-method comparison: a metric × method table with best-value
highlighting (:class:`MetricsTableElement`), a metric-by-method
heatmap, a bar chart of details, and — when applicable — a radar
chart normalised across metrics.

.. code-block:: python

   from coco_pipe.dim_reduction.evaluation import MethodSelector

   selector = MethodSelector(reducers).collect()
   report.add_comparison(selector.to_frame(), name="Method Comparison")

Raises ``ValueError`` if the metrics frame is empty after
normalisation.

---

4. ``make_reduction_report``
==============================

The factory builds a multi-reducer report by calling adders 1-9 for
each reducer.

.. code-block:: python

   from coco_pipe.report import make_reduction_report

   report = make_reduction_report(
       [pca, umap, phate],
       embeddings=[pca_emb, umap_emb, phate_emb],   # required for embedding sections
       labels=class_ids,
       metadata={"subject": subject_ids},
       times=time_axis,                              # required for 3-D embeddings
       sections="default",
       theme="paper",
       output_path="reduction.html",
   )

4.1 Sections argument
-----------------------

Valid section names (see :data:`DEFAULT_REDUCTION_SECTIONS`):

.. code-block:: text

   overview, embedding, metrics, diagnostics, coranking,
   interpretation, components, trajectory, trajectory_separation

Unknown names raise ``ValueError``. Pass a subset to render only the
sections you need:

.. code-block:: python

   make_reduction_report(
       [pca, umap],
       embeddings=[pca_emb, umap_emb],
       sections=["overview", "embedding", "metrics"],
   )

4.2 Error tolerance
---------------------

Per-section ``try / except`` mirrors the decoding factory:
unavailable sections are logged and skipped, the rest of the report
still renders.

---

5. Combining With Comparison
==============================

The natural full-comparison pattern:

.. code-block:: python

   from coco_pipe.report import make_reduction_report
   from coco_pipe.dim_reduction.evaluation import MethodSelector

   selector = MethodSelector(reducers).collect()
   report = make_reduction_report(
       reducers,
       embeddings=embeddings,
       title="PCA vs UMAP vs PHATE",
   )
   report.add_comparison(selector.to_frame())
   report.save("reduction_comparison.html")

---

6. Internal Helpers (Used by Adders)
======================================

Two helpers in :mod:`coco_pipe.report.dim_reduction` are exposed for
custom adders that need the same normalisation contract:

==================================================  ============================================
``_get_reducer_summary(reducer)``                    Return a normalised summary dict (always
                                                     populates ``method``, ``metrics``,
                                                     ``metric_records``, ``quality_metadata``,
                                                     ``diagnostics``, ``interpretation``,
                                                     ``interpretation_records``, ``capabilities``).
                                                     Raises if the reducer lacks ``get_summary()``.
``_metrics_summary_table(metrics)``                  Pivot a tidy metrics frame into the
                                                     ``method`` × ``metric`` wide table used
                                                     by :class:`MetricsTableElement`.
``_trajectory_times(diagnostics, times)``            Resolve the trajectory time axis,
                                                     preferring an explicit ``times=`` argument
                                                     over ``diagnostics["trajectory_times_"]``.
==================================================  ============================================

These are intentionally underscore-prefixed; their signatures may
change. For custom workflows, prefer the public adders.
