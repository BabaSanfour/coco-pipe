.. _viz-dim-reduction:

=========================================
Dimensionality Reduction Plots (Static)
=========================================

This page covers the ~20 static plotting functions in
:mod:`coco_pipe.viz.dim_reduction`. They turn embedded data, quality metrics,
and trajectory tensors into publication-grade figures. All functions return
``(Figure, Axes)``.

The interactive Plotly mirror lives in
:mod:`coco_pipe.viz.interactive.dim_reduction`; see :ref:`viz-interactive` for
the parity table.

---

1. Reducer Source → Plot Mapping
=================================

==============================  ===============================================
What you have                   Plot
==============================  ===============================================
A 2D or 3D embedding array      ``plot_embedding``
A reducer's ``metrics_``        ``plot_metrics``
PCA / SVD eigenvalues           ``plot_eigenvalues``
Iterative reducer loss          ``plot_loss_history``
Original + embedded distances   ``plot_shepard_diagram``
A co-ranking matrix             ``plot_coranking_matrix``
Component loadings              ``plot_component_loadings``
Per-feature scores              ``plot_feature_importance`` (in this module)
Feature correlations            ``plot_feature_correlation_heatmap``
Trajectory tensors              ``plot_trajectory``,
                                ``plot_trajectory_separation``,
                                ``plot_trajectory_metric_series``
Velocity field over embedding   ``plot_streamlines``
==============================  ===============================================

---

2. Embedding Visualization
===========================

2.1 ``plot_embedding``
------------------------

2D or 3D scatter of a precomputed embedding. The ``dims`` argument selects
which embedding columns become the plot axes — two indices produce a 2D plot,
three a 3D plot. Pass ``labels`` for categorical coloring or ``metrics`` +
``metric_name`` to annotate a chosen scalar metric in the corner.

.. code-block:: python

   from coco_pipe.viz import plot_embedding

   fig, ax = plot_embedding(
       X_emb,
       labels=class_ids,
       label_kind="categorical",
       title="UMAP embedding",
   )

   # 3D view, dimensions 0, 1, 3
   fig, ax = plot_embedding(X_emb, dims=(0, 1, 3))

---

3. Quality Metrics
===================

3.1 ``plot_metrics`` — one entrypoint, many shapes
----------------------------------------------------

Accepts any of: a tidy DataFrame, ``{metric: value}`` mapping, list of records,
or any object exposing ``to_frame()`` / ``metrics_``. The ``plot_type`` argument
picks the visualization style:

==================  ==============================================================
``plot_type``       Use
==================  ==============================================================
``bar`` /            Scalar comparisons across methods or metrics.
``grouped_bar`` /
``lollipop``
``box`` / ``violin`` /  Per-observation distributions when multiple subjects /
``boxen`` /             repeats are present.
``raincloud`` /
``strip`` / ``swarm``
``heatmap``         Method × metric grid.
``line``            Metrics across a numeric ``scope`` axis (e.g., n_components).
``dumbbell`` /      Two-method paired comparison.
``slopegraph``
==================  ==============================================================

Use ``metric``, ``scope``, and ``method`` to filter rows before rendering.

.. code-block:: python

   from coco_pipe.viz import plot_metrics

   fig, ax = plot_metrics({"trustworthiness": 0.92, "continuity": 0.88})
   fig, ax = plot_metrics(reducer, plot_type="heatmap")
   fig, ax = plot_metrics(
       eval_df, plot_type="line", scope="n_components",
       metric="trustworthiness",
   )

3.2 ``plot_eigenvalues`` — scree plots
----------------------------------------

Mapping of label → 1-D or 2-D array. 2-D inputs ``(n_subjects, n_pcs)`` get a
mean ± SEM band; 1-D inputs render as a single line.

.. code-block:: python

   from coco_pipe.viz import plot_eigenvalues

   fig, ax = plot_eigenvalues(
       {"PCA": pca.explained_variance_ratio_},
       max_components=20,
   )

3.3 ``plot_loss_history``
---------------------------

Training loss curve for iterative reducers (UMAP, PHATE, autoencoders). Linear
reducers typically have no loss to plot.

---

4. Structure-Preservation Diagnostics
======================================

4.1 ``plot_shepard_diagram``
------------------------------

Original vs. embedded pairwise distances. A subsample of point pairs is drawn
to keep the figure readable on large datasets.

.. code-block:: python

   from coco_pipe.viz import plot_shepard_diagram

   fig, ax = plot_shepard_diagram(X_orig, X_emb, sample_size=2000)

4.2 ``plot_coranking_matrix``
-------------------------------

Heatmap of a co-ranking matrix produced by ``DimReduction.score()``. By default
the top-left 50×50 block is shown; cap with ``max_k``.

---

5. Feature-Level Insight
=========================

==================================  =========================================
Function                            Use
==================================  =========================================
``plot_feature_importance``         Bars of per-feature importance scores
                                    (re-imported into ``viz.decoding`` for
                                    DataFrame routing).
``plot_component_loadings``         Heatmap of loadings (feature × component)
                                    for linear reducers.
``plot_feature_correlation_heatmap``  Pairwise feature correlation matrix —
                                    sanity check before reducing.
==================================  =========================================

---

6. Trajectories on the Embedding
=================================

These plots assume an array of shape ``(n_trajectories, n_times, n_dims)``
along the first axis — typically per-subject, per-condition, or any grouping.

6.1 ``plot_trajectory``
------------------------

2D or 3D trajectory paths. Pass ``values`` (per-point scalar, e.g., speed) to
color-encode by value; ``speed_mode`` selects ``"linecollection"`` (per-segment
color) or ``"alpha"`` (alpha + lightness modulation). ``smooth_window`` and
``downsample`` reduce noise.

.. code-block:: python

   from coco_pipe.viz import plot_trajectory

   fig, ax = plot_trajectory(
       X_traj,                        # (n_trajectories, n_times, n_dims)
       times=times,
       values=speed,
       labels=condition,
       add_start_end_markers=True,
       dimensions=2,
   )

6.2 ``plot_trajectory_separation``
------------------------------------

Pairwise label-separation timecourses, e.g., distance between condition
centroids over time. Pass ``top_n`` to keep only the most separated pairs.

6.3 ``plot_trajectory_metric_series``
---------------------------------------

A metric (e.g., trustworthiness, continuity) plotted across the trajectory time
axis — useful for spotting when an embedding stops being faithful.

6.4 ``plot_streamlines``
--------------------------

Velocity field over the embedding using :func:`plot_streamfield`. Pair with an
underlying scatter of raw points for context.

---

7. Common Patterns
====================

7.1 Composite figure: scree + embedding + Shepard
----------------------------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import (
       plot_eigenvalues, plot_embedding, plot_shepard_diagram,
   )
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper"):
       fig, axes = plt.subplots(
           1, 3, figsize=figure_size(columns=2, aspect_ratio=0.32),
           constrained_layout=True,
       )
       plot_eigenvalues({"PCA": pca.explained_variance_ratio_}, ax=axes[0])
       plot_embedding(X_emb, labels=class_ids, ax=axes[1])
       plot_shepard_diagram(X, X_emb, ax=axes[2])
       save_figure(fig, "figures/reduction_summary.pdf")

7.2 Backend mirror
--------------------

To get the Plotly equivalent of any of these plots, swap the import:

.. code-block:: python

   from coco_pipe.viz.interactive import dim_reduction as iviz
   ifig = iviz.plot_embedding(X_emb, labels=class_ids)
   ifig.write_html("embedding.html")

The interactive side adds three plots not present in the static module
(``plot_channel_traces``, ``plot_raw_preview``, ``plot_radar_comparison``); see
:ref:`viz-interactive`.
