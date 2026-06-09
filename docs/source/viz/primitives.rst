.. _viz-primitives:

==========================
Plotting Primitives (Base)
==========================

The :mod:`coco_pipe.viz.base` module exposes 12 generic plotting primitives.
Every domain plot in ``viz.decoding`` and ``viz.dim_reduction`` ultimately calls
one of them. They are also available directly for custom figures that don't
fit one of the higher-level templates.

All primitives share a consistent contract:

- Accept ``ax`` (existing axes) or ``figsize`` (new axes).
- Return ``(matplotlib.figure.Figure, matplotlib.axes.Axes)``.
- Apply titles/labels/legends through :func:`coco_pipe.viz._utils.finalize_axes`.
- Use the :ref:`theme palettes <viz-theme>` for default colors.

---

1. Catalog
===========

==========================  ====================================================
Primitive                   Use
==========================  ====================================================
``plot_bar``                Ranked bars with optional error bars, sorting,
                            ``top_n``, horizontal/vertical layout, colormap-by-value.
``plot_line``               2D line with optional error band or bars.
``plot_error_points``       Scatter of point estimates with x/y error bars and
                            optional reference lines.
``plot_distribution_groups``  Grouped box or violin with overlaid jittered points.
``plot_scatter2d``          2D scatter with categorical or continuous color
                            encoding, optional error bars and reference lines.
``plot_scatter3d``          3D scatter with categorical or continuous color
                            encoding.
``plot_heatmap``            2D matrix heatmap with diverging-around-center
                            support, optional cell annotations, colorbar.
``plot_hexbin``             2D hexagonal density binning.
``plot_streamfield``        Stream plot from a gridded ``(U, V)`` velocity field.
``plot_topomap``            Sensor-level topographic map (MNE-backed).
==========================  ====================================================

Each primitive accepts a permissive input type (Series, mapping, sequence,
DataFrame, ndarray as appropriate) and coerces internally so callers don't
have to pre-normalize.

---

2. Bars: ``plot_bar``
======================

Ranked horizontal or vertical bars with optional error magnitudes and a colormap
applied across values.

.. code-block:: python

   from coco_pipe.viz import plot_bar

   fig, ax = plot_bar(
       {"accuracy": 0.71, "balanced_accuracy": 0.66, "roc_auc": 0.78},
       errors={"accuracy": 0.02, "balanced_accuracy": 0.03, "roc_auc": 0.015},
       sort=True,
       top_n=10,
       orientation="horizontal",
       cmap="viridis",
       ylabel="Score",
       title="Cross-validated scores",
   )

Used internally by ``plot_decoding_scores(kind="bar")``,
``plot_feature_importance``, ``plot_eigenvalues``, and more.

---

3. Lines: ``plot_line``
========================

A single 2D line with an optional uncertainty band (``error_style="band"``) or
error bars (``error_style="bar"``).

.. code-block:: python

   from coco_pipe.viz import plot_line

   fig, ax = plot_line(
       times, scores,
       yerr=score_std,
       error_style="band",
       label="LogisticRegression",
       legend=True,
       xlabel="Time (s)", ylabel="Accuracy",
   )

Backbone of every temporal-curve plot.

---

4. Grouped Distributions: ``plot_distribution_groups``
========================================================

Box or violin plots across groups with optional jittered individual points
overlaid. Non-finite values are dropped automatically.

.. code-block:: python

   from coco_pipe.viz import plot_distribution_groups

   fig, ax = plot_distribution_groups(
       [fold_scores_lr, fold_scores_rf, fold_scores_svm],
       labels=["LR", "RF", "SVM"],
       kind="violin",
       showmeans=True,
       ylabel="Accuracy",
   )

---

5. Scatters: ``plot_scatter2d`` / ``plot_scatter3d``
======================================================

Categorical (``labels``) **or** continuous (``c``) color encoding — passing
``labels`` disables the continuous branch. Both 2D and 3D variants accept the
same color/legend options. 2D additionally supports x/y error bars and dashed
reference lines.

.. code-block:: python

   from coco_pipe.viz import plot_scatter2d, plot_scatter3d

   # Categorical
   fig, ax = plot_scatter2d(
       emb[:, 0], emb[:, 1],
       labels=class_ids, alpha=0.7,
       legend_title="Class",
   )

   # Continuous
   fig, ax = plot_scatter2d(
       emb[:, 0], emb[:, 1],
       c=time_index, cmap="viridis", colorbar=True,
       reference_x=0.0, reference_y=0.0,
   )

   # 3D
   fig, ax = plot_scatter3d(
       emb[:, 0], emb[:, 1], emb[:, 2],
       labels=class_ids,
   )

---

6. Heatmaps: ``plot_heatmap``
==============================

Numeric 2D matrices with diverging-around-center support (set ``center`` to a
numeric reference and the colormap is centered with ``TwoSlopeNorm``). Use
``annotate=True`` for cell annotations.

.. code-block:: python

   from coco_pipe.viz import plot_heatmap

   fig, ax = plot_heatmap(
       confusion_df,
       cmap="viridis",
       annotate=True,
       annotation_format=".2f",
       colorbar_label="Count",
       title="Confusion matrix",
   )

Backbone of confusion matrices, generalization matrices, and the co-ranking
matrix.

---

7. Density: ``plot_hexbin``
============================

Hexagonal density binning for 2D scatter clouds too dense for individual points.

---

8. Vector Fields: ``plot_streamfield``
=======================================

Stream-line rendering of a gridded ``(U, V)`` vector field. Underlying scatter
points (e.g., raw embedding samples) can be overlaid via ``points``. Colors
encode local flow speed.

---

9. Topographic Maps: ``plot_topomap``
=======================================

Sensor-level topographic visualization backed by ``mne.viz.plot_topomap``.

Sensor positions can come from any of three sources:

- ``info``: an MNE :class:`mne.Info` object (preferred when available).
- ``coords``: an ``(N, 2+)`` array, ``{name: (x, y)}`` mapping, or DataFrame
  with ``x``/``y`` columns and a ``FeatureName`` or ``Sensor`` column.
- The function infers names from ``values`` (Series/mapping) or ``index``.

.. code-block:: python

   from coco_pipe.viz import plot_topomap

   fig, ax = plot_topomap(
       importance_series,            # {sensor_name: value}
       info=raw.info,                # MNE Info
       symmetric=True,
       cbar_label="Importance",
   )

When ``symmetric=True`` (the default) the color scale is anchored at zero with
the diverging palette; otherwise the sequential palette is used.

---

10. Composing Primitives
==========================

Domain plots are deliberately thin. To build a custom figure that mixes plots,
share an axes between primitives:

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import plot_line, plot_distribution_groups
   from coco_pipe.viz.theme import coco_theme, figure_size

   with coco_theme("paper"):
       fig, axes = plt.subplots(
           1, 2, figsize=figure_size(columns=2, aspect_ratio=0.4),
           constrained_layout=True,
       )
       plot_line(times, mean_score, yerr=score_std, ax=axes[0],
                 xlabel="Time (s)", ylabel="Accuracy")
       plot_distribution_groups(
           [fold_lr, fold_rf], labels=["LR", "RF"], ax=axes[1], ylabel="Accuracy"
       )
