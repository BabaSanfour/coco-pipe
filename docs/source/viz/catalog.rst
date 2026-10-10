.. _viz-catalog:

============
Plot Catalog
============

.. _viz-decoding:

Decoding Plots (Static Backend)
===============================

This page covers the ~25 static plotting functions in
:mod:`coco_pipe.viz.decoding`. Every function accepts either an
:class:`coco_pipe.decoding.result.ExperimentResult` or the tidy DataFrame
produced by the corresponding result accessor — the wrapper detects the input
type and routes accordingly. All functions return ``(Figure, Axes)``.

The interactive Plotly mirror lives in :mod:`coco_pipe.viz.interactive.decoding`
and is described in :ref:`viz-interactive`.

---

1. Result Accessor → Plot Mapping
---------------------------------

Each decoding plot maps to a specific :class:`ExperimentResult` accessor. If
you pass a raw DataFrame, it must match the schema that accessor produces.

.. list-table::
   :header-rows: 1

   * - ``viz.decoding`` function
     - ``ExperimentResult`` accessor
     - Required columns (raw input)
   * - ``plot_decoding_scores``
     - ``get_detailed_scores()``
     - ``Model``, ``Metric``, ``Value``
   * - ``plot_fold_score_dispersion``
     - ``get_detailed_scores()``
     - ``Model``, ``Metric``, ``Fold``, ``Value``
   * - ``plot_confusion_matrix``
     - ``get_confusion_matrices()`` / ``get_pooled_confusion_matrix()``
     - ``TrueLabel``, ``PredictedLabel``, ``Value``
   * - ``plot_roc_curve``
     - ``get_roc_curve()``
     - ``Model``, ``FPR``, ``TPR``
   * - ``plot_pr_curve``
     - ``get_pr_curve()``
     - ``Model``, ``Precision``, ``Recall``
   * - ``plot_calibration_curve``
     - ``get_calibration_curve()``
     - ``Model``, ``MeanPredicted``, ``Fraction``
   * - ``plot_probability_diagnostics``
     - ``get_probability_diagnostics()``
     - ``Model``, ``Metric``, ``Value``
   * - ``plot_temporal_score_curve``
     - ``get_temporal_score_summary()``
     - ``Model``, ``Metric``, ``Time``, ``Mean``
   * - ``plot_temporal_generalization_matrix``
     - ``get_generalization_matrix()``
     - ``Model``, ``TrainTime``, ``TestTime``, ``Value``
   * - ``plot_temporal_statistical_assessment``
     - ``get_statistical_assessment()``
     - ``Time``, ``Observed``, ``NullMean``, ``NullStd``
   * - ``plot_null_interval_summary``
     - ``get_statistical_assessment()``
     - ``Model``, ``Metric``, ``Observed``, ``NullLow``, ``NullHigh``
   * - ``plot_model_comparison``
     - ``compare_models()`` / ``compare_models_paired()``
     - ``ModelA``, ``ModelB``, ``Difference``
   * - ``plot_fit_diagnostics``
     - ``get_fit_diagnostics()``
     - ``Model``, ``Fold``, ``FitSeconds``, ``Score``
   * - ``plot_training_history``
     - ``get_model_artifacts()``
     - ``Epoch``, ``Loss`` / ``Metric``
   * - ``plot_search_results``
     - ``get_search_results()``
     - ``Model``, ``Param``, ``MeanScore``
   * - ``plot_subject_diagnostics``
     - ``get_predictions()``
     - ``Subject``, ``y_true``, ``y_pred``
   * - ``plot_group_summary``
     - ``get_predictions()``
     - ``Subject``, ``y_true``, ``y_pred``
   * - ``plot_regression_diagnostics``
     - ``get_predictions()``
     - ``y_true``, ``y_pred``
   * - ``plot_feature_importance``
     - ``get_feature_importances()``
     - ``FeatureName``, ``Mean`` (or ``Importance``)
   * - ``plot_feature_stability``
     - ``get_feature_stability()``
     - ``FeatureName``, ``Stability``
   * - ``plot_feature_scores``
     - ``get_feature_scores()``
     - ``FeatureName``, ``Score``
   * - ``plot_decoding_topomap``
     - ``get_feature_importances()`` + sensor layout
     - ``FeatureName``, value column
   * - ``plot_sensor_feature_heatmap``
     - ``get_feature_importances()`` + sensor layout
     - ``FeatureName``, value column
   * - ``plot_sensor_feature_profile``
     - ``get_feature_importances()`` + sensor layout
     - ``FeatureName``, value column
   * - ``plot_feature_sensor_profile``
     - ``get_feature_importances()`` + sensor layout
     - ``FeatureName``, value column

---

2. Score Summaries
------------------

2.1 ``plot_decoding_scores`` — aggregate scalar scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per-fold strip/box overlay, useful for spotting unstable models or outlier
folds before any aggregation.

.. code-block:: python

   from coco_pipe.viz import plot_fold_score_dispersion

   fig, ax = plot_fold_score_dispersion(result, metric="balanced_accuracy")

---

3. Classification Diagnostics
-----------------------------

3.1 ``plot_confusion_matrix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Aggregated confusion matrix. Pass ``model`` and/or ``fold`` to filter; otherwise
all rows are summed.

.. code-block:: python

   from coco_pipe.viz import plot_confusion_matrix

   fig, ax = plot_confusion_matrix(result, model="logistic_regression")

3.2 ROC, PR, Calibration
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bar summary of log-loss and Brier-score per model. Requires probability
estimates (either native ``predict_proba`` or a calibrated classifier — see
:ref:`decoding-concepts`).

---

4. Temporal Decoding
--------------------

4.1 ``plot_temporal_score_curve``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mean score per time-point with a ±1 SD band. Non-numeric time labels are placed
at integer positions and shown as rotated tick labels.

.. code-block:: python

   from coco_pipe.viz import plot_temporal_score_curve

   fig, ax = plot_temporal_score_curve(result, metric="accuracy")

4.2 ``plot_temporal_generalization_matrix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Train-time × test-time heatmap. Off-diagonal mass indicates stable neural
representations (see :ref:`decoding-concepts`).

.. code-block:: python

   from coco_pipe.viz import plot_temporal_generalization_matrix

   fig, ax = plot_temporal_generalization_matrix(result, metric="accuracy")

4.3 ``plot_temporal_statistical_assessment``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Observed temporal curve drawn against the permutation null band. Significant
time-points (after the chosen correction) are shaded.

.. code-block:: python

   from coco_pipe.viz import plot_temporal_statistical_assessment

   fig, ax = plot_temporal_statistical_assessment(result, metric="accuracy")

---

5. Statistical Assessment
-------------------------

5.1 ``plot_null_interval_summary``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One row per model showing the observed scalar score against the permutation
null interval; ideal for compact scalar summaries.

5.2 ``plot_model_comparison``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairwise model comparisons rendered as either paired-difference distributions
(``paired=True``) or independent ones. Pass ``reference`` to anchor every
comparison against a single model.

.. code-block:: python

   from coco_pipe.viz import plot_model_comparison

   fig, ax = plot_model_comparison(result, metric="accuracy",
                                   reference="logistic_regression")

---

6. Training-Time Diagnostics
----------------------------

- ``plot_fit_diagnostics``: per-fold fit time, score, and convergence warnings
  for catching unstable models.
- ``plot_training_history``: neural-network loss / metric curves from
  ``get_model_artifacts()``.
- ``plot_search_results``: hyperparameter grid scores, useful for sanity-checking
  ``TuningConfig`` outcomes.

---

7. Per-Subject and Regression Outputs
-------------------------------------

- ``plot_subject_diagnostics``: per-subject accuracy histogram and outliers
  derived from ``get_predictions()``.
- ``plot_group_summary``: subject-level aggregated metric summary (uses the
  same accessor).
- ``plot_regression_diagnostics``: residuals and predicted-vs-true scatter for
  ``task="regression"`` experiments.

---

8. Feature Importance, Stability, Selection
-------------------------------------------

8.1 ``plot_feature_importance``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Selection rate across folds, as bar or heatmap. Higher is more stable.

.. code-block:: python

   from coco_pipe.viz import plot_feature_stability

   fig, ax = plot_feature_stability(result, kind="bar", top_n=25)

8.3 ``plot_feature_scores`` / ``plot_search_results``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Univariate selector scores and tuned hyperparameter grids, respectively.

---

9. Sensor-Level Decoding Maps
-----------------------------

These plots accept a sensor-feature DataFrame (typically derived from
``get_feature_importances()``) and resolve sensor positions from either an
MNE :class:`~mne.Info` (``info=``) or a coordinate table (``coords=``).

.. list-table::
   :header-rows: 1

   * - Function
     - Picture
   * - ``plot_decoding_topomap`` ``plot_sensor_feature_heatmap`` Sensor × feature-family heatmap. ``plot_sensor_feature_profile`` Bars: feature importances within one sensor. ``plot_feature_sensor_profile`` Topomap: one feature family across sensors.
     - Topomap of one value column across sensors.

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
-------------------

10.1 Combining a result and a custom axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many functions accept ``model`` / ``metric`` / ``fold`` filters that map to the
same column names in the accessor output. They short-circuit before rendering,
so passing a filter is cheaper than slicing the DataFrame yourself.

10.3 Backend mirror
~~~~~~~~~~~~~~~~~~~

To get the Plotly equivalent of any of these plots, swap the import:

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz
   ifig = iviz.plot_roc_curve(result)
   ifig.write_html("roc.html")

A handful of static-only plots (``plot_decoding_topomap``,
``plot_sensor_feature_*``) are matplotlib-only because they rely on MNE; see
:ref:`viz-interactive` for the current parity table.

.. _viz-dim-reduction:

Dimensionality Reduction Plots (Static)
=======================================

This page covers the ~20 static plotting functions in
:mod:`coco_pipe.viz.dim_reduction`. They turn embedded data, quality metrics,
and trajectory tensors into publication-grade figures. All functions return
``(Figure, Axes)``.

The interactive Plotly mirror lives in
:mod:`coco_pipe.viz.interactive.dim_reduction`; see :ref:`viz-interactive` for
the parity table.

---

1. Reducer Source → Plot Mapping
--------------------------------

.. list-table::
   :header-rows: 1

   * - What you have
     - Plot
   * - A 2D or 3D embedding array
     - ``plot_embedding``
   * - A reducer's ``metrics_``
     - ``plot_metrics``
   * - PCA / SVD eigenvalues
     - ``plot_eigenvalues``
   * - Iterative reducer loss
     - ``plot_loss_history``
   * - Original + embedded distances
     - ``plot_shepard_diagram``
   * - A co-ranking matrix
     - ``plot_coranking_matrix``
   * - Component loadings
     - ``plot_component_loadings``
   * - Per-feature scores
     - ``plot_feature_importance`` (in this module)
   * - Feature correlations
     - ``plot_feature_correlation_heatmap``
   * - Trajectory tensors
     - ``plot_trajectory``, ``plot_trajectory_separation``, ``plot_trajectory_metric_series``
   * - Velocity field over embedding
     - ``plot_streamlines``

---

2. Embedding Visualization
--------------------------

2.1 ``plot_embedding``
~~~~~~~~~~~~~~~~~~~~~~

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
------------------

3.1 ``plot_metrics`` — one entrypoint, many shapes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Accepts any of: a tidy DataFrame, ``{metric: value}`` mapping, list of records,
or any object exposing ``to_frame()`` / ``metrics_``. The ``plot_type`` argument
picks the visualization style:

.. list-table::
   :header-rows: 1

   * - ``plot_type``
     - Use
   * - ``bar`` / ``grouped_bar`` / ``lollipop``
     - Scalar comparisons across methods or metrics.
   * - ``box`` / ``violin`` /
     - Per-observation distributions when multiple subjects /
   * - ``boxen`` / ``raincloud`` / ``strip`` / ``swarm``
     - repeats are present.
   * - ``heatmap``
     - Method × metric grid.
   * - ``line``
     - Metrics across a numeric ``scope`` axis (e.g., n_components).
   * - ``dumbbell`` / ``slopegraph``
     - Two-method paired comparison.

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mapping of label → 1-D or 2-D array. 2-D inputs ``(n_subjects, n_pcs)`` get a
mean ± SEM band; 1-D inputs render as a single line.

.. code-block:: python

   from coco_pipe.viz import plot_eigenvalues

   fig, ax = plot_eigenvalues(
       {"PCA": pca.explained_variance_ratio_},
       max_components=20,
   )

3.3 ``plot_loss_history``
~~~~~~~~~~~~~~~~~~~~~~~~~

Training loss curve for iterative reducers (UMAP, PHATE, autoencoders). Linear
reducers typically have no loss to plot.

---

4. Structure-Preservation Diagnostics
-------------------------------------

4.1 ``plot_shepard_diagram``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Original vs. embedded pairwise distances. A subsample of point pairs is drawn
to keep the figure readable on large datasets.

.. code-block:: python

   from coco_pipe.viz import plot_shepard_diagram

   fig, ax = plot_shepard_diagram(X_orig, X_emb, sample_size=2000)

4.2 ``plot_coranking_matrix``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heatmap of a co-ranking matrix produced by ``DimReduction.score()``. By default
the top-left 50×50 block is shown; cap with ``max_k``.

---

5. Feature-Level Insight
------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Use
   * - ``plot_feature_importance``
     - Bars of per-feature importance scores (re-imported into ``viz.decoding`` for DataFrame routing).
   * - ``plot_component_loadings``
     - Heatmap of loadings (feature × component) for linear reducers.
   * - ``plot_feature_correlation_heatmap``
     - Pairwise feature correlation matrix — sanity check before reducing.

---

6. Trajectories on the Embedding
--------------------------------

These plots assume an array of shape ``(n_trajectories, n_times, n_dims)``
along the first axis — typically per-subject, per-condition, or any grouping.

6.1 ``plot_trajectory``
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pairwise label-separation timecourses, e.g., distance between condition
centroids over time. Pass ``top_n`` to keep only the most separated pairs.

6.3 ``plot_trajectory_metric_series``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A metric (e.g., trustworthiness, continuity) plotted across the trajectory time
axis — useful for spotting when an embedding stops being faithful.

6.4 ``plot_streamlines``
~~~~~~~~~~~~~~~~~~~~~~~~

Velocity field over the embedding using :func:`plot_streamfield`. Pair with an
underlying scatter of raw points for context.

---

7. Common Patterns
------------------

7.1 Composite figure: scree + embedding + Shepard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~

To get the Plotly equivalent of any of these plots, swap the import:

.. code-block:: python

   from coco_pipe.viz.interactive import dim_reduction as iviz
   ifig = iviz.plot_embedding(X_emb, labels=class_ids)
   ifig.write_html("embedding.html")

The interactive side adds a few plots not present in the static module
(``plot_raw_preview``, ``plot_radar_comparison``, ``plot_phase_portrait``,
``plot_scree``); see :ref:`viz-interactive`.

.. _viz-interactive:

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
------------------

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
------------------

All plots in :mod:`coco_pipe.viz.interactive.decoding` accept the same
``ExperimentResult`` or tidy DataFrame as their static twins.

.. list-table::
   :header-rows: 1

   * - Function
     - Static counterpart
   * - ``plot_confusion_matrix``
     - ``viz.decoding.plot_confusion_matrix``
   * - ``plot_roc_curve``
     - ``viz.decoding.plot_roc_curve``
   * - ``plot_pr_curve``
     - ``viz.decoding.plot_pr_curve``
   * - ``plot_calibration_curve``
     - ``viz.decoding.plot_calibration_curve``
   * - ``plot_fold_score_dispersion``
     - ``viz.decoding.plot_fold_score_dispersion``
   * - ``plot_temporal_score_curve``
     - ``viz.decoding.plot_temporal_score_curve``
   * - ``plot_temporal_generalization_matrix``
     - ``viz.decoding.plot_temporal_generalization_matrix``
   * - ``plot_temporal_statistical_assessment``
     - ``viz.decoding.plot_temporal_statistical_assessment``
   * - ``plot_null_interval_summary``
     - ``viz.decoding.plot_null_interval_summary``
   * - ``plot_training_history``
     - ``viz.decoding.plot_training_history``
   * - ``plot_decoding_scores``
     - ``viz.decoding.plot_decoding_scores``
   * - ``plot_model_comparison``
     - ``viz.decoding.plot_model_comparison``
   * - ``plot_fit_diagnostics``
     - ``viz.decoding.plot_fit_diagnostics``
   * - ``plot_probability_diagnostics``
     - ``viz.decoding.plot_probability_diagnostics``
   * - ``plot_subject_diagnostics``
     - ``viz.decoding.plot_subject_diagnostics``
   * - ``plot_group_summary``
     - ``viz.decoding.plot_group_summary``
   * - ``plot_regression_diagnostics``
     - ``viz.decoding.plot_regression_diagnostics``
   * - ``plot_search_results``
     - ``viz.decoding.plot_search_results``
   * - ``plot_feature_stability``
     - ``viz.decoding.plot_feature_stability``
   * - ``plot_feature_scores``
     - ``viz.decoding.plot_feature_scores``

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
----------------------------------

.. list-table::
   :header-rows: 1

   * - Function
     - Static counterpart
   * - ``plot_embedding``
     - ``viz.dim_reduction.plot_embedding``
   * - ``plot_metrics``
     - ``viz.dim_reduction.plot_metrics``
   * - ``plot_loss_history``
     - ``viz.dim_reduction.plot_loss_history``
   * - ``plot_eigenvalues``
     - ``viz.dim_reduction.plot_eigenvalues``
   * - ``plot_shepard_diagram``
     - ``viz.dim_reduction.plot_shepard_diagram``
   * - ``plot_coranking_matrix``
     - ``viz.dim_reduction.plot_coranking_matrix``
   * - ``plot_component_loadings``
     - ``viz.dim_reduction.plot_component_loadings``
   * - ``plot_feature_importance``
     - ``viz.dim_reduction.plot_feature_importance``
   * - ``plot_feature_correlation_heatmap``
     - ``viz.dim_reduction.plot_feature_correlation_heatmap``
   * - ``plot_streamlines``
     - ``viz.dim_reduction.plot_streamlines``
   * - ``plot_trajectory``
     - ``viz.dim_reduction.plot_trajectory``
   * - ``plot_trajectory_metric_series``
     - ``viz.dim_reduction.plot_trajectory_metric_series``
   * - ``plot_trajectory_separation``
     - ``viz.dim_reduction.plot_trajectory_separation``

Interactive-only additions (no static twin yet):

- ``plot_raw_preview`` — scrollable preview of multichannel raw traces.
- ``plot_radar_comparison`` — radar/spider chart for multi-metric model comparison.
- ``plot_phase_portrait`` — 2-D phase portrait of trajectory dynamics.
- ``plot_scree`` — scree plot of explained variance.

---

4. Notebook Workflow
--------------------

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
----------------------------

Use ``fig.to_html(include_plotlyjs="cdn", full_html=False)`` to get an embedded
``<div>`` suitable for inclusion in a report (see :mod:`coco_pipe.report`).

.. code-block:: python

   from coco_pipe.viz.interactive import decoding as iviz

   html_fragment = iviz.plot_roc_curve(result).to_html(
       include_plotlyjs="cdn", full_html=False
   )

---

6. Backend Compatibility Notes
------------------------------

- Interactive plots accept the **same input types** as their static
  counterparts (``ExperimentResult`` or matching DataFrame); the prep layer
  (``viz._utils``) is shared.
- Default colors come from the same ``_COLORBLIND_COLORS`` palette used by
  the matplotlib theme, so static and interactive renderings of the same data
  use matching colors.
- The Plotly template name is :data:`coco_pipe.viz.interactive._utils.COCO_TEMPLATE`.
- If you build figures outside the wrappers, calling
  ``fig.update_layout(template="coco")`` is enough to match the look.
