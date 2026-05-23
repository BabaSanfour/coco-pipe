.. _dim-reduction-concepts:

==================================
Scientific Concepts and Principles
==================================

This page explains the foundational decisions behind ``coco_pipe.dim_reduction``.
Understanding them prevents the most common mistakes when reducing
high-dimensional scientific data — pseudo-clustering, embedding-leakage in
evaluation, and misinterpreting reducer outputs.

---

1. Reduction vs. Evaluation vs. Interpretation
================================================

Three concerns are kept deliberately separate:

- **Reduction** produces an embedding from data. A
  :class:`~coco_pipe.dim_reduction.DimReduction` wraps one fitted reducer.
- **Evaluation** measures whether the embedding preserves the original
  structure. Implemented by ``evaluation.core.evaluate_embedding``.
- **Interpretation** measures which input features appear to drive the
  embedding axes. Implemented by
  :func:`~coco_pipe.dim_reduction.analysis.interpret_features`.

Each step accepts an **explicit embedding array** rather than re-running the
reducer. The manager never silently re-fits or re-embeds.

.. admonition:: Why the explicit-embedding contract?

   Caching embeddings on the manager would make it easy to score an embedding
   the reducer no longer produces (e.g., after parameter changes), or to score
   an embedding that was computed on a different sample. Forcing the user to
   pass the embedding object explicitly makes lineage visible at the call site.

---

2. Sample-Layout Matters: 2D vs. 3D Embeddings
================================================

The evaluator routes by embedding shape, not by method name:

==================================  =====================================
``X_emb.shape``                     Path
==================================  =====================================
``(n_samples, n_components)``        Standard 2D metrics (trustworthiness,
                                    continuity, LCMC, MRRE, Shepard).
``(n_trajectories, n_times, n_dims)``  Trajectory metrics (speed, curvature,
                                    dispersion, separation).
==================================  =====================================

For 2D paths, the **original** data ``X`` is required (the co-ranking matrix
needs both spaces). For 3D paths, ``X`` is optional — most trajectory metrics
operate purely on the embedded tensor.

.. warning::

   Trajectory metrics never reshape a flat 2D embedding into a 3D tensor.
   Any reshape has to happen upstream (e.g., via
   :meth:`coco_pipe.io.DataContainer.unstack`). Silent reshaping would invent
   trajectory structure that is not in the data.

---

3. Strict Configuration
========================

Every reducer has a pydantic :class:`~coco_pipe.dim_reduction.config.BaseReducerConfig`
subclass:

- **No unknown fields**: typos like ``n_neigbors`` fail at construction, not
  at fit time.
- **Canonical method names**: exact strings (``"PCA"``, ``"UMAP"``,
  ``"TopologicalAE"``…). No aliasing.
- **Typed constructors**: the manager can be built from
  ``DimReduction("UMAP", n_neighbors=15)`` *or* from
  ``DimReduction(UMAPConfig(n_neighbors=15))``.

The :class:`~coco_pipe.dim_reduction.config.EvaluationConfig` follows the same
contract: invalid metric names, duplicate entries, and selection metrics not
present in ``metrics`` fail at parse time. This prevents the common pattern
where the experiment runs but downstream ranking silently does nothing.

---

4. Embedding-Aware Metric Selection
=====================================

The evaluator never silently skips a requested metric. If the embedding shape
or required inputs are incompatible, the metric is reported as unavailable
with a clear reason in the metric payload.

Standard 2D metric families:

- **Co-ranking based**: ``trustworthiness``, ``continuity``, ``lcmc``,
  ``mrre_intrusion``, ``mrre_extrusion``, ``mrre_total``. All require a
  ``(n_samples - 1) × (n_samples - 1)`` co-ranking matrix and a chosen
  neighborhood ``k``.
- **Distance preservation**: ``shepard_correlation`` (rank-correlation of
  pairwise original vs. embedded distances).

Trajectory metric families (operating on ``(n_trajectories, n_times, n_dims)``):

- **Kinematics**: ``trajectory_speed``, ``trajectory_acceleration``,
  ``trajectory_curvature``, ``trajectory_turning_angle``.
- **Geometry**: ``trajectory_path_length``, ``trajectory_displacement``,
  ``trajectory_tortuosity``, ``trajectory_dispersion``.
- **Group structure**: ``trajectory_separation`` (requires per-trajectory
  ``labels``).

See :ref:`dim-reduction-evaluation` and :ref:`dim-reduction-trajectories` for
the full metric catalog and the math behind each one.

---

5. Tidy Records and Post-Hoc Comparison
=========================================

Every scored reducer caches the evaluator's tidy long-form output on
``DimReduction.metric_records_``. Records have these columns:

- ``method`` — reducer name.
- ``metric`` — metric name.
- ``value`` — numeric value.
- ``scope`` — what the value is parameterized by (``"k"``, ``"time"``,
  ``"window"``, etc., or ``None`` for global scalars).
- ``scope_value`` — value of ``scope`` for this record.

Optional columns (``group``, ``condition``, ``pair``, ``subject``,
``session``, ``seed``, ``fold``) survive when present. This is the same shape
consumed by:

- :class:`coco_pipe.dim_reduction.evaluation.MethodSelector` for ranking,
- :func:`coco_pipe.viz.plot_metrics` for visualization,
- :meth:`coco_pipe.report.Report.add_comparison` for report sections.

.. admonition:: Why post-hoc?

   Some users score and rank in the same script; others score on a cluster
   and compare later. ``MethodSelector`` accepts either a list of scored
   :class:`~coco_pipe.dim_reduction.DimReduction` objects or a frame of tidy
   records (``MethodSelector.from_records`` / ``from_frame``) so both flows
   share the same ranking semantics.

---

6. Interpretation Is Not Preservation Scoring
==============================================

Preservation tells you *whether* an embedding faithfully represents the
original data; interpretation tells you *which input features* appear to
drive each embedding axis. These are independent questions:

- A PCA embedding can have perfect ``trustworthiness`` *and* a misleading
  feature interpretation if multiple features are collinear.
- A non-linear embedding can be highly informative even with weak Spearman
  correlations between input features and embedding axes.

The three interpretation backends ("correlation", "perturbation", "gradient")
target different reducer classes and computational budgets. See
:ref:`dim-reduction-interpretation`.

---

7. Lazy Optional Dependencies
==============================

Heavy libraries (``torch``, ``umap-learn``, ``dask``, ``pydmd``, ``ivis``…)
are imported inside reducer methods, not at package import. ``import
coco_pipe.dim_reduction`` is safe even if you only have the base scientific
Python stack. See :ref:`dim-reduction-dependencies` for which extras unlock
which methods.

---

8. Reducer Capability Contracts
=================================

Every reducer exposes a ``capabilities`` dict that the manager and the
evaluator inspect. Common flags include:

- ``is_linear`` — whether components are linear projections of inputs.
- ``has_components`` — whether component loadings can be extracted via
  ``DimReduction.get_components()``.
- ``has_loss_history`` — whether training loss is available for plotting.
- ``input_ndim`` / ``input_layout`` — expected input shape, used to validate
  inputs early.

Custom reducers declare their own capabilities by overriding the property;
see :ref:`dim-reduction-custom-reducers`.
