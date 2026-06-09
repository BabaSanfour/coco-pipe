.. _dim-reduction-trajectories:

================================
Trajectory Metrics and Geometry
================================

Trajectory metrics operate on **native 3D embedding tensors** of shape
``(n_trajectories, n_times, n_dims)``. They quantify how groups (trials,
conditions, subjects) move through embedding space — speed, curvature,
dispersion, and time-resolved separation between labeled groups.

These metrics are domain-agnostic: any ordered, grouped embedding (e.g.,
trial-locked EEG, cell-cycle stages, robot trajectories) works.

---

1. Why Native 3D?
==================

A flat ``(n_samples, n_components)`` embedding has lost the trajectory
structure that makes per-time-point metrics meaningful. The evaluation layer
**never reshapes** a 2D embedding into a 3D tensor — any reshape has to happen
upstream:

- :meth:`coco_pipe.io.DataContainer.unstack` to peel off the time axis.
- :meth:`coco_pipe.io.DataContainer.stack` to combine subject + trial axes.
- Custom NumPy reshape with explicit ordering.

If the embedding shape is wrong, the evaluator skips trajectory metrics and
reports them as unavailable.

---

2. Kinematics
===============

All kinematic primitives take ``traj`` with shape
``(n_trials, n_times, n_dims)`` and return per-time-point or
per-trajectory values. They live in
:mod:`coco_pipe.dim_reduction.evaluation.geometry` and are re-exported from
:mod:`coco_pipe.dim_reduction`.

==============================  ================================================
Function                        What it computes
==============================  ================================================
``trajectory_speed``            ``|Δposition / Δt|`` per timepoint.
``trajectory_acceleration``     ``|Δspeed / Δt|`` per timepoint.
``trajectory_curvature``        Local curvature via cosine or Frenet method.
``trajectory_turning_angle``    Angular change between successive segments.
``trajectory_path_length``      Total or cumulative length along the path.
``trajectory_displacement``     ``|endpoint − startpoint|`` per trajectory.
``trajectory_tortuosity``       ``path_length / displacement``.
``trajectory_dispersion``       Spread of trajectories at each timepoint.
``trajectory_distance_from_center``  Distance from each point to its trajectory
                                centroid.
``trajectory_cohesion``         Tight-cluster diagnostic per trajectory.
``trajectory_intra_spread``     Mean within-trajectory pairwise distance.
``trajectory_auc_speed``        Time-integrated speed.
==============================  ================================================

.. code-block:: python

   from coco_pipe.dim_reduction import (
       trajectory_speed,
       trajectory_curvature,
       trajectory_separation,
   )

   speeds = trajectory_speed(traj, dt=1.0)
   curvatures = trajectory_curvature(traj, method="cosine")

When invoked through the manager, the evaluator returns scalar summaries
(``trajectory_speed_mean``, ``trajectory_speed_peak``, etc.) and caches the
full per-timepoint timecourses under ``DimReduction.diagnostics_``.

---

3. Group Separation
=====================

:func:`~coco_pipe.dim_reduction.trajectory_separation` is the high-level
entrypoint for time-resolved separation between labeled trajectory groups.
It returns a mapping ``{(label_a, label_b): timecourse}`` for each label
pair.

Five separation definitions are supported via ``method=``:

================================  ==========================================================
``"centroid"`` (default)          Euclidean distance between per-time-point label
                                  centroids.
``"within_between_ratio"``        Between-centroid distance divided by within-group
                                  spread.
``"mahalanobis"``                 Covariance-aware centroid separation.
``"distributional"``              Energy-distance between trial clouds.
``"margin"``                      Nearest-cross-label minus nearest-within-label
                                  margin.
================================  ==========================================================

.. code-block:: python

   from coco_pipe.dim_reduction import trajectory_separation

   sep = trajectory_separation(traj, labels, method="centroid")
   # {("face", "scrambled"): np.array([0.1, 0.4, 0.9, ...])}

In manager-driven scoring, set ``separation_method`` in
:class:`~coco_pipe.dim_reduction.config.EvaluationConfig` (or pass it
directly to :meth:`DimReduction.score`).

3.1 Choosing a separation method
----------------------------------

- **Centroid**: simple, fast, most interpretable. Good default.
- **Within-between ratio**: when groups have very different spread. Robust
  to scale.
- **Mahalanobis**: when covariance differs between groups. Costs more.
- **Distributional**: when sample sizes are small and you want a
  distribution-aware metric.
- **Margin**: when class boundaries matter more than centroid distance.

---

4. Trajectory Metrics in the Evaluator
========================================

When ``X_emb`` is 3D, the evaluator dispatches to the trajectory path. Each
kinematic family produces both ``_mean`` and ``_peak`` scalar summaries on
``metrics_``; the underlying per-time-point arrays live under
``diagnostics_``.

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction

   reducer = DimReduction("UMAP", n_components=3)
   embedding = reducer.fit_transform(X_flat)

   # Reshape upstream: (n_trials, n_times, 3)
   traj = embedding.reshape(n_trials, n_times, 3)

   scores = reducer.score(
       traj,
       metrics=[
           "trajectory_speed",
           "trajectory_curvature",
           "trajectory_separation",
       ],
       labels=condition_labels,           # required for trajectory_separation
       times=times,                       # optional, used for AUC
       separation_method="centroid",
   )
   scores["metrics"]["trajectory_speed_mean"]
   scores["metrics"]["trajectory_separation_auc_face_vs_scrambled"]

---

5. Visualization
==================

The trajectory plots in :mod:`coco_pipe.viz.dim_reduction` consume the same
``(n_trajectories, n_times, n_dims)`` tensors and the diagnostics emitted by
the evaluator.

.. code-block:: python

   from coco_pipe.viz import (
       plot_trajectory,
       plot_trajectory_separation,
       plot_trajectory_metric_series,
   )

   plot_trajectory(
       traj, times=times, labels=condition_labels,
       values=speeds, speed_mode="linecollection",
   )
   plot_trajectory_separation(reducer.diagnostics_["trajectory_separation"])
   plot_trajectory_metric_series(reducer.diagnostics_["trajectory_speed"], times=times)

See :ref:`viz-dim-reduction` for the full plot reference.

---

6. Conditions and Statistics
==============================

For paired or grouped per-trajectory comparisons (e.g., paired-sample tests
across conditions), use the statistics helpers re-exported from the package:

.. code-block:: python

   from coco_pipe.dim_reduction import (
       grouped_condition_stats,
       paired_condition_stats,
   )

These wrap standard non-parametric tests (Wilcoxon, Friedman) over per-trial
scalar summaries and are convenient when you want a publication-ready
condition comparison without leaving the dim-reduction module.

---

7. Caveats
============

- **Coordinate frame matters**. Distances and curvatures are computed in the
  embedding space — if you compare across reducers, normalize first or rely
  on dimensionless quantities (``tortuosity``, ``turning_angle``).
- **Time spacing**. The default ``dt=1.0`` assumes uniform spacing. Pass an
  explicit ``times`` array to the evaluator when you need physical units.
- **Evaluation-level ``trajectory_dispersion``** is the global unlabeled
  variant. For label-conditioned dispersion, call
  ``geometry.trajectory_dispersion`` directly.
- **Trajectory metrics are descriptive**, not method-selection metrics by
  default. They're emitted to ``metric_records_`` and visualized, but
  ``EvaluationConfig.selection_metric`` only accepts preservation metrics
  (``trustworthiness``, ``continuity``, ``lcmc``, ``mrre_*``,
  ``shepard_correlation``).
