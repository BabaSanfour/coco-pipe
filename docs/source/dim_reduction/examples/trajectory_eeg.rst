.. _dim-reduction-example-trajectory:

=======================================================
Example: Trajectory Analysis on Trial-Locked Embeddings
=======================================================

This example computes a per-trial trajectory through a 3D embedding and
quantifies condition separation over time. The workflow uses native trajectory
metrics — no reshape inside the evaluator.

**Scientific context**: ``n_trials`` trial-locked EEG windows, each with
``n_times`` samples, are embedded into 3D. Each trial belongs to one of two
conditions (e.g., ``"face"`` / ``"scrambled"``), and we want to see when in
the trial window the conditions separate.

---

1. Reshape Upstream
=====================

The evaluator never reshapes a 2D embedding into a 3D tensor. The reshape
happens here, in user code, with explicit ordering:

.. code-block:: python

   import numpy as np
   from coco_pipe.dim_reduction import DimReduction

   # X_flat has shape (n_trials * n_times, n_features)
   # condition_per_trial has shape (n_trials,)
   n_trials, n_times = 40, 100

   reducer = DimReduction("UMAP", n_components=3, n_neighbors=15, random_state=42)
   flat_embedding = reducer.fit_transform(X_flat)

   # Native 3D trajectory tensor
   traj = flat_embedding.reshape(n_trials, n_times, 3)
   times = np.linspace(-0.2, 0.8, n_times)        # seconds, stimulus-locked
   condition = condition_per_trial               # array of "face"/"scrambled"

If the data ships through :class:`coco_pipe.io.DataContainer`, prefer its
``unstack`` to do the reshape with named axes.

---

2. Score the Trajectory
=========================

.. code-block:: python

   scores = reducer.score(
       traj,
       metrics=[
           "trajectory_speed",
           "trajectory_curvature",
           "trajectory_dispersion",
           "trajectory_separation",
       ],
       labels=condition,
       times=times,
       separation_method="centroid",
   )

   scores["metrics"]["trajectory_speed_mean"]
   scores["metrics"]["trajectory_curvature_peak"]

Per-time-point timecourses live under ``reducer.diagnostics_``:

.. code-block:: python

   speed_per_time = reducer.diagnostics_["trajectory_speed"]
   separation = reducer.diagnostics_["trajectory_separation"]
   # {("face", "scrambled"): np.ndarray of shape (n_times,)}

---

3. Visualize
==============

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import (
       plot_trajectory,
       plot_trajectory_separation,
       plot_trajectory_metric_series,
   )
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           2, 2, figsize=figure_size(columns=2, aspect_ratio=0.8),
           constrained_layout=True,
       )

       # (a) Trajectories colored by condition
       plot_trajectory(
           traj, times=times, labels=condition,
           values=reducer.diagnostics_["trajectory_speed"],
           speed_mode="linecollection",
           dimensions=3,
           ax=axes[0, 0],
       )
       axes[0, 0].set_title("a) 3D trajectories (color: speed)")

       # (b) Mean speed over time
       plot_trajectory_metric_series(
           reducer.diagnostics_["trajectory_speed"],
           times=times, ax=axes[0, 1],
       )
       axes[0, 1].set_title("b) Speed timecourse")

       # (c) Curvature over time
       plot_trajectory_metric_series(
           reducer.diagnostics_["trajectory_curvature"],
           times=times, ax=axes[1, 0],
       )
       axes[1, 0].set_title("c) Curvature timecourse")

       # (d) Separation between conditions
       plot_trajectory_separation(
           reducer.diagnostics_["trajectory_separation"],
           times=times, ax=axes[1, 1],
       )
       axes[1, 1].set_title("d) Face vs. Scrambled separation")

       save_figure(fig, "figures/trajectory_summary.pdf")

---

4. Switch Separation Definitions
==================================

The ``centroid`` separation above is the default. For groups with different
within-group spread, a normalized definition is more interpretable:

.. code-block:: python

   reducer.score(
       traj,
       metrics=["trajectory_separation"],
       labels=condition,
       times=times,
       separation_method="within_between_ratio",
   )

See :ref:`dim-reduction-trajectories` for the full list (``mahalanobis``,
``distributional``, ``margin``).

---

5. Compare Across Reducers
============================

To compare how UMAP, PHATE, and DMD organize the same trajectories, score each
in turn and pass the scored ``DimReduction`` objects to ``MethodSelector``:

.. code-block:: python

   from coco_pipe.dim_reduction.evaluation import MethodSelector

   reducers = [
       DimReduction("UMAP", n_components=3, random_state=42),
       DimReduction("PHATE", n_components=3, random_state=42),
       DimReduction("DMD", n_components=3),
   ]
   for r in reducers:
       flat_emb = r.fit_transform(X_flat)
       traj = flat_emb.reshape(n_trials, n_times, 3)
       r.score(traj,
               metrics=["trajectory_speed", "trajectory_separation"],
               labels=condition, times=times)

   selector = MethodSelector(reducers).collect()
   selector.to_frame().query("metric == 'trajectory_separation_auc'")

Trajectory metrics are descriptive (not ranking metrics), so use
``MethodSelector`` primarily to expose them side by side — not to call a
"winner" by separation AUC alone.

---

6. Common Pitfalls
====================

- **Implicit reshape**. Never trust silent reshaping. Always verify
  ``traj.shape[0] == n_trials`` and ``traj.shape[1] == n_times``.
- **Cross-reducer trajectory comparison**. Distances are computed in each
  reducer's embedding space; cross-reducer scalar comparisons (e.g., raw
  speed) are not meaningful. Use dimensionless quantities (``tortuosity``,
  ``turning_angle``) or normalize.
- **Time spacing**. Pass an explicit ``times`` array when sampling is not
  uniform; otherwise the default ``dt=1.0`` is used.
- **Label alignment**. ``labels`` must have shape ``(n_trials,)`` — not
  ``(n_trials * n_times,)``. If the labels are flat, take every
  ``n_times``-th element after sorting.
