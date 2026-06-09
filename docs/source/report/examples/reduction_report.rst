.. _report-example-reduction:

==========================================================
Example: Multi-Reducer Comparison Report
==========================================================

Build a report that runs three dim-reduction methods on the same
data, scores them with a shared
:class:`~coco_pipe.dim_reduction.config.EvaluationConfig`, ranks
them, and renders the comparison + per-method sections in one HTML
file.

---

1. Fit and Score Three Reducers
=================================

(See :ref:`dim-reduction-example-compare` for the scientific
rationale. Requires ``pip install coco-pipe[dim-red]``.)

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.dim_reduction.config import EvaluationConfig
   from coco_pipe.dim_reduction.evaluation import MethodSelector

   eval_cfg = EvaluationConfig(
       metrics=["trustworthiness", "continuity", "lcmc"],
       k_range=[5, 10, 20, 50],
       selection_metric="trustworthiness",
       selection_k=10,
       tie_breakers=["continuity"],
   )

   reducers = [
       DimReduction("PCA", n_components=2, random_state=42),
       DimReduction("UMAP", n_components=2, n_neighbors=15,
                    min_dist=0.1, random_state=42),
       DimReduction("PHATE", n_components=2, knn=5, decay=40,
                    random_state=42),
   ]
   embeddings = []
   for r in reducers:
       emb = r.fit_transform(X)
       r.score(emb, X=X,
               metrics=eval_cfg.metrics,
               k_values=eval_cfg.k_range)
       embeddings.append(emb)

   selector = MethodSelector(reducers).collect()
   ranked = selector.rank_methods(
       selection_metric=eval_cfg.selection_metric,
       selection_k=eval_cfg.selection_k,
       tie_breakers=eval_cfg.tie_breakers,
   )

---

2. One-Shot Multi-Reducer Report
==================================

The simplest path uses :func:`~coco_pipe.report.from_reductions`,
which calls every adder for each reducer.

.. code-block:: python

   from coco_pipe.io import DataContainer
   from coco_pipe.report import from_reductions

   container = DataContainer(X=X, dims=("obs", "feature"), y=class_ids)

   report = from_reductions(
       reductions=reducers,
       container=container,                  # adds a "Data Overview" section
       embeddings=embeddings,
       labels=class_ids,
       metadata={"subject": subject_ids},
       title="PCA vs UMAP vs PHATE",
       config={"eval_metrics": eval_cfg.metrics},
       output_path="reports/reduction_comparison.html",
   )

This renders:

- "Data Overview" — dimensions, missingness, sample histogram.
- "PCA Overview" + embedding + scalar metrics + co-ranking + scree
  + component loadings.
- "UMAP Overview" + embedding + metrics + co-ranking + loss curve.
- "PHATE Overview" + embedding + metrics + co-ranking.

---

3. Adding the Comparison Table
================================

The factory builds per-method sections, but the cross-method
comparison table is a separate adder. Attach it once:

.. code-block:: python

   report.add_comparison(selector.to_frame(),
                         name="Method Comparison")
   report.save("reports/reduction_comparison.html")

The new section adds:

- A :class:`MetricsTableElement` (method × metric) with best-value
  highlighting per metric column.
- A heatmap of every metric across methods.
- A bar chart of metric details (longer, sortable).
- A radar / spider chart when there are at least 3 metrics and 2
  methods evaluated at the same scope (e.g., the same ``k``).

---

4. Hand-Built Variant (Per-Method Control)
============================================

When the default ``add_reduction`` packaging is too coarse, build the
sections individually:

.. code-block:: python

   from coco_pipe.report import Report

   report = Report(title="Reducer Audit")

   for reducer, emb in zip(reducers, embeddings):
       prefix = reducer.method
       (
           report
           .add_reduction_overview(reducer, name=f"{prefix} — Overview")
           .add_reduction_embedding(emb, labels=class_ids,
                                    name=f"{prefix} — Embedding")
           .add_reduction_metrics(reducer, name=f"{prefix} — Metrics")
           .add_reduction_diagnostics(X, emb, name=f"{prefix} — Shepard")
       )

   report.add_comparison(selector.to_frame())
   report.save("reports/audit.html")

---

5. Loading From Tidy Records (No Live Reducers)
=================================================

If the scoring ran on a cluster and you only have the tidy CSV back
on your laptop, build a :class:`MethodSelector` from records and feed
that into the comparison adder:

.. code-block:: python

   import pandas as pd
   from coco_pipe.dim_reduction.evaluation import MethodSelector
   from coco_pipe.report import Report

   frame = pd.read_csv("results/metric_records.csv")
   selector = MethodSelector.from_frame(frame)

   report = Report(title="Cluster Comparison")
   report.add_comparison(selector.to_frame())
   report.save("reports/cluster_comparison.html")

This pattern is useful for nightly cron / Snakemake runs where the
reducer objects are too heavy to ship back but the metric records
are tiny.

---

6. Trajectory Workflow
========================

For native 3-D trajectory tensors (see
:ref:`dim-reduction-example-trajectory`), pass the tensor as
``embeddings=[traj]`` and the time axis as ``times=...``:

.. code-block:: python

   from coco_pipe.report import from_reductions

   report = from_reductions(
       reductions=[reducer],
       embeddings=[traj],                # shape (n_trials, n_times, n_dims)
       labels=condition_ids,
       times=times,
       title="Trajectory Audit",
       output_path="reports/trajectory.html",
   )

The "Embedding" section renders the trajectory plot directly; the
"Trajectory separation" section is appended automatically when
``diagnostics["trajectory_separation_"]`` is present.

---

7. Tips
=========

================================  =========================================================
Tip                               Detail
================================  =========================================================
Pass ``embeddings=``               Embedding/trajectory sections need explicit arrays;
                                  ``DimReduction.get_summary()`` does **not** carry them.
``raw_preview=True``               Adds an interactive raw-data scroller after the data
                                  overview. Off by default to keep file size down.
Custom section subset              ``sections=["overview", "embedding", "metrics"]``
                                  narrows the per-reducer output.
``MethodSelector.from_records``   Skip the live reducer objects entirely — useful for
                                  reports generated downstream of saved tidy CSVs.
================================  =========================================================
