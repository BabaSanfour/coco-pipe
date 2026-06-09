.. _dim-reduction-evaluation:

=================================
Evaluation and Method Comparison
=================================

The evaluation layer answers two questions:

1. **For one embedding**: how well does it preserve the structure of the
   original data?
2. **Across multiple embeddings**: which reducer should I prefer for this
   dataset?

Both flow through a single pure evaluator
(:func:`~coco_pipe.dim_reduction.evaluation.core.evaluate_embedding`) that
emits tidy long-form records, then consumed either through manager scoring or
through :class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` for
ranking.

---

1. Standard 2D Metric Catalog
==============================

All standard metrics operate on an embedding of shape
``(n_samples, n_components)`` and the corresponding original ``X`` with shape
``(n_samples, n_features)``. The first three are computed from a shared
co-ranking matrix; the last is distance-based.

================================  ==================================================
Metric                            What it measures
================================  ==================================================
``trustworthiness``               Penalizes *intrusions* — points that appear in the
                                  embedding's ``k``-nearest neighbors but were not
                                  in the original's ``k``-NN. ``[0, 1]``, higher is
                                  better.
``continuity``                    Penalizes *extrusions* — points that were in the
                                  original's ``k``-NN but were pushed out of the
                                  embedding's ``k``-NN. ``[0, 1]``, higher is better.
``lcmc``                          Local Continuity Meta-Criterion: overlap of the
                                  original and embedding ``k``-NN sets, normalized.
``mrre_intrusion`` /              Mean Relative Rank Error split into intrusion and
``mrre_extrusion`` /              extrusion components, and combined as
``mrre_total``                    ``mrre_total``. Lower is better.
``shepard_correlation``           Spearman correlation between original and embedded
                                  pairwise distances, computed on a subsample.
================================  ==================================================

The co-ranking-based metrics share a per-sample-size validity domain:
``2 * n_samples - 3 * k - 1 > 0``. The evaluator validates this before
computing and surfaces a clear error if it fails.

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction, trustworthiness, continuity, lcmc
   from coco_pipe.dim_reduction.evaluation.metrics import compute_coranking_matrix

   reducer = DimReduction("PCA", n_components=2)
   embedding = reducer.fit_transform(X)

   # Direct use of the primitives (rare — usually done via score()):
   Q = compute_coranking_matrix(X, embedding)
   print(trustworthiness(Q, k=10), continuity(Q, k=10), lcmc(Q, k=10))

In practice, prefer the manager:

.. code-block:: python

   reducer.score(embedding, X=X, k_values=[5, 10, 20])
   reducer.metrics_           # scalar summaries
   reducer.metric_records_    # tidy long-form, one row per (metric, k)

---

2. Trajectory Metrics (Native 3D Paths)
=========================================

When ``X_emb.shape == (n_trajectories, n_times, n_dims)``, the evaluator
switches to trajectory metrics. They are covered in detail in
:ref:`dim-reduction-trajectories`.

---

3. Calling the Pure Evaluator Directly
========================================

Most workflows go through ``DimReduction.score``, but the pure evaluator is
public for advanced use:

.. code-block:: python

   from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

   payload = evaluate_embedding(
       X_emb=embedding,
       X=X,
       method_name="UMAP",
       metrics=["trustworthiness", "continuity"],
       k_values=[5, 10, 20],
       random_state=42,
   )
   payload["metrics"]       # scalar summaries
   payload["records"]       # tidy long-form, ready for plotting / reports
   payload["diagnostics"]   # arrays (e.g., coranking_matrix, shepard_distances)

Inputs:

================================  ==========================================
``X_emb``                         2D ``(n_samples, n_components)`` for
                                  standard metrics; 3D
                                  ``(n_trajectories, n_times, n_dims)`` for
                                  trajectory metrics.
``X``                             Required for 2D paths; optional for 3D.
``metrics``                       Optional metric subset; defaults to "all
                                  applicable for the shape".
``labels`` / ``groups``           Used by supervised separation metrics and
                                  ``trajectory_separation``.
``times``                         Optional time coords for trajectory AUC.
``random_state``                  Seed for sampled Shepard distances.
================================  ==========================================

Output: a dict with keys ``embedding``, ``metrics``, ``metadata``,
``diagnostics``, ``records``, ``artifacts``.

---

4. Tidy Records Schema
========================

Every record is a flat dictionary with at minimum:

================  =================================================
``method``        Reducer name (filled in by the manager / selector).
``metric``        Metric name (e.g., ``"trustworthiness"``).
``value``         Numeric value.
``scope``         Parameter dimension this row is parameterized by
                  (``"k"``, ``"time"``, ``"window"``, ``"pair"``, …) or
                  ``None`` for global scalars.
``scope_value``   Value of ``scope`` for this row.
================  =================================================

Optional columns survive when present: ``group``, ``condition``, ``pair``,
``subject``, ``session``, ``seed``, ``fold``. These are not required by the
selector but pass through to plots and reports unchanged.

This is the same shape consumed by:

- :func:`coco_pipe.viz.plot_metrics` for visualization,
- :class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` for ranking,
- :meth:`coco_pipe.report.Report.add_comparison` for report sections.

---

5. ``MethodSelector``: Post-Hoc Comparison and Ranking
=======================================================

:class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` is a thin
collector + ranker over already-scored reducers. It never fits or scores
anything — only what's already cached.

5.1 Construction
------------------

.. code-block:: python

   from coco_pipe.dim_reduction.evaluation import MethodSelector

   reducers = [DimReduction(m, n_components=2) for m in ["PCA", "UMAP", "Isomap"]]
   for r in reducers:
       emb = r.fit_transform(X)
       r.score(emb, X=X, k_values=[5, 10, 20])

   selector = MethodSelector(reducers).collect()
   # Or: MethodSelector({"pca": pca_reducer, "umap": umap_reducer}).collect()

You can also build from existing records:

.. code-block:: python

   selector = MethodSelector.from_records(metric_records)
   selector = MethodSelector.from_frame(metric_frame)

5.2 Frame Export and Ranking
------------------------------

.. code-block:: python

   frame = selector.to_frame()         # tidy DataFrame
   ranked = selector.rank_methods(
       selection_metric="trustworthiness",
       selection_k=10,
       tie_breakers=["continuity"],
   )
   best_name = ranked.iloc[0]["method"]

``rank_methods`` ranks by **mean** of the selected metric. For ``k``-scoped
metrics, ``selection_k`` narrows comparison to one neighborhood size; ties
are broken using each successive ``tie_breakers`` metric.

5.3 Failure modes the selector catches
----------------------------------------

- Reducers without cached ``metric_records_`` (you forgot to call
  ``score()``).
- Asking to rank by a metric that no reducer ever computed.
- Asking for a ``selection_k`` that none of the records cover.

These all raise :class:`ValueError` at ranking time with a specific message.

---

6. Driving Evaluation From ``EvaluationConfig``
=================================================

When the same metric stack is used across many experiments, drive everything
from one :class:`~coco_pipe.dim_reduction.config.EvaluationConfig`:

.. code-block:: python

   from coco_pipe.dim_reduction.config import EvaluationConfig

   eval_cfg = EvaluationConfig(
       metrics=["trustworthiness", "continuity", "lcmc"],
       k_range=[5, 10, 20],
       selection_metric="trustworthiness",
       selection_k=10,
       tie_breakers=["continuity"],
       separation_method="centroid",
   )

   for r in reducers:
       emb = r.fit_transform(X)
       r.score(emb, X=X,
               metrics=eval_cfg.metrics,
               k_values=eval_cfg.k_range,
               separation_method=eval_cfg.separation_method)

   ranked = MethodSelector(reducers).collect().rank_methods(
       selection_metric=eval_cfg.selection_metric,
       selection_k=eval_cfg.selection_k,
       tie_breakers=eval_cfg.tie_breakers,
   )
