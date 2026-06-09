.. _dim-reduction-example-compare:

=========================================================
Example: Comparing Multiple Reducers with ``MethodSelector``
=========================================================

This example fits four reducers on the same data, scores them with a shared
:class:`~coco_pipe.dim_reduction.config.EvaluationConfig`, and ranks them with
:class:`~coco_pipe.dim_reduction.evaluation.MethodSelector`.

**Scientific context**: deciding whether to use PCA, UMAP, Pacmap, or PHATE on
a single dataset. We use trustworthiness at ``k=10`` as the primary ranking
metric and continuity as a tie-breaker.

Requires the ``[dim-red]`` extra (``pip install coco-pipe[dim-red]``).

---

1. Configure the Comparison
=============================

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
       DimReduction("UMAP", n_components=2, n_neighbors=15, min_dist=0.1,
                    random_state=42),
       DimReduction("Pacmap", n_components=2, n_neighbors=10, random_state=42),
       DimReduction("PHATE", n_components=2, knn=5, decay=40, random_state=42),
   ]

---

2. Fit and Score Each Reducer
===============================

.. code-block:: python

   for reducer in reducers:
       embedding = reducer.fit_transform(X)
       reducer.score(
           embedding,
           X=X,
           metrics=eval_cfg.metrics,
           k_values=eval_cfg.k_range,
       )

Each scored ``DimReduction`` caches scalar ``metrics_``, tidy
``metric_records_``, and diagnostics. We don't need to keep the embedding
around for ranking — only the records.

---

3. Rank with ``MethodSelector``
=================================

.. code-block:: python

   selector = MethodSelector(reducers).collect()
   ranked = selector.rank_methods(
       selection_metric=eval_cfg.selection_metric,
       selection_k=eval_cfg.selection_k,
       tie_breakers=eval_cfg.tie_breakers,
   )
   print(ranked[["method", "trustworthiness", "continuity"]])

   best_name = ranked.iloc[0]["method"]
   best_reducer = selector.reducers[best_name]

---

4. Visualize the Comparison
=============================

A method × metric heatmap makes the picture obvious at a glance.

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import plot_metrics
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           1, 2, figsize=figure_size(columns=2, aspect_ratio=0.45),
           constrained_layout=True,
       )
       plot_metrics(selector, plot_type="heatmap", ax=axes[0])
       axes[0].set_title("a) Method × metric")

       plot_metrics(
           selector.to_frame(),
           plot_type="line",
           metric="trustworthiness",
           scope="k",
           ax=axes[1],
       )
       axes[1].set_title("b) Trustworthiness by k")

       save_figure(fig, "figures/method_comparison.pdf")

---

5. Drop the Winner Into the Rest of the Pipeline
==================================================

.. code-block:: python

   from coco_pipe.viz import plot_embedding

   best_embedding = best_reducer.fit_transform(X)        # re-embed if not kept
   plot_embedding(best_embedding, labels=labels,
                  title=f"Best method: {best_name}")

   best_reducer.interpret(
       X, X_emb=best_embedding,
       analyses=["correlation"],
       feature_names=feature_names,
   )

---

6. Building a Selector From Saved Records
===========================================

If the scoring runs happen on a cluster and you only ship the tidy records back,
construct the selector directly from records:

.. code-block:: python

   import pandas as pd

   records = pd.read_parquet("records.parquet").to_dict(orient="records")
   selector = MethodSelector.from_records(records)
   ranked = selector.rank_methods(
       selection_metric="trustworthiness",
       selection_k=10,
       tie_breakers=["continuity"],
   )

The ranking semantics are identical regardless of how the records were
collected.
