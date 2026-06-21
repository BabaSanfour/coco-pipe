.. _dim-reduction-evaluation:

=============================
Evaluation and Interpretation
=============================

Evaluation and Method Comparison
================================

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
-----------------------------

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
---------------------------------------

When ``X_emb.shape == (n_trajectories, n_times, n_dims)``, the evaluator
switches to trajectory metrics. They are covered in detail in
:ref:`dim-reduction-trajectories`.

---

3. Calling the Pure Evaluator Directly
--------------------------------------

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
----------------------

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
------------------------------------------------------

:class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` is a thin
collector + ranker over already-scored reducers. It never fits or scores
anything — only what's already cached.

5.1 Construction
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Reducers without cached ``metric_records_`` (you forgot to call
  ``score()``).
- Asking to rank by a metric that no reducer ever computed.
- Asking for a ``selection_k`` that none of the records cover.

These all raise :class:`ValueError` at ranking time with a specific message.

---

6. Driving Evaluation From ``EvaluationConfig``
-----------------------------------------------

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

.. _dim-reduction-interpretation:

Feature Interpretation
======================

Interpretation answers: *which input features appear to drive each embedding
axis*? This is independent of preservation scoring (covered in
:ref:`dim-reduction-evaluation`).

Three backends with different cost / reducer-class tradeoffs are available
through :meth:`coco_pipe.dim_reduction.DimReduction.interpret` and the pure
backend :func:`coco_pipe.dim_reduction.analysis.interpret_features`.

---

1. Backends at a Glance
-----------------------

==================  ==========================================  ========================
Backend             What it measures                            Reducer requirements
==================  ==========================================  ========================
``correlation``     Spearman correlation between each input     Any reducer (just needs
                    feature and each embedding axis.            an embedding).
``perturbation``    Mean-squared shift in the embedding when    Any reducer with a
                    each feature is independently shuffled.     fitted ``transform``.
``gradient``        Encoder saliency: ``∂‖z‖ / ∂x`` averaged    Torch-based encoders
                    over samples.                               (``IVIS``,
                                                                ``ParametricUMAP``,
                                                                ``TopologicalAE``).
==================  ==========================================  ========================

All three return tidy long-form records suitable for the same plotting and
report paths.

---

2. Correlation (Default)
------------------------

Spearman correlation between every column of ``X`` and every column of
``X_emb``. Returns a nested mapping of dimension → feature → correlation,
sorted by absolute magnitude within each dimension.

.. code-block:: python

   from coco_pipe.dim_reduction.analysis import correlate_features

   per_dim = correlate_features(X, X_emb, feature_names=feature_names)
   # {"Dimension 1": {"feat_07": -0.81, "feat_12": 0.74, ...}, ...}

When the input is constant or the embedding axis is degenerate, the Spearman
coefficient is undefined; ``correlate_features`` reports those as ``0.0`` so
the output stays sortable.

Cost: ``O(n_features * n_components)`` Spearman calls — essentially free.

---

3. Perturbation Importance
--------------------------

Model-agnostic. For each feature, shuffle it ``n_repeats`` times, ask the
reducer to re-embed, and measure mean squared deviation from the reference
embedding. Aggregate across repeats and normalize so importances sum to 1.

.. code-block:: python

   from coco_pipe.dim_reduction.analysis import perturbation_importance

   scores = perturbation_importance(
       reducer.reducer,                # the underlying fitted reducer
       X,
       feature_names=feature_names,
       X_emb=X_emb,
       n_repeats=5,
       random_state=42,
   )
   # {"feat_07": 0.31, "feat_12": 0.18, ...}

Cost: ``n_features * n_repeats`` calls to ``transform``. For methods where
``transform`` is expensive (PHATE, TSNE — though TSNE doesn't even expose
``transform``), this is slow.

Caveats:

- **Requires ``transform``.** Non-parametric methods (``TSNE``, ``MDS``,
  ``PHATE``, ``Isomap``, ``LLE``, ``SpectralEmbedding``) do not implement it.
  Use ``correlation`` or fit a parametric proxy.
- **Correlated features dilute importance.** If two features are perfectly
  correlated, shuffling one barely changes the embedding — both will appear
  unimportant.
- **Stochastic.** Set ``random_state`` for reproducibility.

---

4. Gradient Saliency
--------------------

Encoder-based methods can compute ``∂‖z‖ / ∂x`` analytically. The backend
calls ``wrapper.get_pytorch_module()``, runs ``z = encoder(x)``, backpropagates
``z.sum()``, and averages absolute gradients across the sample axis.

.. code-block:: python

   from coco_pipe.dim_reduction.analysis import gradient_importance

   scores = gradient_importance(
       reducer.reducer,                # the underlying torch-backed reducer
       X,
       feature_names=feature_names,
   )
   # {"feat_07": 0.41, ...} for 1D inputs;
   # {"importance_matrix": ndarray} for higher-rank inputs.

Cost: one forward + one backward pass. The cheapest option *when applicable*.

Requirements:

- The reducer must expose ``get_pytorch_module()`` returning a module with an
  ``encoder`` submodule.
- Currently supported: ``IVIS``, ``ParametricUMAP``, ``TopologicalAE``.
- ``torch`` must be installed. Use the ``[topology]`` or ``[ivis]`` extras
  depending on the reducer.

---

5. Unified Backend: ``interpret_features``
------------------------------------------

For most workflows, use the high-level backend directly through the manager:

.. code-block:: python

   result = reducer.interpret(
       X,
       X_emb=embedding,
       analyses=["correlation", "perturbation"],
       feature_names=feature_names,
       n_repeats=5,
       random_state=42,
   )
   result["analysis"]   # dict keyed by analysis name
   result["records"]    # tidy long-form records

The same backend is also importable as a pure function:

.. code-block:: python

   from coco_pipe.dim_reduction.analysis import interpret_features

   payload = interpret_features(
       X,
       X_emb=embedding,
       model=reducer.reducer,
       analyses=["correlation", "perturbation"],
       feature_names=feature_names,
       method_name="UMAP",
       n_repeats=5,
       random_state=42,
   )

Outputs are cached on ``DimReduction.interpretation_`` and
``DimReduction.interpretation_records_`` so subsequent plotting and reporting
don't need to recompute.

---

6. Visualization
----------------

Tidy records flow straight into :func:`coco_pipe.viz.plot_feature_importance`
(in the dim-reduction module) and :func:`coco_pipe.viz.plot_feature_correlation_heatmap`.

.. code-block:: python

   from coco_pipe.viz import (
       plot_reduction_feature_importance,
       plot_feature_correlation_heatmap,
   )

   plot_reduction_feature_importance(
       reducer.interpretation_records_,
       analysis="perturbation",
       method=reducer.method,
       top_n=20,
   )
   plot_feature_correlation_heatmap(
       reducer.interpretation_["correlation"],
       method=reducer.method,
   )

---

7. Choosing a Backend
---------------------

- **First pass on any reducer**: ``correlation`` — cheap, always available.
- **Parametric reducer with a non-trivial cost per ``transform``**:
  ``perturbation`` — gives a true input → output sensitivity but is
  ``n_features``-times slower.
- **Encoder-based reducer**: ``gradient`` — by far the cheapest accurate
  measure when it applies.

Combine them: ``correlation`` for ranking, ``perturbation`` or ``gradient``
for the final published interpretation.
