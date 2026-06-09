.. _dim-reduction-interpretation:

============================
Feature Interpretation
============================

Interpretation answers: *which input features appear to drive each embedding
axis*? This is independent of preservation scoring (covered in
:ref:`dim-reduction-evaluation`).

Three backends with different cost / reducer-class tradeoffs are available
through :meth:`coco_pipe.dim_reduction.DimReduction.interpret` and the pure
backend :func:`coco_pipe.dim_reduction.analysis.interpret_features`.

---

1. Backends at a Glance
==========================

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
===========================

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
=============================

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
======================

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
=============================================

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
==================

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
=======================

- **First pass on any reducer**: ``correlation`` — cheap, always available.
- **Parametric reducer with a non-trivial cost per ``transform``**:
  ``perturbation`` — gives a true input → output sensitivity but is
  ``n_features``-times slower.
- **Encoder-based reducer**: ``gradient`` — by far the cheapest accurate
  measure when it applies.

Combine them: ``correlation`` for ranking, ``perturbation`` or ``gradient``
for the final published interpretation.
