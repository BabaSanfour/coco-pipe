.. _dim-reduction:

==========================================================
Dimensionality Reduction Module — Scientific User Guide
==========================================================

The ``coco_pipe.dim_reduction`` module provides a unified, reproducible framework
for dimensionality reduction across linear, manifold, neighbor-graph,
spatiotemporal, neural, and topological reducers. It is designed around a
**single manager class**, **explicit prep-and-score boundaries**, and a
**post-hoc comparison layer** so the same workflow scales from a one-off PCA
to multi-method comparison and trajectory analysis.

.. admonition:: Design Philosophy

   The manager (:class:`~coco_pipe.dim_reduction.DimReduction`) wraps one
   reducer and one workflow. It does **not** cache embeddings — every plotting,
   scoring, or interpretation step is fed an explicit embedding array. Scoring
   is delegated to a pure evaluator (``evaluation.core.evaluate_embedding``)
   so the same metric stack works for any embedding shape — including native
   ``(n_trajectories, n_times, n_dims)`` trajectory tensors.

.. rubric:: Key Features

- 15+ reducers across six families (linear, manifold, neighbor, spatiotemporal,
  neural, topology), exposed through a single
  :class:`~coco_pipe.dim_reduction.DimReduction` interface.
- Strict pydantic configs per reducer with early validation of unknown fields.
- A pure evaluator (``evaluation.core.evaluate_embedding``) shared by manager
  scoring and post-hoc comparison.
- Tidy long-form metric records (``method``, ``metric``, ``value``, ``scope``,
  ``scope_value``) feeding both reports and the
  :class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` ranker.
- Native trajectory metrics (speed, curvature, dispersion, separation) operating
  directly on ``(n_trajectories, n_times, n_dims)`` tensors.
- Feature interpretation backends (``correlation``, ``perturbation``,
  ``gradient``) decoupled from preservation scoring.
- Lazy import of heavy optional libraries (``torch``, ``umap``, ``dask``,
  ``pydmd``…) so base imports stay lightweight.

---

.. rubric:: Quickstart

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction

   reducer = DimReduction("UMAP", n_components=2, n_neighbors=15)
   embedding = reducer.fit_transform(X)
   scores = reducer.score(embedding, X=X, k_values=[5, 10])
   summary = reducer.get_summary()       # cached metrics + metadata + diagnostics

   # Visualize
   from coco_pipe.viz import plot_embedding, plot_shepard_diagram
   plot_embedding(embedding, labels=class_ids)
   plot_shepard_diagram(X, embedding)

Compare multiple reducers post-hoc:

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.dim_reduction.evaluation import MethodSelector

   reducers = [
       DimReduction("PCA", n_components=2),
       DimReduction("UMAP", n_components=2),
       DimReduction("Isomap", n_components=2),
   ]
   for r in reducers:
       emb = r.fit_transform(X)
       r.score(emb, X=X, k_values=[5, 10])

   selector = MethodSelector(reducers).collect()
   ranked = selector.rank_methods(
       selection_metric="trustworthiness",
       selection_k=10,
   )

---

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts
   configs
   core
   reducers

.. toctree::
   :maxdepth: 2
   :caption: Evaluation and Interpretation

   evaluation
   trajectories
   interpretation

.. toctree::
   :maxdepth: 2
   :caption: Visualization and Reports

   visualization

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/custom_reducers
   advanced/dependencies
   advanced/vision.md

.. toctree::
   :maxdepth: 2
   :caption: Worked Examples

   examples/basic_pca
   examples/compare_methods
   examples/trajectory_eeg
