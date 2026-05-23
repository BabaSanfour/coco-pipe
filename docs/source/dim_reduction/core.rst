.. _dim-reduction-core:

==========================
``DimReduction`` Manager
==========================

:class:`~coco_pipe.dim_reduction.DimReduction` is the single public manager
class. It wraps one reducer, drives ``fit`` / ``transform``, delegates scoring
to the pure evaluator, runs interpretation analyses, and exposes a consolidated
``get_summary()`` payload.

The manager does **not** cache embeddings. Every method that needs an embedding
takes it as an explicit argument.

---

1. Construction
================

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.dim_reduction.config import UMAPConfig

   # By method name
   reducer = DimReduction("UMAP", n_components=2, n_neighbors=15)

   # With a typed config
   reducer = DimReduction(UMAPConfig(n_components=2, n_neighbors=15))

Resolution order for reducer kwargs (later overrides earlier):

1. Fields from the config (if a :class:`BaseReducerConfig` is passed).
2. The ``params=`` dictionary.
3. Additional ``**kwargs``.

---

2. Lifecycle Methods
======================

.. code-block:: python

   reducer.fit(X)                        # fit only
   embedding = reducer.transform(X)      # transform an already-fitted reducer
   embedding = reducer.fit_transform(X)  # combined

All three return ``self`` / a NumPy array as appropriate. They reset cached
``metrics_``, ``diagnostics_``, ``metric_records_``, and ``interpretation_``
so stale evaluations cannot leak across runs.

``DimReduction.get_components()`` returns reducer components for linear
methods (PCA, IncrementalPCA, DaskPCA, DaskTruncatedSVD). It raises for
methods that do not expose components.

---

3. Scoring
============

.. code-block:: python

   scores = reducer.score(
       embedding,                        # required, explicit
       X=X,                              # required for 2D metrics
       n_neighbors=5,                    # single-score neighborhood
       metrics=["trustworthiness", "continuity"],
       k_values=[5, 10, 20],             # multi-scale sweep
       labels=labels,                    # used for trajectory_separation
       groups=groups,                    # for supervised separation metrics
       times=times,                      # for trajectory AUC
       separation_method="centroid",
   )

Returns a dict with three keys:

==============  =========================================================
``metrics``     Scalar metric summary (also cached on ``metrics_``).
``metadata``    Scalar descriptive metadata (also cached on
                ``quality_metadata_``).
``diagnostics`` Array / structured diagnostics (e.g., ``shepard_distances``,
                ``coranking_matrix``, trajectory timecourses) — also cached on
                ``diagnostics_``.
==============  =========================================================

Tidy long-form records are cached on ``metric_records_`` for downstream
ranking and reporting.

.. admonition:: 2D vs. 3D shape routing

   The evaluator chooses standard or trajectory metrics from ``embedding.shape``,
   not from the reducer name. Pass a ``(n_samples, n_components)`` embedding
   for standard metrics; pass a ``(n_trajectories, n_times, n_dims)`` tensor
   for trajectory metrics. See :ref:`dim-reduction-trajectories`.

---

4. Interpretation
===================

.. code-block:: python

   result = reducer.interpret(
       X,
       X_emb=embedding,
       analyses=["correlation", "perturbation", "gradient"],
       feature_names=feature_names,
       n_repeats=5,
       random_state=42,
   )

Returns ``{"analysis": ..., "records": [...]}``. The ``analysis`` payload is
keyed by analysis name; ``records`` is tidy long-form ready for plotting and
reports. Both are cached on ``interpretation_`` and ``interpretation_records_``.

Supported analyses:

- ``"correlation"`` — Spearman correlations between input features and embedding
  axes. Works for any reducer.
- ``"perturbation"`` — model-agnostic feature importance from per-feature
  shuffling. Requires a fitted reducer with ``transform``.
- ``"gradient"`` — encoder saliency for supported torch-based reducers
  (``IVIS``, ``ParametricUMAP``, ``TopologicalAE``).

See :ref:`dim-reduction-interpretation` for the math and reducer requirements.

---

5. Inspecting Cached State
============================

.. code-block:: python

   reducer.get_metrics()           # scalar metrics_
   reducer.get_quality_metadata()  # metadata from reducer + evaluator
   reducer.get_diagnostics()       # full diagnostics_ payload
   reducer.get_summary()           # combined: metrics + metadata + diagnostics
                                   # + metric_records + interpretation + capabilities

``get_summary()`` is the canonical input for :meth:`coco_pipe.report.Report.add_reduction`
and is JSON-serializable. **It deliberately does not carry an embedding** —
pass embeddings explicitly to plotting and reporting paths that need them.

---

6. Capabilities
=================

.. code-block:: python

   caps = reducer.capabilities
   # {'is_linear': True, 'has_components': True, 'has_loss_history': False, ...}

The manager exposes the reducer's capability dict directly. Capabilities are
used by:

- the evaluator to skip metrics the reducer cannot support,
- :func:`coco_pipe.viz.dim_reduction.plot_loss_history` to detect available
  loss curves,
- :class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` for capability-aware
  filtering in comparison tables.

---

7. Persistence
================

.. code-block:: python

   reducer.save("models/umap.pkl")
   loaded = DimReduction.load("models/umap.pkl", method="UMAP")

``save`` writes the fitted reducer with its kwargs and method name; ``load``
re-instantiates the manager and restores the reducer's fitted state. Cached
evaluation payloads are **not** persisted — re-run ``score()`` on the
embedding you have.

---

8. End-to-End Skeleton
========================

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.viz import plot_embedding, plot_metrics

   reducer = DimReduction("UMAP", n_components=2, n_neighbors=15, random_state=42)
   embedding = reducer.fit_transform(X)
   reducer.score(embedding, X=X, k_values=[5, 10, 20])
   reducer.interpret(X, X_emb=embedding, analyses=["correlation"],
                     feature_names=feature_names)
   summary = reducer.get_summary()

   plot_embedding(embedding, labels=class_ids)
   plot_metrics(reducer)                    # accepts the manager directly

For a multi-reducer comparison, see
:class:`~coco_pipe.dim_reduction.evaluation.MethodSelector` in
:ref:`dim-reduction-evaluation`.
