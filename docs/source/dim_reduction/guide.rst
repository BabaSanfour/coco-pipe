.. _dim-reduction-guide:

===============================
Core Workflow and Configuration
===============================

.. _dim-reduction-core:

``DimReduction`` Manager
========================

:class:`~coco_pipe.dim_reduction.DimReduction` is the single public manager
class. It wraps one reducer, drives ``fit`` / ``transform``, delegates scoring
to the pure evaluator, runs interpretation analyses, and exposes a consolidated
``get_summary()`` payload.

The manager does **not** cache embeddings. Every method that needs an embedding
takes it as an explicit argument.

---

1. Construction
---------------

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
--------------------

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
----------

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

.. list-table::
   :header-rows: 0
   :widths: 20 80

   * - ``metrics``
     - Scalar metric summary (also cached on ``metrics_``).
   * - ``metadata``
     - Scalar descriptive metadata (also cached on ``quality_metadata_``).
   * - ``diagnostics``
     - Array / structured diagnostics (e.g., ``shepard_distances``,
       ``coranking_matrix``, trajectory timecourses) — also cached on
       ``diagnostics_``.

Tidy long-form records are cached on ``metric_records_`` for downstream
ranking and reporting.

.. admonition:: 2D vs. 3D shape routing

   The evaluator chooses standard or trajectory metrics from ``embedding.shape``,
   not from the reducer name. Pass a ``(n_samples, n_components)`` embedding
   for standard metrics; pass a ``(n_trajectories, n_times, n_dims)`` tensor
   for trajectory metrics. See :ref:`dim-reduction-trajectories`.

---

4. Interpretation
-----------------

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
--------------------------

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
---------------

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
--------------

.. code-block:: python

   reducer.save("models/umap.pkl")
   loaded = DimReduction.load("models/umap.pkl", method="UMAP")

``save`` writes the fitted reducer with its kwargs and method name; ``load``
re-instantiates the manager and restores the reducer's fitted state. Cached
evaluation payloads are **not** persisted — re-run ``score()`` on the
embedding you have.

---

8. End-to-End Skeleton
----------------------

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

.. _dim-reduction-configs:

Configuration Reference
=======================

Every reducer in ``coco_pipe.dim_reduction`` accepts either keyword arguments
to :class:`~coco_pipe.dim_reduction.DimReduction` *or* a typed pydantic config.
Configs validate field names, types, and ranges at parse time, so typos and
incompatible options fail before any data is touched.

---

1. The Two Equivalent Construction Styles
-----------------------------------------

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.dim_reduction.config import UMAPConfig

   # Keyword-style (string method name + kwargs)
   reducer = DimReduction("UMAP", n_components=2, n_neighbors=15, min_dist=0.1)

   # Config-style (typed pydantic model)
   reducer = DimReduction(UMAPConfig(n_components=2, n_neighbors=15, min_dist=0.1))

Prefer the config style when:

- Reducer parameters come from a YAML or JSON file (pass via ``**config``).
- You want a single object that can be serialized, logged, or reused across
  experiments.
- You want strict validation of every field at construction time.

Use the keyword style for one-off scripts and exploration.

---

2. Base Config
--------------

.. code-block:: python

   from coco_pipe.dim_reduction.config import BaseReducerConfig

All reducer configs inherit from :class:`BaseReducerConfig`. Common fields:

- ``method`` — canonical reducer name (``Literal``, immutable per subclass).
- ``n_components`` — target dimensionality, default 2.

A second mixin, :class:`StochasticReducerConfig`, adds a ``random_state``
field (default 42) for reducers that have a seed.

---

3. Reducer Configs
------------------

The full set of typed configs maps 1:1 with the registry in
:data:`coco_pipe.dim_reduction.config.METHODS`.

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Family
     - Config class
     - Key fields
   * - **Linear**
     - ``PCAConfig``
     - ``whiten``, ``svd_solver``
   * -
     - ``IncrementalPCAConfig``
     - ``batch_size``, ``whiten``
   * -
     - ``DaskPCAConfig``
     - ``svd_solver``
   * -
     - ``DaskTruncatedSVDConfig``
     - ``algorithm``
   * - **Manifold**
     - ``IsomapConfig``
     - ``n_neighbors``, ``metric``, ``p``
   * -
     - ``LLEConfig``
     - ``n_neighbors``, ``lle_method``
   * -
     - ``MDSConfig``
     - ``metric``, ``n_init``, ``dissimilarity``
   * -
     - ``SpectralEmbeddingConfig``
     - ``affinity``, ``gamma``
   * - **Neighbor**
     - ``TSNEConfig``
     - ``perplexity``, ``early_exaggeration``, ``learning_rate``, ``max_iter``, ``init``
   * -
     - ``UMAPConfig``
     - ``n_neighbors``, ``min_dist``, ``metric``, ``spread``
   * -
     - ``ParametricUMAPConfig``
     - ``n_neighbors``, ``min_dist``, ``batch_size``
   * -
     - ``PacmapConfig``
     - ``n_neighbors``, ``MN_ratio``, ``FP_ratio``, ``nn_backend``, ``init``
   * -
     - ``TrimapConfig``
     - ``n_inliers``, ``n_outliers``, ``n_random``
   * -
     - ``PHATEConfig``
     - ``knn``, ``decay``, ``t``
   * - **Spatiotemporal**
     - ``DMDConfig``
     - ``tlsq_rank``, ``exact``, ``opt``
   * -
     - ``TRCAConfig``
     - ``sfreq``, ``filterbank``
   * - **Neural / Topology**
     - ``IVISConfig``
     - ``k``, ``model``, ``n_epochs_without_progress``
   * -
     - ``TopologicalAEConfig``
     - ``hidden_dims``, ``lam``, ``lr``, ``batch_size``, ``epochs``, ``device``

See :ref:`dim-reduction-reducers` for what each reducer does and when to use
it.

3.1 Example: full UMAP config
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from coco_pipe.dim_reduction.config import UMAPConfig

   config = UMAPConfig(
       n_components=2,
       n_neighbors=15,
       min_dist=0.1,
       metric="euclidean",
       spread=1.0,
       random_state=42,
   )

3.2 Example: LLE name renaming
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pydantic ``method`` field is reserved for reducer selection, so LLE's
sklearn parameter ``method`` is exposed as ``lle_method`` and renamed back
via :meth:`LLEConfig.to_reducer_kwargs`:

.. code-block:: python

   from coco_pipe.dim_reduction.config import LLEConfig

   config = LLEConfig(n_components=2, n_neighbors=10, lle_method="hessian")
   # to_reducer_kwargs() yields {"n_components": 2, "n_neighbors": 10, "method": "hessian"}

---

4. Evaluation Config
--------------------

.. code-block:: python

   from coco_pipe.dim_reduction.config import EvaluationConfig

   eval_config = EvaluationConfig(
       metrics=["trustworthiness", "continuity", "lcmc"],
       k_range=[5, 10, 20, 50, 100],
       selection_metric="trustworthiness",
       selection_k=10,
       tie_breakers=["continuity"],
       separation_method="centroid",
   )

Fields:

================================  ==================================================
``metrics``                       Metric families to compute. Must be canonical
                                  evaluator names (see :ref:`dim-reduction-evaluation`).
                                  No duplicates; at least one entry.
``k_range``                       Neighborhood sizes for multi-scale metrics
                                  (``trustworthiness``, ``continuity``, ``lcmc``,
                                  ``mrre_*``). Positive integers, no duplicates.
``selection_metric``              Primary ranking metric. Must be in
                                  ``_VALID_RANKING_METRICS`` *and* in
                                  ``metrics``.
``selection_k``                   Neighborhood size used when ranking a
                                  ``k``-scoped metric.
``tie_breakers``                  Ordered list of additional ranking metrics.
                                  Each must also be present in ``metrics``.
``separation_method``             Separation definition for trajectory
                                  separation: ``"centroid"`` (default),
                                  ``"within_between_ratio"``, ``"mahalanobis"``,
                                  ``"distributional"``, ``"margin"``.
================================  ==================================================

.. admonition:: Early validation pays off

   ``EvaluationConfig`` rejects unknown metric names, duplicate entries,
   invalid separation methods, and ranking metrics that are not in ``metrics``.
   You won't run a 10-minute scoring loop only to find the ranker has nothing
   to rank with.

---

5. Configs from YAML / JSON
---------------------------

All configs are standard pydantic models, so loading from a serialized form is
direct:

.. code-block:: python

   import yaml
   from coco_pipe.dim_reduction.config import UMAPConfig

   with open("umap.yaml") as f:
       data = yaml.safe_load(f)
   config = UMAPConfig(**data)             # validation runs here

   reducer = DimReduction(config)
