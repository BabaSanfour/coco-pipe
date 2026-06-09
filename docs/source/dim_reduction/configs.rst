.. _dim-reduction-configs:

==========================================
Configuration Reference
==========================================

Every reducer in ``coco_pipe.dim_reduction`` accepts either keyword arguments
to :class:`~coco_pipe.dim_reduction.DimReduction` *or* a typed pydantic config.
Configs validate field names, types, and ranges at parse time, so typos and
incompatible options fail before any data is touched.

---

1. The Two Equivalent Construction Styles
==========================================

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
================

.. code-block:: python

   from coco_pipe.dim_reduction.config import BaseReducerConfig

All reducer configs inherit from :class:`BaseReducerConfig`. Common fields:

- ``method`` — canonical reducer name (``Literal``, immutable per subclass).
- ``n_components`` — target dimensionality, default 2.

A second mixin, :class:`StochasticReducerConfig`, adds a ``random_state``
field (default 42) for reducers that have a seed.

---

3. Reducer Configs
====================

The full set of typed configs maps 1:1 with the registry in
:data:`coco_pipe.dim_reduction.config.METHODS`.

============================  ========================  ==================================================
Family                        Config class              Key fields
============================  ========================  ==================================================
**Linear**                    ``PCAConfig``             ``whiten``, ``svd_solver``
                              ``IncrementalPCAConfig``  ``batch_size``, ``whiten``
                              ``DaskPCAConfig``         ``svd_solver``
                              ``DaskTruncatedSVDConfig``  ``algorithm``
**Manifold**                  ``IsomapConfig``          ``n_neighbors``, ``metric``, ``p``
                              ``LLEConfig``             ``n_neighbors``, ``lle_method``
                              ``MDSConfig``             ``metric``, ``n_init``, ``dissimilarity``
                              ``SpectralEmbeddingConfig``  ``affinity``, ``gamma``
**Neighbor**                  ``TSNEConfig``            ``perplexity``, ``early_exaggeration``,
                                                        ``learning_rate``, ``max_iter``, ``init``
                              ``UMAPConfig``            ``n_neighbors``, ``min_dist``, ``metric``,
                                                        ``spread``
                              ``ParametricUMAPConfig``  ``n_neighbors``, ``min_dist``, ``batch_size``
                              ``PacmapConfig``          ``n_neighbors``, ``MN_ratio``, ``FP_ratio``,
                                                        ``nn_backend``, ``init``
                              ``TrimapConfig``          ``n_inliers``, ``n_outliers``, ``n_random``
                              ``PHATEConfig``           ``knn``, ``decay``, ``t``
**Spatiotemporal**            ``DMDConfig``             ``tlsq_rank``, ``exact``, ``opt``
                              ``TRCAConfig``            ``sfreq``, ``filterbank``
**Neural / Topology**         ``IVISConfig``            ``k``, ``model``, ``n_epochs_without_progress``
                              ``TopologicalAEConfig``   ``hidden_dims``, ``lam``, ``lr``, ``batch_size``,
                                                        ``epochs``, ``device``
============================  ========================  ==================================================

See :ref:`dim-reduction-reducers` for what each reducer does and when to use
it.

3.1 Example: full UMAP config
-------------------------------

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
--------------------------------

The pydantic ``method`` field is reserved for reducer selection, so LLE's
sklearn parameter ``method`` is exposed as ``lle_method`` and renamed back
via :meth:`LLEConfig.to_reducer_kwargs`:

.. code-block:: python

   from coco_pipe.dim_reduction.config import LLEConfig

   config = LLEConfig(n_components=2, n_neighbors=10, lle_method="hessian")
   # to_reducer_kwargs() yields {"n_components": 2, "n_neighbors": 10, "method": "hessian"}

---

4. Evaluation Config
======================

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
=============================

All configs are standard pydantic models, so loading from a serialized form is
direct:

.. code-block:: python

   import yaml
   from coco_pipe.dim_reduction.config import UMAPConfig

   with open("umap.yaml") as f:
       data = yaml.safe_load(f)
   config = UMAPConfig(**data)             # validation runs here

   reducer = DimReduction(config)
