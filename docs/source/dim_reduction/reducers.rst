.. _dim-reduction-reducers:

============================
Reducer Catalog
============================

This page lists every reducer in the registry, what it does, when to use it,
and which optional install unlocks it. All reducers are reachable through
:class:`~coco_pipe.dim_reduction.DimReduction` using the canonical method
name in the leftmost column.

---

1. Registry at a Glance
=========================

==================  ==================  ===========================================  ========================
Family              Method              When to use it                                Install
==================  ==================  ===========================================  ========================
**Linear**          ``PCA``             Default baseline; fast, deterministic,        core
                                        components inspectable.
                    ``IncrementalPCA``  PCA when data does not fit in memory.         core
                    ``DaskPCA``         PCA on Dask arrays / distributed data.        ``[dask]``
                    ``DaskTruncatedSVD``  Truncated SVD on Dask arrays.               ``[dask]``
**Manifold**        ``Isomap``          Global geodesic distances; smooth manifolds.  core
                    ``LLE``             Local linear reconstruction.                  core
                    ``MDS``             Pairwise distance preservation.               core
                    ``SpectralEmbedding``  Graph-Laplacian embedding for clusters.    core
**Neighbor graph**  ``TSNE``            Cluster visualization at small/medium scale.  core
                    ``UMAP``            Faster, more global than t-SNE.               ``[dim-red]`` or ``[neighbor]``
                    ``Pacmap``          Better global structure than UMAP at scale.   ``[dim-red]`` or ``[neighbor]``
                    ``Trimap``          Triplet-based UMAP/PaCMAP alternative.        ``[dim-red]`` or ``[neighbor]``
                    ``PHATE``           Diffusion-based, good for continuous          ``[dim-red]`` or ``[neighbor]``
                                        trajectories.
                    ``ParametricUMAP``  UMAP backed by a learnable encoder.           ``[parametric-umap]``
**Spatiotemporal**  ``DMD``             Dynamic Mode Decomposition for time series.   ``[spatiotemporal]``
                    ``TRCA``            Task-Related Component Analysis; SSVEP /      ``[spatiotemporal]``
                                        evoked EEG.
**Neural**          ``IVIS``            Siamese-network parametric reducer.           ``[ivis]``
                    ``TopologicalAE``   Autoencoder with topological regularization.  ``[topology]``
==================  ==================  ===========================================  ========================

The full reducer registry is exposed as
:data:`coco_pipe.dim_reduction.METHODS`. Optional methods are imported lazily;
``import coco_pipe.dim_reduction`` does not pull ``torch`` or ``umap-learn``.

---

2. Linear Reducers
===================

2.1 PCA
---------

- Components: ``get_components()`` returns ``(n_components, n_features)``.
- Capabilities: ``is_linear=True``, ``has_components=True``.
- Use when you need a deterministic, fast baseline or a sanity check before
  trying non-linear methods.

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction

   reducer = DimReduction("PCA", n_components=10, whiten=False)
   embedding = reducer.fit_transform(X)
   loadings = reducer.get_components()

2.2 IncrementalPCA
--------------------

- Out-of-core; fit in chunks via ``partial_fit`` (sklearn's machinery).
- Same component contract as PCA.

2.3 DaskPCA / DaskTruncatedSVD
--------------------------------

- For Dask arrays; useful with very large feature matrices.
- Requires the ``[dask]`` extra.

---

3. Manifold Reducers
======================

3.1 Isomap
------------

Global manifold reducer based on geodesic distances over a nearest-neighbor
graph. Sensitive to ``n_neighbors``.

.. code-block:: python

   reducer = DimReduction("Isomap", n_components=2, n_neighbors=10)

3.2 LLE
---------

Local linear reconstruction. The pydantic ``method`` slot is reserved, so the
sklearn ``method`` parameter is exposed as ``lle_method``:

.. code-block:: python

   reducer = DimReduction("LLE", n_components=2, n_neighbors=10, lle_method="hessian")

3.3 MDS
---------

Preserves pairwise distances. ``metric=False`` selects non-metric MDS.

3.4 SpectralEmbedding
-----------------------

Graph-Laplacian eigenmaps. Good for cluster-structured data; sensitive to
``affinity``.

---

4. Neighbor-Graph Reducers
============================

4.1 t-SNE
-----------

Best for cluster visualization at small/medium ``n``. Sensitive to
``perplexity`` (rule of thumb: 5–50). Stochastic; pass ``random_state`` for
reproducibility.

.. code-block:: python

   reducer = DimReduction("TSNE", n_components=2, perplexity=30, init="pca")

4.2 UMAP
----------

Faster than t-SNE, better global structure. ``n_neighbors`` controls local
vs. global tradeoff; ``min_dist`` controls cluster compactness.

.. code-block:: python

   reducer = DimReduction("UMAP", n_components=2, n_neighbors=15, min_dist=0.1)

4.3 Pacmap / Trimap
---------------------

PaCMAP and TriMap optimize triplet relationships; often preserve global
structure better than UMAP on large datasets. PaCMAP defaults
``nn_backend="faiss"``; the ``[neighbor]`` and ``[dim-red]`` extras include
``faiss-cpu``.

4.4 PHATE
-----------

Diffusion-based reducer designed for continuous trajectories (development
gradients, sequential states). Preserves both local and global structure.

.. code-block:: python

   reducer = DimReduction("PHATE", n_components=2, knn=5, decay=40)

4.5 ParametricUMAP
--------------------

UMAP backed by a trainable encoder network. Allows ``transform`` on out-of-sample
points without re-fitting; requires the ``[parametric-umap]`` extra.

---

5. Spatiotemporal Reducers
============================

5.1 DMD
---------

Dynamic Mode Decomposition. Takes a sequence of snapshots and extracts
coherent spatiotemporal modes. Requires the ``[spatiotemporal]`` extra
(``pydmd``). Input layout is ``(n_snapshots, n_features)``; set
``force_transpose=True`` for the alternative layout.

5.2 TRCA
----------

Task-Related Component Analysis: maximizes the inter-trial reproducibility
of evoked responses (e.g., SSVEP). Requires the sampling frequency in Hz.
Optional filterbank for multi-band decomposition.

---

6. Neural and Topological Reducers
====================================

6.1 IVIS
----------

Siamese-network parametric reducer. Supports semi-supervised mode via
``supervise_metric``. Requires the ``[ivis]`` extra.

6.2 TopologicalAE
-------------------

Autoencoder with a topological-loss regularizer. Encoder size, regularization
weight, batch size, learning rate, and epochs are exposed via the config.
Requires the ``[topology]`` extra (``torch`` + ``torch-topological``).

.. code-block:: python

   reducer = DimReduction(
       "TopologicalAE",
       n_components=2,
       hidden_dims=[128, 64],
       lam=0.5,
       lr=1e-3,
       epochs=100,
       device="auto",
   )

---

7. Choosing a Reducer
=======================

Pragmatic decision tree:

- **First pass**: ``PCA`` — always run it. It tells you how much variance
  lives in low-dim and gives a baseline ``trustworthiness`` to beat.
- **Cluster visualization**: ``UMAP`` (fast, robust). Fall back to ``TSNE``
  for very small datasets where UMAP's defaults over-spread.
- **Continuous trajectories / development data**: ``PHATE`` (diffusion).
  ``UMAP`` with low ``min_dist`` is a quick alternative.
- **Trajectory dynamics over time**: ``DMD`` (or evaluate any reducer with
  trajectory metrics from :ref:`dim-reduction-trajectories`).
- **Large global structure**: ``Pacmap`` / ``Trimap`` (triplet methods).
- **Need ``transform`` on new samples**: ``PCA``, ``IncrementalPCA``,
  ``ParametricUMAP``, ``IVIS``, ``TopologicalAE`` — but **not** ``TSNE``,
  ``UMAP`` (non-parametric), ``MDS``, ``PHATE``.

Compare them objectively with :ref:`dim-reduction-evaluation`.

---

8. Capabilities Cheat-Sheet
=============================

============================  ============  ============  ===============  ==========
Reducer                       ``is_linear``  ``has_components``  ``has_loss_history``  ``transform``
============================  ============  ============  ===============  ==========
PCA / IncrementalPCA / Dask\*  ✓             ✓              ✗                ✓
Isomap / LLE / MDS /           ✗             ✗              ✗                ✗ (non-parametric)
SpectralEmbedding
TSNE                           ✗             ✗              ✗                ✗
UMAP                           ✗             ✗              ✓                ✓ (post-fit)
ParametricUMAP                 ✗             ✗              ✓                ✓
PaCMAP / TriMap                ✗             ✗              ✓                ✓ (post-fit)
PHATE                          ✗             ✗              ✗                ✗
DMD                            ✓ (modes)     ✓ (modes)      ✗                ✓
TRCA                           ✓             ✓              ✗                ✓
IVIS                           ✗             ✗              ✓                ✓
TopologicalAE                  ✗             ✗              ✓                ✓
============================  ============  ============  ===============  ==========

Always check ``reducer.capabilities`` at runtime — third-party libraries
occasionally change what they expose.
