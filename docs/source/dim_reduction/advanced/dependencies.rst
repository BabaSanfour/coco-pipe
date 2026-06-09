.. _dim-reduction-dependencies:

==========================
Optional Dependencies
==========================

The dim-reduction module is built so that the **base install does not require
any heavy ML libraries**. Reducers that need ``torch``, ``umap-learn``,
``pydmd``, etc. import them lazily inside ``fit`` / ``transform``. Optional
extras unlock specific reducer families.

---

1. What Stays Lightweight
===========================

``import coco_pipe.dim_reduction`` only pulls in:

- ``numpy``, ``scipy``, ``pandas``, ``scikit-learn``
- ``pydantic`` (for configs)

The following submodules are also import-light:

- ``coco_pipe.io``
- ``coco_pipe.report``
- ``coco_pipe.viz`` (matplotlib + plotly; plotly is imported lazily)

So a script that uses only PCA / Isomap / TSNE / MDS / LLE / SpectralEmbedding
doesn't need any optional installs.

---

2. Extras and What They Unlock
================================

==============================  ======================================================
Extra                           Unlocks
==============================  ======================================================
``[dim-red]``                   Umbrella extra: ``UMAP``, ``Pacmap``, ``Trimap``,
                                ``PHATE``, plus ``faiss-cpu``.
``[neighbor]``                  Same as ``[dim-red]`` for the neighbor-graph family.
``[dask]``                      ``DaskPCA``, ``DaskTruncatedSVD``.
``[parametric-umap]``           ``ParametricUMAP``.
``[ivis]``                      ``IVIS``.
``[topology]``                  ``TopologicalAE`` (pulls ``torch`` +
                                ``torch-topological``).
``[spatiotemporal]``            ``DMD`` (``pydmd``), ``TRCA``.
``[eeg]``                       MNE-Python; used by :func:`coco_pipe.viz.plot_topomap`
                                and EEG-specific examples.
==============================  ======================================================

Install with pip:

.. code-block:: bash

   pip install coco-pipe[dim-red]
   pip install coco-pipe[dask,topology]
   pip install coco-pipe[neighbor,parametric-umap,ivis]

---

3. Choosing What to Install
=============================

- **You only need PCA / Isomap / TSNE / MDS / LLE / SpectralEmbedding** →
  base install. No extras needed.
- **You need fast non-linear dimensionality reduction** → ``[dim-red]``.
- **Your data is on Dask arrays** → ``[dask]``.
- **You want a parametric reducer for transferring to new samples** →
  ``[parametric-umap]`` (preferred) or ``[ivis]``.
- **You want topology-regularized embeddings** → ``[topology]``.
- **You're working with EEG topographic maps** → add ``[eeg]`` for MNE.

For exploration across many methods, ``[dim-red]`` is the most common starting
point.

---

4. Failure Mode When an Extra is Missing
==========================================

Optional reducers are imported the first time you instantiate them. If the
underlying library is missing, you'll see a structured error:

.. code-block:: text

   ImportError: PHATE requires the 'phate' package.
   Install with: pip install coco-pipe[dim-red]

The same message applies to interpretation backends:
:func:`gradient_importance` raises a similar error when ``torch`` is missing.

---

5. ``import_optional_dependency``
===================================

The helper used internally is exposed for custom reducers:

.. code-block:: python

   from coco_pipe.utils import import_optional_dependency

   torch = import_optional_dependency(
       lambda: __import__("torch"),
       feature="my_custom_reducer",
       dependency="torch",
       install_hint="pip install coco-pipe[topology]",
   )

It centralizes the "raise a clear, actionable error when optional dependency
is missing" pattern. See :ref:`dim-reduction-custom-reducers` for usage in
custom reducers.

---

6. PaCMAP and Nearest-Neighbor Backends
=========================================

PaCMAP supports multiple NN backends. The ``[neighbor]`` and ``[dim-red]``
extras include ``faiss-cpu``, so PaCMAP's default
``nn_backend="faiss"`` works out of the box on supported platforms. To force
a different backend:

.. code-block:: python

   reducer = DimReduction("Pacmap", n_components=2, nn_backend="annoy")

Recent PaCMAP versions accept ``"faiss"``, ``"annoy"``, and ``"voyager"``.

---

7. Recommended Install Profiles
=================================

============================  =========================================================
Profile                       Install
============================  =========================================================
Quick exploration             ``pip install coco-pipe``
Standard scientific work      ``pip install coco-pipe[dim-red,eeg]``
Distributed / out-of-core     ``pip install coco-pipe[dim-red,dask]``
Deep learning / parametric    ``pip install coco-pipe[dim-red,parametric-umap,topology]``
Everything                    ``pip install coco-pipe[dim-red,dask,parametric-umap,ivis,topology,spatiotemporal,eeg]``
============================  =========================================================
