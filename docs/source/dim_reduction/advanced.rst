.. _dim-reduction-advanced:

===============
Advanced Topics
===============

.. _dim-reduction-custom-reducers:

Custom Reducers
===============

``coco_pipe.dim_reduction`` is designed to be extended. To add a new reducer,
subclass :class:`~coco_pipe.dim_reduction.BaseReducer`, declare your
``capabilities``, implement ``fit`` / ``transform``, and (optionally) register
the reducer with the method registry. The new reducer then works through
:class:`~coco_pipe.dim_reduction.DimReduction` with no further wiring.

---

1. Minimal Reducer
------------------

.. code-block:: python

   from sklearn.decomposition import PCA
   from coco_pipe.dim_reduction import BaseReducer


   class CustomPCAReducer(BaseReducer):
       @property
       def capabilities(self):
           caps = super().capabilities
           caps.update({"is_linear": True, "has_components": True})
           return caps

       def fit(self, X, y=None):
           self.model = PCA(n_components=self.n_components, **self.params)
           self.model.fit(X)
           return self

       def transform(self, X):
           return self.model.transform(X)

That's enough to make the new reducer plug into the manager:

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction

   class CustomPCAManager(DimReduction):
       pass   # only needed if you also want a typed config — see below

   reducer = CustomPCAReducer(n_components=2)
   embedding = reducer.fit_transform(X)

The manager itself is happy to wrap any ``BaseReducer`` instance once
registered.

---

2. Capabilities Contract
------------------------

The manager and evaluator inspect ``BaseReducer.capabilities`` to know what
your reducer can do. Common flags:

================================  =================================================
``is_linear``                     Set ``True`` if the reducer is a linear
                                  projection.
``has_components``                Set ``True`` if ``get_components()`` returns a
                                  ``(n_components, n_features)`` array.
``has_loss_history``              Set ``True`` if ``quality_metadata_["loss_history"]``
                                  is populated after ``fit``.
``input_ndim``                    Expected input rank. Defaults to 2.
``input_layout``                  ``"samples_x_features"`` (default) or
                                  ``"snapshots_x_features"`` for DMD-like methods.
================================  =================================================

Declaring capabilities accurately lets the evaluator skip incompatible metrics
early and lets the viz layer enable / disable diagnostic plots.

---

3. Non-Standard Input Shapes
----------------------------

If your reducer consumes ``(n_features, n_snapshots)`` (like DMD) or higher-rank
tensors, declare it:

.. code-block:: python

   class CustomDMDReducer(BaseReducer):
       @property
       def capabilities(self):
           caps = super().capabilities
           caps.update({
               "is_linear": True,
               "input_ndim": 2,
               "input_layout": "snapshots_x_features",
           })
           return caps

The manager's ``_validate_input`` step will use these flags to fail fast on
shape mismatches.

---

4. Heavy Optional Dependencies
------------------------------

Keep heavy imports **inside** ``fit`` / ``transform`` so importing the reducer
module stays lightweight:

.. code-block:: python

   from coco_pipe.utils import import_optional_dependency


   class CustomTorchReducer(BaseReducer):
       def fit(self, X, y=None):
           torch = import_optional_dependency(
               lambda: __import__("torch"),
               feature="CustomTorchReducer",
               dependency="torch",
               install_hint="pip install coco-pipe[topology]",
           )
           # ... build and train your torch model ...
           return self

``import_optional_dependency`` raises a clear, actionable error if the
dependency is missing.

---

5. Registering a New Method
---------------------------

To use your reducer through ``DimReduction("MyMethod")`` rather than passing the
reducer instance directly, register it in the method registry. The registry is
intentionally module-local
(``coco_pipe.dim_reduction.config._METHOD_REGISTRY``), so a downstream package
should expose a helper that:

1. Imports the dotted path to the reducer class.
2. Adds ``"MyMethod": (module_path, "MyReducer")`` to the registry.

For a one-process script, prefer passing the reducer instance to
:class:`~coco_pipe.dim_reduction.DimReduction` directly.

---

6. Typed Config (Optional but Recommended)
------------------------------------------

Pair the reducer with a pydantic config so it benefits from strict validation:

.. code-block:: python

   from typing import Literal
   from pydantic import Field
   from coco_pipe.dim_reduction.config import (
       BaseReducerConfig,
       StochasticReducerConfig,
   )


   class CustomPCAConfig(BaseReducerConfig, StochasticReducerConfig):
       method: Literal["CustomPCA"] = "CustomPCA"
       whiten: bool = Field(False, description="Whiten projected components.")

Now you can do:

.. code-block:: python

   reducer = DimReduction(CustomPCAConfig(n_components=2, whiten=True))

---

7. Testing Checklist
--------------------

Before treating a custom reducer as production-ready, verify:

- ``fit`` returns ``self`` and is idempotent on repeated calls with the same
  data + parameters.
- ``transform`` works on **new** samples when the method supports it
  (``capabilities["has_transform"]``).
- ``capabilities`` reflects reality — particularly ``input_ndim`` and
  ``has_components``.
- ``DimReduction(reducer).score(embedding, X=X)`` produces a non-empty
  ``metric_records_`` and no metrics report ``"reason: incompatible"``.
- If you registered a method name, ``DimReduction("MyMethod")`` round-trips
  through your config.

.. _dim-reduction-dependencies:

Optional Dependencies
=====================

The dim-reduction module is built so that the **base install does not require
any heavy ML libraries**. Reducers that need ``torch``, ``umap-learn``,
``pydmd``, etc. import them lazily inside ``fit`` / ``transform``. Optional
extras unlock specific reducer families.

---

1. What Stays Lightweight
-------------------------

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
------------------------------

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
---------------------------

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
----------------------------------------

Optional reducers are imported the first time you instantiate them. If the
underlying library is missing, you'll see a structured error:

.. code-block:: text

   ImportError: PHATE requires the 'phate' package.
   Install with: pip install coco-pipe[dim-red]

The same message applies to interpretation backends:
:func:`gradient_importance` raises a similar error when ``torch`` is missing.

---

5. ``import_optional_dependency``
---------------------------------

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
---------------------------------------

PaCMAP supports multiple NN backends. The ``[neighbor]`` and ``[dim-red]``
extras include ``faiss-cpu``, so PaCMAP's default
``nn_backend="faiss"`` works out of the box on supported platforms. To force
a different backend:

.. code-block:: python

   reducer = DimReduction("Pacmap", n_components=2, nn_backend="annoy")

Recent PaCMAP versions accept ``"faiss"``, ``"annoy"``, and ``"voyager"``.

---

7. Recommended Install Profiles
-------------------------------

============================  =========================================================
Profile                       Install
============================  =========================================================
Quick exploration             ``pip install coco-pipe``
Standard scientific work      ``pip install coco-pipe[dim-red,eeg]``
Distributed / out-of-core     ``pip install coco-pipe[dim-red,dask]``
Deep learning / parametric    ``pip install coco-pipe[dim-red,parametric-umap,topology]``
Everything                    ``pip install coco-pipe[dim-red,dask,parametric-umap,ivis,topology,spatiotemporal,eeg]``
============================  =========================================================
