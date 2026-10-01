===============
Getting Started
===============

This page gets you from a fresh environment to your first ``coco-pipe``
workflow. For the conceptual background, see the :doc:`user_guide`; for the
full interface, see the :doc:`api_reference`.

Installation
============

``coco-pipe`` targets Python 3.10+. Install the core package from PyPI:

.. code-block:: bash

   pip install coco-pipe

The base install stays lightweight. Heavier capabilities ship as optional
extras so you only pull what you need:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Extra
     - Adds
   * - ``[descriptors]``
     - Signal feature extraction (spectral, complexity, parametric).
   * - ``[dim-red]`` / ``[neighbor]``
     - UMAP, PaCMAP, TriMap, PHATE and the FAISS backend.
   * - ``[foundation]``
     - Foundation-model decoding (Hugging Face / braindecode backbones).
   * - ``[dask]``
     - Out-of-core PCA / TruncatedSVD on Dask arrays.
   * - ``[eeg]``
     - MNE-backed M/EEG loading and topographic plotting.

Install one or more extras together with the core package:

.. code-block:: bash

   pip install "coco-pipe[descriptors,dim-red,eeg]"

The ``DataContainer``
=====================

Every module reads and writes a single labelled structure,
:class:`~coco_pipe.io.DataContainer`, so feature extraction, reduction,
decoding, visualization, and reporting all compose without glue code.

.. code-block:: python

   from coco_pipe.io import load_data

   container = load_data("my_dataset.csv")   # CSV / parquet / BIDS / npz
   X, y = container.X, container.y

Your first decoding experiment
==============================

A leakage-safe classification experiment with group-aware cross-validation:

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy", "roc_auc"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

   result = Experiment(config).run(
       X, y, sample_metadata={"Subject": subject_ids}, observation_level="epoch"
   )
   print(result.get_detailed_scores())

Your first embedding
====================

.. code-block:: python

   from coco_pipe.dim_reduction import DimReduction
   from coco_pipe.viz import plot_embedding

   reducer = DimReduction("UMAP", n_components=2, n_neighbors=15)
   embedding = reducer.fit_transform(X)
   reducer.score(embedding, X=X, k_values=[5, 10])
   plot_embedding(embedding, labels=y)

Turn results into a report
==========================

.. code-block:: python

   from coco_pipe.report import from_experiment_result

   from_experiment_result(result, output_path="decoding.html")

Next steps
==========

- Browse the :doc:`user_guide` for each module's scientific guide.
- Explore runnable workflows in the :doc:`gallery <auto_examples/index>`.
- See the :doc:`api_reference` for the complete public API.
