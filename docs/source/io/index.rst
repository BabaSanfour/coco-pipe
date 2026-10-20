.. _io:

=========
Data & IO
=========

The ``coco_pipe.io`` module is the data backbone of coco-pipe. It loads datasets
from many sources into a single labelled structure —
:class:`~coco_pipe.io.DataContainer` — that every other module consumes, and it
provides quality-control and persistence utilities around it.

.. admonition:: Design Philosophy

   One in-memory contract, everywhere. Whether your data starts as a tabular
   feature table, a BIDS M/EEG dataset, or foundation-model embedding files,
   it is loaded into the same :class:`~coco_pipe.io.DataContainer`,
   so feature extraction, reduction, decoding, visualization, and reporting all
   compose without bespoke glue code.

.. rubric:: Key Features

- One loader, :func:`~coco_pipe.io.load_data`, driven by a typed config for
  tabular files, BIDS datasets, and embedding files.
- A labelled, dimension-aware container with tidy selection, aggregation, and
  normalization helpers.
- Inline data-quality checks (missingness, constant columns, outliers,
  flatlines) and a one-call :func:`~coco_pipe.io.run_qc`.
- Typed dataset configs (:class:`~coco_pipe.io.DatasetConfig`,
  :class:`~coco_pipe.io.BIDSConfig`, :class:`~coco_pipe.io.TabularConfig`,
  :class:`~coco_pipe.io.EmbeddingConfig`).

---

1. The ``DataContainer``
========================

A :class:`~coco_pipe.io.DataContainer` wraps a numeric array ``X`` with named
dimensions, coordinate labels, optional targets ``y``, observation ``ids``, and
free-form ``meta``.

.. code-block:: python

   from coco_pipe.io import DataContainer
   import numpy as np

   container = DataContainer(
       X=np.random.randn(100, 32),       # (obs, feature)
       dims=("obs", "feature"),
       y=labels,                          # optional targets
       ids=subject_ids,                   # optional observation ids
   )

   container.X            # the numeric array
   container.y            # targets
   container.shape        # array shape
   container.ids          # observation ids

Common transformations return new containers (nothing mutates in place):

.. code-block:: python

   sub = container.select(feature=["alpha", "beta"])   # label-based selection
   sub = container.isel(obs=slice(0, 10))              # positional selection
   array = container.to_xarray()                        # labelled xarray.DataArray
   pooled = container.aggregate_groups("subject", how="mean")
   flat = container.flatten()                           # collapse to 2D (obs, feature)

   container.save("data.pkl")
   restored = DataContainer.load("data.pkl")

2. Loading Data
===============

:func:`~coco_pipe.io.load_data` is the single entry point. It takes a typed
config, validated when it is built, which says what to load and how:

.. code-block:: python

   from coco_pipe.io import BIDSConfig, EmbeddingConfig, TabularConfig, load_data

   # Tabular feature table (CSV / TSV / Excel); columns map to dimensions.
   container = load_data(
       TabularConfig(path="features.csv", sep=",", target_col="diagnosis")
   )

   # BIDS M/EEG dataset, cut into 2 s windows on load.
   container = load_data(
       BIDSConfig(path="/data/bids", task="rest", datatype="eeg", window_length=2.0)
   )

   # Foundation-model embeddings written by coco-pipe: one row per recording
   # and condition, eyes-closed only.
   container = load_data(
       EmbeddingConfig(
           path="derivatives/coco-pipe-cbramod/task-rest_desc-base_windowembeddings.nc",
           representation="recording",
           conditions="EC",
       )
   )

BIDS and embedding containers name their rows alike: ``obs`` is
``<recording_id>_epoch-NNNN``, so an epoch and the embedding extracted from it
share an ID, and both carry ``recording_id``, ``subject``, ``task``,
``condition`` (the event name) and ``epoch``. With
``representation="recording"`` both return one row per recording and
condition, the mean of its rows, named ``<recording_id>_cond-<condition>``
(:func:`~coco_pipe.io.aggregate_recordings`).

They also carry each subject's ``participants.tsv`` columns along ``obs``. A
derivative dataset without that file reads it from the dataset it comes from,
following ``SourceDatasets`` (embeddings, then preprocessed data, then raw
data). More subject-level columns, which a config cannot hold, are passed to
``load_data`` directly as ``subject_metadata_df``, with ``subject_key`` naming
its subject column. Embedding files are also read with
:func:`~coco_pipe.io.load_embeddings`, which takes several files at once (see
:func:`~coco_pipe.io.find_embeddings`).

3. Quality Control
==================

Run the standard quality suite over a container in one call. It returns the
(optionally cleaned) container plus a :class:`~coco_pipe.io.QCResult`:

.. code-block:: python

   from coco_pipe.io import run_qc

   container, qc = run_qc(container, subject_col="subject")
   print(qc.summary())   # rows in and out, drops, and thresholds

The individual checks live in :mod:`coco_pipe.io.quality` and report their
findings as :class:`~coco_pipe.io.CheckResult` objects:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Check
     - What it flags
   * - :func:`coco_pipe.io.quality.check_missingness`
     - Fraction of missing values over the whole input.
   * - :func:`coco_pipe.io.quality.check_constant_columns`
     - Constant columns: two or more values, all equal.
   * - :func:`coco_pipe.io.quality.check_outliers_zscore`
     - Values whose z-score, over all values pooled, exceeds ``sigma``.
   * - :func:`coco_pipe.io.quality.check_flatline`
     - Near-zero-variance (flatlined) signals.

The same checks run automatically when a ``DataContainer`` is added to a
:doc:`report <../report/index>`.

4. Persistence
==============

Byte-level helpers back the container's ``save``/``load`` and are available
directly for custom artifacts:

.. code-block:: python

   from coco_pipe.io import save_object, load_object, write_json, read_json, save_npz

   save_object(obj, "artifact.pkl")
   obj = load_object("artifact.pkl")
   write_json(metadata, "meta.json")

---

See the :doc:`/api_reference` for the complete ``coco_pipe.io`` API.
