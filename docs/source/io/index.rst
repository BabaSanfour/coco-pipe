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
   feature table, a BIDS M/EEG dataset, or an array of foundation-model
   embeddings, it is loaded into the same :class:`~coco_pipe.io.DataContainer`,
   so feature extraction, reduction, decoding, visualization, and reporting all
   compose without bespoke glue code.

.. rubric:: Key Features

- One loader, :func:`~coco_pipe.io.load_data`, with auto-detected modes for
  tabular files, BIDS datasets, and embedding derivatives.
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
   z = container.zscore(dim="obs")                      # normalize
   pooled = container.aggregate_groups("subject", how="mean")
   flat = container.flatten()                           # collapse to 2D (obs, feature)

   container.save("data.pkl")
   restored = DataContainer.load("data.pkl")

2. Loading Data
===============

:func:`~coco_pipe.io.load_data` is the single entry point. ``mode="auto"``
infers the source type; pass an explicit mode or a typed config for full
control.

.. code-block:: python

   from coco_pipe.io import load_data

   # Tabular feature table (CSV / parquet / Excel); columns map to dimensions.
   container = load_data("features.csv", target_col="diagnosis")

   # BIDS M/EEG dataset, epoched on load (requires the [eeg] extra).
   container = load_data(
       "/data/bids", mode="bids", datatype="eeg",
       loading_mode="epochs", tmin=-0.2, tmax=0.5,
   )

   # Precomputed embedding derivatives.
   container = load_data("/data/embeddings", mode="embeddings")

For reproducible, validated loading, pass a config object instead of loose
keyword arguments:

.. code-block:: python

   from coco_pipe.io import TabularConfig

   container = load_data(config=TabularConfig(path="features.csv", target_col="diagnosis"))

3. Quality Control
==================

Run the standard quality suite over a container in one call. It returns the
(optionally cleaned) container plus a :class:`~coco_pipe.io.QCResult`:

.. code-block:: python

   from coco_pipe.io import run_qc

   container, qc = run_qc(container, subject_col="subject")
   print(qc)   # missingness, outlier, and flatline findings

The individual checks live in :mod:`coco_pipe.io.quality` and each returns a
:class:`~coco_pipe.io.CheckResult`:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Check
     - What it flags
   * - :func:`coco_pipe.io.quality.check_missingness`
     - Fraction of ``NaN`` values per feature.
   * - :func:`coco_pipe.io.quality.check_constant_columns`
     - Near-zero-variance (constant) columns.
   * - :func:`coco_pipe.io.quality.check_outliers_zscore`
     - Z-score outliers above a threshold.
   * - :func:`coco_pipe.io.quality.check_flatline`
     - Zero-variance (flatlined) signal arrays.

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
