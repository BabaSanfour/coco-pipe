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

3. Foundation-model Embeddings
==============================

Embeddings from a foundation model are written as a BIDS derivative dataset,
one per model, as NetCDF files (read and written with xarray's ``h5netcdf``
engine). A file holds one representation of the model's output, its *view*,
named by the BIDS suffix ``<view>embeddings``:

.. list-table::
   :header-rows: 1
   :widths: 14 40 46

   * - View
     - Axes of ``embeddings``
     - Content
   * - ``token``
     - ``window``, the model's token axes, ``feature``
     - The model's token tensor of each window, nothing pooled.
   * - ``window``
     - ``window``, ``feature``
     - One vector per window.
   * - ``channel``
     - ``window``, ``channel``, ``feature``
     - One vector per channel of each window, averaged over time patches.
   * - ``patch``
     - ``window``, ``time_patch``, ``feature``
     - One vector per time patch of each window, averaged over channels.
   * - ``cov``
     - ``window``, ``feature``
     - The tangent-space covariance of each window's tokens.

Window vectors are L2-normalised when the extraction setting
``NormalizeEmbeddings`` says so. Channel and patch views are the tokens
averaged and are never normalised, whatever that setting says: an analysis
normalises them at load as it needs, per channel or, after ``flatten``, as one
vector.

An extractor built with ``extra_poolings`` (``("attention",)`` for REVE's
attention read-out) pools the same tokens again, in the same forward pass. The
writer puts those window vectors in a window file of their own label, the
label followed by the pooling's name (``baseAttention``), whose
``WindowPooling`` setting names that pooling.

The ``desc`` entity names the processing that produced the file, e.g. ``base``,
``baseRandom0`` (random weights of seed 0) or ``baseEaCoral``; it is any BIDS
label (letters, digits and ``+``). A file keeps its source recording's
entities, so one recording's files sit in its subject folder and the files
holding every recording at the root:

.. code-block:: text

   derivatives/coco-pipe-cbramod/
     dataset_description.json  README.md  descriptions.tsv  descriptions.json
     .bidsignore
     task-rest_desc-base_windowembeddings.nc     every recording
     task-rest_desc-base_windowembeddings.json   its sidecar
     desc-base_manifest.json                     skipped and failed recordings
     sub-01/ses-01/eeg/
       sub-01_ses-01_task-rest_run-01_desc-base_windowembeddings.nc
       sub-01_ses-01_task-rest_run-01_desc-base_tokenembeddings.nc

BIDS defines no NetCDF data type, so ``.bidsignore`` lists the embedding files;
the validator then checks the dataset files. The sidecar at the root holds the
extraction settings and, by BIDS inheritance, applies to the recordings' files
of the same label and view.

**File contents.** Every file, one recording's or every recording's, has the
same layout:

- ``embeddings``: float32, with the axes of its view. Token files are float16
  when written with ``token_dtype="float16"``, half the size; both are read as
  float32.
- Along ``window``: ``obs`` (``<recording_id>_epoch-NNNN``), ``recording_id``,
  ``subject``, ``session``, ``task``, ``run``, ``condition``, ``epoch`` (the
  epoch's number in its source file), ``start`` and ``stop`` (samples). Windows
  are in epoch order, every condition of a recording in one file.
- Along ``recording``: the recording's ``source`` (a BIDS URI such as
  ``bids:preproc:sub-01/ses-01/eeg/..._epo.fif``, resolved through
  ``DatasetLinks``), ``sfreq``, ``channels``, ``bad_channels`` and the channels
  the model interpolated, zero-filled or dropped (lists joined by commas).
- ``channel``: the channels the model saw, on views with a channel axis.
- Attributes: ``model_key``, ``config`` (the extraction settings as JSON, see
  :data:`~coco_pipe.io.EMBEDDING_CONFIG_KEYS`), ``config_hash`` (their hash)
  and ``description`` (the label); token files add ``token_source`` and
  ``token_feature_axis``.

**Writing and reading.**

.. code-block:: python

   from mne_bids import BIDSPath

   from coco_pipe.io import (
       consolidate_embeddings,
       find_embeddings,
       load_embeddings,
       write_embedding_dataset,
       write_recording_embeddings,
   )

   root = "derivatives/coco-pipe-cbramod"
   source = BIDSPath(
       root="derivatives/preproc", subject="01", session="01", task="rest",
       run="01", description="base", suffix="epo", extension=".fif",
       datatype="eeg", check=False,
   )
   # One recording: `results` maps each condition to the extractor's result.
   write_recording_embeddings(
       results, root, label="base", source=source, source_dataset="preproc"
   )

   # Every recording of a label and view in one file, then the dataset files.
   files = find_embeddings(root, "base", "window")
   consolidate_embeddings(files, root)
   write_embedding_dataset(
       root,
       {"base": "Pretrained weights"},
       name="CBraMod embeddings",
       dataset_links={"preproc": "../preproc"},
   )

   windows = load_embeddings(f"{root}/task-rest_desc-base_windowembeddings.nc")

A subject transform's output is written from what was loaded, row for row,
with :func:`~coco_pipe.io.write_transformed_embeddings`. The file keeps the
windows, recordings and channels of its source, and its settings add
``Transform``, so it never mixes with the source:

.. code-block:: python

   from coco_pipe.io import write_transformed_embeddings
   from coco_pipe.transforms import EuclideanAlign

   tokens = load_embeddings(find_embeddings(root, "base", "token"))
   aligner = EuclideanAlign("coral", channel_axis=1)
   write_transformed_embeddings(
       tokens,
       aligner.fit_transform(tokens.X, groups=tokens.coords["subject"]),
       root,
       label="baseEaCoralCh",
       transform={"name": aligner.name, **aligner.get_params()},
   )

``view="cov"`` writes a transform that changes the axes, such as
``RiemannAlign``'s tangent vectors, as a ``covembeddings`` file.

Each writer checks before it writes (one model and set of settings per file,
finite values, no epoch twice) and replaces its file atomically, so jobs may
write into one dataset at once. :func:`~coco_pipe.io.load_embeddings` refuses
files that differ in model, settings, label, view or channels.

4. Quality Control
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

5. Persistence
==============

Byte-level helpers back the container's ``save``/``load`` and are available
directly for custom artifacts:

.. code-block:: python

   from coco_pipe.io import save_object, load_object, write_json, read_json, save_npz

   save_object(obj, "artifact.pkl")
   obj = load_object("artifact.pkl")
   write_json("meta.json", metadata)

---

See the :doc:`/api_reference` for the complete ``coco_pipe.io`` API.
