.. _descriptors:

===========
Descriptors
===========

The ``coco_pipe.descriptors`` module turns raw or epoched M/EEG signals into
interpretable feature tables — spectral band powers, parametric (aperiodic /
periodic) descriptors, and complexity measures — emitted as a
:class:`~coco_pipe.io.DataContainer` ready for decoding, reduction, or reporting.

.. admonition:: Design Philosophy

   Feature extraction is declarative and container-native. You describe *which*
   families to compute with a :class:`~coco_pipe.descriptors.DescriptorConfig`,
   and :class:`~coco_pipe.descriptors.DescriptorPipeline` returns a labelled
   ``DataContainer`` whose feature coordinates encode the family, band, and
   channel of every column — so downstream steps stay self-describing.

.. rubric:: Key Features

- Three feature families — **band** (spectral power / ratios), **param**
  (aperiodic + periodic parametrization), and **complexity** — selected and
  tuned through one typed config.
- Container in, container out: :meth:`~coco_pipe.descriptors.DescriptorPipeline.extract`
  returns a :class:`~coco_pipe.io.DataContainer` with self-describing feature names.
- Channel pooling into regions of interest via
  :meth:`~coco_pipe.descriptors.DescriptorPipeline.pool_channels`.
- Tidy table builders (:func:`~coco_pipe.descriptors.build_descriptor_tables`,
  :func:`~coco_pipe.descriptors.save_descriptor_table`) for export and sharing.

---

1. Quickstart
=============

.. code-block:: python

   import numpy as np
   from coco_pipe.descriptors import DescriptorPipeline, DescriptorConfig

   # X: (n_epochs, n_channels, n_times)
   pipeline = DescriptorPipeline(DescriptorConfig())   # defaults: all families

   container = pipeline.extract(
       X,
       ids=epoch_ids,
       sfreq=200.0,
       channel_names=["Fz", "Cz", "Pz", "Oz"],
   )

   container.X        # (n_epochs, n_features) feature matrix
   container.shape

``extract`` returns a :class:`~coco_pipe.io.DataContainer`, so it flows straight
into the rest of coco-pipe:

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig

   result = Experiment(ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="subject"),
   )).run(container.X, container.y, sample_metadata={"subject": subjects})

2. Choosing Feature Families
============================

Each family is a sub-config on :class:`~coco_pipe.descriptors.DescriptorConfig`,
toggled and parameterized independently:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Family
     - What it computes
   * - ``band``
     - Per-band spectral power and aggregated band ratios over canonical bands
       (delta, theta, alpha, beta, gamma).
   * - ``param``
     - Parametric spectral descriptors — aperiodic (offset, exponent) and
       periodic (peak) parameters.
   * - ``complexity``
     - Signal-complexity measures (e.g., entropy / fractal descriptors).

.. code-block:: python

   from coco_pipe.descriptors import DescriptorConfig
   from coco_pipe.descriptors.configs import (
       DescriptorFamiliesConfig, BandDescriptorConfig,
       ParametricDescriptorConfig, ComplexityDescriptorConfig,
   )

   config = DescriptorConfig(
       families=DescriptorFamiliesConfig(
           bands=BandDescriptorConfig(),
           parametric=ParametricDescriptorConfig(),
           complexity=ComplexityDescriptorConfig(),
       ),
       precision="float32",
   )

The family tokens are exposed as
:data:`coco_pipe.descriptors.KNOWN_FAMILY_TOKENS` (``band``, ``param``,
``complexity``) and are embedded in each feature's name so columns remain
traceable to their family.

3. Pooling Channels Into Regions
================================

Reduce channel-level features to regions of interest with
:meth:`~coco_pipe.descriptors.DescriptorPipeline.pool_channels`:

.. code-block:: python

   pooled = pipeline.pool_channels(
       container,
       channel_groups={
           "frontal": ["Fz", "F3", "F4"],
           "posterior": ["Pz", "O1", "O2"],
       },
   )

4. Exporting Feature Tables
===========================

Build and persist tidy descriptor tables for sharing or external analysis:

.. code-block:: python

   from coco_pipe.descriptors import build_descriptor_tables, save_descriptor_table

   tables = build_descriptor_tables(container)
   save_descriptor_table(tables, "descriptors.parquet")

---

See the :doc:`/api_reference` for the complete ``coco_pipe.descriptors`` API.
