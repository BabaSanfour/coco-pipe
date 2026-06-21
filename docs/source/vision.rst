======
Vision
======

How we envisage ``coco-pipe`` evolving — what it is today, the principles that
guide it, and where it is headed.

The idea
--------

``coco-pipe`` is, first, an **engine** that brings the tools of cognitive and
computational neuroscience under one roof — feature extraction, dimensionality
reduction, trajectory analysis, classical decoding, and foundation models — all
speaking a single data structure so they compose without friction.

On top of that engine we are building **end-to-end pipelines**. The goal is
simple: throw your *preprocessed* data at a pipeline and, with a few CLI
commands, run a battery of complementary analyses to understand it — then move
from broad, exploratory analysis to focused, targeted questions, again powered
by the same engine. The audience is both **neuroscientists** who want rigorous
answers without writing boilerplate and **engineers** who want a dependable,
composable toolkit.

We begin with **M/EEG**. The architecture is deliberately modality-agnostic at
its core, and we aim to extend to other modalities as the engine matures.

Roadmap
-------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Phase 1 — Build and harden the engine (current)

      We are building and testing the engine and preparing the tools the pipelines
      will need: a shared :class:`~coco_pipe.io.DataContainer`, leakage-safe decoding
      for classical and foundation models, dimensionality reduction with trajectory
      analysis and interpretation, and self-contained reporting. This phase is about
      correctness, reproducibility, and a clean, composable API.

   .. grid-item-card:: Phase 2 — End-to-end pipelines

      With the engine in place, we will ship opinionated, CLI-driven pipelines that
      chain these tools into ready-made exploratory and targeted analyses — so going
      from preprocessed data to interpretable results is a few commands, not a
      research-engineering project.

Design principles
-----------------

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Valid by construction

      Cross-validation, feature selection, tuning, and calibration run inside
      the appropriate fold. Sound inference is the default, never an afterthought.

   .. grid-item-card:: One data structure, end to end

      Every module reads and writes a single :class:`~coco_pipe.io.DataContainer`,
      so steps compose without glue code and results stay self-describing.

   .. grid-item-card:: Declarative and reproducible

      Workflows are described with validated, typed configs and seeded throughout,
      so an experiment is explicit, serializable, and bit-reproducible.

   .. grid-item-card:: Lightweight core, optional power

      Heavy dependencies — deep learning, distributed compute, manifold libraries —
      load lazily behind extras; the base install stays small.

   .. grid-item-card:: Comparable, shareable results

      Reports are interactive, lineage-aware, and self-contained, built to line up
      many analyses side by side.

   .. grid-item-card:: Modality-agnostic at the core

      M/EEG first, but the contracts are designed so new modalities slot in
      without reworking the engine.

What we have today
------------------

The engine is real and in active use. Highlights per module:

.. grid:: 1 2 2 2
   :gutter: 3
   :class-container: sd-mt-3

   .. grid-item-card:: 📦 Data & IO

      One :func:`~coco_pipe.io.load_data` for tabular, BIDS, and embedding sources
      into a labelled ``DataContainer``; built-in quality control.

   .. grid-item-card:: 🌊 Descriptors

      Spectral, parametric, and complexity feature families with channel pooling,
      emitted container-native.

   .. grid-item-card:: 🧠 Decoding — Classical ML

      Leakage-free CV, group-aware inference, feature selection and tuning, and
      full-pipeline permutation testing.

   .. grid-item-card:: 🤖 Decoding — Foundation Models

      Frozen, fine-tuned, and LoRA/QLoRA backbones treated as ordinary, comparable
      estimators, with leakage-safe cached embeddings.

   .. grid-item-card:: 🌀 Dimensionality Reduction

      15+ reducers behind one interface, preservation metrics, feature interpretation,
      and post-hoc method ranking.

   .. grid-item-card:: 📈 Trajectory Analysis

      Kinematics and time-resolved group separation over native 3D
      ``(trajectory, time, dim)`` embedding tensors.

   .. grid-item-card:: 📊 Visualization

      Mirrored Matplotlib/Plotly backends from one theme, from exploratory
      to publication-ready.

   .. grid-item-card:: 📄 Reports

      Self-contained, interactive HTML that makes many experiments easy to compare.

Where we are heading
--------------------

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: CLI-driven end-to-end pipelines

      **(Phase 2)** — preprocessed data in, exploratory-then-targeted analysis
      out, in a few commands.

   .. grid-item-card:: Deeper foundation-model workflows

      More backbones, richer fine-tuning, and embedding reuse across analyses.

   .. grid-item-card:: Domain-aware methods

      Reducers and metrics that respect the physical and topological structure
      of neural data.

   .. grid-item-card:: Comparative reporting at scale

      Reports that line up models, modalities, and subjects for side-by-side
      interpretation.

   .. grid-item-card:: More modalities

      Extending beyond M/EEG as the engine matures, and broader standardized loading.

Get involved
------------

``coco-pipe`` is in active pre-release development and we welcome contributions.
See the :doc:`contributing` guide to get started.
