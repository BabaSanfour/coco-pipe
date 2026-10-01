coco-pipe
=========

.. raw:: html

   <p style="font-size:1.3rem; margin-top:0.5rem; max-width:52rem; line-height:1.5;">
     An engine for cognitive and computational neuroscience — from feature
     extraction and dimensionality reduction to trajectory analysis, classical
     machine learning, and foundation-model decoding, with publication-grade
     visualization and automated reporting.
   </p>

``coco-pipe`` provides reusable, leakage-safe components for M/EEG and related
biosignal research. Each module works on its own or as part of an end-to-end
workflow, and they all speak the same :class:`~coco_pipe.io.DataContainer`.

.. grid:: 1 2 3 3
   :gutter: 3
   :class-container: sd-mt-4

   .. grid-item-card:: 🚀 Getting Started
      :link: getting_started
      :link-type: doc

      Install ``coco-pipe`` and run your first end-to-end workflow.

   .. grid-item-card:: 📦 Data & IO
      :link: io/index
      :link-type: doc

      Loading (tabular, BIDS, embeddings), the ``DataContainer``, quality
      control, and persistence.

   .. grid-item-card:: 🌊 Descriptors
      :link: descriptors/index
      :link-type: doc

      Signal feature extraction — spectral, parametric, and complexity families.

   .. grid-item-card:: 🧠 Decoding — Classical ML
      :link: decoding/index
      :link-type: doc

      Leakage-safe classification & regression with scikit-learn estimators,
      cross-validation, tuning, and statistical assessment.

   .. grid-item-card:: 🤖 Decoding — Foundation Models
      :link: decoding/foundation_models
      :link-type: doc

      Pretrained backbones (frozen, fine-tuned, LoRA/QLoRA). In the decoding
      module, but a wholly separate design.

   .. grid-item-card:: 🌀 Dimensionality Reduction
      :link: dim_reduction/index
      :link-type: doc

      PCA, UMAP, PHATE, PaCMAP and 15+ reducers behind one interface, with
      quality metrics and method comparison.

   .. grid-item-card:: 📈 Trajectory Analysis
      :link: dim_reduction/trajectories
      :link-type: doc

      Kinematics and time-resolved group separation over native 3D embedding
      tensors.

   .. grid-item-card:: 📊 Visualization
      :link: viz/index
      :link-type: doc

      Mirrored Matplotlib and Plotly backends, one theme, exploratory to
      publication-ready figures.

   .. grid-item-card:: 📄 Reports
      :link: report/index
      :link-type: doc

      Self-contained, interactive HTML reports — lineage-aware and offline-ready.


Where to go next
----------------

.. grid:: 1 1 3 3
   :gutter: 2

   .. grid-item-card:: User Guide
      :link: user_guide
      :link-type: doc

      Scientific guides for every module.

   .. grid-item-card:: API Reference
      :link: api_reference
      :link-type: doc

      Public API and the full module index.

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc

      Executable, end-to-end gallery.


Our vision
----------

``coco-pipe`` is, first, an **engine** that brings the tools of cognitive and
computational neuroscience under one roof — feature extraction, dimensionality
reduction, trajectory analysis, classical decoding, and foundation models — all
speaking one data structure. On top of that engine we are building **end-to-end
pipelines**: throw your preprocessed data at them and, with a few CLI commands,
run complementary analyses to understand your data — then move from broad
exploration to focused, targeted analysis, again powered by the engine.

We start with **M/EEG**, and aim to extend to other modalities as the engine
matures.

.. seealso::

   Read the full :doc:`vision` for the roadmap and design principles.


.. toctree::
   :maxdepth: 1
   :hidden:

   getting_started
   user_guide
   api_reference
   Examples <auto_examples/index>
   Contributing <contributing>
   vision
