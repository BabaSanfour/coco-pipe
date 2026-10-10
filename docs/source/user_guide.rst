==========
User Guide
==========

Scientific user guides for each ``coco-pipe`` module. Every guide covers the
concepts, the workflow, and the catalog of what the module offers. For the
programmatic interface, see the :doc:`api_reference`.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: 📦 Data & IO
      :link: io/index
      :link-type: doc

      Loading (tabular, BIDS, embeddings), the ``DataContainer``, quality
      control, and persistence.

   .. grid-item-card:: 🌊 Descriptors
      :link: descriptors/index
      :link-type: doc

      Signal feature extraction — spectral band, parametric, and complexity
      families — emitted as a ``DataContainer``.

   .. grid-item-card:: 🧠 Decoding — Classical ML
      :link: decoding/index
      :link-type: doc

      Leakage-safe classification/regression with scikit-learn estimators:
      cross-validation, feature selection, tuning, and statistical inference.

   .. grid-item-card:: 🤖 Decoding — Foundation Models
      :link: decoding/foundation_models
      :link-type: doc

      Pretrained backbones (frozen, fine-tuned, LoRA/QLoRA). Lives in the
      decoding module but has a wholly separate design.

   .. grid-item-card:: 🌀 Dimensionality Reduction
      :link: dim_reduction/index
      :link-type: doc

      Reducers, preservation metrics, interpretation, and post-hoc method
      comparison.

   .. grid-item-card:: 📈 Trajectory Analysis
      :link: dim_reduction/trajectories
      :link-type: doc

      Kinematics and time-resolved group separation over native 3D
      ``(trajectory, time, dim)`` embedding tensors.

   .. grid-item-card:: 🔎 Representation Diagnostics
      :link: diagnostics/variance
      :link-type: doc

      Design-aware subject and label variance partitions, permutation nulls,
      omega squared, and subject-identity probes.

   .. grid-item-card:: 🧭 Subject Alignment
      :link: transforms/index
      :link-type: doc

      LEACE, Euclidean/CORAL, and Riemannian transforms with explicit fit-scope
      and leakage guidance.

   .. grid-item-card:: 📊 Visualization
      :link: viz/index
      :link-type: doc

      Theme, primitives, and the static/interactive plot catalog.

   .. grid-item-card:: 📄 Reports
      :link: report/index
      :link-type: doc

      Building reports, section builders, elements, and advanced templating.


.. toctree::
   :maxdepth: 2
   :hidden:

   io/index
   descriptors/index
   decoding/index
   dim_reduction/index
   diagnostics/variance
   transforms/index
   viz/index
   report/index
