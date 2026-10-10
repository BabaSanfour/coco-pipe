.. _decoding-foundation-models:

=================
Foundation Models
=================


``coco_pipe.decoding`` integrates pretrained neural-network backbones for
EEG/MEG decoding (implemented in ``coco_pipe.decoding.foundation_models``)
directly into the ``Experiment`` workflow, configured through
``coco_pipe.decoding.configs`` just like classical estimators. Foundation
models can be used in three modes:

1. **Frozen embedding extraction**: extract features from a fixed backbone,
   then decode with a classical scikit-learn estimator.
2. **Frozen backbone + trainable head**: fine-tune only the output head.
3. **Full fine-tuning**: update all backbone parameters (LoRA, QLoRA, or full).

All modes enter through ``Experiment.run(...)`` and are compatible with the
outer CV loop, meaning the foundation model is fit/embedded inside the
training partition of each fold.

---

1. Embedding Extraction (Frozen Backbone)
-----------------------------------------

The simplest foundation model workflow: freeze the backbone and use it as a
fixed feature extractor. A classical scikit-learn model decodes from the
extracted embeddings.

.. code-block:: python

   from coco_pipe.decoding.configs import (
       ExperimentConfig, CVConfig,
       FoundationEmbeddingModelConfig,
       FrozenBackboneDecoderConfig,
       ClassicalModelConfig,
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "labram_probe": FrozenBackboneDecoderConfig(
               backbone=FoundationEmbeddingModelConfig(
                   backend="braindecode",
                   model_key="labram",
                   pooling="mean",
                   cache_embeddings=True,    # cache to disk for reuse across folds
               ),
               head=ClassicalModelConfig(
                   estimator="LogisticRegression",
                   params={"max_iter": 1000},
               ),
           )
       },
       metrics=["balanced_accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

   result = Experiment(config).run(X_epochs, y, sample_metadata=meta)

1.1 Embedding Cache
~~~~~~~~~~~~~~~~~~~

``cache_embeddings=True`` writes extracted embeddings to a content-addressed
disk cache. Because a frozen backbone is a deterministic, target-independent
function of the window content, the cache key is derived from the input windows
and the backbone fingerprint — so embeddings are computed once and reused across
folds and models without leakage. No manual cache management is required.

1.2 Supported Models & Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

coco-pipe integrates the following foundation models natively, registered in
``coco_pipe.decoding._specs.ESTIMATOR_SPECS`` (browse them at runtime with
:func:`~coco_pipe.decoding.list_foundation_models`):

.. foundation-table::

---

2. Neural Fine-Tuning (LoRA / QLoRA)
------------------------------------

.. code-block:: python

   from coco_pipe.decoding.configs import (
       NeuralFineTuneConfig, LoRAConfig, QuantizationConfig, DeviceConfig, CheckpointConfig
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "reve_qlora": NeuralFineTuneConfig(
               backend="hugging_face",
               model_key="reve",
               input_kind="epoched",
               train_mode="qlora",
               lora=LoRAConfig(r=16, alpha=32, dropout=0.05),
               quantization=QuantizationConfig(enabled=True, load_in_4bit=True),
               device=DeviceConfig(device="auto", precision="bf16"),
               checkpoints=CheckpointConfig(save="best"),
           )
       },
       metrics=["balanced_accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

2.1 Training Modes
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``"full"``
     - Update all backbone parameters. Highest capacity; requires most memory.
   * - ``"lora"``
     - Low-Rank Adaptation. Trains small rank-decomposed matrices injected into
       transformer attention. Memory-efficient.
   * - ``"qlora"``
     - Quantized LoRA. Backbone quantized to 4-bit for inference; LoRA adapters
       trained in higher precision. Most memory-efficient option.

2.2 LoRA Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Parameter
     - Description
   * - ``r``
     - Rank of the LoRA decomposition. Higher rank → more parameters. Default 16.
   * - ``alpha``
     - Scaling factor. ``alpha / r`` scales the LoRA output. Default 32.
   * - ``dropout``
     - Dropout on LoRA layers. Default 0.0.

---

3. Diagnostic Artifacts
-----------------------

Trainable neural models expose training diagnostics via ``NeuralTrainable``
protocol methods:

.. code-block:: python

   artifacts = result.get_model_artifacts()
   # columns: Model, Fold, ArtifactKey, ArtifactValue

   # Per-fold training history
   history = result.get_model_artifacts(artifact_type="training_history")

   # Checkpoint manifest
   checkpoints = result.get_model_artifacts(artifact_type="checkpoints")

The ``NeuralTrainable`` protocol requires:

- ``get_training_history() → list[dict]``: loss/metric per epoch.
- ``get_checkpoint_manifest() → dict``: saved checkpoint paths and best epoch.
- ``get_model_card_info() → dict``: architecture and training summary.
- ``get_failure_diagnostics() → dict``: NaN detection, gradient norms.
- ``get_artifact_metadata() → dict``: aggregated artifact dictionary.

---

4. Required Dependencies
------------------------

Foundation models require optional extras:

- ``braindecode`` provider: ``pip install coco-pipe[braindecode]``
- ``huggingface`` / ``qlora`` provider: ``pip install coco-pipe[hf,peft,quant]``
- ``reve`` provider: Contact the REVE team for access.

.. code-block:: bash

   pip install coco-pipe[hf,peft,quant]  # QLoRA path
   pip install coco-pipe[braindecode]     # Braindecode backbone
