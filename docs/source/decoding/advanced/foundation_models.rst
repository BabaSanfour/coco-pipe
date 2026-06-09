.. _decoding-foundation-models:

===========================
Foundation Models (fm_hub)
===========================

``coco_pipe.decoding.fm_hub`` provides a unified interface to pretrained
neural network backbones for EEG/MEG decoding. Foundation models can be used
in three modes:

1. **Frozen embedding extraction**: extract features from a fixed backbone,
   then decode with a classical scikit-learn estimator.
2. **Frozen backbone + trainable head**: fine-tune only the output head.
3. **Full fine-tuning**: update all backbone parameters (LoRA, QLoRA, or full).

All modes enter through ``Experiment.run(...)`` and are compatible with the
outer CV loop, meaning the foundation model is fit/embedded inside the
training partition of each fold.

---

1. Embedding Extraction (Frozen Backbone)
==========================================

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
                   provider="braindecode",
                   model_name="labram-pretrained",
                   input_kind="epoched",
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
--------------------

``cache_embeddings=True`` writes extracted embeddings to a fold-keyed disk
cache. On subsequent runs with the same data and backbone configuration, the
backbone forward pass is skipped and embeddings are loaded from cache. The cache
key is computed from the training split identity and backbone fingerprint.

.. code-block:: python

   from coco_pipe.decoding import make_feature_cache_key

   key = make_feature_cache_key(
       train_sample_ids=train_ids,
       test_sample_ids=test_ids,
       backbone_fingerprint=backbone_hash,
   )

1.2 Supported Providers
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Provider
     - Description
   * - ``"braindecode"``
     - Pretrained models from the Braindecode library (ShallowFBCSPNet, EEGNetv4,
       LaBraM, etc.).
   * - ``"huggingface"``
     - Any HuggingFace Hub model compatible with the EEG tokenizer interface.
   * - ``"reve"``
     - REVE pretrained EEG backbone.
   * - ``"dummy"``
     - Returns random embeddings. For testing pipeline integrity only.

---

2. Neural Fine-Tuning (LoRA / QLoRA)
=======================================

.. code-block:: python

   from coco_pipe.decoding.configs import (
       NeuralFineTuneConfig, LoRAConfig, QuantizationConfig, DeviceConfig, CheckpointConfig
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "reve_qlora": NeuralFineTuneConfig(
               provider="huggingface",
               model_name="brain-bzh/reve-base",
               input_kind="epoched",
               train_mode="qlora",
               lora=LoRAConfig(r=16, alpha=32, dropout=0.05),
               quantization=QuantizationConfig(enabled=True, bits=4),
               device=DeviceConfig(device="auto", precision="bf16"),
               checkpoints=CheckpointConfig(save="best"),
           )
       },
       metrics=["balanced_accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

2.1 Training Modes
-------------------

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
-----------------------

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
=========================

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
==========================

Foundation models require optional extras:

- ``braindecode`` provider: ``pip install coco-pipe[braindecode]``
- ``huggingface`` / ``qlora`` provider: ``pip install coco-pipe[hf,peft,quant]``
- ``reve`` provider: Contact the REVE team for access.

.. code-block:: bash

   pip install coco-pipe[hf,peft,quant]  # QLoRA path
   pip install coco-pipe[braindecode]     # Braindecode backbone
