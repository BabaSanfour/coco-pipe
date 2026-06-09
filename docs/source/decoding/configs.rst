.. _decoding-configs:

==========================
Configuration Reference
==========================

All experiment configuration is declarative and Pydantic-validated. Every
config class uses ``extra="forbid"`` so misspelled or unsupported field names
raise a ``ValidationError`` immediately — before any training starts.

---

1. ``ExperimentConfig``
========================

Top-level configuration for a decoding experiment.

.. code-block:: python

   from coco_pipe.decoding.configs import ExperimentConfig

   config = ExperimentConfig(
       task="classification",          # required: "classification" or "regression"
       models={"lr": ...},             # required: dict of model configs
       metrics=["accuracy"],           # default: task-appropriate defaults
       cv=CVConfig(...),               # default: StratifiedKFold(5)
       tuning=TuningConfig(...),       # default: disabled
       feature_selection=FeatureSelectionConfig(...),  # default: disabled
       calibration=CalibrationConfig(...),             # default: disabled
       evaluation=StatisticalAssessmentConfig(...),    # default: disabled
       grids={"lr": {"C": [0.1, 1.0]}},  # hyperparameter grids for tuning
       use_scaler=True,                   # prepend StandardScaler to pipeline
       n_jobs=1,                          # outer CV parallelism
       verbose=False,
       tag="my_experiment",               # descriptive label in result metadata
       random_state=42,
   )

---

2. ``CVConfig``
================

Controls the outer cross-validation loop.

.. code-block:: python

   from coco_pipe.decoding.configs import CVConfig

   cv = CVConfig(
       strategy="stratified_group_kfold",
       n_splits=5,
       group_key="Subject",    # column name in sample_metadata
       test_size=0.2,          # for "split" strategy only
       stratify=True,          # for "split" + classification only
       n_groups=2,             # for "leave_p_out" only
       random_state=42,
   )

See :ref:`decoding-cv` for a complete strategy guide.

---

3. ``ClassicalModelConfig``
============================

Configures a classical scikit-learn estimator.

.. code-block:: python

   from coco_pipe.decoding.configs import ClassicalModelConfig

   model = ClassicalModelConfig(
       estimator="LogisticRegression",    # key in ESTIMATOR_SPECS
       params={"C": 1.0, "max_iter": 200},
   )

Short-form aliases are also available for common estimators:

.. code-block:: python

   from coco_pipe.decoding.configs import LogisticRegressionConfig

   model = LogisticRegressionConfig(C=1.0, max_iter=200)

---

4. ``TemporalDecoderConfig``
==============================

Wraps a classical base estimator for 3D temporal inputs.

.. code-block:: python

   from coco_pipe.decoding.configs import TemporalDecoderConfig, ClassicalModelConfig

   model = TemporalDecoderConfig(
       wrapper="sliding",          # or "generalizing"
       base=ClassicalModelConfig(estimator="LogisticRegression"),
       scoring="accuracy",
       n_jobs=-1,
   )

Requires ``mne`` as an optional dependency.

---

5. ``TuningConfig``
=====================

Controls hyperparameter search.

.. code-block:: python

   from coco_pipe.decoding.configs import TuningConfig, CVConfig

   tuning = TuningConfig(
       enabled=True,
       search_type="grid",         # or "random"
       scoring="accuracy",
       n_iter=20,                  # for "random" search only
       n_jobs=1,
       refit=True,
       cv=CVConfig(strategy="stratified", n_splits=3),    # inner CV
       allow_nongroup_inner_cv=False,   # leakage guard
       random_state=42,
   )

---

6. ``FeatureSelectionConfig``
===============================

.. code-block:: python

   from coco_pipe.decoding.configs import FeatureSelectionConfig, CVConfig

   fs = FeatureSelectionConfig(
       enabled=True,
       method="k_best",        # or "sfs"
       n_features=20,
       scoring="accuracy",     # scoring criterion for SFS inner CV
       cv=CVConfig(strategy="stratified", n_splits=3),    # SFS inner CV
       direction="forward",    # for SFS: "forward" or "backward"
       allow_nongroup_inner_cv=False,
   )

---

7. ``CalibrationConfig``
==========================

Enables probability calibration inside the training path.

.. code-block:: python

   from coco_pipe.decoding.configs import CalibrationConfig, CVConfig

   calibration = CalibrationConfig(
       enabled=True,
       method="sigmoid",       # or "isotonic"
       cv=CVConfig(strategy="stratified", n_splits=3),
       allow_nongroup_inner_cv=False,
   )

---

8. ``StatisticalAssessmentConfig``
====================================

.. code-block:: python

   from coco_pipe.decoding.configs import (
       StatisticalAssessmentConfig, ChanceAssessmentConfig, ConfidenceIntervalConfig
   )

   evaluation = StatisticalAssessmentConfig(
       enabled=True,
       random_state=42,
       unit_of_inference="group_mean",   # "sample", "group_mean", "group_majority", "custom"
       chance=ChanceAssessmentConfig(
           method="permutation",         # or "binomial", "auto"
           n_permutations=1000,
           p0=None,                      # required for "binomial"
           temporal_correction="max_stat",  # "max_stat", "fdr_bh", "fdr_by", "none"
           store_null_distribution=False,
       ),
       confidence_intervals=ConfidenceIntervalConfig(
           alpha=0.05,
           method="clopper_pearson",     # or "wilson"
           n_bootstraps=1000,
       ),
   )

---

9. Foundation Model Configs
==============================

.. code-block:: python

   from coco_pipe.decoding.configs import (
       FoundationEmbeddingModelConfig,
       FrozenBackboneDecoderConfig,
       NeuralFineTuneConfig,
       LoRAConfig,
       QuantizationConfig,
       DeviceConfig,
       CheckpointConfig,
   )

   # Frozen embedding
   embed_cfg = FoundationEmbeddingModelConfig(
       provider="braindecode",     # "dummy", "braindecode", "huggingface", "reve"
       model_name="labram-pretrained",
       input_kind="epoched",       # "tabular", "temporal", "epoched", "embeddings", "tokens"
       pooling="mean",             # "mean", "flatten", "last"
       cache_embeddings=True,
       normalize_embeddings=True,
   )

   # Full neural fine-tuning
   ft_cfg = NeuralFineTuneConfig(
       provider="huggingface",
       model_name="brain-bzh/reve-base",
       input_kind="epoched",
       train_mode="qlora",         # "full", "lora", "qlora"
       lora=LoRAConfig(r=16, alpha=32),
       quantization=QuantizationConfig(enabled=True, bits=4),
       device=DeviceConfig(device="auto", precision="bf16"),
       checkpoints=CheckpointConfig(save="best"),
   )
