.. _decoding:

===========================================
Decoding Module — Scientific User Guide
===========================================

The ``coco_pipe.decoding`` module provides a rigorous, reproducible framework for
neural decoding — the statistical inference of cognitive, perceptual, or clinical
states from multivariate brain recordings. It is designed from first principles
around **zero data leakage**, **independence-aware inference**, and a
**declarative configuration** API.

.. admonition:: Design Philosophy

   Every preprocessing step (scaling, feature selection, hyperparameter search,
   calibration) executes **inside** each outer cross-validation fold on the
   training partition only. This eliminates the most common source of inflated
   decoding accuracy in neuroimaging pipelines.

.. rubric:: Key Features

- Outer CV with zero preprocessing leakage guaranteed by architecture.
- Registry-based estimator + metric compatibility contracts (blocked before training).
- Full-pipeline permutation testing with optional Max-Stat temporal correction.
- Sliding and generalizing temporal estimators (MNE meta-estimators) with tidy output.
- Foundation model integration (frozen backbone, fine-tune, LoRA, QLoRA).
- Group-aware cross-validation with automatic inner CV propagation.
- Comprehensive ``ExperimentResult`` API with 20+ diagnostic accessors.

---

.. rubric:: Quickstart

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig

   config = ExperimentConfig(
       task="classification",
       models={
           "lr": ClassicalModelConfig(
               estimator="LogisticRegression",
               params={"max_iter": 200}
           )
       },
       metrics=["accuracy", "roc_auc"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

   result = Experiment(config).run(
       X, y,
       sample_metadata={"Subject": subject_ids, "Session": session_ids},
       observation_level="epoch",
   )

   print(result.get_detailed_scores())
   print(result.get_predictions().head())

---

.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts
   configs
   experiment
   result

.. toctree::
   :maxdepth: 2
   :caption: Statistical Inference

   stats
   cv_strategies
   model_comparison

.. toctree::
   :maxdepth: 2
   :caption: Model Reference

   models
   metrics
   feature_selection
   temporal_decoding

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/foundation_models
   advanced/custom_estimators
   advanced/reproducibility

.. toctree::
   :maxdepth: 2
   :caption: Worked Examples

   examples/basic_classification
   examples/grouped_cv
   examples/temporal_eeg
   examples/model_comparison
