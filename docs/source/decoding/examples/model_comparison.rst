.. _decoding-example-model-comparison:

==============================
Example: Model Comparison
==============================

This example demonstrates the correct workflow for comparing multiple decoding
models using paired permutation tests, with FDR correction for multiple
comparisons.

**Scientific context**: Three candidate classifiers compete to decode cognitive
state from EEG features. We want to know: (1) which models perform above chance,
and (2) which model is significantly better than the others.

---

1. Configure Multi-Model Experiment
======================================

.. code-block:: python

   import numpy as np
   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, CVConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
   )

   rng = np.random.default_rng(42)
   n_subjects = 20
   X = rng.standard_normal((n_subjects * 10, 64))
   y = np.tile([0, 1], n_subjects * 5)
   subject_ids = np.repeat(np.arange(n_subjects), 10)

   config = ExperimentConfig(
       task="classification",
       models={
           "lr": ClassicalModelConfig(estimator="LogisticRegression"),
           "svm": ClassicalModelConfig(estimator="LinearSVC"),
           "rf": ClassicalModelConfig(estimator="RandomForestClassifier"),
       },
       metrics=["accuracy", "balanced_accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       evaluation=StatisticalAssessmentConfig(
           enabled=True,
           chance=ChanceAssessmentConfig(
               method="permutation",
               n_permutations=1000,
               unit_of_inference="group_mean",
           ),
       ),
       use_scaler=True,
       random_state=42,
   )

---

2. Run and Assess Significance
=================================

.. code-block:: python

   result = Experiment(config).run(
       X, y,
       sample_metadata={"Subject": subject_ids},
       observation_level="epoch",
   )

   # Which models are above chance?
   assessment = result.get_statistical_assessment()
   print(assessment[["Model", "Metric", "Observed", "PValue", "Significant"]])

---

3. Pairwise Model Comparison
==============================

.. code-block:: python

   # Paired comparison: LR vs SVM
   lr_vs_svm = result.compare_models_paired("lr", "svm", metric="accuracy", unit="Subject")
   print(lr_vs_svm[["Difference", "PValue", "Significant"]])

   # All pairwise comparisons with FDR correction
   all_pairs = result.compare_models(
       metric="accuracy",
       unit="Subject",
       correction="fdr_bh",
       n_permutations=5000,
   )
   print(all_pairs[["ModelA", "ModelB", "Difference", "CorrectedPValue", "Significant"]])

---

4. Score Distribution Visualization
=====================================

.. code-block:: python

   from coco_pipe.viz import plot_fold_score_dispersion

   # Visualize fold-level score distributions across models
   fig = plot_fold_score_dispersion(result, metric="accuracy")
   fig.savefig("score_dispersion.png", dpi=150)
