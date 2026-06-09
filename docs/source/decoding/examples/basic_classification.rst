.. _decoding-example-basic:

===========================
Example: Basic Classification
===========================

This example walks through a minimal reproducible decoding experiment: binary
classification from a 2D feature matrix, with stratified group-based
cross-validation and post-hoc statistical assessment.

**Scientific context**: Classifying task conditions (e.g., face vs. object) from
subject-level EEG power spectral density features. Each row in ``X`` is one
epoch's feature vector. Multiple epochs from the same subject share a subject ID.

---

1. Prepare Data
================

.. code-block:: python

   import numpy as np
   import pandas as pd
   from sklearn.datasets import make_classification

   # Simulate: 200 epochs, 64 features, 20 subjects (10 per class)
   n_subjects = 20
   n_epochs_per_subject = 10
   n_features = 64

   rng = np.random.default_rng(42)
   X = rng.standard_normal((n_subjects * n_epochs_per_subject, n_features))
   y = np.repeat([0, 1], n_subjects // 2 * n_epochs_per_subject)
   subject_ids = np.repeat(np.arange(n_subjects), n_epochs_per_subject)

---

2. Configure the Experiment
============================

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, CVConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "logistic_regression": ClassicalModelConfig(
               estimator="LogisticRegression",
               params={"max_iter": 500, "solver": "liblinear"},
           ),
           "random_forest": ClassicalModelConfig(
               estimator="RandomForestClassifier",
               params={"n_estimators": 100},
           ),
       },
       metrics=["accuracy", "balanced_accuracy", "roc_auc"],
       cv=CVConfig(
           strategy="stratified_group_kfold",
           n_splits=5,
           group_key="Subject",
       ),
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

3. Run the Experiment
======================

.. code-block:: python

   result = Experiment(config).run(
       X,
       y,
       sample_metadata={"Subject": subject_ids},
       observation_level="epoch",
   )

---

4. Inspect Results
===================

.. code-block:: python

   # Per-fold scores
   scores = result.get_detailed_scores()
   print(scores.groupby(["Model", "Metric"])["Score"].agg(["mean", "std"]))

   # Confusion matrices
   cm = result.get_confusion_matrices(normalize=True)

   # ROC curves
   roc = result.get_roc_curve()

   # Calibration curves
   cal = result.get_calibration_curve()

---

5. Statistical Assessment
==========================

.. code-block:: python

   assessment = result.get_statistical_assessment()
   print(assessment[["Model", "Metric", "Observed", "PValue", "Significant"]])

---

6. Compare Models
==================

.. code-block:: python

   comparison = result.compare_models_paired(
       "logistic_regression", "random_forest",
       metric="accuracy",
       unit="Subject",
       n_permutations=5000,
   )
   print(comparison[["Difference", "PValue", "Significant"]])

---

7. Persist and Load
====================

.. code-block:: python

   path = result.save("results/basic_classification.json")
   loaded = ExperimentResult.load(path)
