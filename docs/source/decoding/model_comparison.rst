.. _decoding-model-comparison:

====================
Model Comparison
====================

After running a decoding experiment with multiple models, ``coco_pipe.decoding``
provides rigorous paired statistical tests to determine whether observed
performance differences are beyond chance. All comparison methods use
within-subject label swaps to control for subject-specific baseline variance.

---

1. Why Paired Tests?
=====================

Independent-sample tests compare two models assuming the samples are drawn
independently. In a within-subject decoding design, the **same subjects** appear
in both models' test folds, making the samples positively correlated. A paired
test exploits this correlation to achieve higher statistical power.

**Paired permutation test**: randomly swap model assignments within each
independent unit (subject) and recompute the observed difference. The resulting
null distribution represents the expected difference under no true effect.

---

2. Quick Paired Comparison
============================

For a fast paired comparison using existing outer-fold predictions:

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig, SVMConfig

   config = ExperimentConfig(
       task="classification",
       models={
           "lr": ClassicalModelConfig(estimator="LogisticRegression"),
           "svm": ClassicalModelConfig(estimator="SVC"),
       },
       metrics=["accuracy", "roc_auc"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

   result = Experiment(config).run(
       X, y,
       sample_metadata={"Subject": subject_ids, "Session": session_ids}
   )

   paired = result.compare_models_paired(
       "lr", "svm",
       metric="accuracy",
       unit="Subject",
       n_permutations=5000,
       random_state=42,
   )

   print(paired[["Metric", "ScoreA", "ScoreB", "Difference", "PValue", "Significant"]])

The returned DataFrame has one row per (temporal coordinate or scalar) with:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - ``ScoreA``
     - Observed score for model A.
   * - ``ScoreB``
     - Observed score for model B.
   * - ``Difference``
     - ``ScoreA - ScoreB``.
   * - ``PValue``
     - Two-sided p-value from the sign-swap permutation distribution.
   * - ``Significant``
     - Boolean: ``PValue <= 0.05``.
   * - ``NUnits``
     - Number of independent units used for swapping.
   * - ``NPermutations``
     - Number of permutations used.

---

3. Multiple Model Comparison
==============================

When comparing more than two models, use ``compare_models`` to compare all
pairs with optional multiple-comparison correction:

.. code-block:: python

   comparison = result.compare_models(
       metric="accuracy",
       unit="Subject",
       correction="fdr_bh",       # or "bonferroni", "none"
       n_permutations=5000,
   )

   print(comparison[["ModelA", "ModelB", "Difference", "PValue", "CorrectedPValue"]])

---

4. Full-Pipeline Paired Permutation Test
==========================================

For rigorous inference where preprocessing, feature selection, and tuning must
be included in the null distribution, use ``run_paired_permutation_assessment``:

.. code-block:: python

   from coco_pipe.decoding.stats import run_paired_permutation_assessment
   from coco_pipe.decoding.configs import StatisticalAssessmentConfig, ChanceAssessmentConfig

   # Run two separate experiments with the same CV folds
   result_a = Experiment(config_a).run(X, y, sample_metadata=meta)
   result_b = Experiment(config_b).run(X, y, sample_metadata=meta)

   eval_config = StatisticalAssessmentConfig(
       chance=ChanceAssessmentConfig(
           n_permutations=1000,
           temporal_correction="max_stat",   # for temporal outputs
       ),
       unit_of_inference="sample",
       random_state=42,
   )

   paired_df = run_paired_permutation_assessment(
       result_a, result_b, "model_name", "accuracy", eval_config
   )

.. note::

   The two experiments must have been run with the **same outer CV configuration**
   and the **same subjects**. The function aligns predictions at the ``SampleID``
   level before computing the difference.

---

5. Interpreting Results
========================

.. rubric:: Effect Size

The ``Difference`` column is the primary effect size. A small but significant
difference is not necessarily scientifically meaningful. Always report both the
magnitude and statistical significance.

.. rubric:: Temporal Generalization Comparison

For generalizing decoders, one comparison row is produced per
``(TrainTime, TestTime)`` cell. Apply temporal correction (``max_stat`` or
``fdr_bh``) to control the family-wise error rate across the matrix.

.. rubric:: Multiple Model Pitfall

If you run ``K`` pairwise comparisons without correction, the expected number of
false positives is ``0.05 × K``. Always apply correction when comparing more than
two models.

---

6. Post-Hoc vs Full-Pipeline Comparison
=========================================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Speed
     - Validity
   * - ``compare_models_paired``
     - Fast (uses existing predictions)
     - Valid if preprocessing did not use the comparison metric during fitting.
   * - ``run_paired_permutation_assessment``
     - Slow (reruns full CV per permutation)
     - Fully valid; recommended for publications.
