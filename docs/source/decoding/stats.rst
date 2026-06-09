.. _decoding-stats:

==================================
Statistical Assessment Guide
==================================

``coco_pipe.decoding`` cleanly separates **descriptive** CV performance from
**inferential** claims. Descriptive metrics (fold scores, confusion matrices,
curves) are always computed. Inferential statistics are opt-in and require
explicit configuration of ``StatisticalAssessmentConfig``.

---

1. Two Levels of Assessment
=============================

1.1 Descriptive Performance
-----------------------------

Every ``ExperimentResult`` provides fold-level and summary scores without
any statistical testing:

.. code-block:: python

   scores = result.get_detailed_scores()
   print(scores[["Model", "Fold", "Metric", "Score"]])

   # Per-model summary: mean ± std across folds
   summary = scores.groupby(["Model", "Metric"])["Score"].agg(["mean", "std"])

This is the correct starting point for all decoding reports. Always report
fold-level variability alongside the mean.

1.2 Finite-Sample Inferential Assessment
------------------------------------------

Statistical significance claims require a null distribution. ``coco_pipe.decoding``
supports two null-generation strategies:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Method
     - How the null is generated
     - When to use
   * - ``"permutation"``
     - Full outer CV rerun under label permutations.
     - Gold standard. Correct for any preprocessing pipeline.
   * - ``"binomial"``
     - Analytical Clopper-Pearson interval on hard accuracy.
     - Only valid for scalar accuracy, one prediction per independent unit.

---

2. Full-Pipeline Permutation Assessment
=========================================

.. code-block:: python

   from coco_pipe.decoding.configs import (
       ExperimentConfig, CVConfig, ClassicalModelConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
   )

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       evaluation=StatisticalAssessmentConfig(
           enabled=True,
           chance=ChanceAssessmentConfig(
               method="permutation",
               n_permutations=1000,
               unit_of_inference="group_mean",
           ),
       ),
   )

   result = Experiment(config).run(
       X, y,
       sample_metadata={"Subject": subject_ids, "Session": session_ids},
       observation_level="epoch",
   )

   assessment = result.get_statistical_assessment()

The returned DataFrame contains, per model and metric:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - ``Observed``
     - Observed score on the true labels.
   * - ``PValue``
     - Empirical p-value from permutation distribution.
   * - ``CorrectedPValue``
     - Multiple-comparison corrected p-value.
   * - ``Significant``
     - Boolean: ``CorrectedPValue <= alpha``.
   * - ``CILower``, ``CIUpper``
     - Bootstrap CI for the observed score.
   * - ``NullMedian``, ``NullLower``, ``NullUpper``
     - Null distribution percentiles.
   * - ``NPermutations``
     - Number of permutations used.
   * - ``NEff``
     - Effective sample size (number of independent units).
   * - ``Time`` / ``TrainTime`` / ``TestTime``
     - Present only for temporal outputs.

2.1 Label Permutation Inside Groups
-------------------------------------

When ``unit_of_inference="group_mean"`` and ``cv.group_key`` is set, labels are
permuted **across groups**, not within them. This preserves within-subject epoch
correlations in the null distribution, yielding a correctly-calibrated p-value
for the group-level null hypothesis.

.. note::

   Permuting only within subjects (swapping epochs within a subject) would be
   the wrong null for testing whether the model performs above chance at the
   **population** level.

---

3. Binomial Assessment
========================

Binomial testing uses the Clopper-Pearson exact interval. It is valid only when:

- Task is classification.
- Metric is plain ``accuracy``.
- Each independent unit contributes **exactly one** prediction (no aggregation needed).
- An explicit chance level ``p0`` is provided.

.. code-block:: python

   evaluation=StatisticalAssessmentConfig(
       enabled=True,
       chance=ChanceAssessmentConfig(
           method="binomial",
           p0=0.5,     # chance level for binary classification
       ),
       confidence_intervals=ConfidenceIntervalConfig(
           method="clopper_pearson",  # or "wilson"
           alpha=0.05,
       ),
   )

The test statistic is:

.. math::

   p = 1 - F(k - 1; n, p_0)

where :math:`F` is the binomial CDF, :math:`k` is the number of correct
predictions, and :math:`n` is the number of independent observations.

---

4. Bootstrap Confidence Intervals
===================================

Confidence intervals for any metric can be computed independently of the
permutation test, using non-parametric bootstrap over independent units:

.. code-block:: python

   ci = result.get_bootstrap_confidence_intervals(
       metric="accuracy",
       unit="Subject",    # or "Session", "sample", etc.
       n_bootstraps=2000,
       ci=0.95,
   )

Bootstrap CI is also automatically included in the permutation assessment output
(``CILower``, ``CIUpper`` columns).

---

5. Temporal Correction Methods
================================

For sliding/generalizing decoders, one p-value per timepoint must be corrected
for multiple comparisons. Set ``temporal_correction`` in ``ChanceAssessmentConfig``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Description
   * - ``"max_stat"``
     - Permutation Max-Stat (default). FWER control. Uses the global maximum
       of the permutation null at each timepoint. Recommended for temporal data
       with moderate-to-high positive correlation between timepoints.
   * - ``"fdr_bh"``
     - Benjamini-Hochberg FDR. Controls the expected proportion of false
       discoveries. More powerful than Max-Stat but weaker guarantees.
   * - ``"fdr_by"``
     - Benjamini-Yekutieli FDR. Valid under arbitrary dependence. More
       conservative than BH.
   * - ``"none"``
     - No correction. For exploratory analysis only.

---

6. Lightweight Post-Hoc Diagnostics
======================================

For quick exploratory inspection without rerunning training:

.. code-block:: python

   # Lightweight label permutation over fixed predictions (fast but biased)
   null = result.get_statistical_assessment(lightweight=True, metric="accuracy")

   # Direct post-hoc permutation (bypasses full retraining)
   from coco_pipe.decoding.stats import assess_post_hoc_permutation
   posthoc = assess_post_hoc_permutation(result.raw["lr"], metric="accuracy", n_permutations=500)

.. warning::

   Post-hoc permutations that shuffle labels over **fixed** predictions do not
   account for preprocessing, feature selection, or hyperparameter search. They
   underestimate the null and can produce overly optimistic p-values if any of
   these steps used the labels. Use ``method="permutation"`` (full-pipeline) for
   any claim of statistical significance in publications.

---

7. Paired Model Comparison
============================

See :ref:`decoding-model-comparison` for a full guide. Quick reference:

.. code-block:: python

   # Paired permutation test: does model A outperform model B?
   paired = result.compare_models_paired("lr", "svm", metric="accuracy")
   print(paired[["Difference", "PValue", "Significant"]])

   # Full-pipeline paired assessment across two result objects
   from coco_pipe.decoding.stats import run_paired_permutation_assessment
   df = run_paired_permutation_assessment(
       result_a, result_b, "lr", "accuracy", config=eval_config
   )

---

8. Unit of Inference Options
==============================

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Value
     - Aggregation behavior
   * - ``"sample"``
     - No aggregation. Each prediction row is treated as independent.
   * - ``"group_mean"``
     - Average probabilities per group, then classify. Recommended for epoch-level EEG.
   * - ``"group_majority"``
     - Majority vote of hard labels per group.
   * - ``"custom"``
     - Aggregate by a named column in ``sample_metadata``.
