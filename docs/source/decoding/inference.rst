.. _decoding-inference:

==========================================
Cross-Validation and Statistical Inference
==========================================

.. _decoding-cv:

Cross-Validation Strategies Guide
=================================

The cross-validation strategy is the most consequential choice in a decoding
experiment. It determines whether the performance estimate is statistically valid,
whether group independence is preserved, and whether the inner model-selection
loops can correctly inherit the outer splitting logic.

---

1. Available Strategies
-----------------------

All strategies are configured via ``CVConfig.strategy``:

.. list-table::
   :header-rows: 1
   :widths: 30 25 20 25

   * - Strategy
     - Group-aware
     - Use case
     - scikit-learn equivalent
   * - ``"stratified"``
     - ❌
     - Balanced class folds (classification)
     - ``StratifiedKFold``
   * - ``"kfold"``
     - ❌
     - Regression, or classification without imbalance
     - ``KFold``
   * - ``"group_kfold"``
     - ✅
     - K folds, subjects exclusive to test
     - ``GroupKFold``
   * - ``"stratified_group_kfold"``
     - ✅
     - K folds, class-balanced, subjects exclusive
     - ``StratifiedGroupKFold``
   * - ``"leave_one_group_out"``
     - ✅
     - Leave-one-subject-out (LOSO)
     - ``LeaveOneGroupOut``
   * - ``"leave_p_out"``
     - ✅
     - Leave-P-subjects-out
     - ``LeavePGroupsOut``
   * - ``"timeseries"``
     - ❌
     - Ordered splits for time-series data
     - ``TimeSeriesSplit``
   * - ``"split"``
     - ❌
     - Single train/test holdout
     - Custom ``ShuffleSplit``
   * - ``"group_shuffle_split"``
     - ✅
     - Randomized group-aware train/test splits
     - ``GroupShuffleSplit``

Set ``CVConfig.auto_reduce_n_splits=True`` to let the splitter shrink
``n_splits`` automatically when a group-aware strategy cannot honor the
requested number of folds (e.g. too few groups).

---

2. Group-Based Strategies
-------------------------

2.1 When Groups Are Required
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use a group-based strategy whenever your data contains **multiple observations
per independent unit** (e.g., multiple epochs per subject). Failure to do so
means the model trains and tests on data from the **same subjects**, producing
inflated accuracy estimates.

Provide groups via sample metadata:

.. code-block:: python

   from coco_pipe.decoding.configs import CVConfig

   cv = CVConfig(
       strategy="stratified_group_kfold",
       n_splits=5,
       group_key="Subject",    # must match a column in sample_metadata
   )

   result = Experiment(config).run(
       X, y,
       sample_metadata={"Subject": subject_ids, "Session": session_ids}
   )

2.2 LOSO (Leave-One-Subject-Out)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LOSO leaves all epochs from one subject out of training per fold. It is the most
conservative and the most clinically-relevant evaluation strategy, but it has
as many folds as subjects, which can be computationally expensive.

.. code-block:: python

   cv = CVConfig(strategy="leave_one_group_out", group_key="Subject")

.. note::

   ``leave_one_group_out`` does not accept an ``n_splits`` parameter. The number
   of folds equals the number of unique subjects.

2.3 Leave-P-Subjects-Out
~~~~~~~~~~~~~~~~~~~~~~~~

Leave-P-groups-out leaves ``p`` subjects out per fold. More powerful than LOSO
when ``p > 1``, but substantially increases the number of folds.

.. code-block:: python

   cv = CVConfig(strategy="leave_p_out", n_splits=2, group_key="Subject")  # leaves 2 groups out per fold

---

3. Group Propagation to Inner CV
--------------------------------

When the outer CV is group-based, ``coco_pipe.decoding`` automatically propagates
group constraints to all inner CV loops:

- **Hyperparameter tuning** (``TuningConfig``): uses a group-based inner CV by default.
- **Sequential Feature Selection** (``FeatureSelectionConfig(method="sfs")``): uses a
  group-based inner CV by default.
- **Calibration** (``CalibrationConfig``): uses a group-based inner calibration
  split by default.

Overriding this requires explicitly setting ``allow_nongroup_inner_cv=True`` on
the relevant config object:

.. code-block:: python

   from coco_pipe.decoding.configs import TuningConfig, CVConfig

   tuning = TuningConfig(
       enabled=True,
       cv=CVConfig(strategy="stratified", n_splits=3),
       allow_nongroup_inner_cv=True,  # explicit acknowledgement of leakage risk
   )

---

4. Stratified Strategies
------------------------

Stratified strategies ensure that class proportions are approximately equal
across folds. This is important for imbalanced datasets where some folds might
contain no minority-class examples.

- ``"stratified"`` and ``"stratified_group_kfold"`` are only valid for classification.
- For regression tasks, use ``"kfold"`` or group-based strategies.

.. code-block:: python

   cv = CVConfig(
       strategy="stratified_group_kfold",
       n_splits=5,
       group_key="Subject",
       random_state=42,    # reproducibility for stratification
   )

---

5. Holdout Split
----------------

For large datasets or as a quick sanity check, a single train/test holdout
avoids the overhead of K outer folds:

.. code-block:: python

   cv = CVConfig(
       strategy="split",
       test_size=0.2,
       stratify=True,     # stratified split for classification
       random_state=42,
   )

The ``n_splits`` field is ignored for ``"split"`` — it always produces exactly
one fold.

---

6. Time Series Split
--------------------

For EEG/MEG data that is **not epoched** (e.g., continuous recordings), or for
temporal regression, use ``"timeseries"``:

.. code-block:: python

   cv = CVConfig(
       strategy="timeseries",
       n_splits=5,
       test_size=0.2,     # optional, overrides sklearn default
   )

``TimeSeriesSplit`` ensures that training data always comes **before** test data
in time, preventing future data from leaking into the model.

---

7. Random State and Reproducibility
-----------------------------------

``CVConfig.random_state`` seeds the splitter. For full reproducibility, also
set ``ExperimentConfig.random_state``, which propagates derived seeds to the CV,
tuning, feature selection, and calibration configs via a ``SeedSequence``.

.. code-block:: python

   config = ExperimentConfig(
       task="classification",
       models={"lr": LogisticRegressionConfig()},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified", n_splits=5),
       random_state=42,    # propagated to all sub-components
   )

See :ref:`decoding-reproducibility` for the full seed propagation architecture.

.. _decoding-stats:

Statistical Assessment Guide
============================

``coco_pipe.decoding`` cleanly separates **descriptive** CV performance from
**inferential** claims. Descriptive metrics (fold scores, confusion matrices,
curves) are always computed. Inferential statistics are opt-in and require
explicit configuration of ``StatisticalAssessmentConfig``.

---

1. Two Levels of Assessment
---------------------------

1.1 Descriptive Performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``ExperimentResult`` provides fold-level and summary scores without
any statistical testing:

.. code-block:: python

   scores = result.get_detailed_scores()
   print(scores[["Model", "Fold", "Metric", "Value"]])

   # Per-model summary: mean ± std across folds
   summary = scores.groupby(["Model", "Metric"])["Value"].agg(["mean", "std"])

This is the correct starting point for all decoding reports. Always report
fold-level variability alongside the mean.

1.2 Finite-Sample Inferential Assessment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
---------------------------------------

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
       statistical_assessment=StatisticalAssessmentConfig(
           enabled=True,
           unit_of_inference="group_mean",
           chance=ChanceAssessmentConfig(
               method="permutation",
               n_permutations=1000,
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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
----------------------

Binomial testing uses the Clopper-Pearson exact interval. It is valid only when:

- Task is classification.
- Metric is plain ``accuracy``.
- Each independent unit contributes **exactly one** prediction (no aggregation needed).
- An explicit chance level ``p0`` is provided.

.. code-block:: python

   statistical_assessment=StatisticalAssessmentConfig(
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
---------------------------------

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
------------------------------

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
   * - ``"none"``
     - No correction. For exploratory analysis only.

---

6. Lightweight Post-Hoc Diagnostics
-----------------------------------

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
--------------------------

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
----------------------------

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

.. _decoding-model-comparison:

Model Comparison
================

After running a decoding experiment with multiple models, ``coco_pipe.decoding``
provides rigorous paired statistical tests to determine whether observed
performance differences are beyond chance. All comparison methods use
within-subject label swaps to control for subject-specific baseline variance.

---

1. Why Paired Tests?
--------------------

Independent-sample tests compare two models assuming the samples are drawn
independently. In a within-subject decoding design, the **same subjects** appear
in both models' test folds, making the samples positively correlated. A paired
test exploits this correlation to achieve higher statistical power.

**Paired permutation test**: randomly swap model assignments within each
independent unit (subject) and recompute the observed difference. The resulting
null distribution represents the expected difference under no true effect.

---

2. Quick Paired Comparison
--------------------------

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
----------------------------

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
----------------------------------------

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
-----------------------

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
---------------------------------------

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
