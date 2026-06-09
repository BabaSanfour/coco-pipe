.. _decoding-concepts:

==================================
Scientific Concepts and Principles
==================================

This page explains the foundational principles that govern every design decision
in ``coco_pipe.decoding``. Understanding these concepts is essential for
interpreting results correctly and avoiding common pitfalls in brain decoding.

---

1. Cross-Validation and Data Leakage
=====================================

1.1 Why Outer-Only Scoring Is Insufficient
-------------------------------------------

A decoding score is an estimate of *generalization performance* — how well a
trained model predicts labels from **unseen** brain data. The critical word is
"unseen". If any information from the test partition is visible during training
(even implicitly, through preprocessing), the score is optimistically biased.

In practice, leakage occurs when:

- A scaler is fit on the **whole dataset** then applied fold-locally.
- A feature selector's statistics are computed on the full feature matrix.
- A hyperparameter is tuned using a validation set that overlaps with the test set.
- A class-label encoder is fit on all samples before splitting.

``coco_pipe.decoding`` prevents all of these by construction: every preprocessing
transformer is created inside a scikit-learn ``Pipeline`` that is fit **only on
the training partition** of each outer fold. The test partition is never touched
during training.

1.2 The Outer Cross-Validation Loop
-------------------------------------

The outer CV loop is controlled by ``ExperimentConfig.cv``. It defines the
*evaluation splits*. For each fold:

1. ``X_train, y_train`` → fit scaler, feature selector, hyperparameter search,
   calibration, and the final model.
2. ``X_test, y_test`` → predict and score using the fold-trained pipeline.
3. Fold scores, predictions, importances, and diagnostics are stored.

The final score (e.g., ``get_detailed_scores()``) is the **average over outer
folds**. It is an unbiased estimate of generalization performance — provided
independence is respected (see section 2).

1.3 Inner CV Loops
-------------------

When hyperparameter tuning (``TuningConfig``) or Sequential Feature Selection
(``FeatureSelectionConfig(method='sfs')``) is enabled, an **inner** CV loop
operates on the training partition of each outer fold. This inner loop selects
the best model configuration without access to the test set.

When the outer CV is group-based (e.g., ``group_kfold``), the inner CV is
**automatically made group-based** as well. Overriding this requires setting
``allow_nongroup_inner_cv=True`` and explicitly acknowledges the data-leakage
trade-off.

.. warning::

   Mixing group-based outer CV with non-group inner CV can cause test-set
   group information to leak into model selection, inflating performance.
   Always use matching group strategies for inner and outer CV when subjects
   must remain exclusive to test folds.

---

2. Independence and the Unit of Inference
==========================================

2.1 Pseudoreplication in Neural Data
--------------------------------------

EEG and MEG experiments typically produce many epochs per subject. If a model
is trained and tested on **epochs** from the same subject, the test scores are
not independent. Each subject's neural patterns are correlated across epochs,
so the effective sample size for statistical inference is the **number of
subjects**, not the number of epochs.

Using epochs as the unit of inference inflates degrees of freedom and produces
incorrect p-values. This is called *pseudoreplication*.

2.2 Group-Based CV
--------------------

The correct solution is to ensure all epochs from a given subject belong
**exclusively** to either the training set or the test set — never both. This
is achieved with ``CVConfig(strategy="group_kfold", group_key="Subject")``.

``coco_pipe.decoding`` accepts two equivalent ways to specify groups:

- ``sample_metadata={"Subject": subject_ids}`` with ``cv.group_key="Subject"``
  (recommended — keeps metadata tidy and allows downstream subject-level analysis).
- ``groups=subject_ids`` (compatibility alias — binds groups directly to the
  splitter).

2.3 Subject-Level Aggregation for Statistical Tests
------------------------------------------------------

Even with group-based CV, the **predictions** stored after each fold are
epoch-level. Before performing a statistical test, predictions must be aggregated
to the independent unit (subjects) to restore correct degrees of freedom.

``coco_pipe.decoding.stats.aggregate_predictions_for_inference()`` handles this:

- ``unit_of_inference="group_mean"``: soft-vote by averaging subject class
  probabilities across epochs, then hard-classify.
- ``unit_of_inference="group_majority"``: hard-vote by majority of epoch labels.
- ``unit_of_inference="sample"``: no aggregation (correct when each row is
  already an independent subject).

The statistical assessment machinery uses this aggregation automatically:

.. code-block:: python

   from coco_pipe.decoding.configs import StatisticalAssessmentConfig, ChanceAssessmentConfig

   eval_cfg = StatisticalAssessmentConfig(
       enabled=True,
       chance=ChanceAssessmentConfig(
           method="permutation",
           n_permutations=1000,
           unit_of_inference="group_mean",
       ),
   )

---

3. Full-Pipeline Permutation Testing
======================================

3.1 Why "Post-Hoc" Permutations Are Biased
--------------------------------------------

The easiest permutation test shuffles labels and scores the **already-fitted**
model's predictions. This is fast but biased: it does not reshuffle labels
during hyperparameter search, feature selection, or calibration. If any of
these steps use the labels (which they all do), the null distribution is too
narrow.

3.2 The Correct Null: Full-Pipeline Permutation
-------------------------------------------------

The correct null distribution is obtained by rerunning the **complete** training
pipeline — scaler, feature selector, inner CV, hyperparameter search, calibration,
and the final model fit — on permuted labels. This is what
``ChanceAssessmentConfig(method="permutation")`` does.

Each permutation:

1. Shuffles ``y`` within each group (or globally for ``unit="sample"``).
2. Reruns the complete outer CV with the shuffled labels.
3. Aggregates the permuted predictions to the unit of inference.
4. Scores the aggregated permuted predictions.

The observed score is then compared against this null distribution:

.. math::

   p = \frac{\#\{\text{null scores} \geq \text{observed score}\} + 1}{B + 1}

where :math:`B` is the number of permutations.

3.3 Multiple Comparison Correction for Temporal Data
------------------------------------------------------

For sliding/generalizing temporal decoders, one p-value is produced per
timepoint. Multiple testing correction is required. ``coco_pipe.decoding``
supports:

- ``temporal_correction="max_stat"`` (default): permutation-based Max-Stat
  correction. The null at each timepoint is the *global maximum* of the
  permutation distribution, yielding family-wise error rate (FWER) control.
  Recommended for temporal decoding with moderate-to-high correlations.
- ``temporal_correction="fdr_bh"``: Benjamini-Hochberg FDR control.
- ``temporal_correction="fdr_by"``: Benjamini-Yekutieli FDR control (more
  conservative, valid under positive dependence).
- ``temporal_correction="none"``: no correction (exploratory use only).

.. math::

   p_t^{\text{max\_stat}} = \frac{\#\{B_b : \max_{t'} s_b(t') \geq s(t)\} + 1}{B + 1}

where :math:`s(t)` is the observed score at time :math:`t` and :math:`s_b(t')`
is the permuted score at any timepoint.

---

4. Probability Calibration
============================

A classifier is *calibrated* if its predicted probability of class 1 matches
the empirical fraction of class-1 samples at that probability level. Poor
calibration does not affect accuracy but matters for:

- Log-loss and Brier score interpretation.
- Clinical decision thresholds.
- Ensemble averaging across models.

``coco_pipe.decoding`` supports ``sklearn.calibration.CalibratedClassifierCV``
inside the training path. The calibration estimator uses **disjoint inner folds**
within each outer training partition, so the test set is never used for
calibration fitting.

Enabling calibration also makes probability metrics (``log_loss``, ``brier_score``)
available for models that do not natively provide ``predict_proba`` (e.g.,
``LinearSVC``).

---

5. Feature Importance and Stability
======================================

Feature importances are extracted per fold (when the fitted model supports them)
and aggregated:

- ``get_feature_importances(fold_level=False)``: mean importance ± std across folds.
- ``get_feature_importances(fold_level=True)``: per-fold importances in long form.
- ``get_feature_stability()``: proportion of folds in which each feature was
  selected (for SFS) or had positive importance.

.. warning::

   Fold-averaged importance is **not** the same as importance computed on the
   full dataset. Because each fold trains on a subset of subjects, the importance
   estimate has higher variance than whole-dataset importance. Always report the
   fold-level standard deviation alongside the mean.

---

6. Temporal Decoding Concepts
================================

6.1 Sliding Decoding
----------------------

A ``SlidingEstimator`` (MNE) fits one independent model per timepoint. Each
model sees the channel-space snapshot at its timepoint across all epochs in the
training fold. The result is a score curve over time.

- *Assumption*: The most discriminative time window is narrow relative to the
  total window length.
- *Strength*: Identifies when (not just whether) neural representations are
  discriminative.

6.2 Generalizing Decoding (Temporal Generalization)
------------------------------------------------------

A ``GeneralizingEstimator`` (MNE) fits one model per training timepoint and
tests each model at **every** test timepoint. The result is a
``(n_train_times, n_test_times)`` matrix of scores.

Off-diagonal entries answer: "Does the representation learned at time :math:`t_1`
generalize to predict the label at time :math:`t_2`?" A diagonal band indicates
a rapidly changing representation; an extended off-diagonal band indicates a
stable neural code.

- *Scientific interpretation*: Off-diagonal generalization is evidence of a
  sustained, format-stable neural representation.
- *Statistical note*: The generalizing matrix has ``n_train × n_test`` cells.
  Temporal correction (Max-Stat) is strongly recommended to control the
  family-wise error rate.
