.. _decoding-cv:

=================================
Cross-Validation Strategies Guide
=================================

The cross-validation strategy is the most consequential choice in a decoding
experiment. It determines whether the performance estimate is statistically valid,
whether group independence is preserved, and whether the inner model-selection
loops can correctly inherit the outer splitting logic.

---

1. Available Strategies
========================

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

---

2. Group-Based Strategies
===========================

2.1 When Groups Are Required
------------------------------

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
----------------------------------

LOSO leaves all epochs from one subject out of training per fold. It is the most
conservative and the most clinically-relevant evaluation strategy, but it has
as many folds as subjects, which can be computationally expensive.

.. code-block:: python

   cv = CVConfig(strategy="leave_one_group_out", group_key="Subject")

.. note::

   ``leave_one_group_out`` does not accept an ``n_splits`` parameter. The number
   of folds equals the number of unique subjects.

2.3 Leave-P-Subjects-Out
--------------------------

Leave-P-groups-out leaves ``p`` subjects out per fold. More powerful than LOSO
when ``p > 1``, but substantially increases the number of folds.

.. code-block:: python

   cv = CVConfig(strategy="leave_p_out", n_groups=2, group_key="Subject")

---

3. Group Propagation to Inner CV
==================================

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
==========================

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
=================

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
======================

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
======================================

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
