.. _decoding-experiment:

===============================
The ``Experiment`` Orchestrator
===============================

``coco_pipe.decoding.Experiment`` is the main entry point for all decoding
experiments. It validates configuration, orchestrates the outer CV loop,
and returns a fully populated ``ExperimentResult``.

---

1. Initialization
==================

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import ClassicalModelConfig, CVConfig

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified", n_splits=5),
   )

   exp = Experiment(config)

At construction time, ``Experiment.__init__`` immediately:

1. Resolves all model specs from ``ESTIMATOR_SPECS``.
2. Validates task/metric/model compatibility (raises ``ValueError`` if any combination is invalid).
3. Propagates the master ``random_state`` to all sub-configs.

---

2. Running an Experiment
==========================

.. code-block:: python

   result = exp.run(
       X,
       y,
       groups=None,                 # or np.ndarray of group labels
       sample_ids=None,             # or array of unique sample identifiers
       sample_metadata=None,        # or dict/DataFrame with Subject, Session, ...
       feature_names=None,          # or list of feature name strings
       time_axis=None,              # or np.ndarray of timepoints for 3D inputs
       observation_level="epoch",   # or "trial", "subject", etc.
       inferential_unit=None,       # auto-inferred from metadata
   )

2.1 ``X`` and ``y``
---------------------

- ``X``: 2D array ``(n_samples, n_features)`` for classical models, or 3D array
  ``(n_samples, n_channels, n_times)`` for temporal estimators.
- ``y``: 1D array ``(n_samples,)`` of class labels (classification) or continuous
  values (regression).

2.2 ``sample_metadata``
-------------------------

A dict or DataFrame with columns for each metadata variable. **Must include
``Subject`` and ``Session``** (capitalized) when the outer CV uses a group key.
Additional columns (e.g., ``Site``, ``Age``) are stored in predictions and splits
for downstream analysis.

.. code-block:: python

   sample_metadata = {
       "Subject": subject_ids,    # unique subject identifiers
       "Session": session_ids,    # recording session identifiers
       "Site":    site_ids,       # optional acquisition site
   }

2.3 ``observation_level``
---------------------------

A string label stored in ``result.meta["observation_level"]``. It describes what
each row of ``X`` represents (``"epoch"``, ``"trial"``, ``"subject"``, etc.).
This metadata does not affect fitting but documents the result for downstream
analysis and reporting.

---

3. Per-Fold Pipeline
======================

For each outer CV fold, ``Experiment`` executes the following sequence:

1. **Split**: divide ``X``, ``y``, and metadata into training and test partitions.
2. **Validate fold integrity**: check for degenerate folds (empty partitions,
   single-class training sets for classification).
3. **Build pipeline**: create a ``sklearn.pipeline.Pipeline`` with steps:
   ``scaler → feature_selector → model``. Each step is instantiated fresh for
   this fold.
4. **Wrap with tuning**: if ``TuningConfig.enabled``, wrap the pipeline in
   ``GridSearchCV`` or ``RandomizedSearchCV``.
5. **Fit**: call ``pipeline.fit(X_train, y_train)`` (with groups if required).
6. **Calibrate**: if ``CalibrationConfig.enabled``, wrap in
   ``CalibratedClassifierCV`` and refit with calibration folds.
7. **Score**: compute all requested metrics on ``X_test``.
8. **Extract diagnostics**: feature importances, predictions, timing, warnings.

---

4. Parallel Execution
=======================

.. code-block:: python

   config = ExperimentConfig(
       ...,
       n_jobs=4,    # number of parallel outer CV jobs
   )

   result = Experiment(config).run(X, y)

``n_jobs`` controls the number of parallel outer-fold evaluations via ``joblib``.
For exact reproducibility, use ``n_jobs=1`` (see :ref:`decoding-reproducibility`).

---

5. Save and Load
==================

.. code-block:: python

   # Save result to JSON
   path = result.save("results/my_experiment.json")

   # Load from JSON
   from coco_pipe.decoding.result import ExperimentResult
   loaded = ExperimentResult.load(path)

The result is serialized as a self-contained JSON payload (schema version
``decoding_result_v1``), including the config, metadata, per-fold outputs,
and provenance information.

---

6. Configuration Reference
============================

See :ref:`decoding-configs` for a full listing of all configuration classes.
The most important fields on ``ExperimentConfig``:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``task``
     - ``"classification"`` or ``"regression"``.
   * - ``models``
     - Dict mapping model names to model configs.
   * - ``metrics``
     - List of metric keys (validated against the task and model capabilities).
   * - ``cv``
     - ``CVConfig`` controlling the outer cross-validation loop.
   * - ``tuning``
     - ``TuningConfig`` for hyperparameter search.
   * - ``feature_selection``
     - ``FeatureSelectionConfig`` for filter/wrapper feature selection.
   * - ``calibration``
     - ``CalibrationConfig`` for probability calibration.
   * - ``evaluation``
     - ``StatisticalAssessmentConfig`` for permutation/binomial testing.
   * - ``use_scaler``
     - Whether to prepend a ``StandardScaler`` to the pipeline.
   * - ``n_jobs``
     - Number of parallel outer CV jobs.
   * - ``random_state``
     - Master seed for reproducibility.
   * - ``tag``
     - Descriptive label stored in the result metadata.
