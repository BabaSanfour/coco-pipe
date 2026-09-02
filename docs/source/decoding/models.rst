.. _decoding-models:

============================
Models and Temporal Decoding
============================

Model Registry and Capabilities
===============================

All estimators available in ``coco_pipe.decoding`` are registered in
``ESTIMATOR_SPECS`` (``coco_pipe.decoding._specs``). The registry is the
**single source of truth** for estimator class lookup, task support, input kind,
prediction interface, temporal compatibility, importance extraction, and default
hyperparameter search spaces.

---

1. Registry API
---------------

.. code-block:: python

   from coco_pipe.decoding import (
       get_estimator_spec,
       list_estimator_specs,
       resolve_estimator_capabilities,
   )

   # Inspect a specific estimator
   spec = get_estimator_spec("LogisticRegression")
   print(spec.name)              # "LogisticRegression"
   print(spec.family)            # "classical"
   print(spec.task)              # ["classification"]
   print(spec.input_kinds)       # ["tabular_2d", "embedding_2d"]
   print(spec.supports_calibration)  # True
   print(spec.supports_feature_selection)  # True

   # List all estimators
   all_specs = list_estimator_specs()

   # Get capability metadata for model selection
   caps = resolve_estimator_capabilities("SVC")
   print(caps.has_response("predict_proba"))  # False (LinearSVC)

---

2. Auto-Generated Capability Table
----------------------------------

The following table is generated automatically at documentation build time
from ``ESTIMATOR_SPECS`` in ``coco_pipe.decoding._specs``. It reflects the
exact state of the registry — no manual maintenance required.

.. rubric:: Classification Estimators

.. capability-table::
   :task: classification

.. rubric:: Regression Estimators

.. capability-table::
   :task: regression

.. rubric:: All Estimators (including temporal and foundation models)

.. capability-table::
   :task: all
   :show-search-space:


The registry blocks incompatible combinations at configuration time:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Blocked combination
     - Error message
   * - ``log_loss`` with ``LinearSVC`` (no calibration)
     - ``"Metric requires probabilities, but model doesn't provide them"``
   * - ``roc_auc`` with a model that lacks both ``predict_proba`` and ``decision_function``
     - ``"Metric requires probabilities or decision scores, but model provides neither"``
   * - Feature selection on 3D temporal input
     - ``"Feature selection is only valid for classical 2D tabular inputs"``
   * - Regression model with classification task
     - ``"Model does not support task 'classification'"``

These checks prevent wasted compute time and silent errors in downstream statistics.

---

6. Default Hyperparameter Search Spaces
---------------------------------------

Each ``EstimatorSpec`` includes a ``default_search_space`` dict for
``GridSearchCV``/``RandomizedSearchCV``. These are reasonable starting points:

.. code-block:: python

   spec = get_estimator_spec("LogisticRegression")
   print(spec.default_search_space)
   # {"C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

When a ``TuningConfig.grids`` key is provided, it overrides the default. The raw
grid keys are automatically prefixed with ``clf__`` to match the pipeline step
name (e.g., ``{"C": [...]}`` becomes ``{"clf__C": [...]}``) unless the key
already contains ``__``.

.. _decoding-temporal:

Temporal Decoding
=================

Temporal decoding applies a classifier or regressor independently at each
timepoint (or pair of timepoints) of an EEG/MEG epoch. The input array has
shape ``(n_samples, n_channels, n_times)`` — sometimes called a 3D or epochs
array. MNE meta-estimators (``SlidingEstimator``, ``GeneralizingEstimator``)
orchestrate the per-timepoint fitting inside the ``coco_pipe.decoding`` outer
CV loop.

---

1. Data Format Requirements
---------------------------

Temporal decoding expects 3D arrays:

- **Axis 0**: samples (trials / epochs).
- **Axis 1**: channels or features.
- **Axis 2**: timepoints.

.. code-block:: python

   X.shape  # (n_epochs, n_channels, n_times)
   y.shape  # (n_epochs,)

Pass the physical time axis (in seconds) for meaningful temporal labels in outputs:

.. code-block:: python

   time_axis = epochs.times  # NumPy array, shape (n_times,)

When omitted, integer timepoint indices are used.

---

2. Sliding Estimator
--------------------

A ``SlidingEstimator`` fits one independent model per timepoint. It is
equivalent to looping over the time axis, extracting each timepoint's
channel-space snapshot, and training a model on it.

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, TemporalDecoderConfig, CVConfig
   )

   config = ExperimentConfig(
       task="classification",
       models={
           "sliding_lr": TemporalDecoderConfig(
               wrapper="sliding",
               base=ClassicalModelConfig(
                   estimator="LogisticRegression",
                   params={"max_iter": 200},
               ),
               scoring="accuracy",
               n_jobs=-1,
           )
       },
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

   result = Experiment(config).run(
       X_epochs,                    # shape (n_epochs, n_channels, n_times)
       y,
       sample_metadata={"Subject": subject_ids, "Session": session_ids},
       time_axis=epochs.times,
   )

2.1 Outputs
~~~~~~~~~~~

.. code-block:: python

   # Score curve: one score per timepoint per fold
   scores = result.get_detailed_scores()
   # columns: Model, Fold, Metric, Value, Time

   # Summary over folds: mean ± std per timepoint
   temporal = result.get_temporal_score_summary()

   # Long-form predictions with Time column
   preds = result.get_predictions()

2.2 Plotting
~~~~~~~~~~~~

.. code-block:: python

   from coco_pipe.viz import plot_temporal_score_curve

   fig = plot_temporal_score_curve(result, metric="accuracy")
   fig.savefig("sliding_accuracy.png")

---

3. Generalizing Estimator (Temporal Generalization)
---------------------------------------------------

A ``GeneralizingEstimator`` fits one model per training timepoint and evaluates
it at **every** test timepoint. The result is a
``(n_train_times, n_test_times)`` matrix of scores.

.. code-block:: python

   config = ExperimentConfig(
       task="classification",
       models={
           "generalizing_lr": TemporalDecoderConfig(
               wrapper="generalizing",
               base=ClassicalModelConfig(
                   estimator="LogisticRegression",
                   params={"max_iter": 200},
               ),
               n_jobs=-1,
           )
       },
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
   )

   result = Experiment(config).run(
       X_epochs, y,
       sample_metadata={"Subject": subject_ids, "Session": session_ids},
       time_axis=epochs.times,
   )

3.1 Outputs
~~~~~~~~~~~

.. code-block:: python

   # Long-form predictions with TrainTime and TestTime columns
   preds = result.get_predictions()

   # Score matrix: mean over folds, shape (n_train_times, n_test_times)
   matrix = result.get_generalization_matrix("accuracy")
   # or long form: matrix = result.get_generalization_matrix("accuracy", long=True)

3.2 Scientific Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Main diagonal** (``TrainTime == TestTime``): equivalent to the sliding
  decoder result.
- **Off-diagonal generalization**: a classifier trained at time :math:`t_1`
  tested at :math:`t_2`. High off-diagonal scores indicate a **sustained neural
  representation** whose format is preserved across the generalizing time window.
- **Asymmetric generalization**: training at late times generalizes to early
  times but not vice versa — suggests temporal ordering of information flow.

3.3 Plotting
~~~~~~~~~~~~

.. code-block:: python

   from coco_pipe.viz import plot_temporal_generalization_matrix

   fig = plot_temporal_generalization_matrix(result, metric="accuracy")
   fig.savefig("generalization_matrix.png")

---

4. Statistical Assessment for Temporal Outputs
----------------------------------------------

Each timepoint (or train-time/test-time cell) produces an independent score.
Multiple comparison correction is required.

.. code-block:: python

   from coco_pipe.decoding.configs import StatisticalAssessmentConfig, ChanceAssessmentConfig

   config = ExperimentConfig(
       ...,
       statistical_assessment=StatisticalAssessmentConfig(
           enabled=True,
           unit_of_inference="group_mean",
           chance=ChanceAssessmentConfig(
               method="permutation",
               n_permutations=1000,
               temporal_correction="max_stat",    # FWER control over timepoints
           ),
       ),
   )

   result = Experiment(config).run(X_epochs, y, ...)
   assessment = result.get_statistical_assessment()
   # One row per (Model, Metric, Time) or (Model, Metric, TrainTime, TestTime)

The ``CorrectedPValue`` and ``Significant`` columns reflect the chosen
temporal correction. See :ref:`decoding-stats` for correction method details.

---

5. Feature Importances in Temporal Models
-----------------------------------------

Feature importances from temporal models are averaged across timepoints and
folds. Each timepoint contributes one importance vector (of length ``n_channels``):

.. code-block:: python

   importances = result.get_feature_importances()
   # columns: FeatureName, MeanImportance, StdImportance, Time

Temporal importance patterns can reveal which channels drive decoding at each
timepoint — a form of spatiotemporal source localization.

---

6. Compatibility Notes
----------------------

- Feature selection (``k_best``, ``sfs``) is **not compatible** with 3D temporal
  inputs. The registry blocks this combination at validation time.
- Standard scalers are not applied to 3D inputs (they expect 2D arrays). Use
  channel-wise normalization within the ``base`` estimator's pipeline if needed.
- ``n_jobs`` inside ``TemporalDecoderConfig`` controls parallelism for the
  per-timepoint model fitting, separate from the outer CV ``n_jobs``.
