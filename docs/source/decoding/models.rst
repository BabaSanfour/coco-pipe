.. _decoding-models:

================================
Model Registry and Capabilities
================================

All estimators available in ``coco_pipe.decoding`` are registered in
``ESTIMATOR_SPECS`` (``coco_pipe.decoding._specs``). The registry is the
**single source of truth** for estimator class lookup, task support, input kind,
prediction interface, temporal compatibility, importance extraction, and default
hyperparameter search spaces.

---

1. Registry API
================

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
=====================================

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
=========================================

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
