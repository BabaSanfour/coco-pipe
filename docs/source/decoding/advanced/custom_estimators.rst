.. _decoding-custom-estimators:

============================
Custom Estimators
============================

``coco_pipe.decoding`` is extensible. You can register any scikit-learn-compatible
estimator with the registry to enable capability contracts, metric compatibility
checks, and diagnostic reporting for your custom model.

---

1. Protocol Requirements
=========================

Any custom estimator used in ``coco_pipe.decoding`` must implement the
``DecoderEstimator`` protocol:

.. code-block:: python

   from coco_pipe.decoding.interfaces import DecoderEstimator

   class MyCustomClassifier:
       def fit(self, X, y=None, **fit_params):
           # ... training logic
           return self

       def predict(self, X):
           # ... inference logic
           return y_pred

       def get_params(self, deep=True):
           return {}

       def set_params(self, **params):
           return self

   assert isinstance(MyCustomClassifier(), DecoderEstimator)  # runtime check

For models that provide probability estimates, also implement ``predict_proba``.
For neural models with training diagnostics, implement the ``NeuralTrainable``
protocol (see :ref:`decoding-foundation-models`).

---

2. Registering an Estimator
=============================

Register your estimator in ``ESTIMATOR_SPECS`` so it is discoverable by the
capability checking system:

.. code-block:: python

   from coco_pipe.decoding._specs import ESTIMATOR_SPECS, EstimatorSpec

   ESTIMATOR_SPECS["MyCustomClassifier"] = EstimatorSpec(
       name="MyCustomClassifier",
       import_path="mypackage.models.MyCustomClassifier",
       family="classical",
       task=["classification"],
       input_kinds=["tabular_2d"],
       response_methods=["predict", "predict_proba"],
       supports_feature_selection=True,
       supports_calibration=True,
       importance_attr="coef_",    # or "feature_importances_"
   )

2.1 EstimatorSpec Fields
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Description
   * - ``name``
     - Must match the registry key and the class name.
   * - ``import_path``
     - Full dotted import path to the class.
   * - ``family``
     - ``"classical"`` | ``"tree_ensemble"`` | ``"boosting"`` | ``"neural"`` | ``"dummy"``
   * - ``task``
     - List of ``"classification"`` and/or ``"regression"``.
   * - ``input_kinds``
     - ``["tabular_2d"]`` for 2D arrays; ``["epoched_3d"]`` for 3D temporal.
   * - ``response_methods``
     - Which prediction interfaces the model provides.
   * - ``supports_feature_selection``
     - Whether it works inside a SelectKBest / SFS pipeline.
   * - ``supports_calibration``
     - Whether CalibratedClassifierCV can wrap it.
   * - ``importance_attr``
     - Attribute name for feature importances (e.g. ``"coef_"``, ``"feature_importances_"``).
   * - ``default_search_space``
     - Optional dict of hyperparameter name → list of values for tuning.

---

3. Using the Custom Estimator in an Experiment
===============================================

After registration, use the estimator name in ``ClassicalModelConfig``:

.. code-block:: python

   from coco_pipe.decoding.configs import ExperimentConfig, ClassicalModelConfig, CVConfig

   config = ExperimentConfig(
       task="classification",
       models={
           "my_model": ClassicalModelConfig(
               estimator="MyCustomClassifier",
               params={"my_param": 1.0},
           )
       },
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified", n_splits=5),
   )

   result = Experiment(config).run(X, y)

---

4. Custom Feature Importances
================================

If your custom model exposes feature importances differently, override the
importance extraction logic by adding a callable to ``importance_attr``:

.. code-block:: python

   class MyCustomClassifier:
       def fit(self, X, y=None):
           self.importances_ = self._compute_importances(X, y)
           return self

       def _compute_importances(self, X, y):
           # custom importance logic
           return np.ones(X.shape[1])

   ESTIMATOR_SPECS["MyCustomClassifier"] = EstimatorSpec(
       ...,
       importance_attr="importances_",   # will be read after .fit()
   )

``coco_pipe.decoding._engine.extract_feature_importances`` reads
``importance_attr`` via ``getattr(fitted_model, importance_attr)`` and handles
1D arrays (feature importance vectors) and 2D arrays (class-specific weights
like ``coef_``).
