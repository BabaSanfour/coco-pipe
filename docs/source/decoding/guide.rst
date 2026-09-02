.. _decoding-guide:

================================
Building and Running Experiments
================================

.. _decoding-experiment:

The ``Experiment`` Orchestrator
===============================

``coco_pipe.decoding.Experiment`` is the main entry point for all decoding
experiments. It validates configuration, orchestrates the outer CV loop,
and returns a fully populated ``ExperimentResult``.

---

1. Initialization
-----------------

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
------------------------

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
~~~~~~~~~~~~~~~~~~~

- ``X``: 2D array ``(n_samples, n_features)`` for classical models, or 3D array
  ``(n_samples, n_channels, n_times)`` for temporal estimators.
- ``y``: 1D array ``(n_samples,)`` of class labels (classification) or continuous
  values (regression).

2.2 ``sample_metadata``
~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

A string label stored in ``result.meta["observation_level"]``. It describes what
each row of ``X`` represents (``"epoch"``, ``"trial"``, ``"subject"``, etc.).
This metadata does not affect fitting but documents the result for downstream
analysis and reporting.

---

3. Per-Fold Pipeline
--------------------

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
---------------------

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
----------------

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
--------------------------

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

.. _decoding-configs:

Configuration Reference
=======================

All experiment configuration is declarative and Pydantic-validated. Every
config class uses ``extra="forbid"`` so misspelled or unsupported field names
raise a ``ValidationError`` immediately — before any training starts.

---

1. ``ExperimentConfig``
-----------------------

Top-level configuration for a decoding experiment.

.. code-block:: python

   from coco_pipe.decoding.configs import ExperimentConfig

   config = ExperimentConfig(
       task="classification",          # required: "classification" or "regression"
       models={"lr": ...},             # required: dict of model configs
       metrics=["accuracy"],           # default: task-appropriate defaults
       cv=CVConfig(...),               # default: StratifiedKFold(5)
       tuning=TuningConfig(...),       # default: disabled
       feature_selection=FeatureSelectionConfig(...),  # default: disabled
       reducer=ReducerConfig(...),                     # default: disabled (in-pipeline reduction)
       calibration=CalibrationConfig(...),             # default: disabled
       statistical_assessment=StatisticalAssessmentConfig(...),  # default: disabled
       grids={"lr": {"C": [0.1, 1.0]}},  # hyperparameter grids for tuning
       use_scaler=True,                   # prepend StandardScaler to pipeline
       n_jobs=1,                          # outer CV parallelism
       verbose=False,
       tag="my_experiment",               # descriptive label in result metadata
       random_state=42,
   )

---

2. ``CVConfig``
---------------

Controls the outer cross-validation loop.

.. code-block:: python

   from coco_pipe.decoding.configs import CVConfig

   cv = CVConfig(
       strategy="stratified_group_kfold",
       n_splits=5,               # also the number of groups left out for "leave_p_out"
       group_key="Subject",      # column name in sample_metadata
       test_size=0.2,            # for "split" / "group_shuffle_split" only
       stratify=True,            # for "split" + classification only
       auto_reduce_n_splits=True,  # shrink n_splits if too few groups
       random_state=42,
   )

See :ref:`decoding-cv` for a complete strategy guide.

---

3. ``ClassicalModelConfig``
---------------------------

Configures a classical scikit-learn estimator.

.. code-block:: python

   from coco_pipe.decoding.configs import ClassicalModelConfig

   model = ClassicalModelConfig(
       estimator="LogisticRegression",    # key in ESTIMATOR_SPECS
       params={"C": 1.0, "max_iter": 200},
   )

Short-form aliases are also available for common estimators:

.. code-block:: python

   from coco_pipe.decoding.configs import LogisticRegressionConfig

   model = LogisticRegressionConfig(C=1.0, max_iter=200)

---

4. ``TemporalDecoderConfig``
----------------------------

Wraps a classical base estimator for 3D temporal inputs.

.. code-block:: python

   from coco_pipe.decoding.configs import TemporalDecoderConfig, ClassicalModelConfig

   model = TemporalDecoderConfig(
       wrapper="sliding",          # or "generalizing"
       base=ClassicalModelConfig(estimator="LogisticRegression"),
       scoring="accuracy",
       n_jobs=-1,
   )

Requires ``mne`` as an optional dependency.

---

5. ``TuningConfig``
-------------------

Controls hyperparameter search.

.. code-block:: python

   from coco_pipe.decoding.configs import TuningConfig, CVConfig

   tuning = TuningConfig(
       enabled=True,
       search_type="grid",         # or "random"
       scoring="accuracy",
       n_iter=20,                  # for "random" search only
       n_jobs=1,
       refit=True,
       cv=CVConfig(strategy="stratified", n_splits=3),    # inner CV
       allow_nongroup_inner_cv=False,   # leakage guard
       random_state=42,
   )

---

6. ``FeatureSelectionConfig``
-----------------------------

.. code-block:: python

   from coco_pipe.decoding.configs import FeatureSelectionConfig, CVConfig

   fs = FeatureSelectionConfig(
       enabled=True,
       method="k_best",        # or "sfs"
       n_features=20,
       scoring="accuracy",     # scoring criterion for SFS inner CV
       cv=CVConfig(strategy="stratified", n_splits=3),    # SFS inner CV
       direction="forward",    # for SFS: "forward" or "backward"
       allow_nongroup_inner_cv=False,
   )

---

7. ``CalibrationConfig``
------------------------

Enables probability calibration inside the training path.

.. code-block:: python

   from coco_pipe.decoding.configs import CalibrationConfig, CVConfig

   calibration = CalibrationConfig(
       enabled=True,
       method="sigmoid",       # or "isotonic"
       cv=CVConfig(strategy="stratified", n_splits=3),
       allow_nongroup_inner_cv=False,
   )

---

8. ``StatisticalAssessmentConfig``
----------------------------------

.. code-block:: python

   from coco_pipe.decoding.configs import (
       StatisticalAssessmentConfig, ChanceAssessmentConfig, ConfidenceIntervalConfig
   )

   assessment = StatisticalAssessmentConfig(   # pass as statistical_assessment=assessment
       enabled=True,
       random_state=42,
       unit_of_inference="group_mean",   # "sample", "group_mean", "group_majority", "custom"
       chance=ChanceAssessmentConfig(
           method="permutation",         # or "binomial", "auto"
           n_permutations=1000,
           p0=None,                      # required for "binomial"
           temporal_correction="max_stat",  # "max_stat", "fdr_bh", "none"
           store_null_distribution=False,
       ),
       confidence_intervals=ConfidenceIntervalConfig(
           alpha=0.05,
           method="clopper_pearson",     # or "wilson"
       ),
   )

---

9. Foundation Model Configs
---------------------------

.. code-block:: python

   from coco_pipe.decoding.configs import (
       FoundationEmbeddingModelConfig,
       FrozenBackboneDecoderConfig,
       NeuralFineTuneConfig,
       LoRAConfig,
       QuantizationConfig,
       DeviceConfig,
       CheckpointConfig,
   )

   # Frozen embedding extractor
   embed_cfg = FoundationEmbeddingModelConfig(
       backend="braindecode",      # "auto" (default), "braindecode", "hugging_face"
       model_key="labram",         # a registered model — see list_foundation_models()
       pooling="mean",             # "mean" or "flatten"
       cache_embeddings=True,
       normalize_embeddings=True,
   )

   # Full / parameter-efficient neural fine-tuning
   ft_cfg = NeuralFineTuneConfig(
       backend="hugging_face",
       model_key="reve",
       input_kind="epoched",       # "temporal", "epoched", "tokens"
       train_mode="qlora",         # "full", "frozen", "linear_probe", "lora", "qlora"
       lora=LoRAConfig(r=16, alpha=32),
       quantization=QuantizationConfig(enabled=True, load_in_4bit=True),
       device=DeviceConfig(device="auto", precision="bf16"),  # "fp32", "fp16", "bf16"
       checkpoints=CheckpointConfig(save="best"),             # "none", "best", "last", "all"
   )

Discover available backbones and their capabilities with
:func:`~coco_pipe.decoding.list_foundation_models` and
:func:`~coco_pipe.decoding.get_foundation_model_spec`.

.. _decoding-result:

``ExperimentResult`` API
========================

``ExperimentResult`` is the structured container returned by ``Experiment.run()``.
It provides 20+ accessor methods for tidy-data inspection, diagnostic reporting,
and statistical inference — all without rerunning the experiment.

---

1. Structure
------------

.. code-block:: python

   result.raw     # per-model dict of fold outputs
   result.meta    # environment provenance, task, model names, capabilities
   result.config  # original ExperimentConfig

---

2. Prediction Accessors
-----------------------

.. code-block:: python

   # All out-of-fold predictions in tidy long form
   preds = result.get_predictions()
   # columns: Model, Fold, SampleIndex, SampleID, Group, y_true, y_pred
   # + y_proba_0, y_proba_1, ... (if probabilities available)
   # + Subject, Session, Site (from sample_metadata)
   # + Time (sliding) or TrainTime, TestTime (generalizing)

---

3. Score Accessors
------------------

.. code-block:: python

   # Per-fold, per-metric scores
   scores = result.get_detailed_scores()
   # columns: Model, Fold, Metric, Value, Time (if temporal)

   # Fold-level split information
   splits = result.get_splits(with_metadata=True)

   # Fit/predict/score timing and convergence warnings
   fit_diag = result.get_fit_diagnostics()

---

4. Curve Diagnostics
--------------------

.. code-block:: python

   # ROC curves (binary or one-vs-rest multiclass)
   roc = result.get_roc_curve()
   # columns: Model, Fold, Class, FPR, TPR, Threshold, AUC

   # Precision-recall curves
   pr = result.get_pr_curve()
   # columns: Model, Fold, Class, Precision, Recall, Threshold

   # Calibration (reliability) curves
   cal = result.get_calibration_curve()

   # Probability quality summary (log-loss + Brier per fold)
   prob_diag = result.get_probability_diagnostics()

   # Summary statistics for ROC AUC
   roc_summary = result.get_roc_auc_summary()

   # Summary statistics for PR AUC
   pr_summary = result.get_pr_auc_summary()

---

5. Confusion Matrices
---------------------

.. code-block:: python

   # Per-fold confusion matrices in long form
   cm = result.get_confusion_matrices(normalize=True)
   # columns: Model, Fold, TrueLabel, PredLabel, Count

   # Pooled (over folds) confusion matrix
   pooled_cm = result.get_pooled_confusion_matrix(normalize="true")

---

6. Temporal Accessors
---------------------

.. code-block:: python

   # Score summary per timepoint (sliding only)
   temporal = result.get_temporal_score_summary()
   # columns: Model, Metric, Time, MeanScore, StdScore

   # Generalization matrix: shape (n_train_times, n_test_times)
   matrix = result.get_generalization_matrix("accuracy")
   # or long form:
   matrix_long = result.get_generalization_matrix("accuracy", long=True)

---

7. Statistical Inference
------------------------

.. code-block:: python

   # Full-pipeline or lightweight permutation/binomial assessment
   assessment = result.get_statistical_assessment()

   # Lightweight (fixed-prediction, fast, biased)
   assessment_fast = result.get_statistical_assessment(lightweight=True, metric="accuracy")

   # Bootstrap CI over independent units
   ci = result.get_bootstrap_confidence_intervals(
       metric="accuracy",
       unit="Subject",
       n_bootstraps=2000,
       ci=0.95,
   )

   # Null distribution (if stored via store_null_distribution=True)
   nulls = result.get_statistical_nulls()

---

8. Model Comparison
-------------------

.. code-block:: python

   # Paired permutation test between two models (in-result)
   paired = result.compare_models_paired("lr", "svm", metric="accuracy", unit="Subject")

   # All pairwise comparisons with correction
   all_pairs = result.compare_models(metric="accuracy", correction="fdr_bh")

---

9. Feature Importances
----------------------

.. code-block:: python

   # Mean ± std feature importance across folds
   importances = result.get_feature_importances()
   # columns: FeatureName, MeanImportance, StdImportance

   # Per-fold importances
   fold_imp = result.get_feature_importances(fold_level=True)

   # Ranked importances (descending by mean)
   ranked = result.get_feature_importances(rank=True)

---

10. Feature Selection Accessors
-------------------------------

.. code-block:: python

   # Selected features per fold
   selected = result.get_selected_features(ordered=True)

   # Feature stability: selection rate across folds
   stability = result.get_feature_stability()

   # Per-fold univariate feature scores (k_best only)
   scores = result.get_feature_scores(with_pvalues=True)

---

11. Hyperparameter Tuning
-------------------------

.. code-block:: python

   # Best hyperparameters per fold
   best = result.get_best_params()

   # Full grid search results
   grid = result.get_search_results()

---

12. Model Artifact Metadata
---------------------------

.. code-block:: python

   # Neural model training history, checkpoints, etc.
   artifacts = result.get_model_artifacts()

---

13. Serialization
-----------------

.. code-block:: python

   # Serialize to JSON-compatible payload
   payload = result.to_payload()

   # Save to file
   path = result.save("results/my_result.json")

   # Load from file
   from coco_pipe.decoding.result import ExperimentResult
   loaded = ExperimentResult.load("results/my_result.json")

.. _decoding-metrics:

Metric Registry
===============

All metrics are registered in ``coco_pipe.decoding._metrics.METRIC_REGISTRY``.
Metric/task compatibility is enforced at config validation time — before any
model is trained — preventing silent misuse of classification metrics for
regression tasks (or vice versa).

---

1. Registry API
---------------

.. code-block:: python

   from coco_pipe.decoding._metrics import (
       get_metric_spec,
       get_metric_names,
       get_metric_families,
       get_scorer,
       METRIC_REGISTRY,
   )

   # Inspect a single metric
   spec = get_metric_spec("accuracy")
   print(spec.name)              # "accuracy"
   print(spec.task)              # "classification"
   print(spec.family)            # "label"
   print(spec.response_method)   # "predict"
   print(spec.greater_is_better) # True

   # List all classification metrics in the "threshold_sweep" family
   names = get_metric_names(task="classification", family="threshold_sweep")

   # Get a callable scorer
   scorer = get_scorer("f1")  # sklearn-compatible callable

Each ``MetricSpec`` contains:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``name``
     - ``str``
     - Unique key in the registry.
   * - ``task``
     - ``str``
     - ``"classification"`` or ``"regression"``.
   * - ``scorer``
     - ``Callable``
     - ``(y_true, y_pred) → float``.
   * - ``response_method``
     - ``str``
     - ``"predict"`` | ``"proba"`` | ``"score"`` | ``"proba_or_score"``.
   * - ``family``
     - ``str``
     - Grouping for reporting (see below).
   * - ``greater_is_better``
     - ``bool``
     - Directionality for permutation p-values and Max-Stat correction.

---

2. Classification Metrics
-------------------------

2.1 Label Metrics (``family="label"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Require only ``predict`` output. Work with any classifier.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``accuracy``
     - Fraction of correctly classified samples. Sensitive to class imbalance.
   * - ``balanced_accuracy``
     - Mean recall per class. Recommended over ``accuracy`` for imbalanced data.
   * - ``zero_one_loss``
     - Fraction misclassified. ``1 - accuracy``. ``greater_is_better=False``.
   * - ``hamming_loss``
     - Per-label Hamming loss (fraction of labels incorrectly predicted).

2.2 Confusion-Derived Metrics (``family="confusion"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Derived from the confusion matrix. Require only ``predict``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``f1``
     - Binary F1 score (harmonic mean of precision and recall).
   * - ``f1_macro``
     - Unweighted macro-average F1 across classes.
   * - ``f1_micro``
     - Global precision/recall pooled across classes.
   * - ``precision``
     - Positive predictive value. TP / (TP + FP).
   * - ``recall``
     - Sensitivity / true positive rate. TP / (TP + FN).
   * - ``sensitivity``
     - Synonym for recall. Binary only; raises ``ValueError`` for multiclass.
   * - ``specificity``
     - True negative rate. TN / (TN + FP). Binary only.
   * - ``jaccard``
     - Intersection-over-union for binary labels.
   * - ``matthews_corrcoef``
     - Matthews correlation coefficient. Balanced for all class distributions.
   * - ``cohen_kappa``
     - Agreement corrected for chance. Range [-1, 1].

2.3 Threshold-Sweep Metrics (``family="threshold_sweep"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Require probability or decision scores. Use ``predict_proba`` when available,
``decision_function`` as fallback for binary classifiers.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``roc_auc``
     - Area under the ROC curve (binary OvR). Insensitive to class threshold.
   * - ``roc_auc_ovr_weighted``
     - Macro-weighted one-vs-rest AUC for multiclass.
   * - ``average_precision``
     - Area under the PR curve using sklearn's interpolated AP (binary).
   * - ``pr_auc``
     - Trapezoidal AUC of the precision-recall curve. Preferred over AP when
       positive fraction is small.

2.4 Probability-Score Metrics (``family="score_probability"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Require ``predict_proba``. Enable calibration diagnostics.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``log_loss``
     - Cross-entropy loss. Lower is better (``greater_is_better=False``).
   * - ``brier_score``
     - Mean squared error of probability predictions. Lower is better.

---

3. Regression Metrics (``family="regression"``)
-----------------------------------------------

Require only ``predict`` output.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Metric
     - Description
   * - ``r2``
     - Coefficient of determination. 1.0 is perfect fit; can be negative.
   * - ``neg_mean_squared_error``
     - Negative MSE. Negated so higher = better for optimization consistency.
   * - ``neg_mean_absolute_error``
     - Negative MAE. More robust than MSE to outliers.
   * - ``neg_root_mean_squared_error``
     - Negative RMSE. Same units as the target variable.
   * - ``explained_variance``
     - Proportion of variance explained. Similar to R² but not penalized for bias.

---

4. Compatibility Rules
----------------------

The registry enforces three compatibility checks at ``ExperimentConfig``
validation time:

1. **Task mismatch**: A metric's ``task`` must match ``ExperimentConfig.task``.
2. **Proba requirement**: If ``response_method == "proba"``, the model must
   declare ``predict_proba`` **or** calibration must be enabled.
3. **Score requirement**: If ``response_method == "proba_or_score"``, the model
   must declare ``predict_proba`` **or** ``decision_function``.

These checks fire before any model is trained, producing a clear ``ValueError``
with the specific metric and model name.

---

5. Custom Metrics
-----------------

You can extend the registry for project-specific metrics:

.. code-block:: python

   from coco_pipe.decoding._metrics import METRIC_REGISTRY, MetricSpec
   from sklearn.metrics import top_k_accuracy_score
   from functools import partial

   top2 = partial(top_k_accuracy_score, k=2, labels=[0, 1, 2])
   METRIC_REGISTRY["top2_accuracy"] = MetricSpec(
       name="top2_accuracy",
       task="classification",
       scorer=top2,
       response_method="proba",
       family="label",
       greater_is_better=True,
   )

.. warning::

   Custom metrics are added to the in-process registry only. They are not
   persisted in saved ``ExperimentResult`` payloads and must be re-registered
   in any new Python process that loads existing results.

.. _decoding-feature-selection:

Feature Selection
=================

``coco_pipe.decoding`` supports two feature selection strategies that execute
**inside** each outer CV fold on the training partition only, guaranteeing
zero test-set leakage.

---

1. Filter Selection (``k_best``)
--------------------------------

``SelectKBest`` selects the top-``k`` features based on a univariate statistical
test. It has no inner CV loop. It is fast and well-suited for high-dimensional data
(e.g., many EEG channels/frequency bins) where a quick, interpretable feature
ranking is desired.

.. code-block:: python

   from coco_pipe.decoding.configs import (
       ExperimentConfig, CVConfig, ClassicalModelConfig, FeatureSelectionConfig
   )

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       feature_selection=FeatureSelectionConfig(
           enabled=True,
           method="k_best",
           n_features=20,
           scoring="accuracy",     # optional; defaults to task-appropriate test
       ),
   )

1.1 Score Function
~~~~~~~~~~~~~~~~~~

For classification, the default univariate test is ``f_classif`` (ANOVA F-value).
For regression, it is ``f_regression``. Override via ``feature_selection.scoring``.

1.2 Accessing Feature Scores
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After fitting, retrieve per-fold and per-feature scores:

.. code-block:: python

   feature_scores = result.get_feature_scores()
   # columns: FeatureName, Fold, Score, PValue

   # Mean score across folds
   mean_scores = feature_scores.groupby("FeatureName")["Score"].mean().sort_values(ascending=False)

---

2. Sequential Feature Selection (``sfs``)
-----------------------------------------

``SequentialFeatureSelector`` is a wrapper-based method. It iteratively adds
(forward SFS) or removes (backward SFS) features by evaluating the model's
cross-validated performance on each candidate feature set. Because it uses the
model's predictive performance as the selection criterion, it is more powerful
than filter methods but significantly more expensive.

.. code-block:: python

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["balanced_accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       feature_selection=FeatureSelectionConfig(
           enabled=True,
           method="sfs",
           n_features=10,
           scoring="balanced_accuracy",    # criterion for SFS inner evaluation
           cv=CVConfig(strategy="stratified_group_kfold", n_splits=3, group_key="Subject"),
           direction="forward",            # or "backward"
       ),
   )

2.1 Inner CV for SFS
~~~~~~~~~~~~~~~~~~~~

SFS requires an inner CV loop to evaluate candidate feature sets. When omitted,
``coco_pipe.decoding`` derives the inner SFS CV from:

1. ``tuning.cv`` if tuning is enabled.
2. The outer CV family (group-based if outer is group-based).

When the outer CV is group-based, the SFS inner CV is automatically group-based.
Overriding requires ``allow_nongroup_inner_cv=True``.

2.2 Group-Aware SFS
~~~~~~~~~~~~~~~~~~~

``coco_pipe.decoding`` uses scikit-learn metadata routing to pass the
outer-fold training groups into the SFS inner CV. This requires
``scikit-learn >= 1.6``.

2.3 SFS with Tuning
~~~~~~~~~~~~~~~~~~~

SFS combined with hyperparameter tuning evaluates feature subsets inside the
tuning inner folds. ``coco_pipe.decoding`` uses a ``sklearn.pipeline.Pipeline``
cache to avoid redundant refitting:

.. code-block:: python

   config = ExperimentConfig(
       ...,
       feature_selection=FeatureSelectionConfig(enabled=True, method="sfs", n_features=10),
       tuning=TuningConfig(enabled=True, scoring="accuracy"),
       grids={"lr": {"C": [0.1, 1.0, 10.0]}},
   )

.. warning::

   SFS + tuning is computationally intensive. Reduce the outer ``n_splits`` or
   the SFS inner ``n_splits`` for development runs.

---

3. Feature Stability Analysis
-----------------------------

For both ``k_best`` and ``sfs``, ``coco_pipe.decoding`` tracks which features
were selected in each fold. The stability score is the proportion of folds in
which a feature was selected:

.. code-block:: python

   stability = result.get_feature_stability()
   # columns: FeatureName, SelectionRate, MeanRank, StdRank

   # Most stable features
   top = stability.sort_values("SelectionRate", ascending=False).head(20)

.. note::

   Feature stability across folds is a measure of **generalizability**, not
   importance. A feature selected in all folds is a robust signal across the
   sampled subjects, regardless of its average selection score.

---

4. Selected Features per Fold
-----------------------------

.. code-block:: python

   selected = result.get_selected_features()
   # columns: FeatureName, Fold, Rank

   # Features selected in every fold
   universal = selected.groupby("FeatureName")["Fold"].count()
   universal = universal[universal == config.cv.n_splits].index.tolist()

---

5. Compatibility Notes
----------------------

- Feature selection is only valid for 2D tabular inputs (``input_kind in {"tabular_2d", "embedding_2d"}``).
- Feature selection is **incompatible** with temporal estimators (``SlidingEstimator``, ``GeneralizingEstimator``).
  The registry blocks this at validation time.
- ``k_best`` does not support ranked importances beyond fold scores/p-values.
  For importance-based selection, use tree ensemble importances via
  ``result.get_feature_importances()``.
