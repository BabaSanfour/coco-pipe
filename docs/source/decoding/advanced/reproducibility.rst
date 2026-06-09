.. _decoding-reproducibility:

=============================
Reproducibility Architecture
=============================

``coco_pipe.decoding`` is designed so that every run with the same configuration
and data produces bit-identical results. This section documents how random seeds
are propagated, where they appear in the result schema, and how to validate
reproducibility.

---

1. Seed Propagation via SeedSequence
======================================

Setting ``ExperimentConfig.random_state`` propagates derived, independent seeds
to every sub-component through NumPy's ``SeedSequence``:

.. code-block:: python

   config = ExperimentConfig(
       task="classification",
       models={"lr": ClassicalModelConfig(estimator="LogisticRegression")},
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified", n_splits=5),
       random_state=42,    # master seed
   )

Internally, ``Experiment._propagate_random_state()`` derives:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Derived seed (offset from master)
   * - ``cv``
     - ``master + 0``
   * - ``feature_selection``
     - ``master + 1``
   * - ``tuning``
     - ``master + 2``
   * - ``calibration``
     - ``master + 3``
   * - Per-model seeds
     - Spawned from ``master + 4`` via ``SeedSequence.spawn``

The per-model seeds are ordered by model name (alphabetically) for determinism.

.. note::

   Even if you add models or change their order in the ``models`` dict,
   alphabetical seed assignment ensures each model always receives the same seed
   regardless of insertion order.

---

2. What Is Seeded
==================

Every stochastic component in the pipeline is seeded:

- **CV splitters**: ``StratifiedKFold``, ``StratifiedGroupKFold``, ``KFold``.
- **Hyperparameter search**: ``RandomizedSearchCV`` uses ``tuning.random_state``.
- **Model initialization**: models with ``random_state`` parameters receive
  model-specific seeds.
- **Calibration**: ``CalibratedClassifierCV`` uses ``calibration.random_state``.
- **Bootstrap CI**: ``get_bootstrap_confidence_intervals`` accepts
  ``random_state``.
- **Permutation tests**: ``ChanceAssessmentConfig.random_state`` seeds the null
  permutation engine.

Not seeded (intentionally):

- **Data loading** and **preprocessing** outside the ``Experiment.run`` call.
- **MNE meta-estimator** internal parallelism (joblib workers), which may vary
  between runs if parallelism order is non-deterministic.

---

3. Result Schema Provenance
==============================

Every ``ExperimentResult`` stores reproducibility metadata in
``result.meta["hardware_provenance"]``:

.. code-block:: python

   print(result.meta["hardware_provenance"])
   # {
   #   "python_version": "3.11.12",
   #   "sklearn_version": "1.6.1",
   #   "numpy_version": "1.26.4",
   #   "platform": "macOS-14.5",
   #   "n_jobs": 1,
   #   "timestamp": "2026-05-14T04:30:00Z",
   # }

This provenance is captured by ``get_environment_info()`` from
``coco_pipe.report.provenance`` at the time of ``Experiment.run``.

---

4. Validating Reproducibility
================================

To verify that two runs produce identical results:

.. code-block:: python

   import numpy as np
   import pandas as pd

   # Run A
   result_a = Experiment(config).run(X, y, sample_metadata=meta)

   # Run B (identical config and data)
   result_b = Experiment(config).run(X, y, sample_metadata=meta)

   scores_a = result_a.get_detailed_scores()
   scores_b = result_b.get_detailed_scores()

   pd.testing.assert_frame_equal(
       scores_a.sort_values(["Model", "Fold", "Metric"]).reset_index(drop=True),
       scores_b.sort_values(["Model", "Fold", "Metric"]).reset_index(drop=True),
   )

---

5. Known Non-Determinism Sources
===================================

Some operations may produce slightly different results even with the same seed:

- **Parallel outer CV** (``n_jobs > 1``): scikit-learn's parallel backends can
  schedule workers in different orders between runs. For exact reproducibility,
  use ``n_jobs=1``.
- **GPU operations** (for foundation models with LoRA/QLoRA): CUDA operations
  are non-deterministic by default unless ``torch.use_deterministic_algorithms(True)``
  is set.
- **OS-level RNG state** leaking into ``random.random()`` or ``os.urandom()``
  calls in third-party libraries.

For fully deterministic publication runs, set ``n_jobs=1`` and document the
exact library versions from ``result.meta["hardware_provenance"]``.
