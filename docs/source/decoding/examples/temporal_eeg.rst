.. _decoding-example-temporal:

=========================
Example: Temporal EEG Decoding
=========================

This example demonstrates sliding and generalizing temporal decoding from a
3D EEG epoch array, including time-resolved scoring, statistical inference
with Max-Stat correction, and visualization of the generalization matrix.

**Scientific context**: Decoding face vs. object from EEG signal amplitude at
each timepoint post-stimulus. The generalizing decoder tests whether the neural
representation learned at one time generalizes to other times.

---

1. Prepare Data
================

.. code-block:: python

   import numpy as np
   import mne  # required for temporal estimators

   # Simulate: 200 epochs, 64 channels, 100 timepoints
   n_epochs = 200
   n_channels = 64
   n_times = 100
   sfreq = 250.0  # Hz
   tmin = -0.1   # seconds pre-stimulus

   rng = np.random.default_rng(42)
   X = rng.standard_normal((n_epochs, n_channels, n_times))
   y = rng.choice([0, 1], size=n_epochs)
   subject_ids = np.repeat(np.arange(20), n_epochs // 20)
   times = np.linspace(tmin, tmin + (n_times - 1) / sfreq, n_times)

---

2. Sliding Decoding
=====================

.. code-block:: python

   from coco_pipe.decoding import Experiment, ExperimentConfig
   from coco_pipe.decoding.configs import (
       ClassicalModelConfig, TemporalDecoderConfig, CVConfig,
       StatisticalAssessmentConfig, ChanceAssessmentConfig,
   )

   sliding_config = ExperimentConfig(
       task="classification",
       models={
           "sliding_lr": TemporalDecoderConfig(
               wrapper="sliding",
               base=ClassicalModelConfig(
                   estimator="LogisticRegression",
                   params={"max_iter": 200},
               ),
               n_jobs=-1,
           )
       },
       metrics=["accuracy"],
       cv=CVConfig(strategy="stratified_group_kfold", n_splits=5, group_key="Subject"),
       evaluation=StatisticalAssessmentConfig(
           enabled=True,
           chance=ChanceAssessmentConfig(
               method="permutation",
               n_permutations=500,
               temporal_correction="max_stat",
               unit_of_inference="group_mean",
           ),
       ),
       random_state=42,
   )

   sliding_result = Experiment(sliding_config).run(
       X, y,
       sample_metadata={"Subject": subject_ids},
       time_axis=times,
       observation_level="epoch",
   )

   # Score curve: one accuracy per timepoint
   temporal = sliding_result.get_temporal_score_summary()
   assessment = sliding_result.get_statistical_assessment()
   # Significant timepoints after Max-Stat correction
   sig = assessment[assessment["Significant"]]
   print(f"Significant window: {sig['Time'].min():.3f}s — {sig['Time'].max():.3f}s")

---

3. Generalization Matrix
==========================

.. code-block:: python

   gen_config = ExperimentConfig(
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
       random_state=42,
   )

   gen_result = Experiment(gen_config).run(
       X, y,
       sample_metadata={"Subject": subject_ids},
       time_axis=times,
       observation_level="epoch",
   )

   # 2D score matrix: shape (n_train_times, n_test_times)
   matrix_df = gen_result.get_generalization_matrix("accuracy")

---

4. Visualization
=================

.. code-block:: python

   from coco_pipe.viz import plot_temporal_score_curve, plot_temporal_generalization_matrix

   # Sliding accuracy curve with significant timepoints highlighted
   fig_curve = plot_temporal_score_curve(sliding_result, metric="accuracy")

   # Generalization matrix heatmap
   fig_matrix = plot_temporal_generalization_matrix(gen_result, metric="accuracy")
   fig_matrix.savefig("generalization_matrix.png", dpi=150)
