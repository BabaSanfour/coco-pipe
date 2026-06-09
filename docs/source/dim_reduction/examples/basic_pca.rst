.. _dim-reduction-example-basic-pca:

===============================================
Example: Basic PCA Workflow End-to-End
===============================================

This example shows the minimum viable dim-reduction workflow: fit a PCA,
score the embedding, run an interpretation, and produce a paper-ready figure.
It uses only the base install — no optional extras required.

**Scientific context**: A 64-feature dataset (e.g., per-channel power-spectral
features from 200 EEG epochs). The goal is to project to 2D for visualization,
verify the embedding preserves neighborhood structure, and identify which
input features drive each principal component.

---

1. Fit and Score
==================

.. code-block:: python

   import numpy as np
   from coco_pipe.dim_reduction import DimReduction

   rng = np.random.default_rng(42)
   X = rng.standard_normal((200, 64))
   labels = rng.integers(0, 3, size=200)
   feature_names = [f"feat_{i:02d}" for i in range(X.shape[1])]

   reducer = DimReduction("PCA", n_components=2, whiten=False, random_state=42)
   embedding = reducer.fit_transform(X)
   scores = reducer.score(
       embedding,
       X=X,
       metrics=["trustworthiness", "continuity", "lcmc", "shepard_correlation"],
       k_values=[5, 10, 20, 50],
   )

   print(scores["metrics"]["trustworthiness"])
   print(reducer.metric_records_[:2])

---

2. Interpret
==============

.. code-block:: python

   interp = reducer.interpret(
       X,
       X_emb=embedding,
       analyses=["correlation", "perturbation"],
       feature_names=feature_names,
       n_repeats=5,
       random_state=42,
   )

   # Top-5 features driving each PC
   for dim, features in interp["analysis"]["correlation"].items():
       top_5 = list(features.items())[:5]
       print(dim, top_5)

---

3. Inspect the Summary
========================

``get_summary()`` returns a JSON-serializable payload combining scalar
metrics, metadata, diagnostics, capability flags, and the tidy metric +
interpretation records.

.. code-block:: python

   summary = reducer.get_summary()
   sorted(summary)
   # ['capabilities', 'diagnostics', 'interpretation', 'interpretation_records',
   #  'metadata', 'metric_records', 'metrics']

---

4. Build a Paper Figure
=========================

.. code-block:: python

   import matplotlib.pyplot as plt
   from coco_pipe.viz import (
       plot_embedding,
       plot_eigenvalues,
       plot_shepard_diagram,
       plot_reduction_feature_importance,
   )
   from coco_pipe.viz.theme import coco_theme, figure_size, save_figure

   with coco_theme("paper", colorblind=True):
       fig, axes = plt.subplots(
           2, 2, figsize=figure_size(columns=2, aspect_ratio=0.8),
           constrained_layout=True,
       )

       # (a) Scree plot
       plot_eigenvalues({"PCA": reducer.reducer.explained_variance_ratio_},
                        max_components=10, ax=axes[0, 0])
       axes[0, 0].set_title("a) Scree plot")

       # (b) Embedding
       plot_embedding(embedding, labels=labels, ax=axes[0, 1])
       axes[0, 1].set_title("b) PCA (2D)")

       # (c) Shepard diagram
       plot_shepard_diagram(X, embedding, ax=axes[1, 0])
       axes[1, 0].set_title("c) Shepard")

       # (d) Feature importance (correlation)
       plot_reduction_feature_importance(
           reducer.interpretation_records_,
           analysis="correlation",
           method="PCA",
           top_n=15,
           ax=axes[1, 1],
       )
       axes[1, 1].set_title("d) Top-15 features by correlation")

       save_figure(fig, "figures/pca_basic.pdf")

---

5. Persist and Reload
=======================

.. code-block:: python

   reducer.save("models/pca.pkl")
   loaded = DimReduction.load("models/pca.pkl", method="PCA")

   # Cached evaluation payloads are not persisted; re-score on the same embedding
   embedding_again = loaded.transform(X)
   loaded.score(embedding_again, X=X, metrics=["trustworthiness"], k_values=[10])
