# ruff: noqa: E402
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from coco_pipe.viz.dim_reduction import (
    plot_embedding,
    plot_metrics,
    plot_shepard_diagram,
)

rng = np.random.default_rng(0)
X_orig = rng.normal(size=(80, 6))
X_emb = rng.normal(size=(80, 2))
labels = np.array(["A", "B"] * 40)
for fig, ax in [
    plot_embedding(X_emb, labels=labels),
    plot_metrics({"trustworthiness": 0.91, "continuity": 0.87}),
    plot_shepard_diagram(X_orig, X_emb),
]:
    plt.close(fig)
