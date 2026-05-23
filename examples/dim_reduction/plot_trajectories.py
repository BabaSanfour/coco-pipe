# ruff: noqa: E402
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from coco_pipe.viz.dim_reduction import (
    plot_embedding,
    plot_trajectory,
    plot_trajectory_metric_series,
)

rng = np.random.default_rng(1)
trajectory = rng.normal(size=(3, 30, 2)).cumsum(axis=1)
embedding = rng.normal(size=(90, 2))
for fig, ax in [
    plot_trajectory(trajectory),
    plot_trajectory_metric_series(rng.random((3, 30))),
    plot_embedding(
        embedding,
        labels=rng.random(90),
        label_kind="continuous",
        title="Local Quality Map",
    ),
]:
    plt.close(fig)
