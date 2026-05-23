# ruff: noqa: E402
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

from coco_pipe.viz.theme import coco_theme, figure_size

with coco_theme("paper", colorblind=True):
    fig, ax = plt.subplots(figsize=figure_size(columns=1, aspect_ratio=0.75))
    ax.plot([0, 1, 2], [0.2, 0.8, 0.5])
    ax.set_title("Coco Theme")
    plt.close(fig)
