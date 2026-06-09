# ruff: noqa: E402
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

from coco_pipe.viz.decoding import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_probability_diagnostics,
    plot_roc_curve,
)
from tests.fixtures.synthetic_result import make_synthetic_result

result = make_synthetic_result()
for fig, ax in [
    plot_confusion_matrix(result),
    plot_roc_curve(result),
    plot_pr_curve(result),
    plot_calibration_curve(result),
    plot_probability_diagnostics(result),
]:
    plt.close(fig)
