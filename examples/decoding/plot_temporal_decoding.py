# ruff: noqa: E402
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

from coco_pipe.viz.decoding import (
    plot_temporal_generalization_matrix,
    plot_temporal_score_curve,
    plot_temporal_statistical_assessment,
)
from tests.fixtures.synthetic_result import make_synthetic_result

result = make_synthetic_result()
for fig, ax in [
    plot_temporal_score_curve(result),
    plot_temporal_generalization_matrix(result),
    plot_temporal_statistical_assessment(result),
]:
    plt.close(fig)
