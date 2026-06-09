# ruff: noqa: E402
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

from coco_pipe.viz.decoding import (
    plot_decoding_scores,
    plot_fold_score_dispersion,
    plot_model_comparison,
)
from tests.fixtures.synthetic_result import make_synthetic_result

result = make_synthetic_result()
for fig, ax in [
    plot_decoding_scores(result),
    plot_fold_score_dispersion(result),
    plot_model_comparison(result),
]:
    plt.close(fig)
