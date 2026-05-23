# ruff: noqa: E402
import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

from coco_pipe.report.dim_reduction import make_reduction_report


class SyntheticReduction:
    diagnostics_ = {"coranking_matrix_": np.eye(8)}

    def get_summary(self):
        return {
            "method": "Synthetic PCA",
            "metrics": {"trustworthiness": 0.9},
            "diagnostics": self.diagnostics_,
        }

    def get_components(self):
        return {"components_": np.random.default_rng(2).normal(size=(6, 2))}


embedding = np.random.default_rng(3).normal(size=(50, 2))
report = make_reduction_report(
    [SyntheticReduction()],
    embeddings=[embedding],
    sections=["overview", "embedding", "metrics", "coranking", "components"],
)
report.save("reduction_report.html")
