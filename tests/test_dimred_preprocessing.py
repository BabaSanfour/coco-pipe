import numpy as np
import pandas as pd

from coco_pipe.dim_reduction.preprocessing import (
    apply_pca_score_baseline,
    flip_pc_scores_for_consistency,
)


def test_apply_pca_score_baseline_n_points():
    time = np.array([-100, 0, 100, 200])
    scores = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], index=["PC1", "PC2"])
    # First 2 points mean: PC1=(1+2)/2=1.5, PC2=(5+6)/2=5.5
    result = apply_pca_score_baseline(time, scores, n_points=2)
    np.testing.assert_allclose(result.loc["PC1"].values, [-0.5, 0.5, 1.5, 2.5])
    np.testing.assert_allclose(result.loc["PC2"].values, [-0.5, 0.5, 1.5, 2.5])


def test_apply_pca_score_baseline_time_window():
    time = np.array([-200, -100, 0, 100])
    scores = pd.DataFrame([[1, 2, 3, 4]], index=["PC1"])
    # Window [-200, 0] -> indices 0, 1, 2. mean = (1+2+3)/3 = 2
    result = apply_pca_score_baseline(
        time, scores, n_points=None, baseline_min_ms=-200, baseline_max_ms=0
    )
    np.testing.assert_allclose(result.loc["PC1"].values, [-1, 0, 1, 2])


def test_flip_pc_scores_for_consistency():
    time = np.array([0, 100, 200])
    # PC1 mean is positive, PC2 mean is negative
    scores = pd.DataFrame([[1, 2, 3], [-1, -2, -3]], index=["PC1", "PC2"])
    result = flip_pc_scores_for_consistency(scores, time, flip_window_ms=(0, 200))
    # PC1 unchanged
    np.testing.assert_allclose(result.loc["PC1"].values, [1, 2, 3])
    # PC2 flipped
    np.testing.assert_allclose(result.loc["PC2"].values, [1, 2, 3])
