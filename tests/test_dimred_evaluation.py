"""
Tests for dim_reduction.evaluation module.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from coco_pipe.dim_reduction.analysis import perturbation_importance
from coco_pipe.dim_reduction.config import EvaluationConfig
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation import (
    MethodSelector,
    compute_coranking_matrix,
    compute_mrre,
    compute_velocity_fields,
    continuity,
    lcmc,
    moving_average,
    shepard_diagram_data,
    trajectory_curvature,
    trajectory_separation,
    trajectory_speed,
    trustworthiness,
)
from coco_pipe.dim_reduction.evaluation.core import _evaluate_single_method
from coco_pipe.dim_reduction.reducers.linear import PCAReducer
from coco_pipe.dim_reduction.reducers.neighbor import UMAPReducer


@pytest.fixture
def data():
    return make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)


@pytest.fixture
def linear_data():
    return np.linspace(0, 10, 100)[:, None] * np.ones((1, 5)) + np.random.normal(
        0, 0.1, (100, 5)
    )


def test_method_selector(data):
    X, y = data

    reducers = [
        DimReduction("PCA", n_components=2),
        DimReduction("Isomap", n_components=2, n_neighbors=10),
    ]

    evaluator = MethodSelector(reducers)
    evaluator.run(X, k_range=[5, 10, 20])

    assert "PCA" in evaluator.results_
    assert "ISOMAP" in evaluator.results_

    res_pca = evaluator.results_["PCA"]
    assert isinstance(res_pca, pd.DataFrame)
    assert "trustworthiness" in res_pca.columns
    assert len(res_pca) == 3


def test_velocity_fields(linear_data):
    X = linear_data
    X_emb = X

    V_emb = compute_velocity_fields(X, X_emb, delta_t=1, n_neighbors=5)

    V_valid = V_emb[1:-2]
    mean_v = np.mean(V_valid, axis=0)

    # Target is [1, 1, 1, 1, 1] because all coords increase linearly
    target_dir = np.ones(5) / np.linalg.norm(np.ones(5))
    calc_dir = mean_v / np.linalg.norm(mean_v)

    dot_prod = np.dot(target_dir, calc_dir)
    assert dot_prod > 0.9


def test_feature_importance():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    X[:, 1] = np.random.randn(100)

    model = DimReduction("PCA", n_components=1)
    model.fit(X)

    scores = perturbation_importance(model, X, feature_names=["Signal", "Noise"])

    assert scores["Signal"] > scores["Noise"]


@pytest.fixture
def sample_data():
    """Create a simple case where we know the ranks."""
    # 5 points on a line
    X = np.array([[0], [1], [2], [3], [4]])
    # Embedding preserves order exactly
    X_emb = np.array([[0], [2], [4], [6], [8]])
    return X, X_emb


def test_perfect_embedding(sample_data):
    """If embedding preserves order perfectly, Q should be diagonal."""
    X, X_emb = sample_data
    Q = compute_coranking_matrix(X, X_emb)

    # Check that Q is diagonal-ish (ranks match)
    # Ranks: [0, 1, 2, 3, 4] -> Neighbors of 0: [1, 2, 3, 4] -> ranks 1, 2, 3, 4
    # Q maps rank_high vs rank_low.
    # For a perfect embedding, rank_high_i(j) == rank_low_i(j) for all i,j.
    # So Q should be diagonal.

    n = X.shape[0]
    assert Q.shape == (n - 1, n - 1)

    # Trustworthiness and Continuity should be 1.0 for all k
    for k in range(1, n - 1):
        t = trustworthiness(Q, k)
        c = continuity(Q, k)
        assert np.isclose(t, 1.0)
        assert np.isclose(c, 1.0)


def test_imperfect_embedding():
    """Create a case with known errors."""
    # High: 0-1-2
    X = np.array([[0], [1], [2]])

    # Low: 0-2-1 (2 and 1 swapped)
    X_emb = np.array([[0], [2], [1]])

    Q = compute_coranking_matrix(X, X_emb)

    # Check T/C for k=1
    # For point 0:
    #   High: neighbor 1 (rank 1), neighbor 2 (rank 2)
    #   Low:  neighbor 2 (rank 1), neighbor 1 (rank 2)
    #   So 2 is rank 2 in High, rank 1 in Low (Intrusion)
    #   So 1 is rank 1 in High, rank 2 in Low (Extrusion)

    # We expect < 1.0
    t = trustworthiness(Q, k=1)
    c = continuity(Q, k=1)

    assert t < 1.0 or c < 1.0


def test_metrics_consistency():
    """Ensure vectorization doesn't break logic on random data."""
    rng = np.random.RandomState(42)
    X = rng.rand(20, 5)
    X_emb = rng.rand(20, 2)

    Q = compute_coranking_matrix(X, X_emb)

    t_5 = trustworthiness(Q, k=5)
    c_5 = continuity(Q, k=5)

    assert 0 <= t_5 <= 1.0
    assert 0 <= c_5 <= 1.0


def test_method_selector_parallel():
    """Test parallel execution of MethodSelector."""
    X = np.random.rand(50, 10).astype(np.float32)

    # Define two simple reducers
    pca = PCAReducer(n_components=2)
    umap = UMAPReducer(n_components=2, n_neighbors=5, n_epochs=10)  # Fast

    reducers = [pca, umap]

    selector = MethodSelector(reducers)
    selector.run(X, k_range=[5, 10])

    assert "PCAReducer" in selector.results_
    assert "UMAPReducer" in selector.results_

    # Check results dict structure
    res_pca = selector.results_["PCAReducer"]
    assert isinstance(res_pca, pd.DataFrame)
    assert len(res_pca) == 2  # 2 k values
    assert "trustworthiness" in res_pca.columns

    # Check embeddings
    assert "PCAReducer" in selector.embeddings_
    assert selector.embeddings_["PCAReducer"].shape == (50, 2)

    # Check Qs
    assert "PCAReducer" in selector.Qs_
    assert selector.Qs_["PCAReducer"].shape == (49, 49)


def test_method_selector_single_method():
    """Test with single method dict to ensure key handling works."""
    X = np.random.rand(20, 5)
    reducers = {"PCA": PCAReducer(n_components=2)}

    selector = MethodSelector(reducers)
    selector.run(X, k_range=[5])

    assert "PCA" in selector.results_
    assert len(selector.results_["PCA"]) == 1


def test_evaluation_plot(monkeypatch, data):
    """Test plotting of evaluation results."""
    X, y = data
    import matplotlib.pyplot as plt

    # Mock running
    selector = MethodSelector([])
    selector.data = X
    selector.results_ = {
        "PCA": pd.DataFrame({"k": [1, 2], "trustworthiness": [0.9, 0.8]}),
        "UMAP": pd.DataFrame({"k": [1, 2], "trustworthiness": [0.95, 0.85]}),
    }

    # Mock viz.plot_comparison
    types.SimpleNamespace(plot_comparison=lambda *a, **k: plt.figure())
    # We need to inject this mock where MethodSelector imports it
    # MethodSelector calls: from ...viz.dim_reduction import plot_comparison
    # This import happens INSIDE plot().
    # So we need to mock coco_pipe.viz.dim_reduction

    mock_viz = types.SimpleNamespace(plot_comparison=lambda *a, **k: plt.figure())
    monkeypatch.setitem(sys.modules, "coco_pipe.viz.dim_reduction", mock_viz)

    # Or force the import if it's already imported?
    # Since the import is inside the function, we can pre-seed sys.modules

    fig = selector.plot(metric="trustworthiness")
    fig = selector.plot(metric="trustworthiness")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# --- Coverage Tests Merged ---


def test_trustworthiness_edge_cases():
    """Test trustworthy edge cases including small N."""
    # Q matrix for N=5
    # perfect matching
    Q = np.diag([1] * 4)

    # 1. Trivial k >= n-1
    # Guardrails: k must be < n-1. So for n=5, max k is 3.
    # Actually, the guardrail CHECK is if k >= n-1, it raises.
    # So we should test the max valid k, which is 3.
    t = trustworthiness(Q, k=3)
    assert t == 1.0

    # 2. Small N logic (N=5)
    t_small = trustworthiness(Q, k=1)
    assert t_small == 1.0


def test_continuity_edge_cases():
    """Test continuity edge cases."""
    Q = np.diag([1] * 4)

    # 1. Trivial k >= n-1
    c = continuity(Q, k=3)
    assert c == 1.0

    # 2. Small N logic
    c_small = continuity(Q, k=1)
    assert c_small == 1.0


def test_lcmc_edge_cases():
    """Test LCMC edge cases."""
    Q = np.diag([1] * 4)
    Q[3, 0] = 1.0  # Add some overlap noise if needed, but diag is fine

    # k >= n-1
    l_score = lcmc(Q, k=3)
    assert isinstance(l_score, float)

    # Normal case
    l_norm = lcmc(Q, k=2)
    assert isinstance(l_norm, float)


def test_mrre_edge_cases():
    """Test MRRE edge cases."""
    Q = np.diag([1] * 4)  # N=5

    # MRRE calculation involves H_k normalization
    m_int, m_ext = compute_mrre(Q, k=2)
    assert isinstance(m_int, float)
    assert m_ext == 0.0  # Perfect embedding has 0 error


def test_shepard_diagram_data():
    """Test shepard diagram data generation."""
    X = np.random.rand(10, 5)
    X_emb = np.random.rand(10, 2)

    # 1. Full sample (N <= sample_size)
    d_orig, d_emb = shepard_diagram_data(X, X_emb, sample_size=20)
    assert len(d_orig) == 45  # 10*9/2
    assert len(d_emb) == 45

    # 2. Subsample (N > sample_size)
    d_orig_sub, d_emb_sub = shepard_diagram_data(X, X_emb, sample_size=5)
    assert len(d_orig_sub) == 10  # 5*4/2


def test_method_selector_config_object():
    """Test passing EvaluationConfig object to run()."""
    reducers = [DimReduction("PCA", n_components=2)]
    X = np.random.rand(20, 5)

    selector = MethodSelector(reducers)

    # Create config
    config = EvaluationConfig(k_range=[2, 5], metrics=["trustworthiness"])

    selector.run(X, k_range=config)

    assert "PCA" in selector.results_
    res = selector.results_["PCA"]
    assert len(res) == 2  # k=2, 5


def test_method_selector_no_data_error():
    """Test error when no data is provided."""
    reducers = [DimReduction("PCA")]
    selector = MethodSelector(reducers)  # No data in init

    with pytest.raises(TypeError):  # Missing X
        selector.run()


def test_evaluate_single_method_skip_large_k():
    """Test that _evaluate_single_method skips k >= n_samples."""
    X = np.random.rand(10, 5)  # 10 samples
    reducer = DimReduction("PCA", n_components=2)

    # Request k=5 (valid) and k=15 (invalid for n=10)
    k_vals = [5, 15]

    name, emb, Q, df = _evaluate_single_method("PCA", reducer, X, None, k_vals)

    assert len(df) == 1
    assert df.iloc[0]["k"] == 5
    # k=15 should be skipped


def test_method_selector_update_data():
    """Test updating data via run() arguments."""
    reducers = [DimReduction("PCA", n_components=2)]
    selector = MethodSelector(reducers)

    X_new = np.random.rand(30, 5)
    selector.run(X=X_new)

    assert selector.data.shape == (30, 5)
    assert selector.embeddings_["PCA"].shape == (30, 2)


def test_coranking_bias():
    """Verify that co-ranking matrix excludes self-neighbors"""
    # Create 5 orthogonal points (distance matrix will have diag 0, others sqrt(2))
    X = np.eye(5)
    X_emb = np.eye(5)[:, :2]

    # compute_coranking_matrix is imported above
    Q = compute_coranking_matrix(X, X_emb)

    # n=5, so Q should be (n-1) x (n-1) = 4x4
    assert Q.shape == (4, 4)
    # The total count in Q should be n * (n-1) = 5 * 4 = 20
    assert np.sum(Q) == 20


def test_reproducibility_shepard_sampling():
    """Verify that shepard_diagram_data is reproducible with random_state."""
    X = np.random.rand(50, 10)
    X_emb = np.random.rand(50, 2)

    # Subsample to a small number
    size = 10
    d1_orig, d1_emb = shepard_diagram_data(X, X_emb, sample_size=size, random_state=42)
    d2_orig, d2_emb = shepard_diagram_data(X, X_emb, sample_size=size, random_state=42)
    d3_orig, d3_emb = shepard_diagram_data(X, X_emb, sample_size=size, random_state=43)

    assert np.allclose(d1_orig, d2_orig)
    assert np.allclose(d1_emb, d2_emb)
    assert not np.allclose(d1_orig, d3_orig)


def test_metrics_pathological_data():
    """Verify behavior on pathological (constant/singular) data."""
    # Constant data: all distances are zero
    X = np.zeros((10, 5))
    X_emb = np.zeros((10, 2))

    Q = compute_coranking_matrix(X, X_emb)
    # With all distances zero, ranks might be arbitrary but should not crash
    assert Q.shape == (9, 9)
    assert np.sum(Q) == 90

    # These should not crash but will raise ValueError now because of the guardrails
    # if k is invalid but here we use valid k.
    t = trustworthiness(Q, k=3)
    assert isinstance(t, float)


def test_evaluation_guardrails_merged():
    """
    Verify that metric functions raise ValueError for invalid k (merged from temp test).
    """
    X = np.random.rand(10, 2)
    X_emb = np.random.rand(10, 2)
    Q = compute_coranking_matrix(X, X_emb)

    # Invalid k <= 0
    with pytest.raises(ValueError, match="must be > 0"):
        trustworthiness(Q, k=0)
    with pytest.raises(ValueError, match="must be > 0"):
        continuity(Q, k=-1)

    # Invalid k >= n-1 (n=10, so k must be < 9)
    with pytest.raises(ValueError, match="must be less than n_samples - 1"):
        trustworthiness(Q, k=9)


def test_velocity_guardrails_merged():
    """
    Verify that velocity computation raises ValueError for invalid params
    (merged from temp test).
    """
    X = np.random.rand(10, 5)
    X_emb = np.random.rand(10, 2)

    # n_samples < 2
    X_small = np.random.rand(1, 5)
    X_emb_small = np.random.rand(1, 2)
    with pytest.raises(ValueError, match="at least 2"):
        compute_velocity_fields(X_small, X_emb_small)

    # n_neighbors invalid
    with pytest.raises(ValueError, match="n_neighbors must be > 0"):
        compute_velocity_fields(X, X_emb, n_neighbors=0)

    # delta_t invalid
    with pytest.raises(ValueError, match="delta_t must be > 0"):
        compute_velocity_fields(X, X_emb, delta_t=0, n_neighbors=5)


def test_moving_average():
    x = np.array([1, 2, 3, 4, 5])
    # Window 3: [2, 3, 4]
    smoothed = moving_average(x, 3)
    np.testing.assert_allclose(smoothed, [2, 3, 4])

    # Window 1: same
    smoothed = moving_average(x, 1)
    np.testing.assert_array_equal(smoothed, x)


def test_trajectory_speed_linear():
    # Linear motion along x: speed should be constant 1.0
    t = np.linspace(0, 10, 11)  # 0, 1, ..., 10
    traj = np.zeros((11, 2))
    traj[:, 0] = t  # x = t, y = 0

    sp = trajectory_speed(traj, dt=1.0)

    # Speed is dx/dt = 1
    # Last point is padded
    expected = np.ones(11)
    np.testing.assert_allclose(sp, expected, atol=1e-5)


def test_trajectory_speed_circle():
    # Circular motion: x=cos(t), y=sin(t)
    # Speed = 1 if parameterized by arc length, but here t is angle
    # x(t) = R cos(w t), y(t) = R sin(w t)
    # v = R w
    R = 2.0
    w = 1.0
    t = np.linspace(0, 2 * np.pi, 100)
    dt = t[1] - t[0]

    traj = np.stack([R * np.cos(w * t), R * np.sin(w * t)], axis=1)

    sp = trajectory_speed(traj, dt=dt)

    # Theoretical speed R*w = 2.0
    # Numerical approx
    np.testing.assert_allclose(sp[:-1], 2.0, rtol=1e-2)


def test_trajectory_curvature_circle():
    # Curvature of circle radius R is 1/R
    R = 4.0
    t = np.linspace(0, 2 * np.pi, 1000)
    traj = np.stack([R * np.cos(t), R * np.sin(t)], axis=1)  # (1000, 2)

    k = trajectory_curvature(traj)

    # Ignore start/end edge effects of gradient
    k_center = k[10:-10]
    np.testing.assert_allclose(k_center, 1 / R, rtol=1e-2)


def test_trajectory_curvature_line():
    # Curvature of line is 0
    t = np.linspace(0, 10, 100)
    traj = np.stack([t, t], axis=1)

    k = trajectory_curvature(traj)
    np.testing.assert_allclose(k[10:-10], 0, atol=1e-5)


def test_trajectory_separation():
    # Two groups: A and B
    # A moves along x=0, B moves along x=2
    # Distance should be 2.0 always

    n_trials = 4
    n_times = 10
    n_dims = 2

    traj = np.zeros((n_trials, n_times, n_dims))
    labels = np.array(["A", "A", "B", "B"])

    # A: x=0, y=t
    traj[0, :, 1] = np.arange(n_times)
    traj[1, :, 1] = np.arange(n_times)

    # B: x=2, y=t
    traj[2, :, 0] = 2.0
    traj[2, :, 1] = np.arange(n_times)
    traj[3, :, 0] = 2.0
    traj[3, :, 1] = np.arange(n_times)

    sep = trajectory_separation(traj, labels)

    assert ("A", "B") in sep or ("B", "A") in sep
    dist = sep.get(("A", "B"), sep.get(("B", "A")))

    np.testing.assert_allclose(dist, 2.0, atol=1e-5)
