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
    shepard_diagram_data,
    trustworthiness,
)
from coco_pipe.dim_reduction.evaluation.core import (
    MethodSelector,
    _evaluate_single_method,
)
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

    evaluator = MethodSelector(reducers, data=X)
    evaluator.run(k_range=[5, 10, 20])

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

    selector = MethodSelector(reducers, data=X)
    selector.run(k_range=[5, 10])

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

    selector = MethodSelector(reducers, data=X)
    selector.run(k_range=[5])

    assert "PCA" in selector.results_
    assert len(selector.results_["PCA"]) == 1


def test_evaluation_plot(monkeypatch, data):
    """Test plotting of evaluation results."""
    X, y = data
    import matplotlib.pyplot as plt

    # Mock running
    selector = MethodSelector([], data=X)
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
    t = trustworthiness(Q, k=4)
    assert t == 1.0

    # 2. Small N logic (N=5, K=1)
    # n=5 is the cutoff for small sample handling
    t_small = trustworthiness(Q, k=1)
    assert t_small == 1.0


def test_continuity_edge_cases():
    """Test continuity edge cases."""
    Q = np.diag([1] * 4)

    # 1. Trivial k >= n-1
    c = continuity(Q, k=4)
    assert c == 1.0

    # 2. Small N logic
    c_small = continuity(Q, k=1)
    assert c_small == 1.0


def test_lcmc_edge_cases():
    """Test LCMC edge cases."""
    Q = np.diag([1] * 4)
    Q[3, 0] = 1.0  # Add some overlap noise if needed, but diag is fine

    # k >= n-1
    l = lcmc(Q, k=4)
    assert l == 0.0

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

    selector = MethodSelector(reducers, data=X)

    # Create config
    config = EvaluationConfig(k_range=[2, 5], metrics=["trustworthiness"])

    selector.run(k_range=config)

    assert "PCA" in selector.results_
    res = selector.results_["PCA"]
    assert len(res) == 2  # k=2, 5


def test_method_selector_no_data_error():
    """Test error when no data is provided."""
    reducers = [DimReduction("PCA")]
    selector = MethodSelector(reducers)  # No data in init

    with pytest.raises(ValueError, match="No data provided"):
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
    X_init = np.random.rand(20, 5)
    selector = MethodSelector(reducers, data=X_init)

    X_new = np.random.rand(30, 5)
    selector.run(X=X_new)

    assert selector.data.shape == (30, 5)
    assert selector.embeddings_["PCA"].shape == (30, 2)
