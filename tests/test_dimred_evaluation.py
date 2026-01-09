"""
Tests for dim_reduction.evaluation module.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_s_curve, make_blobs

from coco_pipe.dim_reduction.evaluation.core import MethodSelector
from coco_pipe.dim_reduction.reducers.linear import PCAReducer
from coco_pipe.dim_reduction.reducers.neighbor import UMAPReducer
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation import (
    compute_coranking_matrix,
    trustworthiness,
    continuity,
    lcmc,
    compute_mrre,
    MethodSelector,
    compute_velocity_fields,
    perturbation_importance,
    compute_feature_importance,
    compute_coranking_matrix, 
    trustworthiness, 
    continuity, 
    lcmc, 
    compute_mrre,
)

@pytest.fixture
def data():
    return make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)

@pytest.fixture
def linear_data():
    return np.linspace(0, 10, 100)[:, None] * np.ones((1, 5)) + np.random.normal(0, 0.1, (100, 5))

def test_method_selector(data):
    X, y = data
    
    reducers = [
        DimReduction("PCA", n_components=2),
        DimReduction("Isomap", n_components=2, n_neighbors=10)
    ]
    
    evaluator = MethodSelector(reducers, data=X)
    evaluator.run(k_range=[5, 10, 20])
    
    assert "PCA" in evaluator.results_
    assert "Isomap" in evaluator.results_
    
    res_pca = evaluator.results_["PCA"]
    assert isinstance(res_pca, pd.DataFrame)
    assert 'trustworthiness' in res_pca.columns
    assert len(res_pca) == 3

def test_velocity_fields(linear_data):
    X = linear_data
    X_emb = X
    
    V_emb = compute_velocity_fields(X, X_emb, delta_t=1, n_neighbors=5)
    
    V_valid = V_emb[1:-2]
    mean_v = np.mean(V_valid, axis=0)
    
    target_dir = np.array([1, 2]) / np.linalg.norm([1, 2])
    calc_dir = mean_v / np.linalg.norm(mean_v)
    
    dot_prod = np.dot(target_dir, calc_dir)
    assert dot_prod > 0.9 

def test_feature_importance():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    X[:, 1] = np.random.randn(100) 
    
    model = DimReduction("PCA", n_components=1)
    model.fit(X)
    
    scores = perturbation_importance(model, X, feature_names=['Signal', 'Noise'])
    
    assert scores['Signal'] > scores['Noise']

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
    assert Q.shape == (n-1, n-1)
    
    # Trustworthiness and Continuity should be 1.0 for all k
    for k in range(1, n-1):
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
    umap = UMAPReducer(n_components=2, n_neighbors=5, n_epochs=10) # Fast
    
    reducers = [pca, umap]
    
    selector = MethodSelector(reducers, data=X)
    selector.run(k_range=[5, 10])
    
    assert "PCAReducer" in selector.results_
    assert "UMAPReducer" in selector.results_
    
    # Check results dict structure
    res_pca = selector.results_["PCAReducer"]
    assert isinstance(res_pca, pd.DataFrame)
    assert len(res_pca) == 2 # 2 k values
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
