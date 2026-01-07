"""
Tests for Benchmarking Metrics
==============================
"""
import pytest
import numpy as np
from coco_pipe.dim_reduction.benchmark.metrics import trustworthiness, continuity, lcmc, shepard_diagram_data

def test_metrics_perfect_match():
    # Identity transform should have perfect scores
    X = np.random.rand(20, 5)
    X_emb = X.copy()
    
    t = trustworthiness(X, X_emb, n_neighbors=5)
    c = continuity(X, X_emb, n_neighbors=5)
    
    # sklearn trustworthiness definition: 1.0 is perfect
    assert t >= 0.99
    assert c >= 0.99
    
    # LCMC for identity should be high/max? 1.0?
    l = lcmc(X, X_emb, n_neighbors=5)
    assert l > 0.9

def test_metrics_random():
    # Random embedding should have low scores
    X = np.random.rand(50, 10)
    X_emb = np.random.rand(50, 2)
    
    t = trustworthiness(X, X_emb, n_neighbors=5)
    # Usually < 0.8 for random
    assert t < 0.9 

def test_shepard_sampling():
    X = np.random.rand(100, 5)
    X_emb = np.random.rand(100, 2)
    
    d_orig, d_emb = shepard_diagram_data(X, X_emb, sample_size=50)
    
    # pdist returns N*(N-1)/2 distances
    n = 50
    expected_len = n * (n - 1) // 2
    assert len(d_orig) == expected_len
    assert len(d_emb) == expected_len
