"""
Tests for Dimensionality Reducers
=================================

Unified test suite for all dimensionality reduction in coco_pipe.
"""

import pytest
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
from sklearn.datasets import make_blobs

import dask.array as da
import dask_ml

# --- Import Reducers ---
from coco_pipe.dim_reduction.reducers.base import BaseReducer
from coco_pipe.dim_reduction.reducers.linear import PCAReducer, IncrementalPCAReducer, DaskPCAReducer, DaskTruncatedSVDReducer
from coco_pipe.dim_reduction.reducers.manifold import (
    IsomapReducer, LLEReducer, MDSReducer, SpectralEmbeddingReducer
)
from coco_pipe.dim_reduction.reducers.neighbor import (
    TSNEReducer, UMAPReducer, PacmapReducer, TrimapReducer, PHATEReducer, ParametricUMAPReducer
)
from coco_pipe.dim_reduction.reducers.spatiotemporal import DMDReducer, TRCAReducer
from coco_pipe.dim_reduction.reducers.neural import IVISReducer
from coco_pipe.dim_reduction.reducers.topology import TopologicalAEReducer
from coco_pipe.dim_reduction.config import ParametricUMAPConfig, DimReductionConfig, METHODS_DICT


# --- Fixtures ---

@pytest.fixture
def data():
    """Standard 2D tabular data (n_samples, n_features)."""
    # Use structured data for PaCMAP validation
    X, _ = make_blobs(n_samples=500, n_features=50, centers=5, random_state=42)
    return X.astype(np.float32)

@pytest.fixture
def data_ts():
    """Time series data (n_samples, n_features) for dynamic/topo methods."""
    rng = np.random.RandomState(42)
    return rng.rand(200, 20).astype(np.float32)

@pytest.fixture
def data_trca():
    """Multi-trial data: (n_trials, n_chans, n_times) for TRCA."""
    rng = np.random.RandomState(42)
    return rng.rand(10, 5, 100).astype(np.float64)

@pytest.fixture
def tmp_save_path(tmp_path):
    return tmp_path / "model.joblib"


# --- 1. Base Functionality (using PCA) ---

def test_base_functionality_pca(data, tmp_save_path):
    """Test standard fit, transform, fit_transform, and save/load logic."""
    reducer = PCAReducer(n_components=3)
    X_emb = reducer.fit_transform(data)
    
    # Check shape
    assert X_emb.shape == (500, 3)
    assert reducer.model is not None
    
    # Check introspection
    assert hasattr(reducer, 'explained_variance_ratio_')
    assert reducer.explained_variance_ratio_.shape == (3,)
    
    # Test transform equivalence
    X_new = reducer.transform(data)
    assert np.allclose(X_emb, X_new)
    
    # Test Save/Load
    reducer.save(tmp_save_path)
    assert tmp_save_path.exists()
    
    loaded = BaseReducer.load(tmp_save_path)
    assert isinstance(loaded, PCAReducer)
    assert loaded.n_components == 3
    
    # Check loaded model works
    X_loaded = loaded.transform(data)
    assert np.allclose(X_new, X_loaded)


# --- 2. Manifold Learners ---

def test_isomap_reducer(data):
    reducer = IsomapReducer(n_components=2, n_neighbors=5)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    assert reducer.model is not None
    
    X_new = reducer.transform(data)
    assert X_new.shape == (500, 2)

def test_lle_reducer(data):
    reducer = LLEReducer(n_components=2, n_neighbors=5)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    X_new = reducer.transform(data)
    assert X_new.shape == (500, 2)

def test_mds_reducer(data):
    reducer = MDSReducer(n_components=2)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    # MDS transform is not implemented in sklearn wrapper we use
    with pytest.raises(NotImplementedError):
        reducer.transform(data)

def test_spectral_reducer(data):
    reducer = SpectralEmbeddingReducer(n_components=2)
    X_emb = reducer.fit_transform(data)
    # Spectral embedding often returns results for the training set only
    assert X_emb.shape == (500, 2)
    
    # Spectral transform not implemented without out-of-sample extension method
    with pytest.raises(NotImplementedError):
        reducer.transform(data)


# --- 3. Neighbor Learners ---

def test_tsne_reducer(data):
    reducer = TSNEReducer(n_components=2, perplexity=10)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    # t-SNE does not support transform
    with pytest.raises(NotImplementedError):
        reducer.transform(data)

def test_umap_reducer(data):
    reducer = UMAPReducer(n_components=2, n_neighbors=10)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    X_new = reducer.transform(data)
    assert X_new.shape == (500, 2)
    assert hasattr(reducer, 'graph_')


def test_pacmap_reducer(data):
    # Use random init to avoid PCA broadcast error on CI
    reducer = PacmapReducer(n_components=2, init="random")
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    assert X_emb.shape == (500, 2)
    
    with pytest.raises(NotImplementedError):
         reducer.transform(data)

def test_trimap_reducer(data):
    reducer = TrimapReducer(n_components=2)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    with pytest.raises(NotImplementedError):
        reducer.transform(data)

def test_phate_reducer(data):
    reducer = PHATEReducer(n_components=2, knn=5)
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)
    
    X_new = reducer.transform(data)
    assert X_new.shape == (500, 2)
    
    # Check properties
    assert reducer.diff_potential is not None


# --- 4. Spatiotemporal Learners ---

def test_dmd_reducer(data_ts):
    reducer = DMDReducer(n_components=0) # keep all
    
    # DMDReducer expects (n_features, n_snapshots)
    # data_ts is (200, 20) -> (samples, features)
    data_dmd = data_ts.T
    
    reducer.fit(data_dmd)
    
    # Transform returns projected dynamics (n_snapshots, n_components)
    X_new = reducer.transform(data_dmd)
    assert X_new.shape[0] == 200 # 200 snapshots preserved
    
    # Check power user access
    assert hasattr(reducer, 'eigs_')
    assert reducer.eigs_ is not None

def test_trca_reducer(data_trca):
    # Filterbank structure: [[(pass_low, pass_high), (stop_low, stop_high)]]
    fb = [[(8, 12), (6, 14)]]
    reducer = TRCAReducer(n_components=1, sfreq=100, filterbank=fb)
    
    y = np.concatenate([np.zeros(5, dtype=int), np.ones(5, dtype=int)]) # 10 trials, 2 classes
    
    # Check if fit works without error
    reducer.fit(data_trca, y=y)
    
    # Transform
    X_out = reducer.transform(data_trca)
    
    # Expected output: (n_trials, n_components_out, n_times)
    # n_components_out = n_bands * n_classes = 1 * 2 = 2
    assert X_out.shape == (10, 2, 100)
    assert hasattr(reducer, 'coef_')


# --- 5. Neural Learners ---

def test_ivis_reducer(data_ts):
    reducer = IVISReducer(n_components=2, k=5, epochs=2, batch_size=16)
    X_emb = reducer.fit_transform(data_ts)
    assert X_emb.shape == (200, 2)
        
    X_new = reducer.transform(data_ts)
    assert X_new.shape == (200, 2)
    
    assert len(reducer.loss_history_) > 0


# --- 6. Topological Learners ---

def test_topo_ae_reducer(data_ts):
    # Basic fit
    reducer = TopologicalAEReducer(n_components=2, epochs=2, batch_size=16, device='cpu')
    reducer.fit(data_ts)
    
    X_emb = reducer.transform(data_ts)
    assert X_emb.shape == (200, 2)
    assert reducer.model is not None
    assert len(reducer.loss_history_) > 0
    
    # Topological loss fit (requires gudhi logic inside)
    reducer_topo = TopologicalAEReducer(n_components=2, epochs=2, lam=0.1, batch_size=16, device='cpu')
    reducer_topo.fit(data_ts)
    X_emb2 = reducer_topo.transform(data_ts)
    assert X_emb2.shape == (200, 2)
    
    # Device selection logic checks
    reducer_auto = TopologicalAEReducer(device='auto')
    assert reducer_auto.device in ['cpu', 'cuda', 'mps']
    
    reducer_cpu = TopologicalAEReducer(device='cpu')
    assert reducer_cpu.device == 'cpu'

def test_all_reducers_instantiation():
    """Test that all registered reducers can be instantiated."""
    for method, cls in METHODS_DICT.items():
        reducer = cls(n_components=2)
        assert isinstance(reducer, BaseReducer)
        assert reducer.n_components == 2

def test_skorch_topological_ae():
    """Test TopologicalAEReducer (Skorch version)."""
    X = np.random.rand(20, 10).astype(np.float32)
    # Reduced epochs for speed
    reducer = TopologicalAEReducer(n_components=2, epochs=2, batch_size=10, lam=0.1, device='cpu')
    reducer.fit(X)
    
    assert reducer.model.initialized_
    
    z = reducer.transform(X)
    assert z.shape == (20, 2)
    
    # Check history
    assert len(reducer.loss_history_) == 2

def test_parametric_umap_config():
    """Test Pydantic configuration for ParametricUMAP."""
    conf = ParametricUMAPConfig(n_epochs=10)
    assert conf.method == "ParametricUMAP"
    assert conf.n_epochs == 10
    
    # Test wrapping in DimReductionConfig
    wrapper = DimReductionConfig(config=conf)
    assert wrapper.config.method == "ParametricUMAP"

def test_parametric_umap_mock():
    """
    Test ParametricUMAPReducer.
    Warning: This requires tensorflow and umap-learn[plot].
    We wrap in try/except to avoid crashing if dependencies are missing in dev environment.
    """
    try:
        import tensorflow as tf
        from umap.parametric_umap import ParametricUMAP
    except ImportError:
        pytest.skip("TensorFlow or ParametricUMAP not installed.")
        
    X = np.random.rand(20, 10).astype(np.float32)
    reducer = ParametricUMAPReducer(n_components=2, n_epochs=2, verbose=False)
    reducer.fit(X)
    z = reducer.transform(X)
    assert z.shape == (20, 2)

def test_incremental_pca_reducer():
    """Test IncrementalPCAReducer (sklearn based)."""
    X = np.random.rand(100, 10)
    
    # Test batch fit (init)
    reducer = IncrementalPCAReducer(n_components=2, batch_size=20)
    reducer.fit(X)
    
    X_emb = reducer.transform(X)
    assert X_emb.shape == (100, 2)
    assert reducer.model is not None
    
    # Test partial fit manually
    reducer2 = IncrementalPCAReducer(n_components=2)
    # Feed 2 batches
    reducer2.partial_fit(X[:50])
    reducer2.partial_fit(X[50:])
    
    X_emb2 = reducer2.transform(X)
    assert X_emb2.shape == (100, 2)

def test_dask_pca_reducer():
    """Test DaskPCAReducer."""
    # Create dask array
    X_np = np.random.rand(100, 10)
    X_da = da.from_array(X_np, chunks=(20, 10))
    
    reducer = DaskPCAReducer(n_components=2)
    reducer.fit(X_da)
    
    # Transform returns dask array
    X_emb_da = reducer.transform(X_da)
    assert isinstance(X_emb_da, da.Array)
    
    X_emb = X_emb_da.compute()
    assert X_emb.shape == (100, 2)
    
    # Check if fit also accepts numpy (should convert implicitly or error depending on dask-ml version)
    # Dask-ML PCA typically handles numpy arrays by converting them or treating them as single chunk
    try:
        reducer.fit(X_np)
    except Exception as e:
        warnings.warn(f"DaskPCA on numpy raised: {e}")

def test_dask_truncated_svd_reducer():
    """Test DaskTruncatedSVDReducer."""
    X_np = np.random.rand(100, 10)
    X_da = da.from_array(X_np, chunks=(20, 10))
    
    reducer = DaskTruncatedSVDReducer(n_components=2)
    reducer.fit(X_da)
    
    X_emb_da = reducer.transform(X_da)
    X_emb = X_emb_da.compute()
    
    assert X_emb.shape == (100, 2)
