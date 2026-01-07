import pytest
import numpy as np
import matplotlib.pyplot as plt
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.reducers.base import BaseReducer

def test_factory():
    dr = DimReduction("PCA", n_components=2)
    assert dr.method == "PCA"
    assert dr.reducer.n_components == 2
    
    with pytest.raises(ValueError, match="Unknown method"):
        DimReduction("UNKNOWN_METHOD")

def test_validation_2d():
    # PCA requires 2D
    dr = DimReduction("PCA")
    X_3d = np.random.rand(10, 5, 20)
    
    with pytest.raises(ValueError, match="requires 2D input"):
        dr.fit(X_3d)

    X_2d = np.random.rand(20, 5)
    dr.fit(X_2d) # Should pass

def test_validation_3d():
    fb = [[(8, 12), (6, 14)]]
    dr = DimReduction("TRCA", params={"filterbank": fb, "sfreq": 100})
    
    X_2d = np.random.rand(20, 5)
    with pytest.raises(ValueError, match="requires 3D input"):
        dr.fit(X_2d)
        
    X_3d = np.random.rand(10, 5, 100)
    # TRCA needs labels y for fit
    y = np.concatenate([np.zeros(5, dtype=int), np.ones(5, dtype=int)])
    
    # Check fit works
    dr.fit(X_3d, y=y)

def test_fit_transform_pca():
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    emb = dr.fit_transform(X)
    assert emb.shape == (20, 2)
    assert dr.embedding_ is not None

def test_score():
    X = np.random.rand(50, 10) # 50 samples
    dr = DimReduction("PCA", n_components=2)
    dr.fit_transform(X)
    
    scores = dr.score(X)
    assert "trustworthiness" in scores
    assert "continuity" in scores
    assert 0 <= scores["trustworthiness"] <= 1.0

def test_plot():
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit_transform(X)
    
    # Test plotting
    fig = dr.plot()
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Test plotting without fit
    dr_new = DimReduction("PCA")
    with pytest.raises(RuntimeError):
        dr_new.plot()

def test_from_config():
    cfg = {"method": "UMAP", "n_components": 3, "params": {"n_neighbors": 5}}
    dr = DimReduction.from_config(cfg)
    assert dr.method == "UMAP"
    assert dr.n_components == 3
    assert dr.reducer_kwargs['n_neighbors'] == 5

def test_save_load(tmp_path):
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit(X)
    
    save_path = tmp_path / "model.joblib"
    dr.save(save_path)
    
    # Load back
    dr_loaded = DimReduction.load(save_path, method="PCA")
    assert dr_loaded.method == "PCA"
    assert dr_loaded.n_components == 2
    
    # Check it still works
    X_new = dr_loaded.transform(X)
    assert X_new.shape == (20, 2)

def test_score_attributes():
    X = np.random.rand(50, 10)
    dr = DimReduction("PCA", n_components=2)
    dr.fit_transform(X)
    
    scores = dr.score(X)
    # PCA should have 'explained_variance_ratio_'
    assert "explained_variance_ratio_" in scores
    assert len(scores["explained_variance_ratio_"]) == 2

class MockMNEObject:
    def __init__(self, data):
        self._data = data
    def get_data(self):
        return self._data

def test_mne_input():
    data = np.random.rand(20, 5)
    mne_obj = MockMNEObject(data)
    
    dr = DimReduction("PCA", n_components=2)
    dr.fit(mne_obj)
    emb = dr.transform(mne_obj)
    assert emb.shape == (20, 2)

def test_transform_independent():
    # Test transform without calling fit_transform
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit(X)
    emb = dr.transform(X)
    assert emb.shape == (20, 2)
    
    # The DimReduction class initializes self.embedding_ = None
    assert dr.embedding_ is None

def test_plot_modes():
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit_transform(X)
    
    # Error: missing X
    with pytest.raises(ValueError, match="requires 'X'"):
        dr.plot(mode='shepard')
        
    # Success case requires 'X'
    fig = dr.plot(mode='shepard', X=X)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
    
    # Streamlines error (requires V_emb)
    with pytest.raises(ValueError, match="requires 'V_emb'"):
        dr.plot(mode='streamlines')
        
    # Unknown mode
    with pytest.raises(ValueError, match="Unknown plot mode"):
        dr.plot(mode='invalid_mode')

def test_score_specific_reducers():
    """Test that score() scrapes specific attributes from different reducers."""
    X = np.random.rand(20, 5)
    
    # 1. MDS -> stress_
    dr_mds = DimReduction("MDS", n_components=2)
    dr_mds.fit_transform(X)
    scores_mds = dr_mds.score(X)
    assert "stress_" in scores_mds
    assert isinstance(scores_mds["stress_"], float)
    
    # 2. t-SNE -> kl_divergence_
    # Note: perplexity must be < n_samples (20).
    dr_tsne = DimReduction("TSNE", n_components=2, perplexity=5)
    dr_tsne.fit_transform(X)
    scores_tsne = dr_tsne.score(X)
    assert "kl_divergence_" in scores_tsne
    assert isinstance(scores_tsne["kl_divergence_"], float)
    
    assert isinstance(scores_tsne["kl_divergence_"], float)
    
    # Repopulate PCA check
    dr_pca = DimReduction("PCA", n_components=2)
    dr_pca.fit_transform(X)
    scores_pca = dr_pca.score(X)
    assert "singular_values_" in scores_pca
    assert len(scores_pca["singular_values_"]) == 2

def test_score_allowlist():
    """Test scraping of non-underscore allowed attributes."""
    class MockModel:
        def __init__(self):
            self.graph = np.zeros((5, 5))
            self.diff_potential = np.random.rand(5)
            self.eigs = np.array([1.0, 0.5])
            self.ignored_attr = "should not be included"

    class MockReducer(BaseReducer):
        def __init__(self):
            super().__init__(n_components=2)
            self.model = MockModel()
            
        def fit(self, X, y=None):
            return self
        def transform(self, X):
            return X[:, :2]
        def fit_transform(self, X, y=None):
            return X[:, :2]
    
    dr = DimReduction("PCA", n_components=2) # Dummy init
    dr.reducer = MockReducer()
    dr.embedding_ = np.zeros((20, 2)) # Dummy embedding (20 samples)
    
    X = np.zeros((20, 5))
    scores = dr.score(X)
    
    assert "graph" in scores
    assert "diff_potential" in scores
    assert "eigs" in scores
    assert "ignored_attr" not in scores
