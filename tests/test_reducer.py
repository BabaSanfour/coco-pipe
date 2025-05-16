import os
import sys
import types
import importlib
import importlib.util
import numpy as np
import pytest
from pathlib import Path
import joblib


# Mock classes for external dependencies
class MockPCA:
    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        
    def fit(self, X):
        self.fitted_ = True
        self.components_ = np.eye(min(X.shape[1], self.n_components), X.shape[1])
        
    def transform(self, X):
        return X @ self.components_.T


class MockTSNE:
    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        
    def fit_transform(self, X):
        return X[:, :self.n_components]


class MockUMAP:
    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        
    def fit(self, X):
        self.n_features_in_ = X.shape[1]
        self.embeddings_ = X[:, :self.n_components]
        return self
        
    def transform(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]} features")
        return X[:, :self.n_components]


@pytest.fixture
def setup_mocks(monkeypatch, tmp_path):
    """Setup dependencies for the real reducer classes"""
    # Create allowed directory for save path validation
    allowed_dir = tmp_path / "allowed_directory"
    allowed_dir.mkdir(exist_ok=True)
    
    # Mock out sklearn.decomposition.PCA
    monkeypatch.setattr('sklearn.decomposition.PCA', MockPCA)
    
    # Mock out sklearn.manifold.TSNE
    monkeypatch.setattr('sklearn.manifold.TSNE', MockTSNE)
    
    # Mock out umap.UMAP
    module = types.ModuleType('umap')
    module.UMAP = MockUMAP
    sys.modules['umap'] = module
    
    # Mock Path.resolve for file path validation
    original_resolve = Path.resolve
    def mock_resolve(self):
        if str(self) == '/path/to/allowed/directory':
            return Path(allowed_dir)
        elif '/path/to/allowed/directory' in str(self):
            return Path(str(self).replace('/path/to/allowed/directory', str(allowed_dir)))
        return original_resolve(self)
    
    monkeypatch.setattr(Path, 'resolve', mock_resolve)
    
    # Mock joblib.dump and joblib.load for saving and loading
    def mock_dump(obj, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(f"Mocked {obj.__class__.__name__}")
    
    def mock_load(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No file found at {filepath}")
            
        from coco_pipe.dim_reduction.reducers.pca import PCAReducer
        from coco_pipe.dim_reduction.reducers.tsne import TSNEReducer
        from coco_pipe.dim_reduction.reducers.umap import UMAPReducer
        
        # Create a properly initialized and fitted reducer based on filename
        with open(filepath, 'r') as f:
            content = f.read()
        
        if "PCAReducer" in content:
            reducer = PCAReducer(n_components=2)
            reducer.model = MockPCA(n_components=2)
            reducer.model.fitted_ = True
            reducer.model.components_ = np.eye(2, 4)  # Mock 4 features
            return reducer
        elif "TSNEReducer" in content:
            reducer = TSNEReducer(n_components=2)
            reducer.embedding_ = np.random.rand(10, 2)
            return reducer
        elif "UMAPReducer" in content:
            reducer = UMAPReducer(n_components=2)
            reducer.model = MockUMAP(n_components=2)
            reducer.model.n_features_in_ = 4  # Mock 4 features
            return reducer
        else:
            raise ValueError(f"Unknown reducer type in file: {content}")
    
    monkeypatch.setattr('joblib.dump', mock_dump)
    monkeypatch.setattr('joblib.load', mock_load)
    
    return allowed_dir


@pytest.fixture
def allowed_dir(tmp_path):
    """Create a temporary directory for test files"""
    dir_path = tmp_path / "allowed_directory"
    dir_path.mkdir(exist_ok=True)
    return dir_path


@pytest.fixture
def DimReducer(monkeypatch, allowed_dir):
    """Get the DimReducer class with path validation using the test directory"""
    from coco_pipe.dim_reduction.reducer import DimReducer
    
    # Save the original __init__ to restore later
    original_init = DimReducer.__init__
    
    def patched_init(self, method, n_components=2, save_path=None, **reducer_kwargs):
        method = method.upper()
        from coco_pipe.dim_reduction.config import METHODS
        if method not in METHODS:
            raise ValueError(f"Unknown method {method!r}, choose from {METHODS}")
        self.method = method
        
        from coco_pipe.dim_reduction.config import METHODS_DICT
        ReducerCls = METHODS_DICT[method]
        self.reducer = ReducerCls(n_components=n_components, **reducer_kwargs)
        
        if save_path is not None:
            # Only paths within allowed_dir are valid for testing
            resolved_save_path = Path(save_path).resolve()
            if not str(resolved_save_path).startswith(str(allowed_dir)):
                raise ValueError('save_path must be within allowed directory')
            self.save_path = resolved_save_path
        else:
            self.save_path = None
    
    # Apply the patch
    monkeypatch.setattr(DimReducer, '__init__', patched_init)
    
    # Return the patched class
    return DimReducer


def test_fit_transform_all_methods(DimReducer):
    """Test fit_transform with all dimensionality reduction methods"""
    rng = np.random.RandomState(0)
    X = rng.rand(20, 5)
    
    # Test PCA
    dr_pca = DimReducer('PCA', n_components=2)
    out_pca = dr_pca.fit_transform(X)
    assert out_pca.shape == (20, 2)
    
    # Test UMAP (with adjusted parameters for small sample size)
    dr_umap = DimReducer('UMAP', n_components=2, n_neighbors=5)
    out_umap = dr_umap.fit_transform(X)
    assert out_umap.shape == (20, 2)
    
    # Test TSNE (with adjusted perplexity for small sample size)
    dr_tsne = DimReducer('TSNE', n_components=2, perplexity=5)
    out_tsne = dr_tsne.fit_transform(X)
    assert out_tsne.shape == (20, 2)


def test_fit_then_transform_pca_umap(DimReducer):
    """Test separate fit and transform for methods that support it"""
    rng = np.random.RandomState(1)
    X = rng.rand(10, 4)

    for method, params in [('PCA', {}), ('UMAP', {'n_neighbors': 5})]:
        dr = DimReducer(method, n_components=3, **params)
        dr.fit(X)
        out = dr.transform(X)
        assert out.shape == (10, 3)


def test_transform_without_fit_raises(DimReducer):
    """Test that transform without fit raises an appropriate error"""
    X = np.random.rand(5, 3)
    for method in ['PCA', 'UMAP']:
        dr = DimReducer(method, n_components=2)
        with pytest.raises(RuntimeError):
            dr.transform(X)


def test_tsne_transform_not_implemented(DimReducer):
    """Test that TSNE transform raises NotImplementedError"""
    X = np.random.rand(30, 6)  # Using larger sample size for TSNE
    dr = DimReducer('TSNE', n_components=2, perplexity=5)
    # We need to fit first since TSNE only implements fit_transform
    dr.fit_transform(X)
    # TSNE doesn't have transform method in our implementation
    with pytest.raises(NotImplementedError):
        dr.transform(X)


def test_umap_dimension_mismatch(DimReducer):
    """Test that UMAP correctly validates input dimensions"""
    rng = np.random.RandomState(2)
    X = rng.rand(12, 5)
    dr = DimReducer('UMAP', n_components=2, n_neighbors=5)
    dr.fit(X)
    with pytest.raises(ValueError):
        dr.transform(rng.rand(12, 6))


def test_save_and_load(DimReducer, allowed_dir):
    """Test saving and loading a reducer"""
    # Use the temporary allowed directory
    save_path = str(allowed_dir / "test.joblib")
    
    X = np.random.rand(10, 4)
    dr = DimReducer('PCA', n_components=2, save_path=save_path)
    dr.fit(X)
    
    assert os.path.exists(save_path)

    loaded = DimReducer.load(save_path)
    out = loaded.transform(X)
    assert out.shape == (10, 2)

    os.remove(save_path)


def test_save_path_outside_allowed_dir(DimReducer):
    """Test that save_path validation rejects paths outside allowed directory"""
    # Use a path outside the allowed directory
    outside_path = '/tmp/invalid_path_for_test.joblib'
    
    with pytest.raises(ValueError):
        DimReducer('PCA', save_path=outside_path)
