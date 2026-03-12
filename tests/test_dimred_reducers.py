"""
Tests for Dimensionality Reducers
=================================

Unified test suite for all dimensionality reduction in coco_pipe.
"""

import importlib
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import dask.array as da
import numpy as np
import pytest
from sklearn.datasets import make_blobs

from coco_pipe.dim_reduction.config import (
    METHODS,
    ParametricUMAPConfig,
    get_reducer_class,
)

# --- Import Reducers ---
from coco_pipe.dim_reduction.reducers.base import BaseReducer
from coco_pipe.dim_reduction.reducers.linear import (
    DaskPCAReducer,
    DaskTruncatedSVDReducer,
    IncrementalPCAReducer,
    PCAReducer,
)
from coco_pipe.dim_reduction.reducers.manifold import (
    IsomapReducer,
    LLEReducer,
    MDSReducer,
    SpectralEmbeddingReducer,
)
from coco_pipe.dim_reduction.reducers.neighbor import (
    PacmapReducer,
    ParametricUMAPReducer,
    PHATEReducer,
    TrimapReducer,
    TSNEReducer,
    UMAPReducer,
)
from coco_pipe.dim_reduction.reducers.neural import IVISReducer
from coco_pipe.dim_reduction.reducers.spatiotemporal import DMDReducer, TRCAReducer
from coco_pipe.dim_reduction.reducers.topology import (
    TopologicalAEReducer,
    TopologicalSignatureDistance,
)

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


class _MockHistory:
    def __init__(self, losses):
        self.losses = losses

    def __getitem__(self, item):
        if item == (slice(None, None, None), "train_loss"):
            return self.losses
        raise KeyError(item)


class _FakeTensor:
    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTopologyTorch:
    def __init__(self, cuda_available=False, mps_available=False):
        self.optim = SimpleNamespace(Adam=object())
        self.cuda = SimpleNamespace(is_available=lambda: cuda_available)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )

    def tensor(self, array):
        return _FakeTensor(array)

    def no_grad(self):
        return _FakeNoGrad()


class _MockTopologyModule:
    def __init__(self, input_dim=10, latent_dim=2, hidden_dims=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [128, 64]

    def eval(self):
        return self

    def __call__(self, x_tensor):
        arr = x_tensor.numpy()
        z = _FakeTensor(arr[:, : self.latent_dim])
        return z, _FakeTensor(arr)


class _MockTopologyLossCriterion:
    def __init__(self, lam=0.0):
        self.lam = lam


class _MockNeuralNetRegressor:
    last_init = None

    def __init__(
        self,
        module,
        module__input_dim,
        module__latent_dim,
        module__hidden_dims,
        criterion,
        criterion__lam,
        optimizer,
        lr,
        max_epochs,
        batch_size,
        device,
        verbose,
        train_split=None,
        iterator_train__shuffle=True,
        callbacks=None,
    ):
        type(self).last_init = {
            "module__input_dim": module__input_dim,
            "module__latent_dim": module__latent_dim,
            "module__hidden_dims": module__hidden_dims,
            "criterion__lam": criterion__lam,
            "lr": lr,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "device": device,
            "verbose": verbose,
            "train_split": train_split,
            "iterator_train__shuffle": iterator_train__shuffle,
            "callbacks": callbacks,
        }
        self.initialized_ = True
        self.fitted_ = False
        self.module_ = module(
            input_dim=module__input_dim,
            latent_dim=module__latent_dim,
            hidden_dims=module__hidden_dims,
        )
        self.history_ = _MockHistory([1.0, 0.5])
        self.callbacks = callbacks

    def set_params(self, **params):
        if self.last_init:
            self.last_init.update(params)
        return self

    def fit(self, X, y):
        self.fitted_ = True
        return self


# --- 1. Base Functionality (using PCA) ---


def test_base_reducer_is_public_extension_point():
    class DummyReducer(BaseReducer):
        def fit(self, X, y=None):
            self.model = object()
            return self

        def transform(self, X):
            X = np.asarray(X)
            return X[:, : self.n_components]

    reducer = DummyReducer(n_components=2)
    caps = reducer.capabilities

    assert caps["input_ndim"] == 2
    assert caps["input_layout"] == "standard"
    assert caps["has_transform"] is True


def test_topology_module_import_is_lazy(monkeypatch):
    module_name = "coco_pipe.dim_reduction.reducers.topology"
    cached_module = sys.modules.get(module_name)
    sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "torch", None)

    try:
        topology_mod = importlib.import_module(module_name)
        assert hasattr(topology_mod, "TopologicalAEReducer")
    finally:
        sys.modules.pop(module_name, None)
        if cached_module is not None:
            sys.modules[module_name] = cached_module


def test_base_functionality_pca(data, tmp_save_path):
    """Test standard fit, transform, fit_transform, and save/load logic."""
    reducer = PCAReducer(n_components=3)
    X_emb = reducer.fit_transform(data)

    # Check shape
    assert X_emb.shape == (500, 3)
    assert reducer.model is not None

    # Check introspection
    assert hasattr(reducer, "explained_variance_ratio_")
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
    # 'dense' solver is more robust for small/singular test data than 'arpack'
    reducer = LLEReducer(n_components=2, n_neighbors=5, eigen_solver="dense")
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
    class MockUMAP:
        def __init__(self, n_components=2, n_neighbors=15):
            self.n_components = n_components
            self.n_neighbors = n_neighbors
            self.graph_ = "mock-graph"

        def fit(self, X, y=None):
            self.embedding_ = np.zeros((len(X), self.n_components))
            return self

        def transform(self, X):
            return np.zeros((len(X), self.n_components))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockUMAP,
    ):
        reducer = UMAPReducer(n_components=2, n_neighbors=10)
        X_emb = reducer.fit_transform(data)
        assert X_emb.shape == (500, 2)

        X_new = reducer.transform(data)
        assert X_new.shape == (500, 2)
        assert reducer.get_diagnostics()["graph_"] == "mock-graph"


@pytest.mark.skip(reason="PaCMAP compilation issues on some CI envs")
def test_pacmap_reducer(data):
    # Use random init to avoid PCA broadcast error on CI
    reducer = PacmapReducer(n_components=2, init="random")
    X_emb = reducer.fit_transform(data)
    assert X_emb.shape == (500, 2)

    assert X_emb.shape == (500, 2)

    with pytest.raises(NotImplementedError):
        reducer.transform(data)


def test_trimap_reducer(data):
    class MockTrimap:
        def __init__(self, n_dims=2, n_inliers=10, n_outliers=5, n_random=5):
            self.n_dims = n_dims

        def fit_transform(self, X):
            return np.zeros((len(X), self.n_dims))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockTrimap,
    ):
        reducer = TrimapReducer(n_components=2)
        X_emb = reducer.fit_transform(data)
        assert X_emb.shape == (500, 2)

        with pytest.raises(NotImplementedError):
            reducer.transform(data)


def test_phate_reducer(data):
    class MockPHATE:
        def __init__(self, n_components=2, knn=5, decay=40, t="auto"):
            self.n_components = n_components
            self.diff_potential = np.ones((3, 3))
            self.diff_op = np.eye(3)
            self.graph = np.eye(3)

        def fit(self, X):
            return self

        def transform(self, X):
            return np.zeros((len(X), self.n_components))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockPHATE,
    ):
        reducer = PHATEReducer(n_components=2, knn=5)
        X_emb = reducer.fit_transform(data)
        assert X_emb.shape == (500, 2)

        X_new = reducer.transform(data)
        assert X_new.shape == (500, 2)

        assert reducer.get_diagnostics()["diff_potential"] is not None


# --- 4. Spatiotemporal Learners ---


def test_dmd_reducer(data_ts):
    reducer = DMDReducer(n_components=0)  # keep all

    # DMDReducer expects (n_features, n_snapshots)
    # data_ts is (200, 20) -> (samples, features)
    data_dmd = data_ts.T

    reducer.fit(data_dmd)

    # Transform returns projected dynamics (n_snapshots, n_components)
    X_new = reducer.transform(data_dmd)
    assert X_new.shape[0] == 200  # 200 snapshots preserved

    # Check power user access
    assert hasattr(reducer, "eigs_")
    assert reducer.eigs_ is not None


def test_trca_reducer(data_trca):
    # Filterbank structure: [[(pass_low, pass_high), (stop_low, stop_high)]]
    fb = [[(8, 12), (6, 14)]]
    reducer = TRCAReducer(n_components=1, sfreq=100, filterbank=fb)

    y = np.concatenate(
        [np.zeros(5, dtype=int), np.ones(5, dtype=int)]
    )  # 10 trials, 2 classes

    # Check if fit works without error
    reducer.fit(data_trca, y=y)

    # Transform
    X_out = reducer.transform(data_trca)

    # Expected output: (n_trials, n_components_out, n_times)
    # n_components_out = n_bands * n_components = 1 * 1 = 1
    assert X_out.shape == (10, 1, 100)
    assert "coef_" in reducer.get_diagnostics()


# --- 5. Neural Learners ---


def test_ivis_reducer(data_ts):
    class MockIvis:
        def __init__(self, embedding_dims=2, k=15, epochs=100, batch_size=32):
            self.embedding_dims = embedding_dims
            self.loss_history_ = [0.1, 0.05]

        def fit(self, X, Y=None):
            self.embedding_ = np.zeros((len(X), self.embedding_dims))
            return self

        def transform(self, X):
            return np.zeros((len(X), self.embedding_dims))

    with patch(
        "coco_pipe.dim_reduction.reducers.neural.import_optional_dependency",
        return_value=MockIvis,
    ):
        reducer = IVISReducer(
            n_components=2,
            k=5,
            epochs=2,
            batch_size=16,
            unsupported_param="ignore-me",
        )

        reducer.fit(data_ts)
        X_new = reducer.transform(data_ts)

        assert X_new.shape == (200, 2)
        assert reducer.model.embedding_dims == 2
        assert len(reducer.get_diagnostics()["loss_history_"]) > 0


# --- 6. Topological Learners ---


def test_topo_ae_reducer(data_ts):
    fake_torch = _FakeTopologyTorch()

    with (
        patch(
            "coco_pipe.dim_reduction.reducers.topology._build_topology_training_classes",
            return_value=(
                fake_torch,
                _MockTopologyModule,
                _MockTopologyLossCriterion,
            ),
        ),
        patch(
            "coco_pipe.dim_reduction.reducers.topology.import_optional_dependency",
            return_value=_MockNeuralNetRegressor,
        ),
        patch(
            "coco_pipe.dim_reduction.reducers.topology._load_torch",
            return_value=fake_torch,
        ),
    ):
        reducer = TopologicalAEReducer(
            n_components=2, epochs=2, batch_size=16, device="cpu"
        )
        reducer.fit(data_ts)

        X_emb = reducer.transform(data_ts)
        assert X_emb.shape == (200, 2)
        assert reducer.model is not None
        assert len(reducer.loss_history_) > 0

        reducer_topo = TopologicalAEReducer(
            n_components=2, epochs=2, lam=0.1, batch_size=16, device="cpu"
        )
        reducer_topo.fit(data_ts)
        X_emb2 = reducer_topo.transform(data_ts)
        assert X_emb2.shape == (200, 2)

    reducer_auto = TopologicalAEReducer(device="auto")
    assert reducer_auto.requested_device == "auto"
    assert reducer_auto.device == "cpu"

    reducer_cpu = TopologicalAEReducer(device="cpu")
    assert reducer_cpu.device == "cpu"


def test_all_reducers_instantiation():
    """Test that all registered reducers can be instantiated."""
    for method in METHODS:
        cls = get_reducer_class(method)
        reducer = cls(n_components=2)
        assert isinstance(reducer, BaseReducer)
        assert reducer.n_components == 2


def test_skorch_topological_ae():
    """Test TopologicalAEReducer with a mocked Skorch backend."""
    X = np.random.rand(20, 10).astype(np.float32)
    fake_torch = _FakeTopologyTorch()

    with (
        patch(
            "coco_pipe.dim_reduction.reducers.topology._build_topology_training_classes",
            return_value=(
                fake_torch,
                _MockTopologyModule,
                _MockTopologyLossCriterion,
            ),
        ),
        patch(
            "coco_pipe.dim_reduction.reducers.topology.import_optional_dependency",
            return_value=_MockNeuralNetRegressor,
        ),
        patch(
            "coco_pipe.dim_reduction.reducers.topology._load_torch",
            return_value=fake_torch,
        ),
    ):
        reducer = TopologicalAEReducer(
            n_components=2,
            epochs=2,
            batch_size=10,
            lam=0.1,
            device="cpu",
            verbose=0,
        )
        reducer.fit(X)

        assert reducer.model.initialized_

        z = reducer.transform(X)
        assert z.shape == (20, 2)
        assert len(reducer.loss_history_) == 2
        assert reducer.get_quality_metadata()["lam"] == 0.1


def test_parametric_umap_config():
    """Test Pydantic configuration for ParametricUMAP."""
    conf = ParametricUMAPConfig(n_epochs=10)
    assert conf.method == "ParametricUMAP"
    assert conf.n_epochs == 10


def test_parametric_umap_mock():
    """
    Test ParametricUMAPReducer with mocking to avoid TF/M1 issues.
    """

    class MockPUMAP:
        init_called = False

        def __init__(self, **kwargs):
            type(self).init_called = True
            self._history = {"loss": [0.1, 0.05]}

        def fit(self, X, y=None):
            self.fitted_ = True
            return self

        def transform(self, X):
            return np.zeros((len(X), 2))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockPUMAP,
    ):
        X = np.random.rand(20, 10).astype(np.float32)
        reducer = ParametricUMAPReducer(n_components=2, n_epochs=1, verbose=False)
        reducer.fit(X)

        assert MockPUMAP.init_called is True
        assert reducer.model.fitted_ is True

        z = reducer.transform(X)
        assert z.shape == (20, 2)
        assert reducer.loss_history_ == [0.1, 0.05]


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
    pytest.skip("Skipping Dask PCA to avoid coverage hangs.")
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

    try:
        reducer.fit(X_np)
    except Exception as e:
        warnings.warn(f"DaskPCA on numpy raised: {e}")


def test_dask_truncated_svd_reducer():
    """Test DaskTruncatedSVDReducer."""
    pytest.skip("Skipping Dask SVD to avoid coverage hangs.")
    X_np = np.random.rand(100, 10)
    X_da = da.from_array(X_np, chunks=(20, 10))

    reducer = DaskTruncatedSVDReducer(n_components=2)
    reducer.fit(X_da)

    X_emb_da = reducer.transform(X_da)
    X_emb = X_emb_da.compute()

    assert X_emb.shape == (100, 2)


# --- BaseReducer Tests ---


class DummyReducer(BaseReducer):
    def fit(self, X, y=None):
        self.model = "fitted"
        return self

    def transform(self, X):
        return X


def test_base_reducer_fit_transform():
    """Test default fit_transform implementation."""
    reducer = DummyReducer(n_components=2)
    X = np.zeros((5, 2))

    # fit_transform should call fit check model is set, then transform
    res = reducer.fit_transform(X)
    assert reducer.model == "fitted"
    assert np.array_equal(res, X)


def test_base_reducer_save_makedirs(tmp_path):
    """Test save creates subdirectories."""
    reducer = DummyReducer()
    save_path = tmp_path / "subdir" / "model.pkl"

    reducer.save(save_path)
    assert save_path.exists()


# --- PCAReducer Tests ---


def test_pca_unfitted_error():
    """Test errors when accessing unfitted PCA."""
    reducer = PCAReducer(n_components=2)

    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.explained_variance_ratio_

    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.components_


# --- IncrementalPCAReducer Tests ---


def test_incremental_pca_partial_fit_logic():
    """Test initialization logic in partial_fit."""
    reducer = IncrementalPCAReducer(n_components=2)
    assert reducer.model is None

    # partial_fit should init model
    reducer.partial_fit(np.random.rand(10, 5))
    assert reducer.model is not None

    # fit should overwrite/re-init
    old_model = reducer.model
    reducer.fit(np.random.rand(10, 5))
    assert reducer.model is not old_model


def test_linear_reducers_filter_unknown_params():
    """Verify linear reducers ignore unsupported constructor kwargs safely."""
    X = np.random.rand(20, 5)

    pca = PCAReducer(n_components=2, unsupported_param=123)
    pca.fit(X)
    assert pca.model is not None

    ipca = IncrementalPCAReducer(n_components=2, batch_size=10, unsupported_param=123)
    ipca.fit(X)
    assert ipca.model is not None

    ipca_stream = IncrementalPCAReducer(n_components=2, unsupported_param=123)
    ipca_stream.partial_fit(X[:10])
    assert ipca_stream.model is not None


def test_incremental_pca_unfitted_error():
    reducer = IncrementalPCAReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))


# --- Dask Reducer Tests (Mocked) ---


def test_dask_pca_mocked():
    """Test DaskPCAReducer logic without running Dask."""

    class MockDaskPCA:
        last_init = None
        last_fit = None
        last_transform = None

        def __init__(self, n_components, svd_solver="auto"):
            type(self).last_init = {
                "n_components": n_components,
                "svd_solver": svd_solver,
            }

        def fit(self, X):
            type(self).last_fit = X

        def transform(self, X):
            type(self).last_transform = X
            return "transformed"

    with patch(
        "coco_pipe.dim_reduction.reducers.linear.import_optional_dependency",
        return_value=MockDaskPCA,
    ):
        reducer = DaskPCAReducer(n_components=2, ignored_param=123)
        X = "dummy_dask_array"

        reducer.fit(X)
        assert MockDaskPCA.last_init == {"n_components": 2, "svd_solver": "auto"}
        assert MockDaskPCA.last_fit == X

        assert reducer.transform(X) == "transformed"
        assert MockDaskPCA.last_transform == X


def test_dask_pca_unfitted():
    reducer = DaskPCAReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform("data")


def test_dask_svd_mocked():
    """Test DaskTruncatedSVDReducer logic."""

    class MockSVD:
        last_init = None
        last_fit = None
        last_transform = None

        def __init__(self, n_components, algorithm="tsqr"):
            type(self).last_init = {
                "n_components": n_components,
                "algorithm": algorithm,
            }

        def fit(self, X):
            type(self).last_fit = X

        def transform(self, X):
            type(self).last_transform = X
            return "transformed"

    with patch(
        "coco_pipe.dim_reduction.reducers.linear.import_optional_dependency",
        return_value=MockSVD,
    ):
        reducer = DaskTruncatedSVDReducer(
            n_components=2,
            algorithm="randomized",
            ignored_param=123,
        )

        reducer.fit("data")
        assert MockSVD.last_init == {"n_components": 2, "algorithm": "randomized"}
        assert MockSVD.last_fit == "data"

        assert reducer.transform("data") == "transformed"
        assert MockSVD.last_transform == "data"
        assert reducer.get_quality_metadata() == {"algorithm": "randomized"}


def test_dask_svd_unfitted():
    reducer = DaskTruncatedSVDReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform("data")


# --- Manifold Reducer Tests ---


def test_isomap_unfitted_error():
    reducer = IsomapReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.reconstruction_error_


def test_lle_unfitted_error():
    reducer = LLEReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.reconstruction_error_


def test_mds_errors():
    reducer = MDSReducer(n_components=2)
    with pytest.raises(NotImplementedError):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.stress_


def test_spectral_errors():
    reducer = SpectralEmbeddingReducer(n_components=2)
    with pytest.raises(NotImplementedError):
        reducer.transform(np.zeros((5, 5)))


def test_manifold_reducers_filter_unknown_params():
    """Verify manifold reducers ignore unsupported constructor kwargs safely."""
    X = np.random.rand(30, 5)

    isomap = IsomapReducer(n_components=2, n_neighbors=5, unsupported_param=123)
    isomap.fit(X)
    assert isomap.model is not None

    lle = LLEReducer(
        n_components=2,
        n_neighbors=5,
        eigen_solver="dense",
        unsupported_param=123,
    )
    lle.fit(X)
    assert lle.model is not None

    mds = MDSReducer(n_components=2, random_state=0, unsupported_param=123)
    mds.fit(X)
    assert mds.model is not None

    spectral = SpectralEmbeddingReducer(
        n_components=2,
        random_state=0,
        unsupported_param=123,
    )
    spectral.fit(X)
    assert spectral.model is not None


# --- Neighbor Reducer Tests ---


def test_neighbor_reducers_filter_unknown_params():
    """Verify neighbor reducers ignore unsupported constructor kwargs safely."""
    X = np.random.rand(30, 5)

    tsne = TSNEReducer(
        n_components=2,
        perplexity=5,
        max_iter=250,
        unsupported_param=123,
    )
    emb = tsne.fit_transform(X)
    assert emb.shape == (30, 2)

    class MockUMAP:
        last_init = None

        def __init__(self, n_components=2, n_neighbors=15):
            type(self).last_init = {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
            }
            self.graph_ = "graph"

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.zeros((len(X), 2))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockUMAP,
    ):
        umap = UMAPReducer(n_components=2, n_neighbors=7, unsupported_param=123)
        umap.fit(X)
        assert MockUMAP.last_init == {"n_components": 2, "n_neighbors": 7}

    class MockPaCMAP:
        last_init = None

        def __init__(
            self,
            n_components=2,
            n_neighbors=10,
            MN_ratio=0.5,
            FP_ratio=2.0,
            nn_backend="faiss",
        ):
            type(self).last_init = {
                "n_components": n_components,
                "n_neighbors": n_neighbors,
                "MN_ratio": MN_ratio,
                "FP_ratio": FP_ratio,
                "nn_backend": nn_backend,
            }

        def fit_transform(self, X, init="pca"):
            return np.zeros((len(X), 2))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockPaCMAP,
    ):
        pacmap = PacmapReducer(
            n_components=2,
            n_neighbors=8,
            nn_backend="faiss",
            unsupported_param=123,
        )
        emb = pacmap.fit_transform(X)
        assert emb.shape == (30, 2)
        assert MockPaCMAP.last_init == {
            "n_components": 2,
            "n_neighbors": 8,
            "MN_ratio": 0.5,
            "FP_ratio": 2.0,
            "nn_backend": "faiss",
        }


def test_tsne_errors():
    reducer = TSNEReducer(n_components=2)
    with pytest.raises(NotImplementedError):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_quality_metadata()


def test_umap_unfitted_error():
    reducer = UMAPReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_diagnostics()


def test_pacmap_errors():
    reducer = PacmapReducer(n_components=2)
    with pytest.raises(NotImplementedError):
        reducer.transform(np.zeros((5, 5)))


def test_trimap_errors():
    reducer = TrimapReducer(n_components=2)
    with pytest.raises(NotImplementedError):
        reducer.transform(np.zeros((5, 5)))


def test_phate_unfitted_error():
    reducer = PHATEReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_diagnostics()


def test_parametric_umap_errors():
    # Test missing dependency
    with patch.dict(sys.modules, {"umap.parametric_umap": None}):
        reducer = ParametricUMAPReducer(n_components=2)
        with pytest.raises(ImportError, match="umap-learn is required"):
            reducer.fit(np.zeros((5, 5)))

    # Test unfitted transform/save
    class MockPUMAP:
        def __init__(self, **kwargs):
            self._history = {"loss": [0.1, 0.05]}

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.zeros((len(X), 2))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=MockPUMAP,
    ):
        reducer = ParametricUMAPReducer(n_components=2)
        reducer.fit(np.zeros((5, 5)))
        reducer_unfitted = ParametricUMAPReducer(n_components=2)
        with pytest.raises(RuntimeError, match="must be fitted"):
            reducer_unfitted.transform(np.zeros((5, 5)))

        with pytest.raises(RuntimeError, match="not fitted"):
            reducer_unfitted.save("dummy.pkl")


def test_neighbor_capabilities_api():
    """Verify neighbor reducer capabilities metadata."""
    tsne = TSNEReducer(n_components=2)
    assert tsne.capabilities["has_transform"] is False
    assert tsne.capabilities["is_stochastic"] is True

    umap = UMAPReducer(n_components=2)
    assert umap.capabilities["has_transform"] is True
    assert umap.capabilities["has_native_plot"] is True
    assert umap.capabilities["supported_diagnostics"] == ["graph_"]

    pacmap = PacmapReducer(n_components=2)
    assert pacmap.capabilities["has_transform"] is False

    trimap = TrimapReducer(n_components=2)
    assert trimap.capabilities["has_transform"] is False

    phate = PHATEReducer(n_components=2)
    assert phate.capabilities["has_transform"] is True
    assert phate.capabilities["has_native_plot"] is True
    assert phate.capabilities["supported_diagnostics"] == ["diff_potential"]

    pumap = ParametricUMAPReducer(n_components=2)
    assert pumap.capabilities["has_transform"] is True
    assert pumap.capabilities["supported_diagnostics"] == ["loss_history_"]


# --- Neural Reducer Tests ---


def test_ivis_errors():
    reducer = IVISReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_diagnostics()


def test_neural_capabilities_api():
    """Verify neural reducer capabilities metadata."""
    reducer = IVISReducer(n_components=2)
    caps = reducer.capabilities
    assert caps["has_transform"] is True
    assert caps["is_linear"] is False
    assert caps["is_stochastic"] is True
    assert caps["supported_diagnostics"] == ["loss_history_"]


# --- Spatiotemporal Reducer Tests ---


def test_dmd_errors():
    reducer = DMDReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.eigs_
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.modes_


def test_trca_errors():
    reducer = TRCAReducer(n_components=1, sfreq=100, filterbank=[[(8, 12), (6, 14)]])
    # Fit validation
    with pytest.raises(ValueError, match="TRCA requires 3D input"):
        reducer.fit(np.zeros((5, 5)))  # 2D
    with pytest.raises(ValueError, match="requires labels"):
        reducer.fit(np.zeros((2, 1, 4)))

    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((1, 1, 1)))
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_diagnostics()


# --- Topological Reducer Tests ---


def test_topo_ae_errors():
    reducer = TopologicalAEReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.transform(np.zeros((5, 5)))
    with pytest.raises(ValueError, match="expects 2D input"):
        reducer.fit(np.zeros((2, 2, 2)))

    # Loss history empty if not fitted
    assert reducer.loss_history_ == []


def test_topo_signature_logic():
    torch = pytest.importorskip("torch")

    from coco_pipe.dim_reduction.reducers.topology import TopologicalSignatureDistance

    sig = TopologicalSignatureDistance()
    x = torch.zeros((5, 2))
    z = torch.zeros((5, 2))
    # Edge case: No edges
    with patch.object(sig, "_get_active_pairs", return_value=[]):
        loss = sig(x, z)
        assert loss.item() == 0.0


def test_topo_device_init():
    from coco_pipe.dim_reduction.reducers.topology import _resolve_device

    fake_cuda = _FakeTopologyTorch(cuda_available=True)
    with patch(
        "coco_pipe.dim_reduction.reducers.topology._load_torch",
        return_value=fake_cuda,
    ):
        assert _resolve_device("auto") == "cuda"

    fake_mps = _FakeTopologyTorch(cuda_available=False, mps_available=True)
    with patch(
        "coco_pipe.dim_reduction.reducers.topology._load_torch",
        return_value=fake_mps,
    ):
        assert _resolve_device("auto") == "mps"

    with patch(
        "coco_pipe.dim_reduction.reducers.topology._load_torch",
        side_effect=RuntimeError("no torch"),
    ):
        assert _resolve_device("auto") == "cpu"


def test_reproducibility_stochastic_reducers(data):
    """Verify that random_state ensures reproducibility."""
    # Test UMAP as a representative stochastic reducer
    reducer1 = UMAPReducer(n_components=2, n_neighbors=10, random_state=42)
    reducer2 = UMAPReducer(n_components=2, n_neighbors=10, random_state=42)

    emb1 = reducer1.fit_transform(data)
    emb2 = reducer2.fit_transform(data)
    assert np.allclose(emb1, emb2)


def test_param_filtering_safety(data_ts):
    """Verify that unsupported params are filtered out to avoid TypeErrors"""
    # DMD does NOT support random_state
    data_dmd = data_ts.T
    reducer = DMDReducer(n_components=2, random_state=42)
    # This should not raise TypeError because of _filter_params
    reducer.fit(data_dmd)


def test_spatiotemporal_reducers_filter_unknown_params():
    """Verify spatiotemporal reducers ignore unsupported constructor kwargs safely."""
    X_dmd = np.random.rand(5, 20)
    X_trca = np.random.rand(6, 4, 30)
    y_trca = np.array([0, 0, 0, 1, 1, 1])

    class MockDMD:
        last_init = None

        def __init__(self, svd_rank=0, exact=False, opt=False, tlsq_rank=0):
            type(self).last_init = {
                "svd_rank": svd_rank,
                "exact": exact,
                "opt": opt,
                "tlsq_rank": tlsq_rank,
            }
            self.eigs = np.ones(max(svd_rank, 1))
            self.modes = np.ones((X_dmd.shape[0], max(svd_rank, 1)))

        def fit(self, X):
            return self

    class MockTRCA:
        last_init = None

        def __init__(self, sfreq=250.0, filterbank=None):
            type(self).last_init = {
                "sfreq": sfreq,
                "filterbank": filterbank,
            }
            self.sfreq = sfreq
            self.filterbank = filterbank
            self.coef_ = np.ones((1, 2, X_trca.shape[1]))

        def fit(self, X, y):
            return self

    with patch(
        "coco_pipe.dim_reduction.reducers.spatiotemporal.import_optional_dependency",
        return_value=MockDMD,
    ):
        dmd = DMDReducer(
            n_components=2,
            exact=True,
            unsupported_param=123,
        )
        dmd.fit(X_dmd)
        assert MockDMD.last_init == {
            "svd_rank": 2,
            "exact": True,
            "opt": False,
            "tlsq_rank": 0,
        }

    with patch(
        "coco_pipe.dim_reduction.reducers.spatiotemporal.import_optional_dependency",
        return_value=MockTRCA,
    ):
        fb = [[(8, 12), (6, 14)]]
        trca = TRCAReducer(
            n_components=1,
            sfreq=100.0,
            filterbank=fb,
            unsupported_param=123,
        )
        trca.fit(X_trca, y=y_trca)
        assert MockTRCA.last_init == {
            "sfreq": 100.0,
            "filterbank": fb,
        }


def test_topology_reducer_filters_unknown_params():
    """Verify the topology reducer filters unsupported raw Skorch kwargs safely."""
    X = np.random.rand(12, 6).astype(np.float32)
    fake_torch = _FakeTopologyTorch()
    _MockNeuralNetRegressor.last_init = None

    with (
        patch(
            "coco_pipe.dim_reduction.reducers.topology._build_topology_training_classes",
            return_value=(
                fake_torch,
                _MockTopologyModule,
                _MockTopologyLossCriterion,
            ),
        ),
        patch(
            "coco_pipe.dim_reduction.reducers.topology.import_optional_dependency",
            side_effect=lambda loader, feature=None, dependency=None, **kwargs: (
                _MockNeuralNetRegressor
                if (dependency or kwargs.get("dependency")) == "skorch"
                else loader()
            ),
        ),
    ):
        reducer = TopologicalAEReducer(
            n_components=2,
            epochs=2,
            batch_size=4,
            device="cpu",
            callbacks=[MagicMock()],
            unsupported_param=123,
        )
        reducer.fit(X)
        # Verify filtering and callback passing
        assert _MockNeuralNetRegressor.last_init is not None
        assert len(_MockNeuralNetRegressor.last_init["callbacks"]) == 1
        assert "unsupported_param" not in _MockNeuralNetRegressor.last_init


def test_capabilities_api(data):
    """Verify the reducer capabilities metadata."""
    reducer = PCAReducer(n_components=2)
    reducer.fit(data)
    caps = reducer.capabilities
    assert caps["has_transform"] is True
    assert caps["is_linear"] is True


def test_manifold_capabilities_api(data):
    """Verify manifold reducer capabilities metadata."""
    isomap = IsomapReducer(n_components=2, n_neighbors=5)
    isomap.fit(data)
    assert isomap.capabilities["has_transform"] is True
    assert isomap.capabilities["is_linear"] is False
    iso_meta = isomap.get_quality_metadata()
    assert "n_features_in_" in iso_meta
    assert "reconstruction_error_" in iso_meta

    mds = MDSReducer(n_components=2, random_state=0)
    mds.fit(data[:40])
    assert mds.capabilities["has_transform"] is False
    assert mds.capabilities["is_stochastic"] is True
    mds_meta = mds.get_quality_metadata()
    assert "stress_" in mds_meta
    assert "n_iter_" in mds_meta

    spectral = SpectralEmbeddingReducer(n_components=2, random_state=0)
    spectral.fit(data[:40])
    assert spectral.capabilities["has_transform"] is False
    spectral_meta = spectral.get_quality_metadata()
    assert "n_features_in_" in spectral_meta


def test_spatiotemporal_capabilities_api():
    """Verify spatiotemporal reducer capabilities metadata."""
    dmd = DMDReducer(n_components=2)
    assert dmd.capabilities["input_layout"] == "features_snapshots"
    assert dmd.capabilities["has_native_plot"] is True
    assert dmd.capabilities["supported_diagnostics"] == [
        "eigs_",
        "modes_",
        "reconstructed_data_",
    ]
    assert dmd.capabilities["is_linear"] is True

    trca = TRCAReducer(n_components=1)
    assert trca.capabilities["input_ndim"] == 3
    assert trca.capabilities["input_layout"] == "trials_channels_times"
    assert trca.capabilities["supported_diagnostics"] == ["coef_"]
    assert trca.capabilities["has_components"] is True
    assert trca.capabilities["is_linear"] is True


def test_topology_capabilities_api():
    """Verify topology reducer capabilities metadata."""
    topo = TopologicalAEReducer(n_components=2)
    assert topo.capabilities["has_transform"] is True
    assert list(topo.capabilities["supported_diagnostics"]) == ["loss_history_"]
    assert topo.capabilities["is_linear"] is False
    assert topo.capabilities["is_stochastic"] is True


def test_dmd_trca_contract_conformance(data_ts, data_trca):
    """Verify get_diagnostics and get_quality_metadata for spatiotemporal reducers."""
    # DMD
    dmd = DMDReducer(n_components=2)
    dmd.fit(data_ts.T)
    diag_dmd = dmd.get_diagnostics()
    assert "eigs_" in diag_dmd
    assert "modes_" in diag_dmd
    meta_dmd = dmd.get_quality_metadata()
    assert meta_dmd["svd_rank"] == 2
    assert "n_modes_" in meta_dmd

    # TRCA
    class MockTRCA:
        def __init__(self, sfreq=250.0, filterbank=None):
            self.sfreq = sfreq
            self.filterbank = filterbank
            self.coef_ = np.ones((1, 2, data_trca.shape[1]))

        def fit(self, X, y):
            return self

    with patch(
        "coco_pipe.dim_reduction.reducers.spatiotemporal.import_optional_dependency",
        return_value=MockTRCA,
    ):
        fb = [[(8, 12), (6, 14)]]
        trca = TRCAReducer(n_components=1, sfreq=100, filterbank=fb)
        y = np.concatenate([np.zeros(5, dtype=int), np.ones(5, dtype=int)])
        trca.fit(data_trca, y=y)
        diag_trca = trca.get_diagnostics()
        assert "coef_" in diag_trca
        meta_trca = trca.get_quality_metadata()
        assert meta_trca["n_components"] == 1
        assert meta_trca["sfreq"] == 100
        assert meta_trca["n_bands"] == 1


def test_lazy_import_stability():
    """Verify that lazy imports handle missing dependencies gracefully."""
    # Mock missing umap-learn for UMAPReducer
    with patch.dict(sys.modules, {"umap": None}):
        reducer = UMAPReducer()
        with pytest.raises(ImportError, match="umap-learn is required"):
            reducer.fit(np.zeros((10, 2)))

    # Mock missing pacmap for PacmapReducer
    with patch.dict(sys.modules, {"pacmap": None}):
        reducer = PacmapReducer()
        with pytest.raises(ImportError, match="pacmap is required"):
            reducer.fit(np.zeros((10, 2)))


def test_base_reducer_filter_params_edge_cases():
    """Test fallback paths in BaseReducer._filter_params."""
    reducer = PCAReducer()

    # 1. Target is a function (not a class)
    def my_func(a, b=1):
        pass

    params = {"a": 1, "c": 2}
    filtered = reducer._filter_params(my_func, params)
    assert filtered == {"a": 1}

    # 2. Target cannot be inspected (e.g. C extension or mock that fails)
    with patch("inspect.signature", side_effect=TypeError):
        assert reducer._filter_params(int, params) == params


def test_base_reducer_attribute_dict_edge_cases():
    """Test BaseReducer._attribute_dict when obj is None."""
    reducer = PCAReducer()
    assert reducer._attribute_dict(None, ("attr",)) == {}


def test_base_reducer_diagnostics_edge_cases():
    """Test get_diagnostics fitted check and property handling."""

    class MinimalReducer(BaseReducer):
        @property
        def capabilities(self):
            return {"supported_diagnostics": ["fail_attr"]}

        def fit(self, X, y=None):
            self.model = object()
            return self

        def transform(self, X):
            return X

    reducer = MinimalReducer()
    # model is None
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_diagnostics()

    # fitted but property access fails
    reducer.model = SimpleNamespace(explained_variance_ratio_=np.array([1]))
    # fail_attr is missing on model -> diag remains {}
    assert reducer.get_diagnostics() == {}


def test_base_reducer_discovery_honors_explicit_allowlists():
    """Base discovery should only query attributes declared in capabilities."""

    class DeclaredOnlyReducer(BaseReducer):
        @property
        def capabilities(self):
            return {
                "supported_diagnostics": ["declared_diag"],
                "supported_metadata": ["declared_meta"],
            }

        def fit(self, X, y=None):
            self.model = SimpleNamespace(
                declared_diag=np.array([1.0, 2.0]),
                declared_meta=3.0,
                hidden_diag=np.array([9.0]),
                hidden_meta=99.0,
            )
            return self

        def transform(self, X):
            return X

    reducer = DeclaredOnlyReducer().fit(np.zeros((2, 2)))
    diagnostics = reducer.get_diagnostics()
    metadata = reducer.get_quality_metadata()

    assert list(diagnostics) == ["declared_diag"]
    assert np.array_equal(diagnostics["declared_diag"], np.array([1.0, 2.0]))
    assert metadata == {"declared_meta": 3.0}


def test_base_reducer_wrapper_values_override_model_values():
    """Wrapper properties should override raw model attributes for declared keys."""

    class OverrideReducer(BaseReducer):
        @property
        def capabilities(self):
            return {
                "supported_diagnostics": ["shared_diag"],
                "supported_metadata": ["shared_meta"],
            }

        def fit(self, X, y=None):
            self.model = SimpleNamespace(
                shared_diag=np.array([1.0]),
                shared_meta="model",
            )
            return self

        def transform(self, X):
            return X

        @property
        def shared_diag(self):
            return np.array([2.0])

        @property
        def shared_meta(self):
            return "wrapper"

    reducer = OverrideReducer().fit(np.zeros((2, 2)))
    diagnostics = reducer.get_diagnostics()
    metadata = reducer.get_quality_metadata()

    assert np.array_equal(diagnostics["shared_diag"], np.array([2.0]))
    assert metadata == {"shared_meta": "wrapper"}


def test_base_reducer_quality_metadata_default():
    """Trigger default empty dict in get_quality_metadata."""
    reducer = DummyReducer(n_components=2)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_quality_metadata()

    _ = reducer.fit(np.zeros((5, 2)))
    assert reducer.get_quality_metadata() == {}


def test_topology_ae_edge_cases(data_ts):
    """Test TopologicalAEReducer edge cases and internal classes."""
    from coco_pipe.dim_reduction.reducers.topology import _resolve_device

    # Global mock for topology training classes to avoid gudhi/skorch issues
    with (
        patch(
            "coco_pipe.dim_reduction.reducers.topology._build_topology_training_classes"
        ) as mock_build,
        patch(
            "coco_pipe.dim_reduction.reducers.topology.import_optional_dependency"
        ) as mock_import,
    ):
        mock_torch = MagicMock()
        mock_ae = MagicMock()
        mock_crit = MagicMock()
        mock_build.return_value = (mock_torch, mock_ae, mock_crit)

        # Mock skorch NeuralNetRegressor
        mock_skorch = MagicMock()
        mock_import.return_value = mock_skorch

        reducer = TopologicalAEReducer(device="cpu")

        # 1. Input dim check in transform
        reducer.fit(data_ts)
        with pytest.raises(ValueError, match="expects 2D input"):
            reducer.transform(data_ts[0])

        # 2. get_pytorch_module when unfitted
        unfitted = TopologicalAEReducer()
        assert unfitted.get_pytorch_module() is None

        # 3. get_diagnostics/metadata when unfitted
        with pytest.raises(RuntimeError, match="must be fitted"):
            unfitted.get_diagnostics()
        with pytest.raises(RuntimeError, match="must be fitted"):
            unfitted.get_quality_metadata()

        # 4. _resolve_device exception path
        with patch(
            "coco_pipe.dim_reduction.reducers.topology._load_torch",
            side_effect=Exception,
        ):
            assert _resolve_device("auto") == "cpu"


def test_topological_signature_distance_and_module():
    """Test internal topology classes directly for coverage."""
    from coco_pipe.dim_reduction.reducers.topology import (
        TopologicalSignatureDistance,
        _build_topology_training_classes,
    )

    # Mock gudhi
    mock_st = MagicMock()
    mock_st.persistence_pairs.return_value = [([1], [2, 3])]

    mock_gd = MagicMock()
    mock_gd.RipsComplex.return_value.create_simplex_tree.return_value = mock_st

    with patch(
        "coco_pipe.dim_reduction.reducers.topology._load_gudhi", return_value=mock_gd
    ):
        dist = TopologicalSignatureDistance()
        x_np = np.random.rand(5, 10).astype(np.float32)

        # Build classes
        torch, AE, Criterion = _build_topology_training_classes()

        model = AE(input_dim=10, latent_dim=2)
        crit = Criterion(lam=0.1)

        x_t = torch.tensor(x_np)
        z_t, recon_t = model(x_t)

        loss = crit((z_t, recon_t), x_t)
        assert loss > 0

        # Test _get_active_pairs
        dist_matrix = torch.cdist(x_t, x_t)
        pairs = dist._get_active_pairs(dist_matrix, dim=0)
        assert isinstance(pairs, list)


def test_unfitted_errors_all_reducers():
    """Verify RuntimeError for all reducers when unfitted (covers miss lines)."""
    # Linear
    with pytest.raises(RuntimeError):
        PCAReducer().components_
    with pytest.raises(RuntimeError):
        IncrementalPCAReducer().get_components()
    with pytest.raises(RuntimeError):
        DaskPCAReducer().get_components()
    with pytest.raises(RuntimeError):
        DaskTruncatedSVDReducer().get_components()

    # Manifold
    with pytest.raises(RuntimeError):
        IsomapReducer().transform(np.zeros((5, 2)))
    with pytest.raises(RuntimeError):
        LLEReducer().transform(np.zeros((5, 2)))
    with pytest.raises(RuntimeError):
        MDSReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        SpectralEmbeddingReducer().get_diagnostics()

    # Neighbor
    with pytest.raises(RuntimeError):
        IVISReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        UMAPReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        TSNEReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        PacmapReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        TrimapReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        PHATEReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        ParametricUMAPReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        IVISReducer().get_diagnostics()


def test_spatiotemporal_diagnostics_unfitted():
    """Test spatiotemporal diagnostics coverage."""
    with pytest.raises(RuntimeError):
        DMDReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        TRCAReducer().get_diagnostics()
    with pytest.raises(RuntimeError):
        DMDReducer().get_quality_metadata()
    with pytest.raises(RuntimeError):
        TRCAReducer().get_quality_metadata()


def test_reducer_property_access_coverage(data_ts, data_trca):
    """Exercise remaining reducer accessors for coverage."""
    # TSNEReducer metadata
    tsne = TSNEReducer(n_components=2)
    tsne.model = MagicMock(kl_divergence_=0.5, n_iter_=100)
    tsne_meta = tsne.get_quality_metadata()
    assert tsne_meta["kl_divergence_"] == 0.5
    assert tsne_meta["n_iter_"] == 100

    # PHATEReducer diagnostics
    phate = PHATEReducer(n_components=2)
    # Using a MagicMock that doesn't trigger LinAlgError in PHATE logic
    phate.model = MagicMock()
    phate.model.diff_potential = np.zeros(2)
    phate.model.diff_op = np.zeros(2)
    phate_diag = phate.get_diagnostics()
    assert phate_diag["diff_potential"] is not None

    # ParametricUMAPReducer
    pumap = ParametricUMAPReducer(n_components=2)
    pumap.model = MagicMock(_history={"loss": [0.1]})
    assert pumap.loss_history_ == [0.1]

    # DMDReducer
    dmd = DMDReducer(n_components=2, force_transpose=True)
    # Mock model properties to avoid pydmd's read-only attributes
    mock_model = MagicMock()
    mock_model.modes = np.zeros((5, 2))
    mock_model.eigs = np.zeros(2)
    mock_model.reconstructed_data = np.zeros((5, 20))
    dmd.model = mock_model

    assert dmd.svd_rank == 2
    assert dmd.n_modes_ == 2
    assert dmd.reconstructed_data_ is not None
    assert dmd.get_components().shape == (2, 5)

    # TRCAReducer
    trca = TRCAReducer(n_components=1)
    trca.model = MagicMock(coef_=np.zeros((1, 2, 3)))
    assert trca.n_bands == 1
    assert trca.n_classes == 2
    assert trca.get_diagnostics()["coef_"].shape == (1, 2, 3)
    assert trca.get_components().shape == (1, 2, 3)

    # TopologicalAEReducer
    topo = TopologicalAEReducer(n_components=2)
    topo.model = MagicMock(module_="mock-module")
    assert topo.get_pytorch_module() == "mock-module"
    topo.model = None
    assert topo.get_pytorch_module() is None

    with patch(
        "coco_pipe.dim_reduction.reducers.topology.import_optional_dependency"
    ) as mock_import:
        from coco_pipe.dim_reduction.reducers.topology import _load_gudhi

        _load_gudhi()
        assert mock_import.called


def test_topology_signature_distance_dim1():
    """Exercise dim=1 path in TopologicalSignatureDistance."""
    dist = TopologicalSignatureDistance()
    mock_st = MagicMock()
    # Mock persistence_pairs to return a pair for dim=1
    # birth_simplex len 2 for dim=1
    mock_st.persistence_pairs.return_value = [([0, 1], [0, 1, 2])]

    mock_gd = MagicMock()
    mock_gd.RipsComplex.return_value.create_simplex_tree.return_value = mock_st

    with patch(
        "coco_pipe.dim_reduction.reducers.topology._load_gudhi", return_value=mock_gd
    ):
        pairs = dist._get_active_pairs(MagicMock(), dim=1)
        assert len(pairs) == 1
        # birth_simplex was [0, 1]
        assert pairs[0] == (0, 1)


def test_reducer_unfitted_property_errors():
    """Verify normalized property accessors raise RuntimeError when unfitted."""
    dmd = DMDReducer()
    with pytest.raises(RuntimeError):
        _ = dmd.eigs_
    with pytest.raises(RuntimeError):
        _ = dmd.modes_
    with pytest.raises(RuntimeError):
        _ = dmd.reconstructed_data_

    trca = TRCAReducer()
    with pytest.raises(RuntimeError):
        trca.get_components()

    tsne = TSNEReducer()
    with pytest.raises(RuntimeError):
        tsne.get_quality_metadata()

    # Extra spatiotemporal coverage
    # 1. DMDReducer.get_components
    dmd = DMDReducer(n_components=2)
    dmd.model = MagicMock(modes=np.zeros((5, 2)))
    assert dmd.get_components().shape == (2, 5)

    # 2. DMD force_transpose in fit
    with patch(
        "coco_pipe.dim_reduction.reducers.spatiotemporal.import_optional_dependency"
    ) as mock_import:
        mock_import.return_value = MagicMock()
        dmd = DMDReducer(force_transpose=True)
        # Mock _build_estimator to return a MagicMock
        with patch.object(dmd, "_build_estimator", return_value=MagicMock()):
            dmd.fit(np.random.rand(10, 5))

    # 3. TRCAReducer.get_components
    trca = TRCAReducer()
    trca.model = MagicMock(coef_=np.zeros((1, 2, 3)))
    assert trca.get_components() is not None

    # Extra neighbor coverage
    pumap = ParametricUMAPReducer()
    pumap.model = MagicMock()
    _ = pumap.transform(np.random.rand(10, 5))
    pumap.model._history = {"loss": [0.1, 0.2]}
    assert pumap.loss_history_ == [0.1, 0.2]
    with patch("joblib.dump"):
        pumap.save("/tmp/test.jbl")

    trimap = TrimapReducer()
    assert trimap.capabilities is not None


def test_manifold_unfitted_properties():
    iso = IsomapReducer()
    with pytest.raises(RuntimeError):
        _ = iso.reconstruction_error_

    lle = LLEReducer()
    with pytest.raises(RuntimeError):
        _ = lle.reconstruction_error_

    mds = MDSReducer()
    with pytest.raises(RuntimeError):
        _ = mds.stress_


def test_spatiotemporal_extra_coverage():
    dmd = DMDReducer()
    assert dmd.n_modes_ is None

    trca = TRCAReducer()
    assert trca.n_bands == 1
    assert trca.n_classes is None
    with pytest.raises(RuntimeError, match="must be fitted"):
        trca.get_components()

    dmd.model = MagicMock(modes=np.zeros((5, 2)))
    res = dmd.transform(np.zeros((5, 10)).tolist())
    assert res.shape == (10, 2)

    dmd.force_transpose = True
    res = dmd.transform(np.zeros((10, 5)))
    assert res.shape == (10, 2)

    # TRCA transform 3D error
    trca = TRCAReducer()
    trca.model = MagicMock(coef_=np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="3D input"):
        trca.transform(np.zeros((5, 5)))

    # TRCA label parity error
    with pytest.raises(ValueError, match="one label per trial"):
        trca.fit(np.zeros((10, 5, 5)), y=np.zeros(5))

    # TRCA get_components
    assert trca.get_components().shape == (1, 2, 3)


def test_linear_extra_coverage():
    # PCAReducer.get_components
    pca = PCAReducer(n_components=2)
    pca.fit(np.random.rand(20, 5))
    assert pca.get_components().shape == (2, 5)

    # IncrementalPCA capabilities
    ipca = IncrementalPCAReducer(n_components=2)
    assert ipca.capabilities["supported_metadata"] == [
        "n_components_",
        "noise_variance_",
        "n_samples_seen_",
    ]

    # DaskPCA capabilities
    dpca = DaskPCAReducer(n_components=2)
    assert dpca.capabilities["is_linear"] is True


def test_manifold_neighbor_extra_coverage():
    # LLEReducer.reconstruction_error_
    lle = LLEReducer(n_components=2)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = lle.reconstruction_error_
    lle.model = MagicMock(reconstruction_error_=0.1, n_features_in_=5)
    assert lle.reconstruction_error_ == 0.1
    assert lle.capabilities["has_transform"] is True  # hits 258

    # ParametricUMAPReducer.loss_history_
    pumap = ParametricUMAPReducer()
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = pumap.loss_history_
