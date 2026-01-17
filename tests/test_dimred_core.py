import sys
import types
from unittest.mock import MagicMock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from coco_pipe.dim_reduction.config import DimReductionConfig
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
    dr.fit(X_2d)  # Should pass


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
    X = np.random.rand(50, 10)  # 50 samples
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
    assert dr.reducer_kwargs["n_neighbors"] == 5


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
    dr.fit_transform(X)  # Ensures embedding_ is set

    # Error: 'shepard' requires X
    with pytest.raises(ValueError, match="requires original data 'X'"):
        dr.plot(mode="shepard")

    # Success case requires 'X'
    fig = dr.plot(mode="shepard", X=X)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # Streamlines error (requires V_emb)
    with pytest.raises(ValueError, match="requires velocity vectors 'V_emb'"):
        dr.plot(mode="streamlines")

    # Unknown mode
    with pytest.raises(ValueError, match="Unknown plot mode"):
        dr.plot(mode="invalid_mode")


def test_plot_enhanced():
    """Test enhanced plotting features (3D, metrics overlay)."""
    # Create 3D data and embedding
    X = np.random.rand(20, 5)

    # Mock a reducer that outputs 3 dims
    dr = DimReduction("PCA", n_components=3)
    # Use fit_transform to set self.embedding_
    dr.fit_transform(X)

    # 1. Test 3D plotting
    fig = dr.plot(dims=(0, 1, 2))
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) > 0
    plt.close(fig)

    # 2. Test metrics overlay (2D default dims)
    fig = dr.plot(X=X, show_metrics=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # 3. Test mode='metrics'
    fig = dr.plot(mode="metrics", X=X)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_diagnostics():
    """Test diagnostic plotting modes."""
    X = np.random.rand(20, 5)

    # 1. Linear (PCA)
    dr = DimReduction("PCA", n_components=3)
    dr.fit_transform(X)  # Set embedding

    fig = dr.plot(mode="diagnostics")
    assert isinstance(fig, plt.Figure)
    assert "Explained Variance" in fig.axes[0].get_title()
    plt.close(fig)

    # 2. Mock Neural
    class MockNeuralReducer(BaseReducer):
        def __init__(self):
            super().__init__(n_components=2)
            self.loss_history_ = [0.5, 0.4, 0.3, 0.2]
            self.model = "MockNetwork"

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[:, :2]

    dr_neural = DimReduction("PCA", n_components=2)
    dr_neural.reducer = MockNeuralReducer()
    dr_neural.embedding_ = np.random.rand(20, 2)  # Manually set embedding
    dr_neural.method = "UseMockNeural"

    fig = dr_neural.plot(mode="diagnostics")
    assert isinstance(fig, plt.Figure)
    assert "Loss History" in fig.axes[0].get_title()
    plt.close(fig)


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

    dr = DimReduction("PCA", n_components=2)  # Dummy init
    dr.reducer = MockReducer()
    dr.embedding_ = np.zeros((20, 2))  # Dummy embedding (20 samples)

    X = np.zeros((20, 5))
    scores = dr.score(X)

    # 'graph' should be filtered out (not in allow_list and possibly large)
    assert "graph" not in scores
    assert "diff_potential" in scores
    assert "eigs" in scores
    assert "ignored_attr" not in scores


def test_plot_native(monkeypatch):
    """Test native plotting calls with mocks."""
    np.random.rand(20, 5)

    # 1. Test PHATE
    dr_phate = DimReduction("PHATE")

    # Mock phate module
    mock_phate = types.SimpleNamespace(
        plot=types.SimpleNamespace(scatter=lambda *a, **k: plt.figure())
    )
    monkeypatch.setitem(sys.modules, "phate", mock_phate)

    dr_phate.reducer.model = "MockPHATEModel"

    fig = dr_phate.plot(mode="native")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # 2. Test UMAP
    # Mock umap.plot (requires umap to be imported)
    import umap

    mock_plot_mod = types.SimpleNamespace(
        points=lambda *a, **k: plt.figure().add_subplot(111)
    )
    monkeypatch.setattr(umap, "plot", mock_plot_mod, raising=False)
    monkeypatch.setitem(sys.modules, "umap.plot", mock_plot_mod)

    dr_umap = DimReduction("UMAP")
    dr_umap.reducer.model = "MockUMAPModel"

    ax = dr_umap.plot(mode="native")
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)

    # 3. Test DMD
    # Mock DMD plotting
    dr_dmd = DimReduction("DMD")
    mock_dmd_model = types.SimpleNamespace(plot_eigs=lambda **k: plt.figure())
    dr_dmd.reducer.model = mock_dmd_model

    fig = dr_dmd.plot(mode="native")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    # 4. Unsupported
    dr_pca = DimReduction("PCA")
    with pytest.raises(NotImplementedError, match="Native plotting not supported"):
        dr_pca.plot(mode="native")


def test_get_components():
    """Test retrieval of linear components/patterns."""
    # 1. PCA (sklearn has components_)
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit(X)

    comps = dr.get_components()
    assert comps.shape == (2, 5)

    # 2. Mock model with patterns_ (e.g. TRCA-like)
    class MockPatternReducer(BaseReducer):
        def __init__(self):
            super().__init__(n_components=2)
            self.patterns_ = np.ones((2, 5))

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    dr_mock = DimReduction(
        "PCA"
    )  # method name doesn't matter for custom reducer injection
    dr_mock.reducer = MockPatternReducer()

    pats = dr_mock.get_components()
    assert np.all(pats == 1)

    # 3. Fail case (Neural/Non-linear)
    dr_umap = DimReduction("UMAP")
    dr_umap.fit(X)
    with pytest.raises(ValueError, match="does not appear to have linear components"):
        dr_umap.get_components()


def test_init_from_config_object():
    """Test initialization with a Pydantic DimReductionConfig object."""
    mock_config = MagicMock(spec=DimReductionConfig)
    mock_inner = MagicMock()
    mock_inner.method = "UMAP"
    mock_inner.n_components = 2
    mock_inner.model_dump.return_value = {"n_neighbors": 15}
    mock_config.config = mock_inner

    dr = DimReduction(mock_config)
    assert dr.method == "UMAP"
    assert dr.n_components == 2
    assert dr.reducer_kwargs["n_neighbors"] == 15


def test_validate_input_errors():
    """Test validation errors for dimension mismatches."""
    # Test 3D method with 2D data
    dr = DimReduction("TRCA", n_components=2)
    X_2d = np.zeros((10, 5))
    with pytest.raises(ValueError, match="requires 3D input"):
        dr._validate_input(X_2d)

    # Test 2D method with 3D data
    dr2 = DimReduction("PCA", n_components=2)
    X_3d = np.zeros((10, 5, 2))
    with pytest.raises(ValueError, match="requires 2D input"):
        dr2._validate_input(X_3d)


def test_score_errors_and_edge_cases():
    """Test score method edge cases."""
    dr = DimReduction("PCA", n_components=2)

    # Error: No embedding available
    X = np.zeros((10, 5))
    with pytest.raises(RuntimeError, match="No embedding available"):
        dr.score(X)

    # Edge case: 3D data returns NaNs
    dr_trca = DimReduction("TRCA", n_components=2)
    dr_trca.embedding_ = np.zeros((10, 2, 5))  # Fake 3D embedding
    scores = dr_trca.score(np.zeros((10, 5, 5)), X_emb=dr_trca.embedding_)
    assert np.isnan(scores["trustworthiness"])
    assert "undefined for 3D" in scores.get("note", "")


def test_plot_errors_coverage():
    """Test plot method error branches."""
    dr = DimReduction("PCA", n_components=2)

    # Error: Not fitted (mode='embedding')
    with pytest.raises(RuntimeError, match="Model is not fitted"):
        dr.plot(mode="embedding")

    dr.embedding_ = np.zeros((10, 2))

    # Error: show_metrics=True without X
    with pytest.raises(ValueError, match="requires 'X'"):
        dr.plot(mode="embedding", show_metrics=True)

    # Error: mode='metrics' without X
    with pytest.raises(ValueError, match="requires 'X'"):
        dr.plot(mode="metrics")

    # Error: mode='streamlines' without V_emb
    with pytest.raises(ValueError, match="requires velocity vectors"):
        dr.plot(mode="streamlines")

    # Error: Unknown mode
    with pytest.raises(ValueError, match="Unknown plot mode"):
        dr.plot(mode="super_cool_mode")


class DummyForPickle(BaseReducer):
    def __init__(self, n_components=2):
        super().__init__(n_components=n_components)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


def test_load_wrapper_reconstruction_coverage(tmp_path):
    """Test that DimReduction.load() correctly reconstructs the manager."""
    dummy = DummyForPickle(n_components=3)
    save_path = tmp_path / "dummy.pkl"
    dummy.save(save_path)

    with patch.dict(
        "coco_pipe.dim_reduction.core.METHODS_DICT", {"DUMMY": DummyForPickle}
    ), patch("coco_pipe.dim_reduction.core.METHODS", ["DUMMY"]):
        loaded = DimReduction.load(save_path, method="DUMMY")

    assert isinstance(loaded, DimReduction)
    assert loaded.method == "DUMMY"
    assert loaded.n_components == 3
