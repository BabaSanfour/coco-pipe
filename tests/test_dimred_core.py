import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from coco_pipe.dim_reduction.config import UMAPConfig
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.reducers.base import BaseReducer


def _install_fake_trca(monkeypatch, n_channels: int = 5):
    class FakeTRCA:
        def __init__(self, sfreq=250.0, filterbank=None, **kwargs):
            self.sfreq = sfreq
            self.filterbank = filterbank or [[(8, 12), (6, 14)]]
            self.coef_ = np.zeros((1, 1, n_channels))

        def fit(self, X, y):
            n_classes = len(np.unique(y))
            self.coef_ = np.ones((1, n_classes, X.shape[1]))
            return self

    fake_trca_mod = types.SimpleNamespace(TRCA=FakeTRCA)
    fake_trca_utils = types.SimpleNamespace(
        bandpass=lambda data, sfreq, Wp=None, Ws=None: data
    )
    fake_meegkit = types.SimpleNamespace(
        trca=fake_trca_mod,
        utils=types.SimpleNamespace(trca=fake_trca_utils),
    )

    monkeypatch.setitem(sys.modules, "meegkit", fake_meegkit)
    monkeypatch.setitem(sys.modules, "meegkit.trca", fake_trca_mod)
    monkeypatch.setitem(sys.modules, "meegkit.utils", fake_meegkit.utils)
    monkeypatch.setitem(sys.modules, "meegkit.utils.trca", fake_trca_utils)


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


def test_validation_3d(monkeypatch):
    fb = [[(8, 12), (6, 14)]]
    dr = DimReduction("TRCA", params={"filterbank": fb, "sfreq": 100})

    X_2d = np.random.rand(20, 5)
    with pytest.raises(ValueError, match="requires 3D input"):
        dr.fit(X_2d)

    X_3d = np.random.rand(10, 5, 100)
    # TRCA needs labels y for fit
    y = np.concatenate([np.zeros(5, dtype=int), np.ones(5, dtype=int)])

    # Check fit works
    _install_fake_trca(monkeypatch)
    dr.fit(X_3d, y=y)


def test_fit_transform_pca():
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    emb = dr.fit_transform(X)
    assert emb.shape == (20, 2)


def test_score():
    X = np.random.rand(50, 10)  # 50 samples
    dr = DimReduction("PCA", n_components=2)
    emb = dr.fit_transform(X)
    scores = dr.score(emb, X=X)
    assert "metrics" in scores
    assert "metadata" in scores
    assert "diagnostics" in scores
    assert "trustworthiness" in scores["metrics"]
    assert "continuity" in scores["metrics"]
    assert 0 <= scores["metrics"]["trustworthiness"] <= 1.0


def test_plot_removed_from_manager():
    dr = DimReduction("PCA", n_components=2)
    assert not hasattr(dr, "plot")
    assert not hasattr(dr, "plot_native")


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


def test_dimreduction_interpret():
    X = np.random.rand(20, 5)
    feature_names = ["A", "B", "C", "D", "E"]
    dr = DimReduction("PCA", n_components=2)
    dr.fit_transform(X)

    res = dr.interpret(
        X, X_emb=dr.transform(X), analyses=["correlation"], feature_names=feature_names
    )
    assert "correlation" in res["analysis"]
    assert len(res["records"]) > 0
    assert dr.interpretation_ == res["analysis"]
    assert dr.interpretation_records_ == res["records"]


def test_dimreduction_summary():
    X = np.random.rand(24, 6)

    reducer = DimReduction("PCA", n_components=2, random_state=42)
    emb = reducer.fit_transform(X)
    reducer.score(emb, X=X)

    summary = reducer.get_summary()
    assert summary["method"] == "PCA"
    assert "metrics" in summary
    assert "quality_metadata" in summary
    assert "diagnostics" in summary
    assert "interpretation" in summary
    assert "interpretation_records" in summary


def test_score_attributes():
    X = np.random.rand(50, 10)
    dr = DimReduction("PCA", n_components=2)
    emb = dr.fit_transform(X)
    scores = dr.score(emb, X=X)
    # PCA should have 'explained_variance_ratio_' in diagnostics
    assert "explained_variance_ratio_" in scores["diagnostics"]
    assert len(scores["diagnostics"]["explained_variance_ratio_"]) == 2


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

    # The DimReduction class no longer caches embedding_
    assert not hasattr(dr, "embedding_")


def test_score_specific_reducers():
    """Test that score() scrapes specific attributes from different reducers."""
    X = np.random.rand(20, 5)

    # 1. MDS -> stress_
    dr_mds = DimReduction("MDS", n_components=2)
    emb_mds = dr_mds.fit_transform(X)
    scores_mds = dr_mds.score(emb_mds, X=X)
    assert "stress_" in scores_mds["metadata"]
    assert isinstance(scores_mds["metadata"]["stress_"], float)

    # 2. t-SNE -> kl_divergence_
    # Note: perplexity must be < n_samples (20).
    dr_tsne = DimReduction("TSNE", n_components=2, perplexity=5)
    emb_tsne = dr_tsne.fit_transform(X)
    scores_tsne = dr_tsne.score(emb_tsne, X=X)
    assert "kl_divergence_" in scores_tsne["metadata"]
    assert isinstance(scores_tsne["metadata"]["kl_divergence_"], float)

    # Repopulate PCA check
    dr_pca = DimReduction("PCA", n_components=2)
    emb_pca = dr_pca.fit_transform(X)
    scores_pca = dr_pca.score(emb_pca, X=X)
    assert "singular_values_" in scores_pca["diagnostics"]
    assert len(scores_pca["diagnostics"]["singular_values_"]) == 2


def test_score_allowlist():
    """Test standard diagnostics contract."""

    class MockModel:
        def __init__(self):
            # These attributes exist on model
            self.graph = np.zeros((5, 5))
            self.diff_potential = np.random.rand(5)
            self.eigs = np.array([1.0, 0.5])
            self.ignored_attr = "should not be included"

    class MockReducer(BaseReducer):
        def __init__(self):
            super().__init__(n_components=2)
            self.model = MockModel()

        @property
        def capabilities(self):
            return {"supported_diagnostics": ["diff_potential", "eigs"]}

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[:, :2]

        def fit_transform(self, X, y=None):
            return X[:, :2]

        @property
        def diff_potential(self):
            return self.model.diff_potential

        @property
        def eigs(self):
            return self.model.eigs

    dr = DimReduction("PCA", n_components=2)  # Dummy init
    dr.reducer = MockReducer()
    X_emb = np.zeros((20, 2))
    X = np.zeros((20, 5))
    scores = dr.score(X_emb, X=X)

    # 'graph' and 'ignored_attr' should be absent because
    # get_diagnostics doesn't return them
    assert "graph" not in scores["diagnostics"]
    assert "ignored_attr" not in scores["diagnostics"]
    # these should be present
    assert "diff_potential" in scores["diagnostics"]
    assert "eigs" in scores["diagnostics"]


def test_get_components(monkeypatch):
    """Test retrieval of linear components/patterns."""
    # 1. PCA (sklearn has components_)
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit(X)
    comps = dr.get_components()
    assert comps.shape == (2, 5)

    # 2. DMD (spatiotemporal)
    X_ts = np.random.rand(5, 50)  # features x snapshots
    dr_dmd = DimReduction("DMD", n_components=2)
    dr_dmd.fit(X_ts)
    assert dr_dmd.get_components().shape == (2, 5)

    # 3. TRCA (spatiotemporal)
    X_3d = np.random.rand(10, 5, 100)  # trials x channels x times
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    dr_trca = DimReduction("TRCA", n_components=1)
    _install_fake_trca(monkeypatch)
    dr_trca.fit(X_3d, y=y)
    assert dr_trca.get_components().shape == (1, 2, 5)

    # 4. Fail case (Neural/Non-linear)
    dr_umap = DimReduction("UMAP")
    dr_umap.reducer.get_components = lambda: (_ for _ in ()).throw(
        ValueError("does not expose public get_components")
    )
    with pytest.raises(ValueError, match="does not expose public get_components"):
        dr_umap.get_components()


def test_init_from_config_object():
    """Test initialization with a typed reducer config object."""
    dr = DimReduction(UMAPConfig(method="UMAP", n_components=2, n_neighbors=15))
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


def test_score_errors_and_edge_cases(monkeypatch):
    """Test score method edge cases."""
    dr = DimReduction("PCA", n_components=2)

    # 4. Pass explicit embedding
    X = np.random.rand(10, 5)
    X_emb = np.random.rand(10, 2)
    dr.fit(X)
    scores = dr.score(X_emb, X=X)
    assert "metrics" in scores

    # Edge case: trajectory-aware scoring on 3D embeddings
    dr_trca = DimReduction("TRCA", n_components=2)
    # TRCA needs a fit for metadata
    X_3d = np.random.rand(10, 5, 100)
    y = np.concatenate([np.zeros(5, dtype=int), np.ones(5, dtype=int)])
    _install_fake_trca(monkeypatch)
    dr_trca.fit(X_3d, y=y)

    traj_3d = np.zeros((10, 3, 2))  # Fake 3D embedding
    scores = dr_trca.score(traj_3d, X=np.zeros((10, 3, 5)))
    assert "trajectory_speed_mean" in scores["metrics"]
    assert "trajectory_acceleration_mean" in scores["metrics"]
    assert "trajectory_curvature_mean" in scores["metrics"]
    assert "trajectory_turning_angle_mean" in scores["metrics"]
    assert "trajectory_dispersion_mean" in scores["metrics"]
    assert "trajectory_path_length_final" in scores["metrics"]
    assert "trajectory_displacement_final" in scores["metrics"]
    assert "trajectory_tortuosity_final" in scores["metrics"]

    dr_sep = DimReduction("TRCA", n_components=2)
    _install_fake_trca(monkeypatch)
    dr_sep.fit(X_3d, y=y)

    traj = np.zeros((4, 6, 2))
    labels = np.array(["A", "A", "B", "B"])
    traj[0, :, 0] = -1.0
    traj[1, :, 0] = 0.0
    traj[2, :, 0] = 2.0
    traj[3, :, 0] = 3.0
    traj[:, :, 1] = np.arange(6, dtype=float)
    centroid_scores = dr_sep.score(
        traj,
        X=np.zeros((4, 5, 5)),
        metrics=["trajectory_separation"],
        labels=labels,
        separation_method="centroid",
    )
    ratio_scores = dr_sep.score(
        traj,
        X=np.zeros((4, 5, 5)),
        metrics=["trajectory_separation"],
        labels=labels,
        separation_method="within_between_ratio",
    )

    assert (
        centroid_scores["metrics"]["trajectory_separation_auc::A::B"]
        != ratio_scores["metrics"]["trajectory_separation_auc::A::B"]
    )


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

    with patch(
        "coco_pipe.dim_reduction.core.get_reducer_class",
        side_effect=lambda m: DummyForPickle if m == "DUMMY" else None,
    ):
        loaded = DimReduction.load(save_path, method="DUMMY")

    assert isinstance(loaded, DimReduction)
    assert loaded.method == "DUMMY"
    assert loaded.n_components == 3


def test_diagnostics_api():
    """Verify the get_diagnostics()."""
    X = np.random.rand(50, 10)
    dr = DimReduction(method="TSNE", n_components=2, perplexity=5, random_state=42)
    emb = dr.fit_transform(X)
    scores = dr.score(emb, X=X)
    # Check for direct attribute extraction via the new API
    assert "kl_divergence_" in scores["metadata"]
    assert "n_iter_" in scores["metadata"]
    # Ensure it doesn't contain broad dir() noise
    assert "__init__" not in scores["metadata"]


def test_summary_and_metadata_api():
    """Test get_summary, get_metrics, and capabilities."""
    X = np.random.rand(20, 5)
    dr = DimReduction("PCA", n_components=2, random_state=42)

    # 1. Initial state
    assert dr.random_state == 42
    assert "input_ndim" in dr.capabilities
    assert dr.get_metrics() == {}

    # 2. Scored state
    emb = dr.fit_transform(X)
    dr.score(emb, X=X)

    metrics = dr.get_metrics()
    assert "trustworthiness" in metrics

    summary = dr.get_summary()
    assert summary["method"] == "PCA"
    assert summary["n_components"] == 2
    assert summary["random_state"] == 42
    assert "trustworthiness" in summary["metrics"]
    assert "explained_variance_ratio_" in summary["diagnostics"]
    assert len(summary["metric_records"]) > 0

    # 3. Reset check (fit clears cache)
    dr.fit(X)
    assert dr.get_metrics() == {}
    assert dr.metric_records_ == []


def test_score_no_metrics_payload_note():
    """Test that a note is added if metrics payload is empty."""
    X = np.random.rand(10, 5)
    dr = DimReduction("PCA", n_components=2)
    dr.fit(X)
    # Request a metric that doesn't exist for 2D to trigger empty payload
    with patch("coco_pipe.dim_reduction.core.evaluate_embedding") as mock_eval:
        mock_eval.return_value = {
            "metrics": {},
            "metadata": {},
            "diagnostics": {},
            "records": [],
        }
        scores = dr.score(np.zeros((10, 2)), X=X)
        assert "note" in scores["metrics"]
        assert "Metrics unavailable" in scores["metrics"]["note"]
