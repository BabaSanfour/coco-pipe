"""Tests for the dimensionality reducers."""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_blobs

import coco_pipe.dim_reduction.reducers as reducers
from coco_pipe.dim_reduction.reducers.base import BaseReducer

X, _ = make_blobs(n_samples=60, n_features=8, centers=3, random_state=0)
X = X.astype(np.float32)

# Small settings so every backend runs in seconds. IVIS and ParametricUMAP run
# on TensorFlow, which deadlocks here once pandas / pyarrow is imported; their
# own libraries are replaced below.
SETTINGS = {
    "IncrementalPCAReducer": {},
    "PCAReducer": {},
    "IsomapReducer": {"n_neighbors": 5},
    "LLEReducer": {"n_neighbors": 5, "eigen_solver": "dense"},
    "MDSReducer": {},
    "SpectralEmbeddingReducer": {},
    "TSNEReducer": {"perplexity": 5},
    "DaskPCAReducer": {},
    "DaskTruncatedSVDReducer": {},
    "UMAPReducer": {"n_neighbors": 5},
    "PacmapReducer": {"init": "random"},
    "TrimapReducer": {},
    "PHATEReducer": {"knn": 25},  # fewer neighbours leave the clusters apart
    "DMDReducer": {},
    "TRCAReducer": {"sfreq": 100, "filterbank": [[(8, 12), (6, 14)]]},
    "TopologicalAEReducer": {"epochs": 2, "batch_size": 16, "device": "cpu"},
}
# These pass extra keyword arguments on to their library, which checks them.
FORWARDS_KWARGS = {"PHATEReducer"}


def _inputs(reducer):
    layout = reducer.capabilities["input_layout"]
    rng = np.random.RandomState(0)
    if layout == "trials_channels_times":
        return rng.rand(10, 5, 100), np.repeat([0, 1], 5)
    if layout == "features_snapshots":
        return rng.rand(20, 60).astype(np.float32), None
    if type(reducer).__name__.startswith("Dask"):
        import dask.array as da

        return da.from_array(X, chunks=(20, 8)), None
    return X, None


def _compute(values):
    return np.asarray(values.compute() if hasattr(values, "compute") else values)


def test_reducer_registry_covers_every_reducer():
    names = {name for name in reducers.__all__ if name != "BaseReducer"}
    assert names == set(SETTINGS) | {"IVISReducer", "ParametricUMAPReducer"}


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_reducer_follows_its_declared_capabilities(name, tmp_path):
    cls = getattr(reducers, name)
    kwargs = {"n_components": 1 if name == "TRCAReducer" else 2, **SETTINGS[name]}
    if name not in FORWARDS_KWARGS:
        kwargs["unsupported_param"] = 1  # unknown arguments are dropped
    reducer = cls(**kwargs)
    caps = reducer.capabilities
    data, y = _inputs(reducer)

    expected = RuntimeError if caps["has_transform"] else NotImplementedError
    with pytest.raises(expected):
        reducer.transform(data)
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_diagnostics()
    # fitted diagnostics are missing or empty before fitting (metadata can
    # include settings, which exist from the start)
    for attr in caps["supported_diagnostics"]:
        if isinstance(getattr(cls, attr, None), property):
            try:
                value = getattr(reducer, attr)
            except RuntimeError:
                continue
            assert value is None or (hasattr(value, "__len__") and len(value) == 0)

    assert (reducer.fit(data, y) if y is not None else reducer.fit(data)) is reducer
    embedding = _compute(
        reducer.fit_transform(data, y) if y is not None else reducer.fit_transform(data)
    )
    n_out = data.shape[1] if caps["input_layout"] == "features_snapshots" else len(data)
    assert embedding.shape[0] == n_out
    assert np.isfinite(embedding).all()

    if caps["has_transform"]:
        assert _compute(reducer.transform(data)).shape == embedding.shape
    else:
        with pytest.raises(NotImplementedError):
            reducer.transform(data)
    assert set(reducer.get_diagnostics()) == set(caps["supported_diagnostics"])
    assert set(reducer.get_quality_metadata()) <= set(caps["supported_metadata"])

    reducer.save(tmp_path / "nested" / "reducer.joblib")
    loaded = BaseReducer.load(tmp_path / "nested" / "reducer.joblib")
    assert type(loaded) is cls
    if caps["has_transform"]:
        np.testing.assert_allclose(
            _compute(loaded.transform(data)), _compute(reducer.transform(data))
        )


@pytest.mark.parametrize("name", sorted(SETTINGS))
def test_unfitted_properties_fail_cleanly(name):
    reducer = getattr(reducers, name)(n_components=2, **SETTINGS[name])
    for attr, member in vars(type(reducer)).items():
        if isinstance(member, property) and not attr.startswith("_"):
            try:
                getattr(reducer, attr)
            except RuntimeError as error:
                assert "fit" in str(error).lower()


def test_reducers_check_input_dimensions():
    trca = reducers.TRCAReducer(n_components=1, sfreq=100)
    with pytest.raises(ValueError, match="3D input"):
        trca.fit(X, np.zeros(60))
    with pytest.raises(ValueError, match="2D input"):
        reducers.TopologicalAEReducer(epochs=1, device="cpu").fit(X[:, :, None])

    # DMD reads (features, snapshots); force_transpose takes (snapshots, features)
    snapshots = np.random.RandomState(0).rand(20, 60)
    plain = reducers.DMDReducer(n_components=2).fit(snapshots)
    flipped = reducers.DMDReducer(n_components=2, force_transpose=True).fit(snapshots.T)
    np.testing.assert_allclose(
        plain.transform(snapshots), flipped.transform(snapshots.T), atol=1e-8
    )


def test_stochastic_reducers_repeat_with_a_seed():
    first, second = (
        reducers.TSNEReducer(
            n_components=2, perplexity=5, random_state=0
        ).fit_transform(X)
        for _ in range(2)
    )
    np.testing.assert_allclose(first, second)


def test_trimap_keeps_separated_clusters_apart():
    # annoy-based TriMap returned only the query as neighbour on Apple silicon
    # and mixed these clusters (trustworthiness 0.74); shapes alone missed it
    from sklearn.manifold import trustworthiness

    blobs, _ = make_blobs(n_samples=300, n_features=20, centers=3, random_state=0)
    embedding = reducers.TrimapReducer(random_state=0).fit_transform(blobs)
    assert trustworthiness(blobs, embedding, n_neighbors=10) > 0.9


# --- linear ---------------------------------------------------------------------


def test_pca_reports_its_variance():
    reducer = reducers.PCAReducer(n_components=3)
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.explained_variance_ratio_
    with pytest.raises(RuntimeError, match="not fitted"):
        _ = reducer.components_
    reducer.fit(X)
    assert reducer.explained_variance_ratio_.shape == (3,)
    assert reducer.components_.shape == (3, 8)

    # variance along one direction only: one effective component
    rng = np.random.default_rng(0)
    line = np.outer(rng.normal(size=60), rng.normal(size=8))
    line += 1e-6 * rng.normal(size=line.shape)
    pr = reducers.PCAReducer(n_components=3).fit(line).participation_ratio_
    assert pr == pytest.approx(1.0, abs=1e-3)


def test_incremental_pca_learns_batch_by_batch():
    data = np.random.default_rng(0).random((100, 10))
    streamed = reducers.IncrementalPCAReducer(n_components=2, unsupported_param=1)
    assert streamed.model is None
    streamed.partial_fit(data[:50])
    streamed.partial_fit(data[50:])
    assert streamed.transform(data).shape == (100, 2)
    assert streamed.get_quality_metadata()["n_samples_seen_"] == 100

    first_model = streamed.model
    streamed.fit(data)  # fit starts over
    assert streamed.model is not first_model


@pytest.mark.parametrize("name", ["DaskPCAReducer", "DaskTruncatedSVDReducer"])
def test_dask_reducers_take_dask_arrays(name):
    import dask.array as da

    reducer = getattr(reducers, name)(n_components=2)
    embedding = reducer.fit(da.from_array(X, chunks=(20, 8))).transform(
        da.from_array(X, chunks=(20, 8))
    )
    assert isinstance(embedding, da.Array)
    with pytest.raises((TypeError, AttributeError)):
        getattr(reducers, name)(n_components=2).fit(X)


# --- reducers on TensorFlow (their libraries replaced) ------------------------


def test_ivis_reducer_wraps_ivis():
    class Ivis:
        def __init__(self, embedding_dims=2, k=15, epochs=100, batch_size=32):
            self.embedding_dims = embedding_dims
            self.loss_history_ = [0.1, 0.05]

        def fit(self, X, Y=None):
            return self

        def transform(self, X):
            return np.zeros((len(X), self.embedding_dims))

    with patch(
        "coco_pipe.dim_reduction.reducers.neural.import_optional_dependency",
        return_value=Ivis,
    ):
        reducer = reducers.IVISReducer(n_components=2, k=5, unsupported_param=1)
        assert reducer.fit(X).transform(X).shape == (60, 2)
    assert reducer.model.embedding_dims == 2
    assert reducer.get_diagnostics()["loss_history_"] == [0.1, 0.05]


def test_parametric_umap_reducer_wraps_parametric_umap():
    class ParametricUMAP:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._history = {"loss": [0.1, 0.05]}

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return np.zeros((len(X), 2))

    with patch(
        "coco_pipe.dim_reduction.reducers.neighbor.import_optional_dependency",
        return_value=ParametricUMAP,
    ):
        reducer = reducers.ParametricUMAPReducer(n_components=2, n_epochs=1)
        assert reducer.fit(X).transform(X).shape == (60, 2)
    assert reducer.model.kwargs["n_epochs"] == 1
    assert reducer.loss_history_ == [0.1, 0.05]


# --- topology -------------------------------------------------------------------


def test_topology_module_imports_without_torch(monkeypatch):
    name = "coco_pipe.dim_reduction.reducers.topology"
    cached = sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, "torch", None)
    try:
        assert hasattr(importlib.import_module(name), "TopologicalAEReducer")
    finally:
        sys.modules.pop(name, None)
        if cached is not None:
            sys.modules[name] = cached


def test_topological_signature_distance():
    import torch

    from coco_pipe.dim_reduction.reducers.topology import TopologicalSignatureDistance

    points = torch.tensor(np.random.default_rng(0).random((12, 4)), dtype=torch.float32)
    distance = TopologicalSignatureDistance()
    assert distance(points, points).item() == pytest.approx(0.0, abs=1e-6)
    assert distance(points, points[:, :2] * 10).item() > 0

    with_loss = reducers.TopologicalAEReducer(
        n_components=2, epochs=2, lam=0.1, batch_size=16, device="cpu"
    )
    assert with_loss.fit_transform(X).shape == (60, 2)
    assert len(with_loss.loss_history_) == 2
    assert reducers.TopologicalAEReducer(device="auto").device in {"cpu", "cuda", "mps"}


# --- extending BaseReducer ------------------------------------------------------------


class _Declared(BaseReducer):
    """A minimal reducer declaring one diagnostic and one metadata value."""

    @property
    def capabilities(self):
        return self._merge_capabilities(
            super().capabilities,
            supported_diagnostics=["diag"],
            supported_metadata=["meta"],
        )

    def fit(self, X, y=None):
        self.model = SimpleNamespace(diag=np.array([1.0]), meta="model", hidden=9)
        return self

    def transform(self, X):
        return np.asarray(X)[:, : self.n_components]

    @property
    def meta(self):
        return "wrapper"


def test_base_reducer_is_an_extension_point(tmp_path):
    reducer = _Declared(n_components=2)
    caps = reducer.capabilities
    assert (caps["input_ndim"], caps["input_layout"]) == (2, "standard")
    with pytest.raises(RuntimeError, match="must be fitted"):
        reducer.get_quality_metadata()

    assert reducer.fit_transform(X).shape == (60, 2)
    # only declared names are reported, and the wrapper's value wins
    assert list(reducer.get_diagnostics()) == ["diag"]
    assert reducer.get_quality_metadata() == {"meta": "wrapper"}

    reducer.save(tmp_path / "sub" / "reducer.joblib")
    assert isinstance(BaseReducer.load(tmp_path / "sub" / "reducer.joblib"), _Declared)
    (tmp_path / "other.joblib").write_bytes(b"")
    import joblib

    joblib.dump({"not": "a reducer"}, tmp_path / "other.joblib")
    with pytest.raises(TypeError, match="not a BaseReducer"):
        BaseReducer.load(tmp_path / "other.joblib")
