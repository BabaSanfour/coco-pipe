import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer

from coco_pipe.dim_reduction.analysis import (
    correlate_features,
    gradient_importance,
    interpret_features,
    perturbation_importance,
)


class _LinearEncoder:
    """A fitted encoder wrapper whose module encodes with known weights."""

    def __init__(self, weight):
        torch = pytest.importorskip("torch")
        weight = torch.as_tensor(weight, dtype=torch.float32)
        self.module = torch.nn.Module()
        self.module.encoder = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(weight.shape[1], weight.shape[0])
        )
        with torch.no_grad():
            self.module.encoder[1].weight.copy_(weight)

    def get_pytorch_module(self):
        return self.module


def _saliency(weight):
    """d sum(W x + b) / dx = sum of W over its outputs, normalised."""
    scores = np.abs(np.asarray(weight).sum(axis=0))
    return scores / scores.sum()


def test_correlate_features():
    """Each dimension ranks the features by absolute Spearman correlation."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    X_emb = np.column_stack([X[:, 0] * 0.9 + rng.normal(size=50) * 0.1, -X[:, 3]])

    corrs = correlate_features(X, X_emb, feature_names=[f"F{i}" for i in range(5)])

    assert list(corrs) == ["Dimension 1", "Dimension 2"]
    assert next(iter(corrs["Dimension 1"])) == "F0"
    assert corrs["Dimension 1"]["F0"] > 0.8
    # a decreasing function of a feature is a rank correlation of exactly -1
    assert corrs["Dimension 2"]["F3"] == pytest.approx(-1.0)
    magnitudes = [abs(value) for value in corrs["Dimension 2"].values()]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_correlate_features_errors():
    X = np.random.default_rng(0).normal(size=(10, 3))
    X_emb = np.random.default_rng(1).normal(size=(10, 2))
    names = ["A", "B", "C"]

    with pytest.raises(ValueError, match="`X_orig` must be a 2D array"):
        correlate_features(np.zeros(10), X_emb, names)
    with pytest.raises(ValueError, match="`X_emb` must be a 2D array"):
        correlate_features(X, np.zeros(10), names)
    with pytest.raises(ValueError, match="matching sample counts"):
        correlate_features(np.zeros((5, 3)), X_emb, names)
    with pytest.raises(ValueError, match="Length of `feature_names`"):
        correlate_features(X, X_emb, ["A", "B"])

    # a constant feature or dimension has no defined correlation: reported as 0
    res = correlate_features(np.zeros((10, 3)), np.zeros((10, 2)), names)
    assert res["Dimension 1"]["A"] == 0.0


def test_perturbation_importance():
    """Shuffling a feature the embedding ignores costs nothing."""
    X = np.random.default_rng(0).normal(size=(20, 5))
    X[:, 0] *= 10
    model = PCA(n_components=2).fit(X)

    scores = perturbation_importance(
        model, X, [f"F{i}" for i in range(5)], model.transform(X), random_state=0
    )
    assert max(scores, key=scores.get) == "F0"
    assert sum(scores.values()) == pytest.approx(1.0)

    first_only = FunctionTransformer(lambda X: X[:, :1])
    scores = perturbation_importance(
        first_only, X, [f"F{i}" for i in range(5)], X[:, :1], random_state=0
    )
    assert scores == {"F0": 1.0, "F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0}


def test_reproducibility_perturbation():
    """The same random_state shuffles the same way; another does not."""
    X = np.random.default_rng(0).normal(size=(20, 10))
    model = PCA(n_components=2).fit(X)
    X_emb = model.transform(X)
    names = [f"F{i}" for i in range(10)]

    s1 = perturbation_importance(model, X, names, X_emb, random_state=42)
    s2 = perturbation_importance(model, X, names, X_emb, random_state=42)
    s3 = perturbation_importance(model, X, names, X_emb, random_state=43)

    assert s1 == s2
    assert any(abs(s1[k] - s3[k]) > 1e-10 for k in s1)


def test_perturbation_importance_errors():
    constant = FunctionTransformer(lambda X: np.zeros((len(X), 2)))
    X = np.random.default_rng(0).normal(size=(10, 3))
    X_emb = np.zeros((10, 2))
    names = ["A", "B", "C"]

    with pytest.raises(ValueError, match="`X` must be a 2D array"):
        perturbation_importance(constant, np.zeros(10), names, X_emb)
    with pytest.raises(ValueError, match="same number of samples"):
        perturbation_importance(constant, np.zeros((5, 3)), names, X_emb)
    with pytest.raises(ValueError, match="match the number of input features"):
        perturbation_importance(constant, X, ["A"], X_emb)

    # an embedding that no feature moves gives every feature a score of 0
    res = perturbation_importance(constant, X, names, X_emb, n_repeats=1)
    assert res == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_gradient_importance_of_a_linear_encoder():
    weight = np.array([[1.0, -2.0, 0.5], [3.0, 1.0, 0.0]])
    X = np.random.default_rng(0).normal(size=(10, 3))

    scores = gradient_importance(_LinearEncoder(weight), X, feature_names="ABC")

    assert list(scores) == ["A", "B", "C"]
    np.testing.assert_allclose(list(scores.values()), _saliency(weight), rtol=1e-6)


def test_gradient_importance_keeps_the_shape_of_3d_input():
    weight = np.random.default_rng(0).normal(size=(2, 50))
    X = np.random.default_rng(1).normal(size=(4, 5, 10))

    res = gradient_importance(_LinearEncoder(weight), X)

    np.testing.assert_allclose(
        res["importance_matrix"], _saliency(weight).reshape(5, 10), rtol=1e-5
    )


def test_gradient_importance_of_a_fitted_topological_autoencoder():
    pytest.importorskip("skorch")
    from coco_pipe.dim_reduction.reducers import TopologicalAEReducer

    X = np.random.default_rng(0).normal(size=(20, 4))
    reducer = TopologicalAEReducer(n_components=2, epochs=1, device="cpu").fit(X)

    scores = gradient_importance(reducer, X, feature_names="abcd")

    assert list(scores) == ["a", "b", "c", "d"]
    assert all(value > 0 for value in scores.values())
    assert sum(scores.values()) == pytest.approx(1.0)


def test_gradient_importance_errors():
    with pytest.raises(ValueError, match="must have at least 2 dimensions"):
        gradient_importance(_LinearEncoder(np.ones((2, 3))), np.zeros(10))

    # an encoder that no input moves gives every feature a score of 0
    res = gradient_importance(_LinearEncoder(np.zeros((2, 3))), np.ones((2, 3)))
    np.testing.assert_array_equal(res["importance_matrix"], np.zeros(3))

    with pytest.raises(ValueError, match="one-dimensional"):
        gradient_importance(
            _LinearEncoder(np.ones((2, 6))), np.ones((2, 2, 3)), feature_names="AB"
        )
    with pytest.raises(ValueError, match="match the number of reduced features"):
        gradient_importance(
            _LinearEncoder(np.ones((2, 3))), np.ones((2, 3)), feature_names=["A"]
        )


def test_interpret_features_packs_each_analysis_with_its_records():
    X = np.random.default_rng(0).normal(size=(20, 3))
    model = PCA(n_components=2).fit(X)
    X_emb = model.transform(X)
    weight = np.array([[1.0, 0.0, 2.0], [1.0, 1.0, 0.0]])
    names = ["A", "B", "C"]

    results = interpret_features(
        X,
        X_emb=X_emb,
        model=model,
        feature_names=names,
        analyses=["correlation", "perturbation"],
        method_name="PCA",
        random_state=0,
    )
    gradient = interpret_features(
        X, model=_LinearEncoder(weight), analyses=["gradient"], feature_names=names
    )

    analysis = results["analysis"]
    assert analysis["correlation"] == correlate_features(X, X_emb, names)
    assert analysis["perturbation"] == perturbation_importance(
        model, X, names, X_emb, random_state=0
    )
    np.testing.assert_allclose(
        list(gradient["analysis"]["gradient"].values()), _saliency(weight), rtol=1e-6
    )
    # one record per dimension and feature, then one per feature
    records = results["records"]
    assert [r["analysis"] for r in records] == ["correlation"] * 6 + [
        "perturbation"
    ] * 3
    assert {r["method"] for r in records} == {"PCA"}
    assert records[-1]["feature"] == "C"
    assert records[-1]["value"] == analysis["perturbation"]["C"]
    assert [r["feature"] for r in gradient["records"]] == names
    assert {r["analysis"] for r in gradient["records"]} == {"gradient"}


def test_interpret_features_errors():
    X = np.random.default_rng(0).normal(size=(10, 3))
    X_emb = np.random.default_rng(1).normal(size=(10, 2))
    model = PCA(n_components=2).fit(X)
    names = ["A", "B", "C"]

    with pytest.raises(ValueError, match="`X_emb` is required for correlation"):
        interpret_features(X, analyses=["correlation"], feature_names=names)
    with pytest.raises(ValueError, match="`feature_names` is required for correlation"):
        interpret_features(X, X_emb=X_emb, analyses=["correlation"])
    with pytest.raises(ValueError, match="`model` is required for perturbation"):
        interpret_features(
            X, X_emb=X_emb, feature_names=names, analyses=["perturbation"]
        )
    with pytest.raises(ValueError, match="`X_emb` is required for perturbation"):
        interpret_features(
            X, model=model, analyses=["perturbation"], feature_names=names
        )
    with pytest.raises(
        ValueError, match="`feature_names` is required for perturbation"
    ):
        interpret_features(X, X_emb=X_emb, model=model, analyses=["perturbation"])
    with pytest.raises(ValueError, match="`model` is required for gradient"):
        interpret_features(X, analyses=["gradient"], feature_names=names)
    with pytest.raises(ValueError, match="Unknown analysis selector"):
        interpret_features(X, analyses=["unknown"])
