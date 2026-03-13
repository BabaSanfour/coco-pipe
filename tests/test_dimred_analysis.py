import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coco_pipe.dim_reduction.analysis import (
    correlate_features,
    gradient_importance,
    interpret_features,
    perturbation_importance,
)


def test_correlate_features():
    """Test feature correlation with embedding components."""
    X = np.random.randn(50, 5)
    # create embedding correlated with Feature 0
    X_emb = np.zeros((50, 2))
    # Strong correlation with Feature 0 on Component 0
    X_emb[:, 0] = X[:, 0] * 0.9 + np.random.randn(50) * 0.1
    # Random for Component 1
    X_emb[:, 1] = np.random.randn(50)

    feat_names = [f"F{i}" for i in range(5)]
    corrs = correlate_features(X, X_emb, feature_names=feat_names)

    # Check structure
    assert "Dimension 1" in corrs
    assert "Dimension 2" in corrs

    # Check content
    comp1_res = corrs["Dimension 1"]
    # Feature 0 should be top correlated
    top_feat = list(comp1_res.keys())[0]
    assert top_feat == "F0"
    assert abs(comp1_res["F0"]) > 0.8


def test_perturbation_importance():
    """Test perturbation-based feature importance."""
    from sklearn.decomposition import PCA

    X = np.random.randn(20, 5)
    # Make feature 0 very important (high variance)
    X[:, 0] *= 10

    model = PCA(n_components=2).fit(X)

    # PCA.transform(X) returns a 2D array (n_samples, n_components)
    X_emb = model.transform(X)
    scores = perturbation_importance(model, X, [f"F{i}" for i in range(5)], X_emb)

    assert scores["F0"] > scores["F1"]
    assert np.isclose(sum(scores.values()), 1.0)


def test_correlate_features_defaults():
    """Test correlate_features with default feature names."""
    X = np.random.randn(10, 3)
    X_emb = np.random.randn(10, 2)

    feat_names = ["F0", "F1", "F2"]
    corrs = correlate_features(X, X_emb, feature_names=feat_names)
    assert "Dimension 1" in corrs
    assert "F0" in corrs["Dimension 1"]
    assert len(corrs["Dimension 1"]) == 3


def test_perturbation_importance_defaults():
    """Test default feature names in perturbation."""

    class MockModel:
        def transform(self, X):
            # Return slightly noisy transform to avoid sum=0 and div by zero warning
            return X[:, :2] + np.random.randn(X.shape[0], 2) * 0.1

    X = np.random.randn(10, 3)
    feat_names = [f"Feature {i}" for i in range(3)]
    scores = perturbation_importance(MockModel(), X, feat_names, X[:, :2])
    assert "Feature 0" in scores
    assert np.isclose(sum(scores.values()), 1.0)


def test_gradient_importance_mocked():
    """Test gradient importance using mocked Torch to avoid dependencies."""
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch

        # Setup mock tensor behavior
        mock_tensor_cls = torch.tensor
        mock_tensor_instance = MagicMock()
        mock_tensor_cls.return_value = mock_tensor_instance

        # Mock .to(device) to return self
        mock_tensor_instance.to.return_value = mock_tensor_instance

        # Mock grads
        mock_grads = MagicMock()
        mock_tensor_instance.grad = mock_grads

        # Mock torch.abs(grads) -> mock_abs
        mock_abs = MagicMock()
        torch.abs.return_value = mock_abs

        # Mock torch.mean(mock_abs, dim=0) -> mock_mean
        mock_mean = MagicMock()
        torch.mean.return_value = mock_mean

        # Mock .detach().cpu().numpy() chain
        # Return a numpy array of importances [0.1, 0.2, 0.7]
        expected_scores = np.array([0.1, 0.2, 0.7])
        mock_mean.detach.return_value.cpu.return_value.numpy.return_value = (
            expected_scores
        )

        # Mock Wrapper and Model
        wrapper = MagicMock()
        wrapper.get_pytorch_module.return_value = MagicMock()
        model = wrapper.get_pytorch_module.return_value
        model.encoder = MagicMock()
        model.parameters.return_value = iter([])  # Handle empty parameters case
        mock_z = MagicMock()
        model.encoder.return_value = mock_z

        # Run
        X = np.zeros((10, 3))
        scores = gradient_importance(wrapper, X, feature_names=["A", "B", "C"])

        assert scores["A"] == 0.1
        assert scores["C"] == 0.7

        # Verify torch calls
        torch.tensor.assert_called()
        model.encoder.assert_called()
        mock_z.sum.return_value.backward.assert_called()


def test_gradient_importance_complex_shape():
    """Test gradient importance return for 3D data (returns matrix if no names)."""
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch

        mock_mean = MagicMock()
        # Assume output is (5, 10) matrix
        mock_matrix = np.ones((5, 10))
        mock_mean.detach.return_value.cpu.return_value.numpy.return_value = mock_matrix
        torch.mean.return_value = mock_mean
        torch.tensor.return_value.to.return_value = MagicMock()

        wrapper = MagicMock()

        X_3d = np.zeros((2, 5, 10))
        res = gradient_importance(wrapper, X_3d)  # No feature names

        assert "importance_matrix" in res
        assert res["importance_matrix"].shape == (5, 10)


def test_reproducibility_perturbation():
    """Verify that random_state ensures reproducible perturbation importance."""
    from sklearn.decomposition import PCA

    X = np.random.randn(20, 10)
    model = PCA(n_components=2).fit(X)

    # Run twice with same seed
    X_emb = model.transform(X)
    s1 = perturbation_importance(
        model, X, [f"F{i}" for i in range(10)], X_emb, random_state=42
    )
    s2 = perturbation_importance(
        model, X, [f"F{i}" for i in range(10)], X_emb, random_state=42
    )

    # Run with different seed
    s3 = perturbation_importance(
        model, X, [f"F{i}" for i in range(10)], X_emb, random_state=43
    )

    for k in s1:
        assert np.isclose(s1[k], s2[k])

    # S3 should be different (stochastically, but highly likely for 10 features)
    diffs = [abs(s1[k] - s3[k]) for k in s1]
    assert np.any(np.array(diffs) > 1e-10)


def test_interpret_features_standard():
    """Test the main entry point for interpret_features."""
    from sklearn.decomposition import PCA

    X = np.random.randn(20, 5)
    model = PCA(n_components=2).fit(X)
    X_emb = model.transform(X)
    feature_names = [f"F{i}" for i in range(5)]

    results = interpret_features(
        X,
        X_emb=X_emb,
        model=model,
        feature_names=feature_names,
        analyses=["correlation", "perturbation"],
    )

    assert "analysis" in results
    assert "records" in results
    assert "correlation" in results["analysis"]
    assert "perturbation" in results["analysis"]
    assert len(results["analysis"]["perturbation"]) == 5
    assert "Dimension 1" in results["analysis"]["correlation"]


def test_interpret_features_errors():
    """Test error handling in interpret_features."""
    # Missing X_emb for correlation
    with pytest.raises(
        ValueError, match="`X_emb` is required for correlation analysis"
    ):
        interpret_features(
            np.zeros((10, 5)), analyses=["correlation"], feature_names=["A"]
        )

    # Missing model for perturbation
    with pytest.raises(
        ValueError, match="`model` is required for perturbation importance"
    ):
        interpret_features(
            np.zeros((10, 5)),
            X_emb=np.zeros((10, 2)),
            feature_names=["A"],
            analyses=["perturbation"],
        )

    # Unknown method
    with pytest.raises(ValueError, match="Unknown analysis selector"):
        interpret_features(np.zeros((10, 5)), analyses=["unknown"])


def test_correlate_features_errors():
    """Test error handling in correlate_features."""
    X = np.random.randn(10, 3)
    X_emb = np.random.randn(10, 2)
    names = ["A", "B", "C"]

    # ndim checks
    with pytest.raises(ValueError, match="`X_orig` must be a 2D array"):
        correlate_features(np.zeros(10), X_emb, names)
    with pytest.raises(ValueError, match="`X_emb` must be a 2D array"):
        correlate_features(X, np.zeros(10), names)

    # Sample match
    with pytest.raises(ValueError, match="matching sample counts"):
        correlate_features(np.zeros((5, 3)), X_emb, names)

    # Name length
    with pytest.raises(ValueError, match="Length of `feature_names`"):
        correlate_features(X, X_emb, ["A", "B"])

    # Finite rho check (constant data)
    res = correlate_features(np.zeros((10, 3)), np.zeros((10, 2)), names)
    assert res["Dimension 1"]["A"] == 0.0


def test_perturbation_importance_errors_extended():
    """Test additional error paths in perturbation_importance."""
    model = MagicMock()
    model.transform.return_value = np.zeros((10, 2))
    X = np.random.randn(10, 3)
    X_emb = np.zeros((10, 2))
    names = ["A", "B", "C"]

    # ndim
    with pytest.raises(ValueError, match="`X` must be a 2D array"):
        perturbation_importance(model, np.zeros(10), names, X_emb)

    # samples
    with pytest.raises(ValueError, match="same number of samples"):
        perturbation_importance(model, np.zeros((5, 3)), names, X_emb)

    # length
    with pytest.raises(ValueError, match="match the number of input features"):
        perturbation_importance(model, X, ["A"], X_emb)

    # total sum zero
    res = perturbation_importance(model, X, names, X_emb, n_repeats=1)
    assert res["A"] == 0.0


def test_gradient_importance_errors_extended():
    """Test additional error paths in gradient_importance."""
    wrapper = MagicMock()

    # X ndim
    with pytest.raises(ValueError, match="must have at least 2 dimensions"):
        gradient_importance(wrapper, np.zeros(10))

    # Mocked torch for sum zero and shape mismatch
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch

        mock_res = torch.mean.return_value.detach.return_value.cpu.return_value
        mock_res.numpy.return_value = np.zeros(3)

        # total sum zero
        res = gradient_importance(wrapper, np.zeros((2, 3)))
        assert res["importance_matrix"][0] == 0.0

        # ndim != 1 for named
        mock_res = torch.mean.return_value.detach.return_value.cpu.return_value
        mock_res.numpy.return_value = np.zeros((2, 2))
        with pytest.raises(ValueError, match="one-dimensional"):
            gradient_importance(wrapper, np.zeros((2, 3)), feature_names=["A", "B"])

        # name length mismatch
        mock_res = torch.mean.return_value.detach.return_value.cpu.return_value
        mock_res.numpy.return_value = np.zeros(3)
        with pytest.raises(ValueError, match="match the number of reduced features"):
            gradient_importance(wrapper, np.zeros((2, 3)), feature_names=["A"])


def test_interpret_features_extended():
    """Cover remaining branches in interpret_features."""
    X = np.random.randn(10, 3)
    X_emb = np.random.randn(10, 2)
    names = ["A", "B", "C"]

    # missing names for correlation
    with pytest.raises(ValueError, match="`feature_names` is required for correlation"):
        interpret_features(X, X_emb=X_emb, analyses=["correlation"])

    # missing names for perturbation
    with pytest.raises(
        ValueError, match="`feature_names` is required for perturbation"
    ):
        interpret_features(X, X_emb=X_emb, model=MagicMock(), analyses=["perturbation"])

    # missing X_emb for perturbation
    with pytest.raises(ValueError, match="`X_emb` is required for perturbation"):
        interpret_features(
            X, model=MagicMock(), analyses=["perturbation"], feature_names=names
        )

    # missing model for gradient
    with pytest.raises(ValueError, match="`model` is required for gradient"):
        interpret_features(X, analyses=["gradient"], feature_names=names)

    # Gradient branch
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch

        # mock returns 1D scores
        mock_scores = np.array([0.1, 0.2, 0.7])
        mock_res = torch.mean.return_value.detach.return_value.cpu.return_value
        mock_res.numpy.return_value = mock_scores

        wrapper = MagicMock()
        wrapper.get_pytorch_module.return_value = MagicMock()

        res = interpret_features(
            X, model=wrapper, analyses=["gradient"], feature_names=names
        )
        assert "gradient" in res["analysis"]
        assert len(res["records"]) == 3
        assert res["records"][0]["analysis"] == "gradient"
