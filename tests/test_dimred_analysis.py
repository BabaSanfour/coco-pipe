import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coco_pipe.dim_reduction.analysis import (
    compute_feature_importance,
    correlate_features,
    gradient_importance,
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
    assert "Component 1" in corrs
    assert "Component 2" in corrs

    # Check content
    comp1_res = corrs["Component 1"]
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

    scores = perturbation_importance(
        model, X, feature_names=[f"F{i}" for i in range(5)]
    )

    assert scores["F0"] > scores["F1"]
    assert np.isclose(sum(scores.values()), 1.0)


def test_compute_feature_importance_wrapper():
    """Test the main wrapper function."""
    from sklearn.decomposition import PCA

    X = np.random.randn(10, 3)
    model = PCA(n_components=2).fit(X)

    scores = compute_feature_importance(model, X, method="perturbation")
    assert len(scores) == 3


def test_correlate_features_defaults():
    """Test correlate_features with default feature names."""
    X = np.random.randn(10, 3)
    X_emb = np.random.randn(10, 2)

    corrs = correlate_features(X, X_emb, feature_names=None)

    assert "Component 1" in corrs
    # Check default naming "Feature 0", "Feature 1"...
    assert "Feature 0" in corrs["Component 1"]
    assert len(corrs["Component 1"]) == 3


def test_perturbation_importance_errors():
    """Test error when model has no transform."""

    class BadModel:
        pass

    X = np.random.randn(10, 2)
    with pytest.raises(ValueError, match="must have a transform method"):
        perturbation_importance(BadModel(), X)


def test_perturbation_importance_defaults():
    """Test default feature names in perturbation."""

    class MockModel:
        def transform(self, X):
            # Return slightly noisy transform to avoid sum=0 and div by zero warning
            return X[:, :2] + np.random.randn(X.shape[0], 2) * 0.1

    X = np.random.randn(10, 3)
    # feature names = None
    scores = perturbation_importance(MockModel(), X, feature_names=None)
    assert "Feature 0" in scores
    assert np.isclose(sum(scores.values()), 1.0)


def test_compute_feature_importance_dispatch():
    """Test dispatch logic."""

    class MockModel:
        def transform(self, X):
            return X

    X = np.zeros((5, 2))

    # 1. Default (perturbation)
    s1 = compute_feature_importance(MockModel(), X)
    assert len(s1) == 2

    # 3. Unknown method
    with pytest.raises(ValueError, match="Unknown method"):
        compute_feature_importance(MockModel(), X, method="magic_wand")


def test_compute_feature_importance_gradient_check():
    """Test that gradient method checks for torch attributes."""

    class SimpleModel:
        def transform(self, X):
            return X

    X = np.zeros((5, 2))
    with pytest.raises(NotImplementedError, match="Gradient method requires"):
        compute_feature_importance(SimpleModel(), X, method="gradient")


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
        wrapper.model.encoder = MagicMock()
        mock_z = MagicMock()
        wrapper.model.encoder.return_value = mock_z

        # Run
        X = np.zeros((10, 3))
        scores = gradient_importance(wrapper, X, feature_names=["A", "B", "C"])

        assert scores["A"] == 0.1
        assert scores["C"] == 0.7

        # Verify torch calls
        torch.tensor.assert_called()
        wrapper.model.encoder.assert_called()
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
