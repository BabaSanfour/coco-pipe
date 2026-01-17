import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from coco_pipe.io.structures import DataContainer
from coco_pipe.io.transform import SklearnWrapper, SpatialWhitener

# --- SklearnWrapper Tests ---


def test_sklearn_wrapper_standard_scaler():
    """Test wrapping a simple scaler."""
    X = np.array([[1.0, 2.0], [3.0, 4.0]])  # Mean=[2, 3], Std=[1, 1]
    dc = DataContainer(X, dims=("obs", "feat"))

    scaler = SklearnWrapper(StandardScaler())

    # fit_transform
    res = scaler.fit_transform(dc)
    assert isinstance(res, DataContainer)
    assert res.dims == dc.dims
    assert np.allclose(res.X, [[-1.0, -1.0], [1.0, 1.0]])

    # inverse_transform (StandardScaler supports it)
    orig = scaler.inverse_transform(res)
    assert np.allclose(orig.X, X)


def test_sklearn_wrapper_validation():
    """Test validation logic."""
    # 1. Not a container
    scaler = SklearnWrapper(StandardScaler())
    with pytest.raises(TypeError, match="expects DataContainer"):
        scaler.fit("not_a_container")

    # 2. 3D input (should fail)
    dc_3d = DataContainer(np.zeros((2, 2, 2)), dims=("obs", "c", "t"))
    with pytest.raises(ValueError, match="expects 2D input"):
        scaler.fit(dc_3d)

    # 3. Inverse transform without fit
    with pytest.raises(RuntimeError, match="not fitted"):
        scaler.inverse_transform(DataContainer(np.zeros((2, 2)), dims=("o", "f")))

    # 4. Inverse transform unimplemented
    SklearnWrapper(PCA(n_components=1))
    from sklearn.preprocessing import Normalizer

    wrapper_norm = SklearnWrapper(Normalizer())
    wrapper_norm.fit(DataContainer(np.zeros((2, 2)), dims=("o", "f")))
    with pytest.raises(NotImplementedError):
        wrapper_norm.inverse_transform(DataContainer(np.zeros((2, 2)), dims=("o", "f")))


# --- SpatialWhitener Tests ---


def test_spatial_whitener_methods():
    """Test PCA, ZCA, and Shrinkage methods."""
    # Create correlated data: 2 channels, 100 obs.
    # CH1 = signal, CH2 = signal + noise.
    rng = np.random.RandomState(42)
    signal = rng.randn(100)
    X_Raw = np.vstack([signal, signal + 0.1 * rng.randn(100)]).T  # (100, 2)
    # Add time dim to simulate EEG epoch structure
    X_3d = X_Raw[:, :, np.newaxis]  # (100, 2, 1)

    dc = DataContainer(X_3d, dims=("obs", "channel", "time"))

    # 1. PCA Whitening
    w_pca = SpatialWhitener(method="pca")
    res_pca = w_pca.fit_transform(dc)
    X_out = res_pca.X[:, :, 0]
    cov = np.cov(X_out.T)
    assert np.allclose(cov, np.eye(2), atol=0.2)
    assert w_pca.whitener_.shape == (2, 2)
    recon_pca = w_pca.inverse_transform(res_pca)
    assert np.allclose(recon_pca.X, dc.X)

    # 2. ZCA Whitening
    w_zca = SpatialWhitener(method="zca")
    res_zca = w_zca.fit_transform(dc)
    X_out_z = res_zca.X[:, :, 0]
    cov_z = np.cov(X_out_z.T)
    assert np.allclose(cov_z, np.eye(2), atol=0.2)

    # 3. Shrinkage (OAS)
    w_oas = SpatialWhitener(method="shrinkage")
    res_oas = w_oas.fit_transform(dc)
    recon_oas = w_oas.inverse_transform(res_oas)
    assert np.allclose(recon_oas.X, dc.X)


def test_spatial_whitener_errors():
    """Test error modes."""
    # Missing 'channel' dim
    dc = DataContainer(np.zeros((10, 2)), dims=("obs", "feat"))
    w = SpatialWhitener()
    with pytest.raises(ValueError, match="requires 'channel'"):
        w.fit(dc)

    w2 = SpatialWhitener()
    dc_valid = DataContainer(np.zeros((2, 2)), dims=("obs", "channel"))
    with pytest.raises(AttributeError):
        w2.transform(dc_valid)
