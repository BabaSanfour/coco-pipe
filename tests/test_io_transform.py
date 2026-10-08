import numpy as np
import pytest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from coco_pipe.io.structures import DataContainer
from coco_pipe.io.transform import SklearnWrapper


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


def _wrapper_container_2d():
    return DataContainer(np.random.randn(6, 3), dims=("obs", "feature"))


def _wrapper_container_3d():
    return DataContainer(np.random.randn(6, 3, 2), dims=("obs", "channel", "time"))


def test_sklearn_wrapper_transform_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        SklearnWrapper(StandardScaler()).transform(_wrapper_container_2d())


def test_sklearn_wrapper_transform_non_2d():
    wrapper = SklearnWrapper(StandardScaler()).fit(_wrapper_container_2d())
    with pytest.raises(ValueError, match="2D"):
        wrapper.transform(_wrapper_container_3d())


def test_sklearn_wrapper_inverse_transform_non_2d():
    wrapper = SklearnWrapper(StandardScaler()).fit(_wrapper_container_2d())
    with pytest.raises(ValueError, match="2D"):
        wrapper.inverse_transform(_wrapper_container_3d())
