import numpy as np

from coco_pipe.decoding.scalers import SubjectStandardScaler


def test_subject_standard_scaler_no_groups():
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    scaler = SubjectStandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)
    assert X_scaled.shape == X.shape


def test_subject_standard_scaler_with_groups():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]])
    groups = np.array(["a", "a", "b", "b"])
    scaler = SubjectStandardScaler()
    X_scaled = scaler.fit_transform(X, groups=groups)
    assert X_scaled.shape == X.shape
    # Ensure centering within subject 'a'
    np.testing.assert_allclose(X_scaled[0:2].mean(axis=0), [0.0, 0.0], atol=1e-7)
    # Ensure centering within subject 'b'
    np.testing.assert_allclose(X_scaled[2:4].mean(axis=0), [0.0, 0.0], atol=1e-7)


def test_subject_standard_scaler_transform_with_groups():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0], [30.0, 40.0]])
    groups = np.array(["a", "a", "b", "b"])
    scaler = SubjectStandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X, groups=groups)
    assert X_scaled.shape == X.shape
