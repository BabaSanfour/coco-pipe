"""
Tests for dim_reduction.evaluation module.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_blobs

from coco_pipe.dim_reduction.analysis import perturbation_importance
from coco_pipe.dim_reduction.core import DimReduction
from coco_pipe.dim_reduction.evaluation import (
    MethodSelector,
    compute_coranking_matrix,
    compute_mrre,
    compute_velocity_fields,
    continuity,
    lcmc,
    moving_average,
    shepard_diagram_data,
    trajectory_acceleration,
    trajectory_curvature,
    trajectory_dispersion,
    trajectory_displacement,
    trajectory_path_length,
    trajectory_separation,
    trajectory_speed,
    trajectory_tortuosity,
    trajectory_turning_angle,
    trustworthiness,
)
from coco_pipe.viz.dim_reduction import plot_metrics


@pytest.fixture
def data():
    return make_blobs(n_samples=50, n_features=10, centers=3, random_state=42)


@pytest.fixture
def linear_data():
    return np.linspace(0, 10, 100)[:, None] * np.ones((1, 5)) + np.random.normal(
        0, 0.1, (100, 5)
    )


def test_method_selector(data):
    X, y = data
    sweep_metrics = [
        "trustworthiness",
        "continuity",
        "lcmc",
        "mrre_intrusion",
        "mrre_extrusion",
        "mrre_total",
    ]

    reducers = [
        DimReduction("PCA", n_components=2),
        DimReduction("Isomap", n_components=2, n_neighbors=10),
    ]
    for reducer in reducers:
        reducer.fit_transform(X, y=y)
        reducer.score(
            reducer.transform(X), X=X, metrics=sweep_metrics, k_values=[5, 10, 20]
        )

    evaluator = MethodSelector(reducers)
    evaluator.collect()

    metrics_df = evaluator.to_frame()
    assert set(metrics_df["method"]) == {"PCA", "Isomap"}

    res_pca = metrics_df[
        (metrics_df["method"] == "PCA") & (metrics_df["metric"] == "trustworthiness")
    ]
    assert len(res_pca) == 3
    assert set(pd.to_numeric(res_pca["scope_value"])) == {5, 10, 20}


def test_method_selector_to_frame_global_records():
    X = np.random.rand(20, 5)
    reducer = DimReduction("PCA", n_components=2)
    emb = reducer.fit_transform(X)
    reducer.score(emb, X=X, metrics=["trustworthiness"], k_values=[5])
    selector = MethodSelector([reducer]).collect()

    frame = selector.to_frame()
    assert not frame.empty
    assert {"method", "metric", "value", "scope", "scope_value"} <= set(frame.columns)


def test_method_selector_to_frame(data):
    X, y = data
    sweep_metrics = [
        "trustworthiness",
        "continuity",
        "lcmc",
        "mrre_intrusion",
        "mrre_extrusion",
        "mrre_total",
    ]
    reducer = DimReduction("PCA", n_components=2)
    reducer.fit_transform(X, y=y)
    reducer.score(reducer.transform(X), X=X, metrics=sweep_metrics, k_values=[5, 10])
    selector = MethodSelector([reducer]).collect()

    metrics_df = selector.to_frame()
    assert {"method", "metric", "value", "scope", "scope_value"} <= set(
        metrics_df.columns
    )
    assert set(metrics_df["scope"]) == {"k"}
    assert set(metrics_df["method"]) == {"PCA"}


def test_velocity_fields(linear_data):
    X = linear_data
    X_emb = X

    V_emb = compute_velocity_fields(X, X_emb, delta_t=1, n_neighbors=5)

    V_valid = V_emb[1:-2]
    mean_v = np.mean(V_valid, axis=0)

    # Target is [1, 1, 1, 1, 1] because all coords increase linearly
    target_dir = np.ones(5) / np.linalg.norm(np.ones(5))
    calc_dir = mean_v / np.linalg.norm(mean_v)

    dot_prod = np.dot(target_dir, calc_dir)
    assert dot_prod > 0.9

    # 1. Multi-group (independent sequences)
    X_mg = np.array(
        [
            [1.0],
            [2.1],
            [3.2],  # Group 0: Increasing
            [10.0],
            [8.9],
            [7.8],  # Group 1: Decreasing
        ]
    )
    groups = np.array([0, 0, 0, 1, 1, 1])
    V_mg = compute_velocity_fields(X_mg, X_mg, n_neighbors=2, groups=groups)
    assert V_mg[0, 0] > 0
    assert V_mg[3, 0] < 0
    assert V_mg[2, 0] == 0  # Last in group 0

    # 2. Timed sequence (scrambled order + scaling)
    X_time = np.array([[1.0], [2.0], [3.0]])
    times = np.array([1, 2, 0])  # 3.0(t0) -> 1.0(t1) -> 2.0(t2)
    V_t = compute_velocity_fields(X_time, X_time, n_neighbors=1, times=times)
    assert V_t[2, 0] < 0  # 3.0 -> 1.0
    assert V_t[0, 0] > 0  # 1.0 -> 2.0

    # 3. Time sorting within groups
    X_mg_t = np.array([[1.0], [2.0], [3.0], [4.0]])
    mg_groups = np.array([0, 0, 1, 1])
    mg_times = np.array([2, 1, 10, 5])  # Scrambled within groups
    V_mg_t = compute_velocity_fields(
        X_mg_t, X_mg_t, n_neighbors=1, groups=mg_groups, times=mg_times
    )
    assert V_mg_t.shape == (4, 1)


def test_velocity_guardrails():
    """
    Verify that velocity computation raises ValueError for invalid params
    """
    X = np.random.rand(10, 5)
    X_emb = np.random.rand(10, 2)

    # n_samples < 2
    X_small = np.random.rand(1, 5)
    X_emb_small = np.random.rand(1, 2)
    with pytest.raises(ValueError, match="at least 2"):
        compute_velocity_fields(X_small, X_emb_small)

    # n_neighbors invalid
    with pytest.raises(ValueError, match="n_neighbors must be > 0"):
        compute_velocity_fields(X, X_emb, n_neighbors=0)

    # delta_t invalid
    with pytest.raises(ValueError, match="delta_t must be > 0"):
        compute_velocity_fields(X, X_emb, delta_t=0, n_neighbors=5)

    with pytest.raises(ValueError, match="same number of samples"):
        compute_velocity_fields(X, X_emb[:5], n_neighbors=5)

    # n_neighbors invalid
    with pytest.raises(ValueError, match="less than n_samples"):
        compute_velocity_fields(X, X_emb, n_neighbors=10)

    # Misaligned inputs
    with pytest.raises(ValueError, match="same number of samples"):
        compute_velocity_fields(X, X_emb, n_neighbors=5, groups=np.array([0, 1]))
    with pytest.raises(ValueError, match="same number of samples"):
        compute_velocity_fields(X, X_emb, n_neighbors=5, times=np.array([0, 1]))

    with pytest.raises(ValueError, match="strictly increasing"):
        compute_velocity_fields(X, X_emb, n_neighbors=5, times=np.zeros(10))

    # Input dimensionality
    with pytest.raises(ValueError, match="2D array"):
        compute_velocity_fields(X[:, 0], X_emb)
    with pytest.raises(ValueError, match="2D array"):
        compute_velocity_fields(X, X_emb[:, 0])

    # Shape checks
    with pytest.raises(ValueError, match="sigma must be > 0"):
        compute_velocity_fields(X, X_emb, n_neighbors=5, sigma=0.0)
    with pytest.raises(ValueError, match="1D array"):
        compute_velocity_fields(X, X_emb, n_neighbors=5, groups=X)
    with pytest.raises(ValueError, match="1D array"):
        compute_velocity_fields(X, X_emb, n_neighbors=5, times=X)
    with pytest.raises(ValueError, match="at least 2"):
        compute_velocity_fields(X[:1], X_emb[:1])
    with pytest.raises(ValueError, match="less than n_samples"):
        compute_velocity_fields(X, X_emb, delta_t=10, n_neighbors=5)


def test_velocity_pathological_cases():
    """Verify behavior on singular or duplicate data."""
    # 1. Zero velocity (all identical points)
    X = np.zeros((5, 2))
    V = compute_velocity_fields(X, X, n_neighbors=2)
    assert np.all(V == 0)

    # 2. Small groups (smaller than delta_t)
    X_small = np.random.rand(5, 2)
    groups = np.array([0, 1, 2, 3, 4])
    V_small = compute_velocity_fields(X_small, X_small, n_neighbors=2, groups=groups)
    assert np.all(V_small == 0)

    # 3. Duplicate neighbors (forces 'valid' mask branch)
    X_dupe = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    groups = np.array([0, 0, 1, 1, 1])
    V_dupe = compute_velocity_fields(X_dupe, X_dupe, n_neighbors=2, groups=groups)
    assert V_dupe.shape == (5, 2)

    # 4. Underflow probs (zero_probs_sum branch)
    X_under = np.array([[0.0], [1.0], [2.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        V_under = compute_velocity_fields(X_under, X_under, n_neighbors=1, sigma=1e-10)
    assert V_under.shape == (3, 1)

    # 5. All neighbors invalid
    X_patho = np.array([[0.0], [0.0], [0.0], [1.0]])
    V_patho = compute_velocity_fields(X_patho, X_patho, delta_t=2, n_neighbors=2)
    assert V_patho[1, 0] == 0


def test_moving_average():
    x = np.array([1, 2, 3, 4, 5])
    # Window 3: [2, 3, 4]
    smoothed = moving_average(x, 3)
    np.testing.assert_allclose(smoothed, [2, 3, 4])

    # Window 1: same
    smoothed = moving_average(x, 1)
    np.testing.assert_array_equal(smoothed, x)


def test_moving_average_guardrails():
    x = np.array([1, 2, 3])

    with pytest.raises(ValueError, match="1D array"):
        moving_average(np.eye(2), 2)

    with pytest.raises(ValueError, match="positive integer"):
        moving_average(x, 0)

    with pytest.raises(ValueError, match="larger than the input length"):
        moving_average(x, 4)


def test_trajectory_acceleration_linear_zero():
    t = np.linspace(0, 10, 11)
    traj = np.stack([t, np.zeros_like(t)], axis=1)

    acc = trajectory_acceleration(traj, dt=1.0)
    np.testing.assert_allclose(acc, 0.0, atol=1e-8)


def test_trajectory_speed_linear():
    # Linear motion along x: speed should be constant 1.0
    t = np.linspace(0, 10, 11)  # 0, 1, ..., 10
    traj = np.zeros((11, 2))
    traj[:, 0] = t  # x = t, y = 0

    sp = trajectory_speed(traj, dt=1.0)

    # Speed is dx/dt = 1
    expected = np.ones(11)
    np.testing.assert_allclose(sp, expected, atol=1e-5)


def test_trajectory_speed_circle():
    # Circular motion: x=cos(t), y=sin(t)
    R = 2.0
    w = 1.0
    t = np.linspace(0, 2 * np.pi, 100)
    dt = t[1] - t[0]

    traj = np.stack([R * np.cos(w * t), R * np.sin(w * t)], axis=1)

    sp = trajectory_speed(traj, dt=dt)

    # Theoretical speed R*w = 2.0
    np.testing.assert_allclose(sp[:-1], 2.0, rtol=1e-2)


def test_trajectory_curvature_circle():
    # Curvature of circle radius R is 1/R
    R = 4.0
    t = np.linspace(0, 2 * np.pi, 1000)
    traj = np.stack([R * np.cos(t), R * np.sin(t)], axis=1)  # (1000, 2)

    k = trajectory_curvature(traj)

    k_center = k[10:-10]
    np.testing.assert_allclose(k_center, 1 / R, rtol=1e-2)


def test_trajectory_curvature_line():
    t = np.linspace(0, 10, 100)
    traj = np.stack([t, t], axis=1)

    k = trajectory_curvature(traj)
    np.testing.assert_allclose(k[10:-10], 0, atol=1e-5)


def test_trajectory_path_length_displacement_and_tortuosity():
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

    total_length = trajectory_path_length(traj)
    cumulative_length = trajectory_path_length(traj, cumulative=True)
    displacement = trajectory_displacement(traj)
    tortuosity = trajectory_tortuosity(traj)

    np.testing.assert_allclose(total_length, 2.0)
    np.testing.assert_allclose(cumulative_length, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(displacement, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(tortuosity, 1.0)

    closed_loop = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    assert np.isinf(trajectory_tortuosity(closed_loop))


def test_trajectory_turning_angle_right_angle():
    traj = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    angles = trajectory_turning_angle(traj)
    np.testing.assert_allclose(angles, np.pi / 2.0, atol=1e-8)


def test_trajectory_separation():
    # Two groups: A and B
    n_trials = 4
    n_times = 10
    n_dims = 2

    traj = np.zeros((n_trials, n_times, n_dims))
    labels = np.array(["A", "A", "B", "B"])

    # A: x=0, y=t
    traj[0, :, 1] = np.arange(n_times)
    traj[1, :, 1] = np.arange(n_times)

    # B: x=2, y=t
    traj[2, :, 0] = 2.0
    traj[2, :, 1] = np.arange(n_times)
    traj[3, :, 0] = 2.0
    traj[3, :, 1] = np.arange(n_times)

    sep = trajectory_separation(traj, labels, method="centroid")

    assert ("A", "B") in sep or ("B", "A") in sep
    dist = sep.get(("A", "B"), sep.get(("B", "A")))

    np.testing.assert_allclose(dist, 2.0, atol=1e-5)


def test_trajectory_dispersion_and_within_between_ratio():
    n_times = 6
    t = np.arange(n_times, dtype=float)
    traj = np.zeros((4, n_times, 2))
    labels = np.array(["A", "A", "B", "B"])

    traj[0, :, 0] = -1.0
    traj[1, :, 0] = 0.0
    traj[2, :, 0] = 2.0
    traj[3, :, 0] = 3.0
    traj[:, :, 1] = t

    global_dispersion = trajectory_dispersion(traj)
    labeled_dispersion = trajectory_dispersion(traj, labels)
    ratio = trajectory_separation(traj, labels, method="within_between_ratio")

    assert global_dispersion.shape == (n_times,)
    np.testing.assert_allclose(labeled_dispersion["A"], 0.5)
    np.testing.assert_allclose(labeled_dispersion["B"], 0.5)
    np.testing.assert_allclose(ratio[("A", "B")], 3.0, atol=1e-6)


def test_distributional_mahalanobis_and_margin_separation():
    n_times = 5
    traj = np.zeros((4, n_times, 2))
    labels = np.array(["A", "A", "B", "B"])

    traj[0, :, 0] = -1.0
    traj[1, :, 0] = 0.0
    traj[2, :, 0] = 2.0
    traj[3, :, 0] = 3.0

    energy = trajectory_separation(traj, labels, method="distributional")
    mahal = trajectory_separation(traj, labels, method="mahalanobis")
    margin = trajectory_separation(traj, labels, method="margin", agg="median")

    np.testing.assert_allclose(energy[("A", "B")], 5.0, atol=1e-6)
    np.testing.assert_allclose(mahal[("A", "B")], np.sqrt(18.0), atol=1e-4)
    np.testing.assert_allclose(margin[("A", "B")], 1.5, atol=1e-6)


def test_trajectory_separation_method_variants():
    traj = np.zeros((4, 4, 2))
    labels = np.array(["A", "A", "B", "B"])
    traj[2:, :, 0] = 2.0

    centroid_default = trajectory_separation(traj, labels)
    centroid = trajectory_separation(traj, labels, method="centroid")
    ratio = trajectory_separation(traj, labels, method="within_between_ratio")
    mahal = trajectory_separation(traj, labels, method="mahalanobis")
    dist = trajectory_separation(traj, labels, method="distributional")
    margin = trajectory_separation(traj, labels, method="margin")

    np.testing.assert_allclose(
        centroid[("A", "B")],
        centroid_default[("A", "B")],
    )
    assert ratio[("A", "B")].shape == (4,)
    assert mahal[("A", "B")].shape == (4,)
    assert dist[("A", "B")].shape == (4,)
    assert margin[("A", "B")].shape == (4,)


def test_trajectory_geometry_guardrails():
    """Verify generic trajectory guardrails from geometry.py."""
    # 1. 1D trajectory (ndim < 2)
    with pytest.raises(ValueError, match="at least 2D"):
        trajectory_speed(np.zeros(5))

    # 2. Acceleration dt=0
    traj_ok = np.zeros((5, 2))
    with pytest.raises(ValueError, match="`dt` must be > 0"):
        trajectory_acceleration(traj_ok, dt=0.0)

    # 3. Label dimensionality
    traj_3d = np.zeros((2, 5, 2))
    with pytest.raises(ValueError, match="shape"):
        # The code checks dim != 3 for trial-wise
        trajectory_separation(np.zeros((2, 2)), np.array(["A", "B"]))

    with pytest.raises(ValueError, match="1D array"):
        trajectory_separation(traj_3d, np.zeros((2, 2)))

    with pytest.raises(ValueError, match="one entry per trial"):
        trajectory_separation(traj_3d, np.array(["A"]))

    with pytest.raises(ValueError, match="unique labels"):
        trajectory_separation(traj_3d, np.array(["A", "A"]))

    # 4. Unknown methods/metrics
    with pytest.raises(ValueError, match="Unsupported separation method"):
        trajectory_separation(traj_3d, np.array(["A", "B"]), method="unknown")

    with pytest.raises(ValueError, match="Only metric='energy'"):
        from coco_pipe.dim_reduction.evaluation.geometry import (
            _distributional_separation_timecourse,
        )

        _distributional_separation_timecourse(traj_3d[:1], traj_3d[1:], metric="cosine")

    with pytest.raises(ValueError, match="must be either 'median' or 'mean'"):
        from coco_pipe.dim_reduction.evaluation.geometry import (
            _margin_separation_timecourse,
        )

        _margin_separation_timecourse(traj_3d[:1], traj_3d[1:], agg="max")

    # 5. Missing Coverage Hooks
    with pytest.raises(ValueError, match="at least 2 time points"):
        _ = trajectory_speed(np.zeros((1, 2)))

    with pytest.raises(ValueError, match="`dt` must be > 0"):
        trajectory_speed(traj_ok, dt=-1.0)

    # 6. Empty collections or single points (internal coverage)
    from coco_pipe.dim_reduction.evaluation.geometry import (
        _mean_self_pairwise_distance,
        _nearest_within_distances,
    )

    assert _mean_self_pairwise_distance(np.zeros((0, 2))) == 0.0
    assert np.all(_nearest_within_distances(np.zeros((1, 2))) == 0.0)


def test_feature_importance():
    X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=42)
    X[:, 1] = np.random.randn(100)

    model = DimReduction("PCA", n_components=1)
    X_emb = model.fit_transform(X)
    scores = perturbation_importance(model, X, ["Signal", "Noise"], X_emb)

    assert scores["Signal"] > scores["Noise"]


@pytest.fixture
def sample_data():
    """Create a simple case where we know the ranks."""
    # 5 points on a line
    X = np.array([[0], [1], [2], [3], [4]])
    # Embedding preserves order exactly
    X_emb = np.array([[0], [2], [4], [6], [8]])
    return X, X_emb


def test_perfect_embedding(sample_data):
    """If embedding preserves order perfectly, Q should be diagonal."""
    X, X_emb = sample_data
    Q = compute_coranking_matrix(X, X_emb)

    n = X.shape[0]
    assert Q.shape == (n - 1, n - 1)

    # Trustworthiness and Continuity should be 1.0 for all valid k
    max_valid_k = (2 * n - 2) // 3
    for k in range(1, max_valid_k + 1):
        t = trustworthiness(Q, k)
        c = continuity(Q, k)
        assert np.isclose(t, 1.0)
        assert np.isclose(c, 1.0)


def test_imperfect_embedding():
    """Create a case with known errors."""
    # High: 0-1-2
    X = np.array([[0], [1], [2]])

    # Low: 0-2-1 (2 and 1 swapped)
    X_emb = np.array([[0], [2], [1]])

    Q = compute_coranking_matrix(X, X_emb)

    # We expect < 1.0
    t = trustworthiness(Q, k=1)
    c = continuity(Q, k=1)

    assert t < 1.0 or c < 1.0


def test_metrics_consistency():
    """Ensure vectorization doesn't break logic on random data."""
    rng = np.random.RandomState(42)
    X = rng.rand(20, 5)
    X_emb = rng.rand(20, 2)

    Q = compute_coranking_matrix(X, X_emb)

    t_5 = trustworthiness(Q, k=5)
    c_5 = continuity(Q, k=5)

    assert 0 <= t_5 <= 1.0
    assert 0 <= c_5 <= 1.0


def test_coranking_bias():
    """Verify that co-ranking matrix excludes self-neighbors"""
    # Create 5 orthogonal points (distance matrix will have diag 0, others sqrt(2))
    X = np.eye(5)
    X_emb = np.eye(5)[:, :2]

    # compute_coranking_matrix is imported above
    Q = compute_coranking_matrix(X, X_emb)

    # n=5, so Q should be (n-1) x (n-1) = 4x4
    assert Q.shape == (4, 4)
    # The total count in Q should be n * (n-1) = 5 * 4 = 20
    assert np.sum(Q) == 20


def test_metrics_pathological_data():
    """Verify behavior on pathological (constant/singular) data."""
    # Constant data: all distances are zero
    X = np.zeros((10, 5))
    X_emb = np.zeros((10, 2))

    Q = compute_coranking_matrix(X, X_emb)
    # With all distances zero, ranks might be arbitrary but should not crash
    assert Q.shape == (9, 9)
    assert np.sum(Q) == 90

    # These should not crash but will raise ValueError now because of the guardrails
    # if k is valid but here we use valid k.
    t = trustworthiness(Q, k=3)
    assert isinstance(t, float)


def test_trustworthiness_edge_cases():
    """Test trustworthy edge cases including small N."""
    # Q matrix for N=5
    # perfect matching
    Q = np.diag([1] * 4)

    # 1. Largest valid small-N neighborhood
    t = trustworthiness(Q, k=2)
    assert t == 1.0

    # 2. Smaller valid neighborhood
    t_small = trustworthiness(Q, k=1)
    assert t_small == 1.0

    with pytest.raises(ValueError, match="normalization term must stay positive"):
        trustworthiness(Q, k=3)


def test_continuity_edge_cases():
    """Test continuity edge cases."""
    Q = np.diag([1] * 4)

    # 1. Largest valid small-N neighborhood
    c = continuity(Q, k=2)
    assert c == 1.0

    # 2. Smaller valid neighborhood
    c_small = continuity(Q, k=1)
    assert c_small == 1.0

    with pytest.raises(ValueError, match="normalization term must stay positive"):
        continuity(Q, k=3)


def test_lcmc_edge_cases():
    """Test LCMC edge cases."""
    Q = np.diag([1] * 4)
    Q[3, 0] = 1.0  # Add some overlap noise if needed, but diag is fine

    # k >= n-1
    l_score = lcmc(Q, k=3)
    assert isinstance(l_score, float)

    # Normal case
    l_norm = lcmc(Q, k=2)
    assert isinstance(l_norm, float)


def test_mrre_edge_cases():
    """Test MRRE edge cases."""
    Q = np.diag([1] * 4)  # N=5

    # MRRE calculation involves H_k normalization
    m_int, m_ext = compute_mrre(Q, k=2)
    assert isinstance(m_int, float)
    assert m_ext == 0.0  # Perfect embedding has 0 error


def test_shepard_diagram_data():
    """Test shepard diagram data generation."""
    X = np.random.rand(10, 5)
    X_emb = np.random.rand(10, 2)

    # 1. Full sample (N <= sample_size)
    d_orig, d_emb = shepard_diagram_data(X, X_emb, sample_size=20)
    assert len(d_orig) == 45  # 10*9/2
    assert len(d_emb) == 45

    # 2. Subsample (N > sample_size)
    d_orig_sub, d_emb_sub = shepard_diagram_data(X, X_emb, sample_size=5)
    assert len(d_orig_sub) == 10  # 5*4/2


def test_reproducibility_shepard_sampling():
    """Verify that shepard_diagram_data is reproducible with random_state."""
    X = np.random.rand(50, 10)
    X_emb = np.random.rand(50, 2)

    # Subsample to a small number
    size = 10
    d1_orig, d1_emb = shepard_diagram_data(X, X_emb, sample_size=size, random_state=42)
    d2_orig, d2_emb = shepard_diagram_data(X, X_emb, sample_size=size, random_state=42)
    d3_orig, d3_emb = shepard_diagram_data(X, X_emb, sample_size=size, random_state=43)

    assert np.allclose(d1_orig, d2_orig)
    assert np.allclose(d1_emb, d2_emb)
    assert not np.allclose(d1_orig, d3_orig)


def test_metric_guardrails():
    X = np.random.rand(10, 2)
    X_emb = np.random.rand(10, 2)
    Q = compute_coranking_matrix(X, X_emb)

    with pytest.raises(ValueError, match="must be > 0"):
        trustworthiness(Q, k=0)
    with pytest.raises(ValueError, match="must be > 0"):
        continuity(Q, k=-1)
    with pytest.raises(ValueError, match="must be less than n_samples - 1"):
        trustworthiness(Q, k=9)

    small_Q = np.diag([1] * 4)
    with pytest.raises(ValueError, match="normalization term must stay positive"):
        trustworthiness(small_Q, k=3)
    with pytest.raises(ValueError, match="normalization term must stay positive"):
        continuity(small_Q, k=3)

    with pytest.raises(ValueError, match="same number of samples"):
        compute_coranking_matrix(X, np.random.rand(9, 2))
    with pytest.raises(ValueError, match="2D array"):
        compute_coranking_matrix(X[:, 0], X_emb)
    with pytest.raises(ValueError, match="`X_emb` must be a 2D array"):
        compute_coranking_matrix(X, X_emb[:, 0])
    with pytest.raises(ValueError, match="At least 2 samples"):
        compute_coranking_matrix(X[:1], X_emb[:1])

    with pytest.raises(ValueError, match="must be an integer"):
        trustworthiness(Q, k=0.5)
    with pytest.raises(ValueError, match="square 2D co-ranking matrix"):
        trustworthiness(np.zeros((4, 3)), k=1)

    with pytest.raises(ValueError, match="greater than 1"):
        shepard_diagram_data(X, X_emb, sample_size=1)
    with pytest.raises(ValueError, match="same number of samples"):
        shepard_diagram_data(X, np.random.rand(9, 2))
    with pytest.raises(ValueError, match="must be an integer"):
        shepard_diagram_data(X, X_emb, sample_size=0.5)

    # 5. Internal scale function
    from coco_pipe.dim_reduction.evaluation.metrics import _trust_continuity_scale

    with pytest.raises(ValueError, match="normalization term must stay positive"):
        _trust_continuity_scale(5, 3, "test")


def test_method_selector_collects_scored_reducers():
    """Test collection of already-scored reducers."""
    X = np.random.rand(50, 10).astype(np.float32)
    sweep_metrics = [
        "trustworthiness",
        "continuity",
        "lcmc",
        "mrre_intrusion",
        "mrre_extrusion",
        "mrre_total",
    ]

    pca = DimReduction("PCA", n_components=2)
    isomap = DimReduction("Isomap", n_components=2, n_neighbors=5)

    for reducer in (pca, isomap):
        reducer.fit_transform(X)
        reducer.score(
            reducer.transform(X), X=X, metrics=sweep_metrics, k_values=[5, 10]
        )

    selector = MethodSelector([pca, isomap]).collect()

    metrics_df = selector.to_frame()
    assert {"PCA", "Isomap"} <= set(metrics_df["method"])

    # Check tidy results structure
    res_pca = metrics_df[
        (metrics_df["method"] == "PCA") & (metrics_df["metric"] == "trustworthiness")
    ]
    assert len(res_pca) == 2  # 2 k values
    ranked = selector.rank_methods(selection_metric="trustworthiness")
    assert not ranked.empty


def test_method_selector_single_method():
    """Test with single method dict to ensure key handling works."""
    X = np.random.rand(20, 5)
    sweep_metrics = [
        "trustworthiness",
        "continuity",
        "lcmc",
        "mrre_intrusion",
        "mrre_extrusion",
        "mrre_total",
    ]
    reducer = DimReduction("PCA", n_components=2)
    reducer.fit_transform(X)
    reducer.score(reducer.transform(X), X=X, metrics=sweep_metrics, k_values=[5])
    reducers = {"PCA": reducer}

    selector = MethodSelector(reducers).collect()

    metrics_df = selector.to_frame()
    res_pca = metrics_df[
        (metrics_df["method"] == "PCA") & (metrics_df["metric"] == "trustworthiness")
    ]
    assert len(res_pca) == 1


def test_evaluation_plot(data):
    """Test plotting of selector metric records through plot_metrics."""
    X, y = data
    import matplotlib.pyplot as plt

    selector = MethodSelector([])
    selector.metric_records_ = [
        {
            "method": "PCA",
            "metric": "trustworthiness",
            "value": 0.9,
            "scope": "k",
            "scope_value": 1,
        },
        {
            "method": "PCA",
            "metric": "trustworthiness",
            "value": 0.8,
            "scope": "k",
            "scope_value": 2,
        },
        {
            "method": "UMAP",
            "metric": "trustworthiness",
            "value": 0.95,
            "scope": "k",
            "scope_value": 1,
        },
        {
            "method": "UMAP",
            "metric": "trustworthiness",
            "value": 0.85,
            "scope": "k",
            "scope_value": 2,
        },
    ]

    fig = plot_metrics(selector, metric="trustworthiness")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_dimreduction_score_respects_metric_selection():
    X = np.random.rand(30, 6)
    reducer = DimReduction("PCA", n_components=2)
    reducer.fit_transform(X)

    payload = reducer.score(reducer.transform(X), X=X, metrics=["trustworthiness"])

    assert set(payload["metrics"]) == {"trustworthiness"}
    assert "coranking_matrix_" in payload["diagnostics"]
    assert "shepard_distances_" not in payload["diagnostics"]


def test_method_selector_best_method_uses_primary_metric():
    selector = MethodSelector([])
    selector.metric_records_ = [
        {
            "method": "PCA",
            "metric": "trustworthiness",
            "value": 0.9,
            "scope": "k",
            "scope_value": 5,
        },
        {
            "method": "UMAP",
            "metric": "trustworthiness",
            "value": 0.95,
            "scope": "k",
            "scope_value": 5,
        },
        {
            "method": "PCA",
            "metric": "continuity",
            "value": 0.93,
            "scope": "k",
            "scope_value": 5,
        },
        {
            "method": "UMAP",
            "metric": "continuity",
            "value": 0.92,
            "scope": "k",
            "scope_value": 5,
        },
    ]

    ranked = selector.rank_methods(
        selection_metric="trustworthiness",
        selection_k=5,
        tie_breakers=["continuity"],
    )

    assert list(ranked["method"]) == ["UMAP", "PCA"]
    assert ranked.iloc[0]["method"] == "UMAP"


def test_method_selector_rejects_unsupported_ranking_metric():
    selector = MethodSelector([])
    selector.metric_records_ = [
        {
            "method": "PCA",
            "metric": "trajectory_speed_mean",
            "value": 1.0,
            "scope": "global",
            "scope_value": "global",
        }
    ]

    with pytest.raises(ValueError, match="Unsupported selection metric"):
        selector.rank_methods(selection_metric="trajectory_speed_mean")


def test_evaluate_embedding_trajectory_orchestration():
    """Test trajectory orchestration in evaluate_embedding."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    # 3 trajectories, 10 timepoints, 2 dimensions
    traj = np.random.rand(3, 10, 2)
    labels = np.array([0, 1, 0])
    times = np.arange(10)

    payload = evaluate_embedding(
        X_emb=traj,
        method_name="TrajectoryMethod",
        labels=labels,
        times=times,
        quality_metadata={"source": "synthetic"},
    )

    assert "metrics" in payload
    metrics = payload["metrics"]
    assert "trajectory_speed_mean" in metrics
    assert "trajectory_separation_auc::0::1" in metrics
    assert payload["metadata"]["trajectory_count"] == 3
    assert payload["metadata"]["source"] == "synthetic"

    # Test with metric selection
    payload_sub = evaluate_embedding(
        X_emb=traj,
        metrics=["trajectory_speed", "trajectory_separation"],
        labels=labels,
    )
    assert "trajectory_speed_mean" in payload_sub["metrics"]
    assert "trajectory_curvature_mean" not in payload_sub["metrics"]

    # Test separation AUC integration with/without times
    payload_no_times = evaluate_embedding(X_emb=traj, labels=labels)
    assert "trajectory_separation_auc::0::1" in payload_no_times["metrics"]


def test_evaluate_embedding_guardrails_extended():
    """Test extended guardrails for evaluate_embedding."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(10, 2)

    # 1. Invalid quality_metadata
    with pytest.raises(TypeError, match="metadata must be a dictionary"):
        evaluate_embedding(X_emb, quality_metadata="not-a-dict")

    # 2. Invalid diagnostics
    with pytest.raises(TypeError, match="diagnostics must be a dictionary"):
        evaluate_embedding(X_emb, diagnostics="not-a-dict")

    # 3. Missing X for 2D evaluation
    with pytest.raises(ValueError, match="Original data `X` is required"):
        evaluate_embedding(X_emb, metrics=["trustworthiness"])

    # 4. Dimension mismatch for Standard Evaluation
    with pytest.raises(ValueError, match="matching sample counts"):
        evaluate_embedding(X_emb, X=np.random.rand(9, 2), metrics=["trustworthiness"])

    # 5. Invalid X_emb dimensions (e.g. 4D)
    with pytest.raises(ValueError, match="must be either 2D or 3D"):
        evaluate_embedding(np.random.rand(2, 2, 2, 2))


def test_method_selector_guardrails_extended():
    """Test extended guardrails for MethodSelector."""
    # 1. Invalid reducer type in init
    with pytest.raises(TypeError, match="only accepts scored DimReduction objects"):
        MethodSelector(["not-a-reducer"])

    # 2. Collect error: no embedding
    reducer_no_emb = DimReduction("PCA")
    with pytest.raises(ValueError, match="has no metric records"):
        MethodSelector([reducer_no_emb]).collect()

    # 3. Collect error: no records
    X = np.random.rand(10, 5)
    reducer_no_recs = DimReduction("PCA")
    reducer_no_recs.fit_transform(X)
    # No score() call
    # With stateless DimReduction, this now checks metric_records_
    with pytest.raises(ValueError, match="has no metric records"):
        MethodSelector([reducer_no_recs]).collect()

    # 4. Rank methods: invalid tie-breaker
    reducer_ok = DimReduction("PCA")
    reducer_ok.fit_transform(X)
    reducer_ok.score(
        reducer_ok.transform(X), X=X, metrics=["trustworthiness"], k_values=[5, 10]
    )
    selector = MethodSelector([reducer_ok]).collect()

    with pytest.raises(ValueError, match="Unsupported tie-breaker"):
        selector.rank_methods(
            selection_metric="trustworthiness", tie_breakers=["invalid"]
        )

    # 5. Rank methods: metric not available
    with pytest.raises(ValueError, match="is not available"):
        selector.rank_methods(selection_metric="continuity")

    # 6. Rank methods: k not found
    with pytest.raises(ValueError, match="has no observations at k=99"):
        selector.rank_methods(selection_metric="trustworthiness", selection_k=99)


def test_internal_trajectory_summary_guardrails():
    """Test internal _summarize_trajectory_metric error paths."""
    from coco_pipe.dim_reduction.evaluation.core import _summarize_trajectory_metric

    with pytest.raises(ValueError, match="Unsupported trajectory summary type"):
        _summarize_trajectory_metric("test", np.zeros(5), summary_type="invalid")


def test_evaluate_embedding_shepard_only():
    """Test evaluate_embedding with shepard_correlation specifically requested."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X = np.random.rand(10, 5)
    X_emb = np.random.rand(10, 2)

    payload = evaluate_embedding(X_emb, X=X, metrics=["shepard_correlation"])
    assert "shepard_correlation" in payload["metrics"]
    assert "trustworthiness" not in payload["metrics"]


def test_evaluate_embedding_trajectory_insufficient_points():
    """Test trajectory evaluation with too few points for certain metrics."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    # 3 trajectories, 2 points (cannot do acceleration which needs 3)
    traj = np.random.rand(3, 2, 2)
    payload = evaluate_embedding(X_emb=traj)

    assert "trajectory_speed_mean" in payload["metrics"]
    assert "trajectory_acceleration_mean" not in payload["metrics"]


def test_evaluate_embedding_empty_metric_selection():
    """Test evaluate_embedding with a metric selection that yields nothing."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(10, 2)
    X = np.random.rand(10, 5)

    # Requesting standard metrics for a 3D embedding (or vice versa)
    payload = evaluate_embedding(X_emb, X=X, metrics=["trajectory_speed"])
    assert payload["metrics"] == {}


def test_method_selector_empty_frame():
    """Test to_frame() when selector is empty."""
    selector = MethodSelector([])
    assert selector.to_frame().empty
    assert "method" in selector.to_frame().columns


def test_method_selector_collect_empty_records():
    """Trigger ValueError in collect if a reducer has empty records."""
    X = np.random.rand(10, 5)
    reducer = DimReduction("PCA")
    reducer.fit_transform(X)
    reducer.metric_records_ = []  # Explicitly empty

    selector = MethodSelector([reducer])
    with pytest.raises(ValueError, match="has no metric records"):
        selector.collect()


def test_evaluate_embedding_trajectory_separation_edge_cases():
    """Test edge cases in _evaluate_trajectory_metrics separation logic."""
    # 1. Empty values_arr for AUC
    # We mock trajectory_separation to return an empty array for a pair
    import coco_pipe.dim_reduction.evaluation.core as core_mod
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    original_sep = core_mod.trajectory_separation

    def mock_sep_empty(*a, **k):
        return {("A", "B"): np.array([])}

    try:
        core_mod.trajectory_separation = mock_sep_empty
        traj = np.random.rand(3, 10, 2)
        payload = evaluate_embedding(X_emb=traj, labels=[0, 1, 0])
        # Should be NaN for AUC
        assert np.isnan(payload["metrics"]["trajectory_separation_auc::A::B"])
    finally:
        core_mod.trajectory_separation = original_sep

    # 2. Time mismatch for AUC integration
    traj = np.random.rand(2, 5, 2)
    # 3 points in times, but 5 in trajectory
    payload_mismatch = evaluate_embedding(X_emb=traj, labels=[0, 1], times=[0, 1, 2])
    assert "trajectory_separation_auc::0::1" in payload_mismatch["metrics"]


def test_evaluate_embedding_standard_metrics_family_mismatch():
    """Test evaluate_embedding when specific metrics are requested for wrong family."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(10, 2)
    X = np.random.rand(10, 5)

    # 1. 2D embedding but only trajectory metrics requested -> empty
    payload = evaluate_embedding(X_emb, X=X, metrics=["trajectory_speed"])
    assert payload["metrics"] == {}

    # 2. 3D embedding but only standard metrics requested -> empty
    traj = np.random.rand(2, 5, 2)
    payload_traj = evaluate_embedding(X_emb=traj, metrics=["trustworthiness"])
    assert payload_traj["metrics"] == {}


def test_evaluate_embedding_standard_metrics_guardrails_n_samples():
    """Test standard metrics k-loop guards (lines 261, 274)."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(10, 2)
    X = np.random.rand(10, 5)

    # 1. No requested metrics after filtering
    payload = evaluate_embedding(X_emb, X=X, metrics=["invalid"])
    assert payload["metrics"] == {}

    # 2. k >= n_samples - 1 skipped
    payload_k = evaluate_embedding(
        X_emb, X=X, metrics=["trustworthiness"], k_values=[15]
    )
    assert payload_k["metrics"] == {}
    assert payload_k["records"] == []


def test_evaluate_embedding_trajectory_ndim_guard():
    """Test _evaluate_trajectory_metrics ndim guard."""
    # 2D instead of 3D
    X_emb = np.random.rand(10, 2)
    from coco_pipe.dim_reduction.evaluation.core import _evaluate_trajectory_metrics

    m, meta, d, r = _evaluate_trajectory_metrics("test", X_emb, None)
    assert m == {}
    assert meta == {}


def test_evaluate_embedding_standard_metrics_negative_normalizer():
    """Test standard metrics normalizer guard."""
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(5, 2)
    X = np.random.rand(5, 5)

    payload = evaluate_embedding(X_emb, X=X, metrics=["trustworthiness"], k_values=[4])
    assert "trustworthiness" not in payload["metrics"]


def test_method_selector_rank_methods_missing_observations():
    """Test rank_methods when a metric is missing observations for k."""
    X = np.random.rand(10, 5)
    reducer = DimReduction("PCA")
    reducer.fit_transform(X)
    reducer.score(reducer.transform(X), X=X, metrics=["trustworthiness"], k_values=[5])

    selector = MethodSelector([reducer]).collect()

    with pytest.raises(ValueError, match="is not available"):
        selector.rank_methods(selection_metric="continuity")

    with pytest.raises(ValueError, match="has no observations at k=10"):
        selector.rank_methods(selection_metric="trustworthiness", selection_k=10)


def test_evaluate_embedding_standard_metrics_empty_selection_orchestration():
    X_emb = np.random.rand(10, 2)
    X = np.random.rand(10, 5)

    from coco_pipe.dim_reduction.evaluation.core import _evaluate_standard_metrics

    m, d, r = _evaluate_standard_metrics(
        "test", X, X_emb, {"trajectory_speed"}, 5, None, None
    )
    assert m == {}


def test_evaluate_embedding_standard_metrics_k_lower_bound():
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(10, 2)
    X = np.random.rand(10, 5)
    payload = evaluate_embedding(X_emb, X=X, metrics=["trustworthiness"], k_values=[0])
    assert "trustworthiness" not in payload["metrics"]

    payload_mismatch = evaluate_embedding(X_emb, X=X, metrics=["trajectory_speed"])
    assert payload_mismatch["metrics"] == {}


def test_method_selector_rank_methods_nan_mean():
    X = np.random.rand(10, 5)
    reducer = DimReduction("PCA")
    reducer.fit_transform(X)
    reducer.score(reducer.transform(X), X=X, metrics=["trustworthiness"])

    selector = MethodSelector([reducer]).collect()
    ranked = selector.rank_methods(selection_metric="trustworthiness")
    assert not ranked.empty


def test_evaluate_embedding_standard_metrics_k_boundary_checks():
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(3, 2)
    X = np.random.rand(3, 5)
    payload = evaluate_embedding(X_emb, X=X, metrics=["trustworthiness"], k_values=[2])
    assert "trustworthiness" not in payload["metrics"]


def test_method_selector_collect_unfitted_reducer():
    reducer = DimReduction("PCA")
    with pytest.raises(ValueError, match="has no metric records"):
        MethodSelector([reducer]).collect()

    reducer.embedding_ = np.random.rand(10, 2)
    with pytest.raises(ValueError, match="has no metric records"):
        MethodSelector([reducer]).collect()


def test_evaluate_embedding_ndim_guard_final():
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    X_emb = np.random.rand(10, 2)
    X = np.random.rand(10, 5)
    payload = evaluate_embedding(X_emb, X=X, metrics=["trajectory_speed"])
    assert payload["metrics"] == {}


def test_evaluate_embedding_ndim_guard_branch():
    from coco_pipe.dim_reduction.evaluation.core import evaluate_embedding

    with pytest.raises(ValueError, match="must be either 2D or 3D"):
        evaluate_embedding(np.random.rand(2, 2, 2, 2))


def test_method_selector_collect_not_fitted_guard():
    # DimReduction starts with embedding_ = None
    reducer = DimReduction("PCA")
    with pytest.raises(ValueError, match="has no metric records"):
        MethodSelector([reducer]).collect()


def test_method_selector_rank_methods_missing_records():
    """Trigger ValueError in rank_methods if no metrics are available."""
    selector = MethodSelector([])
    with pytest.raises(ValueError, match="No evaluation metrics available"):
        selector.rank_methods(selection_metric="trustworthiness")
