"""Model-free subject and label variance diagnostics for embeddings."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from coco_pipe.decoding import CVConfig, Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    LogisticRegressionConfig,
    MLPClassifierConfig,
)
from coco_pipe.io import DataContainer


def _arrays(features, subject, label):
    x = np.asarray(features, dtype=np.float64)
    groups = np.asarray(subject)
    y = np.asarray(label)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D, got {x.shape}.")
    if groups.ndim != 1 or y.ndim != 1 or len(x) != len(groups) or len(x) != len(y):
        raise ValueError(
            "features, subject, and label must have matching sample counts."
        )
    if len(x) < 2:
        raise ValueError("Variance decomposition requires at least two observations.")
    if not np.isfinite(x).all():
        raise ValueError("features must contain only finite values.")
    if pd.isna(groups).any() or pd.isna(y).any():
        raise ValueError("subject and label must not contain missing values.")
    return x, groups, y


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    codes, levels = pd.factorize(values, sort=False)
    if np.any(codes < 0):
        raise ValueError("Categorical vectors must not contain missing values.")
    return codes, np.asarray(levels)


def _subject_labels(
    groups: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return subject codes, one label per subject, and whether labels are pure."""
    subject_codes, subjects = _factorize(groups)
    labels = np.empty(len(subjects), dtype=object)
    pure = True
    for code in range(len(subjects)):
        values = pd.unique(y[subject_codes == code])
        labels[code] = values[0]
        pure &= len(values) == 1
    return subject_codes, labels, pure


def _scale_features(
    x: np.ndarray,
    scaling: Literal["none", "zscore"],
) -> tuple[np.ndarray, int]:
    if scaling == "none":
        return x, int(np.sum(np.ptp(x, axis=0) == 0.0))
    if scaling != "zscore":
        raise ValueError("feature_scaling must be 'none' or 'zscore'.")
    std = x.std(axis=0)
    constant = std <= np.finfo(float).eps
    scaled = np.zeros_like(x)
    scaled[:, ~constant] = (x[:, ~constant] - x[:, ~constant].mean(axis=0)) / std[
        ~constant
    ]
    return scaled, int(constant.sum())


def _compute_total_sample_variance_and_participation_ratio(
    x: np.ndarray,
) -> dict[str, float]:
    """Return total sample variance and the covariance participation ratio.

    The covariance eigenvalues are obtained from the singular values of the
    centered feature matrix without materializing the covariance matrix.
    The participation ratio is
    ``sum(eigenvalues) ** 2 / sum(eigenvalues ** 2)`` and its fraction uses
    ``min(n_observations - 1, n_features)`` as the maximum possible rank.
    A numerically zero variance spectrum has participation ratio zero.
    """
    centered_features = x - x.mean(axis=0)
    sample_degrees_of_freedom = len(x) - 1
    singular_values = np.linalg.svd(centered_features, compute_uv=False)
    sample_covariance_eigenvalues = (
        np.square(singular_values) / sample_degrees_of_freedom
    )
    total_sample_variance = float(sample_covariance_eigenvalues.sum())
    sum_squared_eigenvalues = float(np.square(sample_covariance_eigenvalues).sum())
    numerically_zero_variance = sum_squared_eigenvalues <= np.finfo(float).eps
    participation_ratio = (
        0.0
        if numerically_zero_variance
        else total_sample_variance**2 / sum_squared_eigenvalues
    )
    maximum_possible_rank = max(min(len(x) - 1, x.shape[1]), 1)
    return {
        "total_sample_variance": total_sample_variance,
        "variance_participation_ratio": participation_ratio,
        "variance_participation_ratio_fraction": (
            participation_ratio / maximum_possible_rank
        ),
    }


def _pooled_fraction(effect: np.ndarray, total: np.ndarray) -> float:
    denominator = float(np.asarray(total).sum())
    if denominator <= np.finfo(float).eps:
        raise ValueError("Total feature variance is zero; fractions are undefined.")
    return float(np.asarray(effect).sum() / denominator)


def _crossed_ss(
    x: np.ndarray, groups: np.ndarray, y: np.ndarray
) -> dict[str, float | bool]:
    """Compute the marginal SS fractions from already-validated arrays."""
    grand = x.mean(axis=0)
    ss_total = float(np.square(x - grand).sum())

    def between(values: np.ndarray) -> float:
        codes, levels = _factorize(values)
        counts = np.bincount(codes, minlength=len(levels)).astype(float)
        sums = np.zeros((len(levels), x.shape[1]), dtype=float)
        np.add.at(sums, codes, x)
        deltas = sums / counts[:, None] - grand
        return float((counts[:, None] * np.square(deltas)).sum())

    ss_label = between(y)
    ss_subject = between(groups)
    if ss_total <= np.finfo(float).eps:
        raise ValueError("Total feature variance is zero; fractions are undefined.")
    denominator = ss_total
    label_frac = ss_label / denominator
    subject_frac = ss_subject / denominator
    return {
        "SS_total": ss_total,
        "SS_label": ss_label,
        "SS_subject": ss_subject,
        "label_frac": label_frac,
        "subject_frac": subject_frac,
        "raw_sum_exceeds_one": bool(label_frac + subject_frac > 1.0 + 1e-9),
    }


def crossed_ss_fractions(features, subject, label) -> dict[str, float | bool]:
    """Return separate marginal label and subject eta-squared diagnostics.

    These effects may overlap and must not be added or subtracted to obtain a
    residual. Use :func:`variance_decomposition_report` for an additive
    design-aware partition.
    """
    return _crossed_ss(*_arrays(features, subject, label))


def nested_ss(features, subject, label) -> dict[str, np.ndarray | int]:
    """Return the additive subject-within-label ANOVA decomposition."""
    x, groups, y = _arrays(features, subject, label)
    n, d = x.shape
    subject_codes, subject_labels, pure = _subject_labels(groups, y)
    if not pure:
        raise ValueError(
            "Some subjects carry multiple labels; nested decomposition requires "
            "pure-label subjects."
        )

    grand = x.mean(axis=0)
    total = np.square(x - grand).sum(axis=0)
    label_ss = np.zeros(d)
    subject_ss = np.zeros(d)
    subject_mean_per_row = np.zeros_like(x)
    df_subject = 0
    labels = pd.unique(y)
    for value in labels:
        label_rows = y == value
        label_mean = x[label_rows].mean(axis=0)
        label_ss += int(label_rows.sum()) * np.square(label_mean - grand)
        label_subjects = pd.unique(groups[label_rows])
        df_subject += max(len(label_subjects) - 1, 0)
        for sid in label_subjects:
            rows = label_rows & (groups == sid)
            subject_mean = x[rows].mean(axis=0)
            subject_ss += int(rows.sum()) * np.square(subject_mean - label_mean)
            subject_mean_per_row[rows] = subject_mean
    residual = np.square(x - subject_mean_per_row).sum(axis=0)

    subject_means = np.vstack(
        [x[subject_codes == code].mean(axis=0) for code in range(len(subject_labels))]
    )
    subject_grand = subject_means.mean(axis=0)
    subject_level_total = np.square(subject_means - subject_grand).sum(axis=0)
    subject_level_label = np.zeros(d)
    for value in labels:
        rows = subject_labels == value
        label_mean = subject_means[rows].mean(axis=0)
        subject_level_label += int(rows.sum()) * np.square(label_mean - subject_grand)
    subject_level_error = subject_level_total - subject_level_label

    return {
        "label": label_ss,
        "subject_within_label": subject_ss,
        "residual": residual,
        "total": total,
        "subject_level_label": subject_level_label,
        "subject_level_error": subject_level_error,
        "subject_level_total": subject_level_total,
        "df_label": int(len(labels) - 1),
        "df_subject_within_label": int(df_subject),
        "df_residual": int(n - len(pd.unique(groups))),
        "df_subject_level_error": int(len(subject_labels) - len(labels)),
        "n_recordings": int(n),
        "n_subjects": len(subject_labels),
    }


def omega_squared_from_ss(ss: dict[str, Any]) -> dict[str, float]:
    """Compute design-appropriate omega squared for a nested decomposition.

    The label effect is estimated from subject means so recordings from the
    same subject are not treated as independent replicates. Subject-within-label
    is reported as partial omega squared against the recording-level residual.
    """
    df_subject_error = int(ss["df_subject_level_error"])
    if df_subject_error <= 0:
        raise ValueError(
            "At least two subjects per label are required to estimate label omega "
            "squared."
        )
    df_residual = int(ss["df_residual"])
    if df_residual <= 0:
        raise ValueError("df_residual <= 0; not enough observations per subject.")

    label_error_ms = np.asarray(ss["subject_level_error"]) / df_subject_error
    label_numerator = (
        np.asarray(ss["subject_level_label"]) - int(ss["df_label"]) * label_error_ms
    )
    label_denominator = np.asarray(ss["subject_level_total"]) + label_error_ms
    label = float(
        np.clip(label_numerator.sum() / max(label_denominator.sum(), 1e-12), 0.0, 1.0)
    )

    residual = np.asarray(ss["residual"])
    residual_ms = residual / df_residual
    subject_effect = np.asarray(ss["subject_within_label"])
    subject_numerator = (
        subject_effect - int(ss["df_subject_within_label"]) * residual_ms
    )
    subject_denominator = subject_effect + residual + residual_ms
    subject_value = float(
        np.clip(
            subject_numerator.sum() / max(subject_denominator.sum(), 1e-12),
            0.0,
            1.0,
        )
    )
    return {
        "omega2_label_subject_level": label,
        "partial_omega2_subject_within_label": subject_value,
    }


def _dummy_matrix(values: np.ndarray) -> np.ndarray:
    """Return a full indicator-matrix block for one categorical fixed effect.

    Each row is an observation and each column is one factor level. Keeping all
    levels makes the fitted-value projection independent of reference coding;
    ``numpy.linalg.lstsq`` handles the redundant intercept when factor blocks
    are combined.
    """
    codes, levels = _factorize(values)
    matrix = np.zeros((len(values), len(levels)), dtype=float)
    matrix[np.arange(len(values)), codes] = 1.0
    return matrix


def _residual_ss(x: np.ndarray, design: np.ndarray) -> np.ndarray:
    coefficients = np.linalg.lstsq(design, x, rcond=None)[0]
    return np.square(x - design @ coefficients).sum(axis=0)


def _crossed_adjusted_ss(
    x: np.ndarray, groups: np.ndarray, y: np.ndarray
) -> dict[str, np.ndarray | int]:
    """Partition a crossed design into unique, shared, and residual SS.

    Unique effects are extra sums of squares from an additive fixed-effects
    model. The shared term makes confounding/non-orthogonality explicit rather
    than assigning it silently to either factor.
    """
    intercept = np.ones((len(x), 1))
    subject_design = _dummy_matrix(groups)
    label_design = _dummy_matrix(y)
    full_design = np.column_stack((subject_design, label_design))
    total = _residual_ss(x, intercept)
    residual = _residual_ss(x, full_design)
    subject_only_residual = _residual_ss(x, subject_design)
    label_only_residual = _residual_ss(x, label_design)
    unique_label = np.maximum(subject_only_residual - residual, 0.0)
    unique_subject = np.maximum(label_only_residual - residual, 0.0)
    shared = total - unique_label - unique_subject - residual
    tolerance = np.finfo(float).eps * np.maximum(total, 1.0) * len(x) * 10
    shared[np.abs(shared) <= tolerance] = 0.0
    return {
        "unique_label": unique_label,
        "unique_subject": unique_subject,
        "shared": shared,
        "residual": residual,
        "total": total,
        "df_label": int(
            np.linalg.matrix_rank(full_design) - np.linalg.matrix_rank(subject_design)
        ),
        "df_subject": int(
            np.linalg.matrix_rank(full_design) - np.linalg.matrix_rank(label_design)
        ),
        "df_residual": int(len(x) - np.linalg.matrix_rank(full_design)),
    }


def _partial_omega_squared(
    effect: np.ndarray,
    df_effect: int,
    residual: np.ndarray,
    df_residual: int,
) -> float:
    if df_effect <= 0 or df_residual <= 0:
        raise ValueError("Omega squared requires positive effect and residual df.")
    residual_ms = residual / df_residual
    numerator = effect - df_effect * residual_ms
    denominator = effect + residual + residual_ms
    return float(np.clip(numerator.sum() / max(denominator.sum(), 1e-12), 0.0, 1.0))


def null_control(
    features,
    subject,
    label,
    *,
    n_null_permutations: int = 200,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Estimate hierarchy-preserving permutation nulls for marginal effects."""
    if n_null_permutations < 1:
        raise ValueError("n_null_permutations must be at least 1.")
    x, groups, y = _arrays(features, subject, label)
    rng = rng or np.random.default_rng(0)
    real = _crossed_ss(x, groups, y)
    subject_codes, subject_labels, pure_labels = _subject_labels(groups, y)
    label_null = np.empty(n_null_permutations)
    subject_null = np.empty(n_null_permutations)
    for index in range(n_null_permutations):
        if pure_labels:
            permuted_subject_labels = rng.permutation(subject_labels)
            permuted_y = permuted_subject_labels[subject_codes]
        else:
            permuted_y = y.copy()
            for code in range(len(subject_labels)):
                rows = np.flatnonzero(subject_codes == code)
                permuted_y[rows] = rng.permutation(permuted_y[rows])

        permuted_groups = groups.copy()
        for value in pd.unique(y):
            rows = np.flatnonzero(y == value)
            permuted_groups[rows] = rng.permutation(permuted_groups[rows])

        label_null[index] = _crossed_ss(x, groups, permuted_y)["label_frac"]
        subject_null[index] = _crossed_ss(x, permuted_groups, y)["subject_frac"]

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "samples": values.tolist(),
        }

    n = len(x)
    n_subjects = len(pd.unique(groups))
    n_labels = len(pd.unique(y))
    return {
        "real": real,
        "null_label_frac": summary(label_null),
        "null_subject_frac": summary(subject_null),
        "excess_label": float(real["label_frac"] / max(label_null.mean(), 1e-18)),
        "excess_subject": float(real["subject_frac"] / max(subject_null.mean(), 1e-18)),
        "p_label": float(
            (1 + np.count_nonzero(label_null >= real["label_frac"]))
            / (n_null_permutations + 1)
        ),
        "p_subject": float(
            (1 + np.count_nonzero(subject_null >= real["subject_frac"]))
            / (n_null_permutations + 1)
        ),
        "df_label_pred": (n_labels - 1) / max(n - 1, 1),
        "df_subject_pred": (n_subjects - 1) / max(n - 1, 1),
        "method": "hierarchy_preserving_permutation",
        "n_null_permutations": n_null_permutations,
        "n": n,
        "n_subjects": n_subjects,
        "n_labels": n_labels,
    }


def subject_probe(
    features,
    subject,
    *,
    kind: str = "linear",
    cap: int = 100,
    n_splits: int = 5,
    seed: int = 42,
    blocks=None,
) -> tuple[float | None, int]:
    """Cross-validate subject identity, optionally holding out whole blocks."""
    x = np.asarray(features, dtype=np.float64)
    groups = np.asarray(subject)
    if x.ndim != 2 or groups.ndim != 1 or len(x) != len(groups):
        raise ValueError("features and subject must have matching sample counts.")
    if not np.isfinite(x).all():
        raise ValueError("features must contain only finite values.")
    if pd.isna(groups).any():
        raise ValueError("subject must not contain missing values.")
    if cap < 1:
        raise ValueError("cap must be at least 1.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    block_values = None if blocks is None else np.asarray(blocks)
    if block_values is not None and (
        block_values.ndim != 1 or len(block_values) != len(x)
    ):
        raise ValueError("blocks must be one-dimensional and align with features.")
    if block_values is not None and pd.isna(block_values).any():
        raise ValueError("blocks must not contain missing values.")
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for sid in pd.unique(groups):
        indices = np.flatnonzero(groups == sid)
        if len(indices) > cap:
            indices = rng.choice(indices, cap, replace=False)
        keep.extend(indices.tolist())
    xs = x[keep]
    encoded = groups[keep]
    selected_blocks = block_values[keep] if block_values is not None else None
    if selected_blocks is None:
        counts = {sid: int((encoded == sid).sum()) for sid in pd.unique(encoded)}
    else:
        counts = {
            sid: len(pd.unique(selected_blocks[encoded == sid]))
            for sid in pd.unique(encoded)
        }
    mask = np.asarray([counts[sid] >= n_splits for sid in encoded])
    xs, encoded = xs[mask], encoded[mask]
    if selected_blocks is not None:
        selected_blocks = selected_blocks[mask]
    n_subjects = len(pd.unique(encoded))
    if n_subjects < 2:
        return None, n_subjects
    encoded = _factorize(encoded)[0]
    if kind == "linear":
        estimator_config = LogisticRegressionConfig(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    elif kind == "mlp":
        estimator_config = MLPClassifierConfig(
            hidden_layer_sizes=(64,),
            max_iter=200,
            early_stopping=True,
            random_state=seed,
        )
    else:
        raise ValueError(f"kind must be 'linear' or 'mlp', got {kind!r}.")

    cv_strategy = "stratified" if selected_blocks is None else "stratified_group_kfold"
    config = ExperimentConfig(
        task="classification",
        models={"subject_probe": estimator_config},
        metrics=["balanced_accuracy"],
        cv=CVConfig(
            strategy=cv_strategy,
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        ),
        use_scaler=True,
        n_jobs=1,
        verbose=False,
    )
    cv_groups = None
    if selected_blocks is not None:
        cv_groups = pd.MultiIndex.from_arrays(
            [encoded, selected_blocks], names=("subject", "block")
        ).factorize(sort=False)[0]
    if cv_groups is None:
        result = Experiment(config).run(xs, encoded)
    else:
        result = Experiment(config).run(xs, encoded, groups=cv_groups)
    probe_result = result.raw["subject_probe"]
    if "error" in probe_result:
        raise RuntimeError(f"Subject probe experiment failed: {probe_result['error']}")
    mean_ba = probe_result["metrics"]["balanced_accuracy"]["mean"]
    return float(mean_ba), n_subjects


def _container_vector(
    container: DataContainer,
    value: str | Any | None,
    *,
    role: str,
) -> np.ndarray:
    if isinstance(value, str):
        observation_metadata = container.obs_table()
        if value not in observation_metadata:
            raise KeyError(f"{role} coordinate not found: {value}")
        return observation_metadata[value].to_numpy()
    if value is not None:
        return np.asarray(value)
    if role == "label" and container.y is not None:
        return np.asarray(container.y)
    raise ValueError(
        f"{role} must be supplied as an array or DataContainer coordinate name."
    )


def variance_decomposition_report(
    features,
    subject=None,
    label=None,
    *,
    feature_scaling: Literal["none", "zscore"] = "zscore",
    n_null_permutations: int = 200,
    rng: np.random.Generator | None = None,
    probe_blocks=None,
    probe_cap: int = 100,
    probe_n_splits: int = 5,
    probe_seed: int = 42,
) -> pd.DataFrame:
    """Return design-aware subject and label diagnostics for an embedding.

    Pure-label subjects use an additive subject-within-label decomposition.
    When labels vary within subject, an adjusted additive fixed-effects model
    reports unique label, unique subject, shared/confounded, and residual SS.

    Parameters
    ----------
    features : array-like or DataContainer
        Embedding observations. Containers are flattened with ``obs`` as rows.
    subject, label : array-like or str, optional
        Observation-aligned vectors, or explicit coordinate names for a
        ``DataContainer``. ``container.y`` is the only implicit label source.
    feature_scaling : {"none", "zscore"}, default="zscore"
        Z-scoring makes the pooled SS partition invariant to feature units.
        Constant features are retained as zeros and counted in the output.
    n_null_permutations : int, default=200
        Number of hierarchy-preserving permutations used for marginal nulls.
    rng : numpy.random.Generator, optional
        Permutation generator. A deterministic generator is used by default.
    probe_blocks : array-like or str, optional
        Acquisition/session block identifier. When supplied, subject-probe CV
        holds out complete subject-block combinations.
    probe_cap, probe_n_splits, probe_seed : int
        Subject-probe sampling and cross-validation settings.

    Returns
    -------
    pandas.DataFrame
        One row per metric with design, scaling, null, and probe metadata.
        Interpretive caveats are stored in ``report.attrs["notes"]``.
    """
    if isinstance(features, DataContainer):
        container = features.flatten(preserve="obs")
        features = container.X
        subject = _container_vector(container, subject, role="subject")
        label = _container_vector(container, label, role="label")
        if isinstance(probe_blocks, str):
            probe_blocks = _container_vector(
                container, probe_blocks, role="probe_blocks"
            )
    elif isinstance(probe_blocks, str):
        raise ValueError(
            "probe_blocks may be a coordinate name only when features is a "
            "DataContainer."
        )

    x_raw, groups, y = _arrays(features, subject, label)
    x, n_constant_features = _scale_features(x_raw, feature_scaling)
    control = null_control(
        x,
        groups,
        y,
        n_null_permutations=n_null_permutations,
        rng=rng,
    )
    real = control["real"]
    values = {
        **_compute_total_sample_variance_and_participation_ratio(x_raw),
        "marginal_label_eta2": real["label_frac"],
        "between_subject_eta2": real["subject_frac"],
        "null_marginal_label_eta2": control["null_label_frac"]["mean"],
        "null_marginal_label_eta2_std": control["null_label_frac"]["std"],
        "null_between_subject_eta2": control["null_subject_frac"]["mean"],
        "null_between_subject_eta2_std": control["null_subject_frac"]["std"],
        "marginal_label_excess_over_null": control["excess_label"],
        "between_subject_excess_over_null": control["excess_subject"],
        "permutation_p_marginal_label_eta2": control["p_label"],
        "permutation_p_between_subject_eta2": control["p_subject"],
    }

    _, _, pure_labels = _subject_labels(groups, y)
    omega_errors: list[str] = []
    if pure_labels:
        design = "nested_subject_within_label"
        ss = nested_ss(x, groups, y)
        values.update(
            {
                "label_fraction": _pooled_fraction(ss["label"], ss["total"]),
                "subject_within_label_fraction": _pooled_fraction(
                    ss["subject_within_label"], ss["total"]
                ),
                "residual_fraction": _pooled_fraction(ss["residual"], ss["total"]),
            }
        )
        try:
            values.update(omega_squared_from_ss(ss))
        except ValueError as exc:
            omega_errors.append(str(exc))
    else:
        design = "crossed_adjusted_fixed_effects"
        ss = _crossed_adjusted_ss(x, groups, y)
        values.update(
            {
                "unique_label_fraction": _pooled_fraction(
                    ss["unique_label"], ss["total"]
                ),
                "unique_subject_fraction": _pooled_fraction(
                    ss["unique_subject"], ss["total"]
                ),
                "shared_or_confounding_fraction": _pooled_fraction(
                    ss["shared"], ss["total"]
                ),
                "residual_fraction": _pooled_fraction(ss["residual"], ss["total"]),
            }
        )
        for name, effect, df_key in (
            ("partial_omega2_label", "unique_label", "df_label"),
            ("partial_omega2_subject", "unique_subject", "df_subject"),
        ):
            try:
                values[name] = _partial_omega_squared(
                    np.asarray(ss[effect]),
                    int(ss[df_key]),
                    np.asarray(ss["residual"]),
                    int(ss["df_residual"]),
                )
            except ValueError as exc:
                omega_errors.append(f"{name}: {exc}")

    probe_score, probe_subjects = subject_probe(
        x_raw,
        groups,
        kind="linear",
        cap=probe_cap,
        n_splits=probe_n_splits,
        seed=probe_seed,
        blocks=probe_blocks,
    )
    if probe_score is not None:
        values["subject_probe_linear_balanced_accuracy"] = probe_score
        values["subject_probe_chance"] = 1.0 / probe_subjects
    common = {
        "n_observations": len(x),
        "n_features": x.shape[1],
        "n_constant_features": n_constant_features,
        "n_subjects": len(pd.unique(groups)),
        "n_labels": len(pd.unique(y)),
        "design": design,
        "feature_scaling": feature_scaling,
        "representation_variance_normalization": "sample_covariance_ddof_1",
        "representation_rank_metric": "participation_ratio",
        "representation_maximum_rank": "min(n_observations - 1, n_features)",
        "null_method": control["method"],
        "n_null_permutations": n_null_permutations,
        "probe_split_unit": "observation" if probe_blocks is None else "block",
    }
    rows = [
        {**common, "metric": metric, "value": float(value), "status": "ok"}
        for metric, value in values.items()
    ]
    if probe_score is None:
        rows.append(
            {
                **common,
                "metric": "subject_probe_linear_balanced_accuracy",
                "value": None,
                "status": "not_applicable",
                "reason": "Fewer than two subjects met the probe CV requirements.",
            }
        )
    for reason in omega_errors:
        rows.append(
            {
                **common,
                "metric": "omega_squared",
                "value": None,
                "status": "not_applicable",
                "reason": reason,
            }
        )
    report = pd.DataFrame.from_records(rows)
    report.attrs["notes"] = [
        "Fractions are descriptive and depend on the selected feature scaling.",
        "Permutation p-values are Monte Carlo diagnostics, not multiplicity-corrected "
        "confirmatory tests.",
        (
            "Observation-level probe CV can retain session/temporal dependence; pass "
            "probe_blocks to hold out whole acquisition blocks."
            if probe_blocks is None
            else "Subject-probe CV holds out complete acquisition blocks."
        ),
    ]
    return report
