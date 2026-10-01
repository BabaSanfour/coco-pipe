"""Exact variance diagnostics over re-iterable feature batches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd

from .variance import (
    _factorize,
    _partial_omega_squared,
    _pooled_fraction,
    _subject_labels,
    omega_squared_from_ss,
)

FeatureBatch: TypeAlias = tuple[np.ndarray, np.ndarray]
FeatureBatchSource: TypeAlias = (
    Iterable[FeatureBatch] | Callable[[], Iterable[FeatureBatch]]
)

DEFAULT_PARTICIPATION_RATIO_MAX_GRAM_BYTES = 4 * 1024**3


def _batches(source: FeatureBatchSource) -> Iterator[FeatureBatch]:
    iterable = source() if callable(source) else source
    return iter(iterable)


def _validated_batches(
    source: FeatureBatchSource,
    *,
    n_observations: int,
    n_features: int | None,
    require_coverage: bool,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    assigned = np.zeros(n_observations, dtype=bool) if require_coverage else None
    observed_features = n_features
    saw_batch = False
    for row_indices, features in _batches(source):
        rows = np.asarray(row_indices)
        values = np.asarray(features)
        saw_batch = True
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError(
                "Batch row_indices must be a one-dimensional integer array."
            )
        rows = rows.astype(np.intp, copy=False)
        if values.ndim != 2 or len(values) != len(rows):
            raise ValueError(
                "Each feature batch must be two-dimensional and align with row_indices."
            )
        if observed_features is None:
            observed_features = int(values.shape[1])
        if values.shape[1] != observed_features:
            raise ValueError(
                "Feature batches have inconsistent dimensions: expected "
                f"{observed_features}, got {values.shape[1]}."
            )
        if np.any(rows < 0) or np.any(rows >= n_observations):
            raise ValueError("Batch row_indices fall outside the observation range.")
        if not np.isfinite(values).all():
            raise ValueError("Feature batches must contain only finite values.")
        if assigned is not None:
            if len(np.unique(rows)) != len(rows) or assigned[rows].any():
                raise ValueError("Feature batches contain duplicate row assignments.")
            assigned[rows] = True
        yield rows, values
    if not saw_batch:
        raise ValueError("Feature batch source yielded no batches.")
    if assigned is not None and not assigned.all():
        missing = int((~assigned).sum())
        raise ValueError(
            "Feature batches do not exactly cover the observations: "
            f"{missing} row(s) are missing."
        )


def _metadata_arrays(subject, label) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(subject)
    y = np.asarray(label)
    if groups.ndim != 1 or y.ndim != 1 or len(groups) != len(y):
        raise ValueError("subject and label must be matching one-dimensional arrays.")
    if len(groups) < 2:
        raise ValueError("Variance decomposition requires at least two observations.")
    if pd.isna(groups).any() or pd.isna(y).any():
        raise ValueError("subject and label must not contain missing values.")
    return groups, y


def _initial_statistics(
    source: FeatureBatchSource,
    groups: np.ndarray,
    y: np.ndarray,
    *,
    participation_ratio_max_gram_bytes: int,
) -> dict[str, Any]:
    subject_codes, subjects = _factorize(groups)
    label_codes, labels = _factorize(y)
    n = len(groups)
    result: dict[str, Any] | None = None
    for rows, batch in _validated_batches(
        source,
        n_observations=n,
        n_features=None,
        require_coverage=True,
    ):
        values = np.asarray(batch, dtype=np.float64)
        if result is None:
            d = int(values.shape[1])
            gram_bytes = d * d * np.dtype(np.float64).itemsize
            keep_gram = gram_bytes <= participation_ratio_max_gram_bytes
            result = {
                "n_features": d,
                "sum": np.zeros(d),
                "sum_squares": np.zeros(d),
                "minimum": np.full(d, np.inf),
                "maximum": np.full(d, -np.inf),
                "subject_sums": np.zeros((len(subjects), d)),
                "label_sums": np.zeros((len(labels), d)),
                "gram": True if keep_gram else None,
                "gram_bytes": gram_bytes,
            }
        result["sum"] += values.sum(axis=0)
        result["sum_squares"] += np.einsum("ij,ij->j", values, values)
        result["minimum"] = np.minimum(result["minimum"], values.min(axis=0))
        result["maximum"] = np.maximum(result["maximum"], values.max(axis=0))
        np.add.at(result["subject_sums"], subject_codes[rows], values)
        np.add.at(result["label_sums"], label_codes[rows], values)
    if result is None:  # pragma: no cover - enforced by _validated_batches
        raise RuntimeError("Feature statistics were not initialized.")
    result.update(
        {
            "subject_codes": subject_codes,
            "subjects": subjects,
            "subject_counts": np.bincount(
                subject_codes, minlength=len(subjects)
            ).astype(float),
            "label_codes": label_codes,
            "labels": labels,
            "label_counts": np.bincount(label_codes, minlength=len(labels)).astype(
                float
            ),
        }
    )
    return result


def _accumulate_centered_gram(
    source: FeatureBatchSource,
    *,
    mean: np.ndarray,
    n_observations: int,
    n_features: int,
) -> np.ndarray:
    gram = np.zeros((n_features, n_features), dtype=np.float64)
    for _, batch in _validated_batches(
        source,
        n_observations=n_observations,
        n_features=n_features,
        require_coverage=False,
    ):
        centered = np.asarray(batch, dtype=np.float64) - mean
        gram += centered.T @ centered
    return gram


def _scaled_statistics(
    raw: dict[str, Any],
    *,
    n_observations: int,
    feature_scaling: Literal["none", "zscore"],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if feature_scaling not in {"none", "zscore"}:
        raise ValueError("feature_scaling must be 'none' or 'zscore'.")
    raw_sum = np.asarray(raw["sum"])
    raw_sum_squares = np.asarray(raw["sum_squares"])
    if feature_scaling == "none":
        center = np.zeros_like(raw_sum)
        scale = np.ones_like(raw_sum)
        constant = np.asarray(raw["maximum"]) - np.asarray(raw["minimum"]) == 0.0
    else:
        center = raw_sum / n_observations
        centered_ss = np.maximum(
            raw_sum_squares - np.square(raw_sum) / n_observations, 0.0
        )
        std = np.sqrt(centered_ss / n_observations)
        constant = std <= np.finfo(float).eps
        scale = std.copy()
        scale[constant] = 1.0

    def transform_sums(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
        transformed = (sums - counts[:, None] * center) / scale
        transformed[:, constant] = 0.0
        return transformed

    total_sum = (raw_sum - n_observations * center) / scale
    total_sum[constant] = 0.0
    total_sum_squares = (
        raw_sum_squares - 2 * center * raw_sum + n_observations * np.square(center)
    ) / np.square(scale)
    total_sum_squares[constant] = 0.0
    scaled = {
        **raw,
        "sum": total_sum,
        "sum_squares": np.maximum(total_sum_squares, 0.0),
        "subject_sums": transform_sums(
            np.asarray(raw["subject_sums"]), np.asarray(raw["subject_counts"])
        ),
        "label_sums": transform_sums(
            np.asarray(raw["label_sums"]), np.asarray(raw["label_counts"])
        ),
    }
    return scaled, constant, center, scale


def _between_ss(
    sums: np.ndarray, counts: np.ndarray, grand_term: np.ndarray
) -> np.ndarray:
    return (np.square(sums) / counts[:, None]).sum(axis=0) - grand_term


def _crossed_from_statistics(stats: dict[str, Any], n: int) -> dict[str, float | bool]:
    grand_term = np.square(stats["sum"]) / n
    total = np.asarray(stats["sum_squares"]) - grand_term
    label = _between_ss(stats["label_sums"], stats["label_counts"], grand_term)
    subject = _between_ss(stats["subject_sums"], stats["subject_counts"], grand_term)
    denominator = float(total.sum())
    if denominator <= np.finfo(float).eps:
        raise ValueError("Total feature variance is zero; fractions are undefined.")
    label_fraction = float(label.sum() / denominator)
    subject_fraction = float(subject.sum() / denominator)
    return {
        "SS_total": denominator,
        "SS_label": float(label.sum()),
        "SS_subject": float(subject.sum()),
        "label_frac": label_fraction,
        "subject_frac": subject_fraction,
        "raw_sum_exceeds_one": bool(label_fraction + subject_fraction > 1.0 + 1e-9),
    }


def _nested_from_statistics(
    stats: dict[str, Any], groups: np.ndarray, y: np.ndarray
) -> dict[str, np.ndarray | int]:
    n = len(groups)
    subject_counts = stats["subject_counts"]
    label_counts = stats["label_counts"]
    subject_sums = np.asarray(stats["subject_sums"])
    label_sums = np.asarray(stats["label_sums"])
    grand_term = np.square(stats["sum"]) / n
    total = np.asarray(stats["sum_squares"]) - grand_term
    label_ss = _between_ss(label_sums, label_counts, grand_term)
    subject_between = _between_ss(subject_sums, subject_counts, grand_term)
    subject_within_label = subject_between - label_ss
    residual = np.asarray(stats["sum_squares"]) - (
        np.square(subject_sums) / subject_counts[:, None]
    ).sum(axis=0)

    _, subject_labels, pure = _subject_labels(groups, y)
    if not pure:
        raise ValueError("Nested statistics require pure-label subjects.")
    subject_means = subject_sums / subject_counts[:, None]
    subject_grand = subject_means.mean(axis=0)
    subject_level_total = np.square(subject_means - subject_grand).sum(axis=0)
    subject_label_codes, subject_label_levels = _factorize(subject_labels)
    subject_label_counts = np.bincount(
        subject_label_codes, minlength=len(subject_label_levels)
    ).astype(float)
    subject_label_sums = np.zeros((len(subject_label_levels), stats["n_features"]))
    np.add.at(subject_label_sums, subject_label_codes, subject_means)
    subject_level_label = (
        np.square(subject_label_sums) / subject_label_counts[:, None]
    ).sum(axis=0) - len(subject_means) * np.square(subject_grand)
    subject_level_error = subject_level_total - subject_level_label
    df_subject = sum(
        max(int(np.sum(subject_label_codes == code)) - 1, 0)
        for code in range(len(subject_label_levels))
    )
    return {
        "label": label_ss,
        "subject_within_label": subject_within_label,
        "residual": residual,
        "total": total,
        "subject_level_label": subject_level_label,
        "subject_level_error": subject_level_error,
        "subject_level_total": subject_level_total,
        "df_label": int(len(subject_label_levels) - 1),
        "df_subject_within_label": int(df_subject),
        "df_residual": int(n - len(subject_counts)),
        "df_subject_level_error": int(len(subject_counts) - len(subject_label_levels)),
        "n_recordings": n,
        "n_subjects": len(subject_counts),
    }


def _crossed_adjusted_from_statistics(
    stats: dict[str, Any], groups: np.ndarray, y: np.ndarray
) -> dict[str, np.ndarray | int]:
    subject_codes = stats["subject_codes"]
    label_codes = stats["label_codes"]
    subject_counts = np.asarray(stats["subject_counts"])
    label_counts = np.asarray(stats["label_counts"])
    n_subjects = len(subject_counts)
    n_labels = len(label_counts)
    cross_counts = np.zeros((n_subjects, n_labels))
    np.add.at(cross_counts, (subject_codes, label_codes), 1.0)
    normal = np.block(
        [
            [np.diag(subject_counts), cross_counts],
            [cross_counts.T, np.diag(label_counts)],
        ]
    )
    rhs = np.vstack((stats["subject_sums"], stats["label_sums"]))
    coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    full_residual = np.asarray(stats["sum_squares"]) - np.sum(
        rhs * coefficients, axis=0
    )
    subject_residual = np.asarray(stats["sum_squares"]) - (
        np.square(stats["subject_sums"]) / subject_counts[:, None]
    ).sum(axis=0)
    label_residual = np.asarray(stats["sum_squares"]) - (
        np.square(stats["label_sums"]) / label_counts[:, None]
    ).sum(axis=0)
    total = np.asarray(stats["sum_squares"]) - np.square(stats["sum"]) / len(groups)
    residual = np.maximum(full_residual, 0.0)
    unique_label = np.maximum(subject_residual - residual, 0.0)
    unique_subject = np.maximum(label_residual - residual, 0.0)
    shared = total - unique_label - unique_subject - residual
    tolerance = np.finfo(float).eps * np.maximum(total, 1.0) * len(groups) * 10
    shared[np.abs(shared) <= tolerance] = 0.0
    full_rank = int(np.linalg.matrix_rank(normal))
    return {
        "unique_label": unique_label,
        "unique_subject": unique_subject,
        "shared": shared,
        "residual": residual,
        "total": total,
        "df_label": full_rank - n_subjects,
        "df_subject": full_rank - n_labels,
        "df_residual": len(groups) - full_rank,
    }


def _permuted_between_fractions(
    source: FeatureBatchSource,
    label_codes: np.ndarray,
    label_counts: np.ndarray,
    subject_codes: np.ndarray,
    subject_counts: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    constant: np.ndarray,
    total_ss: float,
    n_features: int,
) -> tuple[float, float]:
    label_sums = np.zeros((len(label_counts), n_features))
    subject_sums = np.zeros((len(subject_counts), n_features))
    for rows, batch in _validated_batches(
        source,
        n_observations=len(label_codes),
        n_features=n_features,
        require_coverage=False,
    ):
        values = (np.asarray(batch, dtype=np.float64) - center) / scale
        values[:, constant] = 0.0
        np.add.at(label_sums, label_codes[rows], values)
        np.add.at(subject_sums, subject_codes[rows], values)
    grand = label_sums.sum(axis=0)
    grand_term = np.square(grand) / len(label_codes)
    label_between = (np.square(label_sums) / label_counts[:, None]).sum(
        axis=0
    ) - grand_term
    subject_between = (np.square(subject_sums) / subject_counts[:, None]).sum(
        axis=0
    ) - grand_term
    return (
        float(label_between.sum() / total_ss),
        float(subject_between.sum() / total_ss),
    )


def _null_control_streamed(
    source: FeatureBatchSource,
    stats: dict[str, Any],
    groups: np.ndarray,
    y: np.ndarray,
    *,
    center: np.ndarray,
    scale: np.ndarray,
    constant: np.ndarray,
    n_null_permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if n_null_permutations < 1:
        raise ValueError("n_null_permutations must be at least 1.")
    real = _crossed_from_statistics(stats, len(groups))
    subject_codes, subject_labels, pure_labels = _subject_labels(groups, y)
    label_null = np.empty(n_null_permutations)
    subject_null = np.empty(n_null_permutations)
    total_ss = float(real["SS_total"])
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

        permuted_label_codes, permuted_labels = _factorize(permuted_y)
        permuted_subject_codes, permuted_subjects = _factorize(permuted_groups)
        label_null[index], subject_null[index] = _permuted_between_fractions(
            source,
            permuted_label_codes,
            np.bincount(permuted_label_codes, minlength=len(permuted_labels)).astype(
                float
            ),
            permuted_subject_codes,
            np.bincount(
                permuted_subject_codes, minlength=len(permuted_subjects)
            ).astype(float),
            center=center,
            scale=scale,
            constant=constant,
            total_ss=total_ss,
            n_features=stats["n_features"],
        )

    def summary(values: np.ndarray) -> dict[str, Any]:
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "samples": values.tolist(),
        }

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
        "method": "hierarchy_preserving_permutation",
    }


def streamed_subject_probe(
    feature_batches: FeatureBatchSource,
    subject,
    *,
    cap: int = 100,
    n_splits: int = 5,
    seed: int = 42,
    blocks=None,
    epochs: int = 10,
    learning_rate: float = 0.01,
    weight_decay: float = 1e-4,
    _n_features: int | None = None,
    _validated: bool = False,
) -> tuple[float | None, int]:
    """Fit a deterministic CPU PyTorch multinomial probe over streamed batches."""
    groups = np.asarray(subject)
    if groups.ndim != 1 or pd.isna(groups).any():
        raise ValueError(
            "subject must be a one-dimensional array without missing values."
        )
    if cap < 1 or n_splits < 2 or epochs < 1:
        raise ValueError("cap and epochs must be positive and n_splits at least 2.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError(
            "learning_rate must be positive and weight_decay non-negative."
        )
    block_values = None if blocks is None else np.asarray(blocks)
    if block_values is not None and (
        block_values.ndim != 1
        or len(block_values) != len(groups)
        or pd.isna(block_values).any()
    ):
        raise ValueError(
            "blocks must align with subject and contain no missing values."
        )

    n_features = _n_features
    if not _validated:
        for _, batch in _validated_batches(
            feature_batches,
            n_observations=len(groups),
            n_features=n_features,
            require_coverage=True,
        ):
            if n_features is None:
                n_features = int(batch.shape[1])
    if n_features is None:
        raise ValueError("Could not determine the streamed feature dimension.")

    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for sid in pd.unique(groups):
        indices = np.flatnonzero(groups == sid)
        if len(indices) > cap:
            indices = rng.choice(indices, cap, replace=False)
        keep.extend(indices.tolist())
    selected = np.asarray(keep, dtype=int)
    selected_subjects = groups[selected]
    selected_blocks = block_values[selected] if block_values is not None else None
    if selected_blocks is None:
        counts = {
            sid: int(np.sum(selected_subjects == sid))
            for sid in pd.unique(selected_subjects)
        }
    else:
        counts = {
            sid: len(pd.unique(selected_blocks[selected_subjects == sid]))
            for sid in pd.unique(selected_subjects)
        }
    eligible = np.asarray([counts[sid] >= n_splits for sid in selected_subjects])
    selected = selected[eligible]
    selected_subjects = selected_subjects[eligible]
    if selected_blocks is not None:
        selected_blocks = selected_blocks[eligible]
    encoded, levels = _factorize(selected_subjects)
    n_subjects = len(levels)
    if n_subjects < 2:
        return None, n_subjects

    try:
        import torch
        from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    except ImportError as exc:  # torch is deliberately lazy/optional
        if exc.name == "torch":
            raise ImportError(
                "Streamed subject probes require PyTorch; install coco-pipe[torch]."
            ) from exc
        raise

    if selected_blocks is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(selected, encoded)
    else:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
        cv_groups = pd.MultiIndex.from_arrays(
            [encoded, selected_blocks], names=("subject", "block")
        ).factorize(sort=False)[0]
        splits = splitter.split(selected, encoded, cv_groups)

    selected_position = np.full(len(groups), -1, dtype=int)
    selected_position[selected] = np.arange(len(selected))
    fold_scores: list[float] = []
    for fold_index, (train_positions, test_positions) in enumerate(splits):
        train_rows = selected[train_positions]
        test_rows = selected[test_positions]
        train_mask = np.zeros(len(groups), dtype=bool)
        test_mask = np.zeros(len(groups), dtype=bool)
        train_mask[train_rows] = True
        test_mask[test_rows] = True

        train_sum = np.zeros(n_features)
        train_sum_squares = np.zeros(n_features)
        train_count = 0
        for rows, batch in _validated_batches(
            feature_batches,
            n_observations=len(groups),
            n_features=n_features,
            require_coverage=False,
        ):
            use = train_mask[rows]
            if not use.any():
                continue
            values = np.asarray(batch[use], dtype=np.float64)
            train_sum += values.sum(axis=0)
            train_sum_squares += np.square(values).sum(axis=0)
            train_count += len(values)
        mean = train_sum / train_count
        variance = np.maximum(train_sum_squares / train_count - np.square(mean), 0.0)
        scale = np.sqrt(variance)
        scale[scale <= np.finfo(float).eps] = 1.0
        mean32 = mean.astype(np.float32)
        scale32 = scale.astype(np.float32)

        torch.manual_seed(seed + fold_index)
        model = torch.nn.Linear(
            n_features, n_subjects, device="cpu", dtype=torch.float32
        )
        torch.nn.init.zeros_(model.weight)
        torch.nn.init.zeros_(model.bias)
        class_counts = np.bincount(encoded[train_positions], minlength=n_subjects)
        class_weights = train_count / (n_subjects * class_counts)
        criterion = torch.nn.CrossEntropyLoss(
            weight=torch.as_tensor(class_weights, dtype=torch.float32),
            reduction="sum",
        )
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            for rows, batch in _validated_batches(
                feature_batches,
                n_observations=len(groups),
                n_features=n_features,
                require_coverage=False,
            ):
                use = train_mask[rows]
                if not use.any():
                    continue
                global_rows = rows[use]
                x = (np.asarray(batch[use], dtype=np.float32) - mean32) / scale32
                positions = selected_position[global_rows]
                targets = encoded[positions]
                logits = model(torch.from_numpy(np.ascontiguousarray(x)))
                loss = (
                    criterion(logits, torch.as_tensor(targets, dtype=torch.long))
                    / train_count
                )
                loss.backward()
            optimizer.step()

        correct = np.zeros(n_subjects, dtype=int)
        totals = np.zeros(n_subjects, dtype=int)
        model.eval()
        with torch.no_grad():
            for rows, batch in _validated_batches(
                feature_batches,
                n_observations=len(groups),
                n_features=n_features,
                require_coverage=False,
            ):
                use = test_mask[rows]
                if not use.any():
                    continue
                global_rows = rows[use]
                x = (np.asarray(batch[use], dtype=np.float32) - mean32) / scale32
                positions = selected_position[global_rows]
                targets = encoded[positions]
                predictions = (
                    model(torch.from_numpy(np.ascontiguousarray(x))).argmax(1).numpy()
                )
                np.add.at(totals, targets, 1)
                np.add.at(correct, targets, predictions == targets)
        present = totals > 0
        fold_scores.append(float(np.mean(correct[present] / totals[present])))
    return float(np.mean(fold_scores)), n_subjects


def streamed_variance_decomposition_report(
    feature_batches: FeatureBatchSource,
    subject,
    label,
    *,
    feature_scaling: Literal["none", "zscore"] = "zscore",
    n_null_permutations: int = 200,
    rng: np.random.Generator | None = None,
    probe_blocks=None,
    probe_cap: int = 100,
    probe_n_splits: int = 5,
    probe_seed: int = 42,
    probe_epochs: int = 10,
    probe_learning_rate: float = 0.01,
    probe_weight_decay: float = 1e-4,
    participation_ratio_max_gram_bytes: int = (
        DEFAULT_PARTICIPATION_RATIO_MAX_GRAM_BYTES
    ),
) -> pd.DataFrame:
    """Return the dense diagnostic schema without assembling a feature matrix.

    ``feature_batches`` must be re-iterable (or a callable returning an iterable)
    and yield ``(row_indices, features)`` pairs. Row indices refer to the supplied
    subject/label vectors and must cover them exactly once per complete pass.
    """
    groups, y = _metadata_arrays(subject, label)
    if participation_ratio_max_gram_bytes < 0:
        raise ValueError("participation_ratio_max_gram_bytes must be non-negative.")
    raw = _initial_statistics(
        feature_batches,
        groups,
        y,
        participation_ratio_max_gram_bytes=participation_ratio_max_gram_bytes,
    )
    if raw["gram"] is not None:
        raw["gram"] = _accumulate_centered_gram(
            feature_batches,
            mean=np.asarray(raw["sum"]) / len(groups),
            n_observations=len(groups),
            n_features=raw["n_features"],
        )
    scaled, constant, center, scale = _scaled_statistics(
        raw, n_observations=len(groups), feature_scaling=feature_scaling
    )
    control = _null_control_streamed(
        feature_batches,
        scaled,
        groups,
        y,
        center=center,
        scale=scale,
        constant=constant,
        n_null_permutations=n_null_permutations,
        rng=rng or np.random.default_rng(0),
    )
    real = control["real"]
    raw_centered_ss = np.maximum(
        np.asarray(raw["sum_squares"]) - np.square(raw["sum"]) / len(groups), 0.0
    )
    if raw["gram"] is None:
        total_sample_variance = float(raw_centered_ss.sum() / (len(groups) - 1))
    else:
        total_sample_variance = float(
            np.trace(np.asarray(raw["gram"])) / (len(groups) - 1)
        )
    values = {
        "total_sample_variance": total_sample_variance,
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
    pr_reason = None
    if raw["gram"] is None:
        pr_reason = (
            "Exact participation ratio was skipped because its float64 feature Gram "
            f"matrix requires {raw['gram_bytes']} bytes, exceeding the configured "
            f"limit of {participation_ratio_max_gram_bytes} bytes."
        )
    else:
        centered_gram = np.asarray(raw["gram"])
        trace = float(np.trace(centered_gram))
        trace_squared = float(np.square(centered_gram).sum())
        covariance_trace_squared = trace_squared / (len(groups) - 1) ** 2
        participation_ratio = (
            0.0
            if covariance_trace_squared <= np.finfo(float).eps
            else trace**2 / trace_squared
        )
        maximum_rank = max(min(len(groups) - 1, raw["n_features"]), 1)
        values["variance_participation_ratio"] = participation_ratio
        values["variance_participation_ratio_fraction"] = (
            participation_ratio / maximum_rank
        )

    _, _, pure_labels = _subject_labels(groups, y)
    omega_errors: list[str] = []
    if pure_labels:
        design = "nested_subject_within_label"
        ss = _nested_from_statistics(scaled, groups, y)
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
        ss = _crossed_adjusted_from_statistics(scaled, groups, y)
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

    probe_score, probe_subjects = streamed_subject_probe(
        feature_batches,
        groups,
        cap=probe_cap,
        n_splits=probe_n_splits,
        seed=probe_seed,
        blocks=probe_blocks,
        epochs=probe_epochs,
        learning_rate=probe_learning_rate,
        weight_decay=probe_weight_decay,
        _n_features=raw["n_features"],
        _validated=True,
    )
    if probe_score is not None:
        values["subject_probe_linear_balanced_accuracy"] = probe_score
        values["subject_probe_chance"] = 1.0 / probe_subjects
    common = {
        "n_observations": len(groups),
        "n_features": raw["n_features"],
        "n_constant_features": int(constant.sum()),
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
    if pr_reason is not None:
        rows.extend(
            {
                **common,
                "metric": metric,
                "value": None,
                "status": "skipped",
                "reason": pr_reason,
            }
            for metric in (
                "variance_participation_ratio",
                "variance_participation_ratio_fraction",
            )
        )
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
    rows.extend(
        {
            **common,
            "metric": "omega_squared",
            "value": None,
            "status": "not_applicable",
            "reason": reason,
        }
        for reason in omega_errors
    )
    report = pd.DataFrame.from_records(rows)
    report.attrs["notes"] = [
        "Fractions are descriptive and depend on the selected feature scaling.",
        "Permutation p-values are Monte Carlo diagnostics, not "
        "multiplicity-corrected confirmatory tests.",
        "The streamed subject probe uses fold-local standardization and a CPU "
        "float32 PyTorch linear model.",
    ]
    return report
