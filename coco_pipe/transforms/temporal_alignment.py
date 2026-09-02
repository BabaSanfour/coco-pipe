"""Group-aware alignment of temporal trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import hilbert
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

_AUGMENT_CHOICES = frozenset({"envelope", "velocity", "scalar", "trajectory"})
_TRAJECTORY_METRICS = (
    "speed",
    "acceleration",
    "jerk",
    "curvature",
    "path_length",
    "displacement",
    "distance_from_center",
    "turning_angle",
    "tortuosity",
    "auc_speed",
)


class TemporalProcrustesAlignment(BaseEstimator, TransformerMixin):
    """Align participant-specific PCA trajectories to a training template.

    ``fit`` learns a shared PCA template and participant mappings from the
    training fold. ``transform`` reuses stored mappings for known participants
    and estimates a label-free PCA/rotation from the transform batch for unseen
    participants. The latter is explicit transductive adaptation.

    Two options turn this into its own controls. ``rotate=False`` keeps the
    per-participant PCA but drops the Procrustes step, which isolates whether a
    participant-specific basis alone transfers (it does not) or whether the
    rotation is what does the work. ``adaptation="calibration"`` estimates an
    unseen participant's PCA and rotation from one random half of their trials
    and applies it to the other half (and vice versa), so no trial ever
    contributes to its own mapping — a non-transductive variant that still
    returns every row.

    ``use_shared_basis=True`` is a third control: every participant is
    projected through the pooled training template directly (the same
    ``shared_pca_`` fit ``fit()`` already uses as the rotation target),
    ignoring per-participant identity and ``rotate`` entirely. This is the
    continuous, temporally-stable analogue of a single spatial filter shared
    by everyone — as opposed to ``coco_pipe.decoding``'s ``ReducerConfig``
    PCA, which is fold-local but refit independently at every sliding-window
    latency.

    ``augment`` concatenates extra channels derived from each component's own
    trajectory onto the raw scores, still computed fold-locally inside
    ``transform`` so nothing about a held-out participant leaks into them:
    ``"envelope"`` (Hilbert amplitude), ``"velocity"`` (index-based first
    derivative — only rank-invariant reparametrizations of time reach the
    classifier, and any downstream ``StandardScaler`` absorbs a fixed
    sampling-rate scale factor, so the true sampling interval isn't needed),
    and ``"scalar"`` (each component's peak amplitude, index-based peak
    latency, mean power, and signed AUC over the whole supplied window,
    broadcast as constant channels across time). Requesting ``n`` of these
    turns ``n_components`` raw channels into ``n_components * (1 + n)``.

    ``augment_only=True`` drops the raw per-component ``scores`` channels
    entirely, leaving only whatever ``augment`` requests (e.g. just the ~10
    ``"trajectory"`` geometry channels, independent of ``n_components`` -
    for analyses where the raw PC scores aren't wanted as features at all,
    only derived channels computed from the embedding they define).
    ``trajectory_metrics`` restricts ``"trajectory"`` to a chosen subset of
    its ~10 metrics (by name, e.g. ``("speed",)``) instead of computing and
    returning all of them - ``None`` (default) means all.
    """

    fold_local = True

    def __init__(
        self,
        n_components: int = 30,
        *,
        adaptation: str = "transductive",
        rotate: bool = True,
        use_shared_basis: bool = False,
        augment: tuple[str, ...] = (),
        augment_only: bool = False,
        trajectory_metrics: tuple[str, ...] | None = None,
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.adaptation = adaptation
        self.rotate = rotate
        self.use_shared_basis = use_shared_basis
        self.augment = augment
        self.augment_only = augment_only
        self.trajectory_metrics = trajectory_metrics
        self.random_state = random_state
        with config_context(enable_metadata_routing=True):
            self.set_fit_request(groups=True)
            self.set_transform_request(groups=True)

    def fit(self, X: Any, y: Any = None, groups: Any = None):
        """Fit the training-fold template and participant mappings."""
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(
                "TemporalProcrustesAlignment expects "
                f"(observation, feature, time), got {x.shape}."
            )
        if groups is None:
            raise ValueError("Temporal alignment requires participant groups.")
        subject = np.asarray(groups)
        if subject.ndim != 1 or len(subject) != len(x):
            raise ValueError("groups must be one-dimensional and match X.")
        if self.adaptation not in {"transductive", "calibration"}:
            raise ValueError(
                "adaptation must be 'transductive' or 'calibration', got "
                f"{self.adaptation!r}."
            )
        invalid_augment = set(self.augment) - _AUGMENT_CHOICES
        if invalid_augment:
            raise ValueError(
                f"augment values must be a subset of {sorted(_AUGMENT_CHOICES)}, "
                f"got unknown values {sorted(invalid_augment)}."
            )
        if self.augment_only and not self.augment:
            raise ValueError("augment_only=True requires a non-empty augment.")
        if self.trajectory_metrics is not None:
            invalid_metrics = set(self.trajectory_metrics) - set(_TRAJECTORY_METRICS)
            if invalid_metrics:
                raise ValueError(
                    f"trajectory_metrics values must be a subset of "
                    f"{sorted(_TRAJECTORY_METRICS)}, got unknown values "
                    f"{sorted(invalid_metrics)}."
                )
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if self.n_components > x.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} exceeds {x.shape[1]} features."
            )

        pooled = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
        self.shared_pca_ = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        ).fit(pooled)
        shared = self.shared_pca_.transform(pooled)
        shared = shared.reshape(len(x), x.shape[2], self.n_components).transpose(
            0, 2, 1
        )
        subjects = np.unique(subject)
        self.template_ = shared.mean(axis=0).T

        self.subject_pcas_ = {}
        self.rotations_ = {}
        self.alignment_diagnostics_ = {}
        for sid in subjects:
            rows = subject == sid
            pca, rotation, diagnostics = self._fit_subject_mapping(x[rows])
            self.subject_pcas_[sid] = pca
            self.rotations_[sid] = rotation
            self.alignment_diagnostics_[sid] = {"seen_in_training": True, **diagnostics}

        self.n_features_in_ = x.shape[1]
        self.n_times_in_ = x.shape[2]
        self.training_groups_ = subjects
        return self

    def _subject_scores(self, participant: np.ndarray, pca: PCA) -> np.ndarray:
        """Project one participant's trials into their own PCA space."""
        pooled = participant.transpose(0, 2, 1).reshape(-1, participant.shape[1])
        scores = pca.transform(pooled)
        return scores.reshape(
            len(participant), participant.shape[2], self.n_components
        ).transpose(0, 2, 1)

    def _shared_scores(self, x: np.ndarray) -> np.ndarray:
        """Project every row through the pooled template, no per-subject fit."""
        pooled = x.transpose(0, 2, 1).reshape(-1, x.shape[1])
        scores = self.shared_pca_.transform(pooled)
        return scores.reshape(len(x), x.shape[2], self.n_components).transpose(0, 2, 1)

    def _augment_channels(self, scores: np.ndarray) -> np.ndarray:
        """Extra channels derived from each fold's trajectory.

        Concatenated onto ``scores`` (observation, component, time) along the
        component axis - a no-op when ``augment`` is empty. ``"envelope"``,
        ``"velocity"``, and ``"scalar"`` are per-component (contribute
        ``n_components`` channels each); ``"trajectory"`` instead reduces the
        *joint* (component, time) path through the full embedded space to a
        handful of geometry channels (speed, curvature, ...), contributing a
        fixed count independent of ``n_components``.
        """
        if not self.augment:
            return scores
        channels = [] if self.augment_only else [scores]
        if "envelope" in self.augment:
            channels.append(np.abs(hilbert(scores, axis=-1)))
        if "velocity" in self.augment:
            channels.append(np.gradient(scores, axis=-1))
        if "scalar" in self.augment:
            n_times = scores.shape[-1]
            peak_idx = np.abs(scores).argmax(axis=-1)
            peak_amplitude = np.take_along_axis(scores, peak_idx[..., None], axis=-1)
            peak_latency = peak_idx.astype(scores.dtype)[..., None]
            mean_power = np.mean(scores**2, axis=-1, keepdims=True)
            auc = np.trapezoid(scores, axis=-1)[..., None]
            for scalar in (peak_amplitude, peak_latency, mean_power, auc):
                channels.append(np.repeat(scalar, n_times, axis=-1))
        if "trajectory" in self.augment:
            channels.extend(self._trajectory_channels(scores))
        return np.concatenate(channels, axis=1)

    def _trajectory_channels(self, scores: np.ndarray) -> list[np.ndarray]:
        """Geometry channels from coco_pipe.dim_reduction's trajectory metrics.

        These reduce the *joint* path through the embedded (component) space
        at each instant, so every metric is exactly one channel regardless of
        ``n_components`` - unlike the per-component ``envelope``/``velocity``/
        ``scalar`` options above. Index-based ``dt=1.0`` throughout, for the
        same reason ``velocity`` doesn't need real timestamps: only a fixed
        sampling-rate scale factor is at stake, which downstream standardizing
        absorbs.
        """
        from coco_pipe.dim_reduction.evaluation.geometry import (
            trajectory_acceleration,
            trajectory_auc_speed,
            trajectory_curvature,
            trajectory_displacement,
            trajectory_distance_from_center,
            trajectory_jerk,
            trajectory_path_length,
            trajectory_speed,
            trajectory_tortuosity,
            trajectory_turning_angle,
        )

        n_times = scores.shape[-1]
        # geometry.py expects (..., time, dims); scores is (obs, dims, time).
        traj = scores.transpose(0, 2, 1)

        def _pad_to_n_times(curve: np.ndarray) -> np.ndarray:
            deficit = n_times - curve.shape[-1]
            if deficit <= 0:
                return curve
            left = deficit // 2
            return np.pad(curve, ((0, 0), (left, deficit - left)), mode="edge")

        # Lazy (thunk) so trajectory_metrics can select a subset without
        # paying for the ones that aren't requested.
        time_varying = {
            "speed": lambda: trajectory_speed(traj, dt=1.0),
            "acceleration": lambda: trajectory_acceleration(traj, dt=1.0),
            "jerk": lambda: trajectory_jerk(traj, dt=1.0),
            "curvature": lambda: trajectory_curvature(traj, method="gradient"),
            "path_length": lambda: trajectory_path_length(traj, cumulative=True),
            "displacement": lambda: trajectory_displacement(traj, final=False),
            "distance_from_center": lambda: trajectory_distance_from_center(traj),
            "turning_angle": lambda: _pad_to_n_times(trajectory_turning_angle(traj)),
        }
        scalar = {
            "tortuosity": lambda: trajectory_tortuosity(traj),
            "auc_speed": lambda: trajectory_auc_speed(traj, dt=1.0),
        }
        requested = self.trajectory_metrics or _TRAJECTORY_METRICS
        channels = []
        for name in _TRAJECTORY_METRICS:
            if name not in requested:
                continue
            if name in time_varying:
                channels.append(time_varying[name]()[:, None, :])
            else:
                value = scalar[name]()
                channels.append(np.repeat(value[:, None, None], n_times, axis=-1))
        return channels

    def _fit_subject_mapping(
        self, participant: np.ndarray
    ) -> tuple[PCA, np.ndarray, dict[str, float]]:
        """Fit one participant's PCA and its rotation onto the shared template.

        Returns the PCA, the rotation (identity when ``rotate`` is False), and
        scalar geometry diagnostics describing how far the participant's mean
        path sat from the template before and after rotating.
        """
        pooled = participant.transpose(0, 2, 1).reshape(-1, participant.shape[1])
        pca = PCA(
            n_components=self.n_components,
            random_state=self.random_state,
        ).fit(pooled)
        mean_trajectory = self._subject_scores(participant, pca).mean(axis=0).T
        u, singular_values, vt = np.linalg.svd(mean_trajectory.T @ self.template_)
        rotation = u @ vt
        if not self.rotate:
            rotation = np.eye(self.n_components)

        # Scale-free shape agreement with the template, before and after the
        # rotation: 1.0 is a perfect match, 0.0 is orthogonal. `rotated` is the
        # optimal attainable value, so `rotated - unrotated` is exactly the
        # geometry the Procrustes step buys for this participant.
        template_norm = float(np.linalg.norm(self.template_))
        subject_norm = float(np.linalg.norm(mean_trajectory))
        denominator = template_norm * subject_norm
        unrotated = float(np.sum(mean_trajectory * self.template_)) / denominator
        rotated = float(np.sum(singular_values)) / denominator
        # How far the rotation is from doing nothing, as a single angle.
        identity_similarity = float(np.trace(rotation)) / self.n_components
        diagnostics = {
            "template_similarity_unrotated": unrotated,
            "template_similarity_rotated": rotated,
            "similarity_gain": rotated - unrotated,
            "procrustes_disparity": 1.0 - rotated,
            "rotation_angle_deg": float(
                np.degrees(np.arccos(np.clip(identity_similarity, -1.0, 1.0)))
            ),
        }
        return pca, rotation, diagnostics

    def transform(self, X: Any, groups: Any = None) -> np.ndarray:
        """Project and align known or unseen participant trajectories."""
        if not hasattr(self, "template_"):
            raise RuntimeError(
                "TemporalProcrustesAlignment.transform called before fit."
            )
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 3:
            raise ValueError(
                "TemporalProcrustesAlignment expects "
                f"(observation, feature, time), got {x.shape}."
            )
        if x.shape[1:] != (self.n_features_in_, self.n_times_in_):
            raise ValueError(
                "Temporal alignment input shape changed after fit: "
                f"expected (*, {self.n_features_in_}, {self.n_times_in_}), "
                f"got {x.shape}."
            )
        if groups is None:
            raise ValueError("Temporal alignment requires participant groups.")
        subject = np.asarray(groups)
        if subject.ndim != 1 or len(subject) != len(x):
            raise ValueError("groups must be one-dimensional and match X.")

        if self.use_shared_basis:
            # No per-subject fitting at all: every row goes through the same
            # pooled template regardless of identity, so there is nothing
            # subject-specific to look up or estimate for unseen participants.
            return self._augment_channels(self._shared_scores(x))

        aligned = np.empty((len(x), self.n_components, x.shape[2]), dtype=np.float64)
        for sid in np.unique(subject):
            rows = np.flatnonzero(subject == sid)
            participant = x[rows]
            pca = self.subject_pcas_.get(sid)
            if pca is not None:
                # Seen during fit: reuse the stored mapping unchanged.
                scores = self._subject_scores(participant, pca)
                aligned[rows] = np.einsum("nkt,kj->njt", scores, self.rotations_[sid])
                continue
            if self.adaptation == "transductive":
                pca, rotation, diagnostics = self._fit_subject_mapping(participant)
                scores = self._subject_scores(participant, pca)
                aligned[rows] = np.einsum("nkt,kj->njt", scores, rotation)
            else:
                # Calibration: split this unseen participant's trials in two and
                # map each half with the mapping estimated from the *other* half,
                # so no trial contributes to the mapping applied to it.
                halves = self._calibration_halves(len(rows))
                diagnostics_halves = []
                for fit_rows, apply_rows in (halves, halves[::-1]):
                    pca, rotation, diagnostics = self._fit_subject_mapping(
                        participant[fit_rows]
                    )
                    scores = self._subject_scores(participant[apply_rows], pca)
                    aligned[rows[apply_rows]] = np.einsum(
                        "nkt,kj->njt", scores, rotation
                    )
                    diagnostics_halves.append(diagnostics)
                diagnostics = {
                    key: float(np.mean([half[key] for half in diagnostics_halves]))
                    for key in diagnostics_halves[0]
                }
            self.alignment_diagnostics_[sid] = {
                "seen_in_training": False,
                **diagnostics,
            }
        return self._augment_channels(aligned)

    def _calibration_halves(self, n_trials: int) -> tuple[np.ndarray, np.ndarray]:
        """Split trial positions into two random halves for cross-fitted mapping."""
        if n_trials < 2:
            raise ValueError(
                "adaptation='calibration' needs at least two trials per unseen "
                f"participant to form two halves, got {n_trials}."
            )
        order = np.random.default_rng(self.random_state).permutation(n_trials)
        return order[: n_trials // 2], order[n_trials // 2 :]

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """Fit and align one training fold."""
        return self.fit(X, y=y, groups=groups).transform(X, groups=groups)


__all__ = ["TemporalProcrustesAlignment"]
