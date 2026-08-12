"""Group-aware alignment of temporal trajectories."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


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
    """

    fold_local = True

    def __init__(
        self,
        n_components: int = 30,
        *,
        adaptation: str = "transductive",
        rotate: bool = True,
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.adaptation = adaptation
        self.rotate = rotate
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
        return aligned

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
