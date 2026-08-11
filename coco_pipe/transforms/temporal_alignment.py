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
    """

    fold_local = True

    def __init__(
        self,
        n_components: int = 30,
        *,
        adaptation: str = "transductive",
        random_state: int | None = 42,
    ):
        self.n_components = n_components
        self.adaptation = adaptation
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
        if self.adaptation != "transductive":
            raise ValueError("adaptation currently supports only 'transductive'.")
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
        for sid in subjects:
            rows = subject == sid
            participant = x[rows]
            participant_pooled = participant.transpose(0, 2, 1).reshape(-1, x.shape[1])
            pca = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            ).fit(participant_pooled)
            scores = pca.transform(participant_pooled)
            scores = scores.reshape(
                len(participant), x.shape[2], self.n_components
            ).transpose(0, 2, 1)
            mean_trajectory = scores.mean(axis=0).T
            u, _, vt = np.linalg.svd(mean_trajectory.T @ self.template_)
            self.subject_pcas_[sid] = pca
            self.rotations_[sid] = u @ vt

        self.n_features_in_ = x.shape[1]
        self.n_times_in_ = x.shape[2]
        self.training_groups_ = subjects
        return self

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
            rows = subject == sid
            participant = x[rows]
            pooled = participant.transpose(0, 2, 1).reshape(-1, x.shape[1])
            pca = self.subject_pcas_.get(sid)
            rotation = self.rotations_.get(sid)
            if pca is None:
                pca = PCA(
                    n_components=self.n_components,
                    random_state=self.random_state,
                ).fit(pooled)
                scores = pca.transform(pooled)
                scores = scores.reshape(
                    len(participant), x.shape[2], self.n_components
                ).transpose(0, 2, 1)
                mean_trajectory = scores.mean(axis=0).T
                u, _, vt = np.linalg.svd(mean_trajectory.T @ self.template_)
                rotation = u @ vt
            else:
                scores = pca.transform(pooled)
                scores = scores.reshape(
                    len(participant), x.shape[2], self.n_components
                ).transpose(0, 2, 1)
            aligned[rows] = np.einsum("nkt,kj->njt", scores, rotation)
        return aligned

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
