"""Experimental: subject-axis removal for frozen embedding vectors."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn import config_context
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import ledoit_wolf


def _require_groups(groups: Any, n_samples: int) -> np.ndarray:
    if groups is None:
        raise ValueError("Subject alignment requires groups (subject identifiers).")
    values = np.asarray(groups)
    if values.ndim != 1 or len(values) != n_samples:
        raise ValueError(
            "groups must be one-dimensional and match X: "
            f"got shape {values.shape} for {n_samples} samples."
        )
    return values


class SubjectAlignment(BaseEstimator, TransformerMixin):
    """Base class for subject-alignment estimators."""

    name = "identity"
    fold_local: bool = False

    def fingerprint(self) -> str:
        from coco_pipe.utils import stable_hash

        return stable_hash(
            {"name": str(self.name), "params": self.get_params(deep=False)}, length=16
        )


def _whiten(features: np.ndarray, *, shrinkage: bool = True):
    x = np.asarray(features, dtype=np.float64)
    mu = x.mean(0)
    xc = x - mu
    n = len(x)
    if shrinkage:
        sigma, _ = ledoit_wolf(xc, assume_centered=True)
    else:
        sigma = xc.T @ xc / max(n, 1)
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.clip(evals, 0.0, None)
    sq = np.sqrt(evals)
    smax = sq.max() if sq.size else 0.0
    pos = sq > 1e-8 * smax if smax > 0 else np.zeros_like(sq, bool)
    inv = np.where(pos, 1.0 / np.where(pos, sq, 1.0), 0.0)
    w = (evecs * inv) @ evecs.T
    w_plus = (evecs * sq) @ evecs.T
    cond = (
        float(evals[pos].max() / evals[pos].min()) if pos.sum() >= 2 else float("inf")
    )
    return mu, xc, w, w_plus, cond


def _subject_eraser(
    xc: np.ndarray,
    w: np.ndarray,
    w_plus: np.ndarray,
    groups: np.ndarray,
):
    n, d = xc.shape
    subjects, inverse = np.unique(groups, return_inverse=True)
    z = np.zeros((n, len(subjects)), dtype=np.float64)
    z[np.arange(n), inverse] = 1.0
    zc = z - z.mean(0)
    sigma_xz = xc.T @ zc / max(n, 1)
    u, singular, _ = np.linalg.svd(w @ sigma_xz, full_matrices=False)
    rank = (
        int((singular > 1e-6 * singular.max()).sum())
        if singular.size and singular.max() > 0
        else 0
    )
    ur = u[:, :rank]
    p_perp = np.eye(d) - w_plus @ (ur @ ur.T) @ w
    return p_perp, rank


class LeaceEraser(SubjectAlignment):
    """Fold-local LEACE projection of the between-subject linear subspace."""

    name = "leace"
    fold_local = True

    def __init__(self, *, shrinkage: bool = True, degenerate_frac: float = 0.95):
        self.shrinkage = shrinkage
        self.degenerate_frac = degenerate_frac
        with config_context(enable_metadata_routing=True):
            self.set_fit_request(groups=True)
            self.set_transform_request(groups=True)

    def fit(self, X, y=None, groups=None):
        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"LeaceEraser expects a 2-D matrix, got {x.shape}.")
        subject = _require_groups(groups, len(x))
        mu, xc, w, w_plus, cond = _whiten(x, shrinkage=self.shrinkage)
        p_perp, rank = _subject_eraser(xc, w, w_plus, subject)
        self.mu_ = mu
        self.p_perp_ = p_perp
        self.rank_ = rank
        self.cond_ = cond
        self.n_subjects_ = int(np.unique(subject).size)
        self.degenerate_ = bool(
            self.n_subjects_ - 1 >= x.shape[1]
            or rank >= self.degenerate_frac * x.shape[1]
        )
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, X, groups=None):
        if not hasattr(self, "p_perp_"):
            raise RuntimeError("LeaceEraser.transform called before fit.")
        x = np.asarray(X, dtype=np.float64)
        return (x - self.mu_) @ self.p_perp_.T + self.mu_

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit the LEACE projector and transform the same observations."""
        return self.fit(X, y=y, groups=groups).transform(X, groups=groups)


def _inv_sqrt_cov(xc: np.ndarray, *, shrinkage: bool = True) -> np.ndarray:
    n = len(xc)
    if n < 2:
        return np.eye(xc.shape[1])
    if shrinkage:
        sigma, _ = ledoit_wolf(xc, assume_centered=True)
    else:
        sigma = xc.T @ xc / n
    evals, evecs = np.linalg.eigh(sigma)
    evals = np.clip(evals, 0.0, None)
    smax = evals.max() if evals.size else 0.0
    floor = 1e-12 * smax if smax > 0 else 1e-12
    inv_sqrt = np.where(
        evals > floor,
        1.0 / np.sqrt(np.where(evals > floor, evals, 1.0)),
        0.0,
    )
    return (evecs * inv_sqrt) @ evecs.T


class EuclideanAlign(SubjectAlignment):
    """Per-subject mean or CORAL alignment, including unseen subjects."""

    def __init__(self, mode: str = "coral", *, shrinkage: bool = True):
        if mode not in {"coral", "mean"}:
            raise ValueError(f"mode must be 'coral' or 'mean', got {mode!r}.")
        self.mode = mode
        self.name = f"ea_{mode}"
        self.shrinkage = shrinkage
        with config_context(enable_metadata_routing=True):
            self.set_fit_request(groups=True)
            self.set_transform_request(groups=True)

    def _stats(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        mu = xs.mean(0)
        if self.mode == "mean":
            return mu, None
        return mu, _inv_sqrt_cov(xs - mu, shrinkage=self.shrinkage)

    def fit(self, X, y=None, groups=None):
        x = np.asarray(X, dtype=np.float64)
        subject = _require_groups(groups, len(x))
        self.mu_ = {}
        self.w_ = {}
        for sid in np.unique(subject):
            mu, w = self._stats(x[subject == sid])
            self.mu_[sid] = mu
            if w is not None:
                self.w_[sid] = w
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, X, groups=None):
        if not hasattr(self, "mu_"):
            raise RuntimeError("EuclideanAlign.transform called before fit.")
        x = np.asarray(X, dtype=np.float64)
        subject = _require_groups(groups, len(x))
        out = np.empty_like(x)
        for sid in np.unique(subject):
            rows = subject == sid
            if sid in self.mu_:
                mu = self.mu_[sid]
                w = self.w_.get(sid)
            else:
                mu, w = self._stats(x[rows])
            centered = x[rows] - mu
            out[rows] = centered @ w.T if w is not None else centered
        return out

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit subject statistics and transform the same observations."""
        return self.fit(X, y=y, groups=groups).transform(X, groups=groups)


def tokens_to_covariances(tokens: np.ndarray, *, shrinkage: float = 0.1) -> np.ndarray:
    """Convert ``(window, token, feature)`` tensors to regularized SPD matrices."""
    z = np.asarray(tokens, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"RiemannAlign expects (N, T, d) tensors, got {z.shape}.")
    _, t, d = z.shape
    covs = np.einsum("ntd,nte->nde", z, z) / max(t, 1)
    eye = np.eye(d)
    for index in range(len(covs)):
        trace = np.trace(covs[index]) / d
        covs[index] = (1.0 - shrinkage) * covs[index] + shrinkage * trace * eye
    return covs


class RiemannAlign(SubjectAlignment):
    """Per-subject Riemannian recentering followed by tangent projection."""

    name = "ra"

    def __init__(self, *, shrinkage: float = 0.1):
        self.shrinkage = shrinkage
        with config_context(enable_metadata_routing=True):
            self.set_fit_request(groups=True)
            self.set_transform_request(groups=True)

    def _covs(self, tokens: np.ndarray) -> np.ndarray:
        return tokens_to_covariances(tokens, shrinkage=self.shrinkage)

    def fit(self, X, y=None, groups=None):
        from pyriemann.utils.mean import mean_riemann

        covs = self._covs(X)
        subject = _require_groups(groups, len(covs))
        self.ref_ = {
            sid: mean_riemann(covs[subject == sid]) for sid in np.unique(subject)
        }
        self.n_features_in_ = covs.shape[1]
        self.n_output_features_ = covs.shape[1] * (covs.shape[1] + 1) // 2
        return self

    def transform(self, X, groups=None):
        from pyriemann.utils.mean import mean_riemann
        from pyriemann.utils.tangentspace import tangent_space

        if not hasattr(self, "ref_"):
            raise RuntimeError("RiemannAlign.transform called before fit.")
        covs = self._covs(X)
        subject = _require_groups(groups, len(covs))
        dim = covs.shape[1]
        out = np.empty((len(covs), dim * (dim + 1) // 2), dtype=np.float64)
        for sid in np.unique(subject):
            rows = np.flatnonzero(subject == sid)
            ref = self.ref_.get(sid)
            if ref is None:
                ref = mean_riemann(covs[rows])
            out[rows] = tangent_space(covs[rows], ref)
        return out

    def fit_transform(self, X, y=None, groups=None, **fit_params):
        """Fit subject references and transform the same token windows."""
        return self.fit(X, y=y, groups=groups).transform(X, groups=groups)


#: Shape-preserving 2-D subject transforms (usable as ``ErasureConfig`` methods).
VECTOR_TRANSFORMS = frozenset({"leace", "ea_coral", "ea_mean"})
#: 3-D ``(N, T, d)`` token transforms (token/covariance decoding path).
TOKEN_TRANSFORMS = frozenset({"ra"})

#: name -> constructor registry. ``ea_coral``/``ea_mean`` share ``EuclideanAlign``.
_REGISTRY: dict[str, Callable[..., SubjectAlignment]] = {
    "leace": LeaceEraser,
    "ea_coral": lambda **p: EuclideanAlign(mode="coral", **p),
    "ea_mean": lambda **p: EuclideanAlign(mode="mean", **p),
    "ra": RiemannAlign,
}


def make_subject_transform(name: str, **params: Any) -> SubjectAlignment:
    """Instantiate a registered subject transform by name."""
    try:
        factory = _REGISTRY[str(name).strip().lower()]
    except KeyError:
        raise ValueError(
            f"Unknown subject transform {name!r}; expected one of {sorted(_REGISTRY)}."
        ) from None
    return factory(**params)
