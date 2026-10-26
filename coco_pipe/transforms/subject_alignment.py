"""Subject alignment of embeddings: LEACE, Euclidean and Riemannian alignment."""

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


def _subject_scatter(
    x: np.ndarray,
    subject: np.ndarray,
    condition: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The mean, the total covariance (Ledoit-Wolf) and the between-subject
    covariance of ``x``.

    With ``condition``, the between-subject covariance is that of the subject x
    condition cell means, each condition's own mean removed first: a subject's
    share of each condition is then not counted as its identity.
    """
    mu = x.mean(0)
    xc = x - mu
    total, _ = ledoit_wolf(xc, assume_centered=True)
    _, cells = np.unique(subject, return_inverse=True)
    if condition is not None:
        levels, conditions = np.unique(condition, return_inverse=True)
        xc = xc.copy()
        for level in range(len(levels)):
            xc[conditions == level] -= xc[conditions == level].mean(0)
        _, cells = np.unique(cells * len(levels) + conditions, return_inverse=True)
    counts = np.bincount(cells).astype(np.float64)
    means = np.zeros((len(counts), x.shape[1]))
    np.add.at(means, cells, xc)
    means /= counts[:, None]
    between = (means * (counts / len(x))[:, None]).T @ means
    return mu, total, between


class LeaceEraser(SubjectAlignment):
    """Erase the directions in which subjects differ most (rank-limited LEACE).

    Each direction ``v`` of the feature space has a subject eta^2, the share of
    its variance that lies between subjects: the solutions of
    ``S_b v = eta^2 S_t v``, with ``S_t`` the total covariance and ``S_b`` the
    between-subject covariance of the subject means. The eraser removes the
    ``r`` directions of largest eta^2 with LEACE's oblique projector
    ``P = I - S_t V_r V_r^T`` (``V_r^T S_t V_r = I``): no linear function of the
    erased data then varies along them, and the rest of the space is kept.
    Erasing every direction in which subject means differ (full LEACE) leaves
    nothing once there are more subjects than features.

    ``r`` is the number of directions with eta^2 of at least ``eta2``, at most
    ``max_rank``: ``eta2`` leaves alone the directions that are mostly
    within-subject variance, ``max_rank`` bounds how much is erased, and
    ``eta2=0`` with ``max_rank`` erases exactly that many.

    With ``condition`` in :meth:`fit`, ``S_b`` is taken from the
    subject x condition cell means around their condition's mean, so a subject
    with more windows of one condition does not count that as identity, and a
    difference between conditions shared by the subjects is kept.

    The projector is fixed after :meth:`fit`, so it applies to subjects it has
    not seen, without their groups or conditions.

    Parameters
    ----------
    eta2 : float, default=0.5
        Erase the directions whose subject eta^2 is at least this.
    max_rank : int or None, default=None
        Erase at most this many of them, the largest eta^2 first.

    Attributes
    ----------
    rank_ : int
        Number of directions erased.
    eta2_ : ndarray of shape (n_features,)
        Subject eta^2 of every direction, largest first.
    components_ : ndarray of shape (n_features, n_features)
        Every direction as a row, in the order of ``eta2_``; the first
        ``rank_`` are erased. ``(X - mu_) @ components_.T`` is each row's
        score on them, with unit total variance.
    n_subjects_ : int
        Number of subjects fitted.

    References
    ----------
    Belrose, N. et al. (2023). LEACE: Perfect linear concept erasure in closed
    form. NeurIPS.
    """

    name = "leace"
    fold_local = True

    def __init__(
        self,
        *,
        eta2: float = 0.5,
        max_rank: int | None = None,
    ):
        self.eta2 = eta2
        self.max_rank = max_rank
        with config_context(enable_metadata_routing=True):
            self.set_fit_request(groups=True, condition=True)
            self.set_transform_request(groups=True)

    def fit(self, X, y=None, groups=None, condition=None):
        """Find the directions to erase.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The vectors, e.g. one embedding per window.
        y : None
            Ignored.
        groups : array-like of shape (n_samples,)
            The subject of each row.
        condition : array-like of shape (n_samples,), optional
            The condition of each row, e.g. eyes open or closed.

        Raises
        ------
        ValueError
            If the eraser would remove 95% of the feature directions or more.
        """
        from scipy.linalg import eigh

        x = np.asarray(X, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"LeaceEraser expects a 2-D matrix, got {x.shape}.")
        subject = _require_groups(groups, len(x))
        if condition is not None:
            condition = np.asarray(condition)
            if condition.shape != subject.shape:
                raise ValueError(
                    f"condition has shape {condition.shape}, groups {subject.shape}."
                )
        mu, total, between = _subject_scatter(x, subject, condition)
        eta2, vectors = eigh(between, total)
        eta2, vectors = np.clip(eta2[::-1], 0.0, 1.0), vectors[:, ::-1]
        rank = int((eta2 >= self.eta2).sum())
        if self.max_rank is not None:
            rank = min(rank, int(self.max_rank))
        n_features = x.shape[1]
        if rank >= 0.95 * n_features:
            raise ValueError(
                f"LEACE would erase {rank} of {n_features} feature directions "
                f"({np.unique(subject).size} subjects): too little would be left. "
                "Raise eta2 or set max_rank."
            )
        erased = vectors[:, :rank]
        self.mu_ = mu
        self.projector_ = np.eye(n_features) - total @ erased @ erased.T
        self.rank_ = rank
        self.eta2_ = eta2
        self.components_ = vectors.T
        self.n_subjects_ = int(np.unique(subject).size)
        self.n_features_in_ = n_features
        return self

    def transform(self, X, groups=None):
        if not hasattr(self, "projector_"):
            raise RuntimeError("LeaceEraser.transform called before fit.")
        x = np.asarray(X, dtype=np.float64)
        return (x - self.mu_) @ self.projector_.T + self.mu_

    def fit_transform(self, X, y=None, groups=None, condition=None, **fit_params):
        """Fit the eraser and erase the same observations."""
        return self.fit(X, groups=groups, condition=condition).transform(X)


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
    """Per-subject mean or CORAL alignment, including unseen subjects.

    ``X`` is ``(window, feature)`` or tokens ``(window, ..., feature)``: a
    subject's mean, and for ``"coral"`` its feature covariance, are taken over
    all its tokens. With ``channel_axis``, each channel's tokens are centered
    on their own mean, and ``"coral"`` whitens every channel with the
    subject's one covariance of the centered tokens. With ``shrinkage``,
    ``"coral"`` scales each subject's output to the identity's mean squared norm
    (the number of features), so it does not depend on how many samples the
    subject has. The
    output keeps the shape and float dtype of ``X``.

    Each subject loses its mean, so a trait constant within a subject (a
    diagnosis, age) is removed with it; and when a subject's samples mix
    conditions in uneven shares, its reference moves with its share. Use it for
    targets that vary within a subject, or pass ``subject x condition`` as
    ``groups`` to take a reference per condition.
    """

    def __init__(
        self,
        mode: str = "coral",
        *,
        shrinkage: bool = True,
        channel_axis: int | None = None,
    ):
        if mode not in {"coral", "mean"}:
            raise ValueError(f"mode must be 'coral' or 'mean', got {mode!r}.")
        self.mode = mode
        self.name = f"ea_{mode}"
        self.shrinkage = shrinkage
        self.channel_axis = channel_axis
        with config_context(enable_metadata_routing=True):
            self.set_fit_request(groups=True)
            self.set_transform_request(groups=True)

    def _stats(self, xs: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        xs = np.asarray(xs, dtype=np.float64)
        axes = tuple(i for i in range(xs.ndim - 1) if i != self.channel_axis)
        mu = xs.mean(axis=axes, keepdims=True)[0]
        if self.mode == "mean":
            return mu, None
        centered = (xs - mu).reshape(-1, xs.shape[-1])
        w = _inv_sqrt_cov(centered, shrinkage=self.shrinkage)
        # Shrinkage whitens a subject with few samples to less than unit variance,
        # so the output size would follow the sample count: give every subject the
        # identity's mean squared norm.
        spread = np.mean(np.sum((centered @ w.T) ** 2, axis=1))
        if self.shrinkage and spread > 0:
            w = w * np.sqrt(xs.shape[-1] / spread)
        return mu, w

    def _check(self, X) -> np.ndarray:
        x = np.asarray(X)
        if x.ndim < 2:
            raise ValueError(
                f"EuclideanAlign expects (window, ..., feature), got {x.shape}."
            )
        axis = self.channel_axis
        if axis is not None and (
            isinstance(axis, bool)
            or not isinstance(axis, int)
            or not 1 <= axis < x.ndim - 1
        ):
            raise ValueError(
                "channel_axis must be a token axis, between the window and "
                f"feature axes of {x.shape}, got {axis!r}."
            )
        return x

    def fit(self, X, y=None, groups=None):
        x = self._check(X)
        subject = _require_groups(groups, len(x))
        self.mu_ = {}
        self.w_ = {}
        for sid in np.unique(subject):
            mu, w = self._stats(x[subject == sid])
            self.mu_[sid] = mu
            if w is not None:
                self.w_[sid] = w
        self.n_features_in_ = x.shape[-1]
        return self

    def transform(self, X, groups=None):
        if not hasattr(self, "mu_"):
            raise RuntimeError("EuclideanAlign.transform called before fit.")
        x = self._check(X)
        subject = _require_groups(groups, len(x))
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        out = np.empty(x.shape, dtype=dtype)
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
    """The feature covariance of each window's tokens, shrunk to be positive definite.

    Each window's tokens are centered on their mean first, so the matrix describes how
    the tokens co-vary, not where they sit. A window has rank at most ``T - 1``, often
    below ``d`` (e.g. 64 tokens of 256 features), so the covariance is moved by
    ``shrinkage`` toward the identity scaled to its mean variance: every eigenvalue is
    then at least ``shrinkage * trace / d``, positive, as the Riemannian mean and the
    tangent map need. A window whose tokens are all the same is refused.
    """
    z = np.asarray(tokens, dtype=np.float64)
    if z.ndim != 3:
        raise ValueError(f"RiemannAlign expects (N, T, d) tensors, got {z.shape}.")
    _, t, d = z.shape
    if t < 2:
        raise ValueError(f"A covariance needs at least two tokens per window, got {t}.")
    z = z - z.mean(axis=1, keepdims=True)
    covs = np.einsum("ntd,nte->nde", z, z) / (t - 1)
    flat = np.flatnonzero(np.trace(covs, axis1=1, axis2=2) <= 0)
    if flat.size:
        raise ValueError(
            f"{flat.size} window(s) have identical tokens (e.g. window {flat[0]}): "
            "their covariance is zero, so no shrinkage makes it positive definite."
        )
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
        from pyriemann.geometry.mean import mean_riemann

        covs = self._covs(X)
        subject = _require_groups(groups, len(covs))
        self.ref_ = {
            sid: mean_riemann(covs[subject == sid]) for sid in np.unique(subject)
        }
        self.n_features_in_ = covs.shape[1]
        self.n_output_features_ = covs.shape[1] * (covs.shape[1] + 1) // 2
        return self

    def transform(self, X, groups=None):
        from pyriemann.geometry.mean import mean_riemann
        from pyriemann.geometry.tangentspace import tangent_space

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
