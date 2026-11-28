"""
Multiple-Instance Learning Estimators
=====================================

Scikit-learn compatible bag-level classifiers backed by ``torchmil``.
Instances (e.g. windows or epochs) are grouped into variable-length bags via
the ``groups`` metadata parameter, pooled with attention, and scored once per
bag while exposing per-instance attention weights through
:meth:`ABMILClassifier.get_artifact_metadata`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

from coco_pipe.utils import import_optional_dependency

__all__ = ["ABMILClassifier"]


class ABMILClassifier(ClassifierMixin, BaseEstimator):
    """Attention-Based Multiple Instance Learning classifier backed by ``torchmil``.

    Wraps :class:`torchmil.models.ABMIL` (Ilse et al., 2018) as a scikit-learn
    binary classifier that operates on 2D instance features grouped by bag ID
    (passed via ``groups``). Preprocessing such as scaling or per-block PCA
    runs upstream in :class:`~coco_pipe.decoding.Experiment`'s fold-local
    pipeline before instances reach this estimator.

    Parameters
    ----------
    att_dim : int, default=64
        Hidden dimension of the attention network.
    att_act : str, default="tanh"
        Activation inside the attention layer (``"tanh"``, ``"relu"``, or ``"gelu"``).
    gated : bool, default=True
        Use Ilse et al.'s gated attention mechanism (``tanh * sigmoid``).
    hidden_dim : int or None, default=64
        Width of the optional instance projection MLP
        (``Linear -> ReLU -> Dropout``) passed as ``feat_ext`` to ``ABMIL``.
        When ``None``, uses ``nn.Identity()``.
    dropout : float, default=0.25
        Dropout probability inside the instance projection MLP.
    max_instances : int or None, default=128
        Maximum instances sampled per bag per training step (``None`` keeps all).
    max_epochs : int, default=200
        Maximum training epochs per restart.
    patience : int, default=20
        Early-stopping patience on bag-level validation ROC-AUC.
    batch_size : int, default=32
        Number of bags per optimizer step.
    lr : float, default=1e-3
        Learning rate for ``AdamW``.
    weight_decay : float, default=1e-2
        Weight decay for ``AdamW``.
    val_fraction : float, default=0.2
        Fraction of training bags held out for early stopping (``0.0`` disables
        validation splitting and trains for ``max_epochs``).
    n_restarts : int, default=1
        Number of independent random initializations averaged at prediction time.
    device : str, default="cpu"
        Torch device (``"cpu"``, ``"cuda"``, or ``"auto"``).
    random_state : int or None, default=0
        Base random seed; restart *r* uses ``(random_state or 0) + r``.
    """

    def __init__(
        self,
        att_dim: int = 64,
        att_act: str = "tanh",
        gated: bool = True,
        hidden_dim: int | None = 64,
        dropout: float = 0.25,
        max_instances: int | None = 128,
        max_epochs: int = 200,
        patience: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        val_fraction: float = 0.2,
        n_restarts: int = 1,
        device: str = "cpu",
        random_state: int | None = 0,
    ) -> None:
        self.att_dim = att_dim
        self.att_act = att_act
        self.gated = gated
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.max_instances = max_instances
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.n_restarts = n_restarts
        self.device = device
        self.random_state = random_state

    def get_metadata_routing(self):
        """Request bag-level ``groups`` on fit, predict, and predict_proba."""
        requests = super().get_metadata_routing()
        requests.fit.add_auto_request("groups")
        requests.predict.add_auto_request("groups")
        requests.predict_proba.add_auto_request("groups")
        return requests

    @staticmethod
    def _load_torchmil() -> tuple[Any, Any]:
        torch = import_optional_dependency(
            lambda: __import__("torch"),
            feature="ABMILClassifier",
            dependency="torch",
            install_hint="pip install coco-pipe[mil]",
        )
        torchmil_models = import_optional_dependency(
            lambda: __import__("torchmil.models", fromlist=["ABMIL"]),
            feature="ABMILClassifier",
            dependency="torchmil",
            install_hint="pip install coco-pipe[mil]",
        )
        return torch, torchmil_models.ABMIL

    @staticmethod
    def _pad_bags(
        torch: Any,
        bags: dict[Any, Any],
        bag_ids: Any,
        device: Any,
        rng: np.random.Generator | None = None,
        max_instances: int | None = None,
    ) -> tuple[Any, Any]:
        tensors = []
        for bag_id in bag_ids:
            bag = bags[bag_id]
            if rng is not None and max_instances and len(bag) > max_instances:
                choice = rng.choice(len(bag), size=max_instances, replace=False)
                bag = bag[torch.from_numpy(choice)]
            tensors.append(bag)
        lengths = torch.tensor([len(t) for t in tensors], dtype=torch.long)
        padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True).to(device)
        mask = (torch.arange(padded.shape[1])[None, :] < lengths[:, None]).to(device)
        return padded, mask

    def fit(self, X: Any, y: Any, groups: Any = None) -> ABMILClassifier:
        """Fit ``n_restarts`` ABMIL models on bags defined by *groups*.

        Parameters
        ----------
        X : array-like of shape (n_instances, n_features)
            Instance-level feature matrix.
        y : array-like of shape (n_instances,)
            Binary target labels (constant within each bag).
        groups : array-like of shape (n_instances,)
            Bag identifier for each instance row (e.g., subject ID).

        Returns
        -------
        self : ABMILClassifier
            Fitted estimator.
        """
        if groups is None:
            raise ValueError("ABMILClassifier requires bag identifiers via groups=.")
        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D instance features, got shape {X_arr.shape}.")
        y_arr = np.asarray(y)
        group_arr = np.asarray(groups)
        if len(group_arr) != len(X_arr):
            raise ValueError(
                f"groups length ({len(group_arr)}) must match X ({len(X_arr)})."
            )

        classes = np.unique(y_arr)
        if len(classes) != 2:
            raise ValueError(
                "ABMILClassifier requires binary targets; "
                f"got classes {classes.tolist()}."
            )
        y_bin = (y_arr == classes[1]).astype(int)

        row_series = pd.Series(np.arange(len(group_arr)))
        bag_rows = row_series.groupby(group_arr, sort=False).apply(np.asarray)
        y_by_bag = pd.Series(y_bin).groupby(group_arr, sort=False)
        if (y_by_bag.nunique() > 1).any():
            raise ValueError("Target labels must be constant within each bag (group).")
        y_bag = y_by_bag.first()
        bag_ids = np.asarray(y_bag.index)

        torch, abmil_cls = self._load_torchmil()
        device = torch.device(
            "cuda"
            if self.device == "auto" and torch.cuda.is_available()
            else ("cpu" if self.device == "auto" else self.device)
        )
        bags = {bag_id: torch.from_numpy(X_arr[bag_rows[bag_id]]) for bag_id in bag_ids}
        n_features = X_arr.shape[1]
        base_seed = 0 if self.random_state is None else int(self.random_state)

        models = []
        for restart in range(self.n_restarts):
            seed = base_seed + restart
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)

            y_counts = y_bag.value_counts()
            n_val = round(len(bag_ids) * self.val_fraction)
            use_val = (
                0.0 < self.val_fraction < 1.0
                and n_val >= 2
                and len(bag_ids) - n_val >= 2
                and int(y_counts.min()) >= 2
            )
            if use_val:
                fit_ids, val_ids = train_test_split(
                    bag_ids,
                    test_size=self.val_fraction,
                    stratify=y_bag.loc[bag_ids],
                    random_state=seed,
                )
            else:
                fit_ids, val_ids = bag_ids, None

            y_fit = torch.tensor(
                y_bag.loc[fit_ids].to_numpy(dtype=np.float32), device=device
            )
            pos_share = float(y_fit.mean().item())
            criterion = (
                torch.nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(
                        (1.0 - pos_share) / pos_share, device=device
                    )
                )
                if 0.0 < pos_share < 1.0
                else torch.nn.BCEWithLogitsLoss()
            )
            feat_ext = (
                torch.nn.Sequential(
                    torch.nn.Linear(n_features, self.hidden_dim),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(self.dropout),
                )
                if self.hidden_dim is not None
                else torch.nn.Identity()
            )
            model = abmil_cls(
                in_shape=(n_features,),
                att_dim=self.att_dim,
                att_act=self.att_act,
                gated=self.gated,
                feat_ext=feat_ext,
                criterion=criterion,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

            y_val_tensor = (
                torch.tensor(
                    y_bag.loc[val_ids].to_numpy(dtype=np.float32), device=device
                )
                if val_ids is not None
                else None
            )
            best_score = (-np.inf, -np.inf)
            best_state: dict[str, Any] | None = None
            waited = 0
            for _ in range(self.max_epochs):
                model.train()
                order = rng.permutation(len(fit_ids))
                for start in range(0, len(fit_ids), self.batch_size):
                    batch_idx = order[start : start + self.batch_size]
                    x_batch, mask_batch = self._pad_bags(
                        torch,
                        bags,
                        fit_ids[batch_idx],
                        device,
                        rng=rng,
                        max_instances=self.max_instances,
                    )
                    optimizer.zero_grad()
                    _, loss_dict = model.compute_loss(
                        y_fit[batch_idx], x_batch, mask=mask_batch
                    )
                    sum(loss_dict.values()).backward()
                    optimizer.step()

                if val_ids is not None:
                    model.eval()
                    with torch.no_grad():
                        x_val, mask_val = self._pad_bags(torch, bags, val_ids, device)
                        val_logits, val_loss_dict = model.compute_loss(
                            y_val_tensor, x_val, mask=mask_val
                        )
                        val_loss = float(sum(val_loss_dict.values()).item())
                        val_scores = val_logits.detach().cpu().numpy()
                    y_val = y_bag.loc[val_ids].to_numpy()
                    auc = (
                        float(roc_auc_score(y_val, val_scores))
                        if len(np.unique(y_val)) == 2
                        else 0.0
                    )
                    score = (auc, -val_loss)
                    if score > best_score:
                        best_score, waited = score, 0
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in model.state_dict().items()
                        }
                    else:
                        waited += 1
                        if waited >= self.patience:
                            break

            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            models.append(model)

        self.models_ = models
        self.device_ = device
        self.classes_ = classes
        self.n_features_in_ = n_features
        self.last_attention_: np.ndarray | None = None
        return self

    def predict_proba(self, X: Any, groups: Any = None) -> np.ndarray:
        """Predict bag-level class probabilities mapped back to each instance row.

        Also records normalized per-instance attention weights (summing to 1
        within each bag) in :attr:`last_attention_`.
        """
        check_is_fitted(self, ["models_", "classes_"])
        if groups is None:
            raise ValueError("ABMILClassifier requires bag identifiers via groups=.")
        X_arr = np.asarray(X, dtype=np.float32)
        group_arr = np.asarray(groups)
        if len(group_arr) != len(X_arr):
            raise ValueError(
                f"groups length ({len(group_arr)}) must match X ({len(X_arr)})."
            )

        torch, _ = self._load_torchmil()
        bag_rows = (
            pd.Series(np.arange(len(group_arr)))
            .groupby(group_arr, sort=False)
            .apply(np.asarray)
        )
        order = list(bag_rows.index)
        bags = {bag_id: torch.from_numpy(X_arr[bag_rows[bag_id]]) for bag_id in order}
        x_pad, mask = self._pad_bags(torch, bags, order, self.device_)

        bag_probs = []
        bag_atts: list[list[np.ndarray]] = []
        with torch.no_grad():
            for model in self.models_:
                logits, raw_att = model(x_pad, mask=mask, return_att=True)
                bag_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
                att_per_bag = []
                for i in range(len(order)):
                    bag_len = int(mask[i].sum().item())
                    weights = torch.softmax(raw_att[i, :bag_len], dim=0)
                    att_per_bag.append(weights.detach().cpu().numpy())
                bag_atts.append(att_per_bag)

        mean_prob = np.mean(bag_probs, axis=0)
        p1 = np.empty(len(X_arr), dtype=float)
        attn_out = np.empty(len(X_arr), dtype=float)
        for i, bag_id in enumerate(order):
            idx = bag_rows[bag_id]
            p1[idx] = mean_prob[i]
            attn_out[idx] = np.mean([att[i] for att in bag_atts], axis=0)

        self.last_attention_ = attn_out
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any, groups: Any = None) -> np.ndarray:
        """Predict bag-level class labels mapped back to each instance row."""
        proba = self.predict_proba(X, groups=groups)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]

    def get_artifact_metadata(self) -> dict[str, Any]:
        """Return the most recent prediction pass's per-instance attention weights."""
        return {"attention": self.last_attention_}
