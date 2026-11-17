"""Pieces of the fine-tuning loop: loss, learning-rate schedule, accumulation,
dropout and optimizer."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from skorch import NeuralNetClassifier, NeuralNetRegressor
from skorch.utils import TeeGenerator


class FocalLoss(torch.nn.Module):
    """Cross-entropy that down-weights the samples a model already gets right.

    The loss of a sample is ``(1 - p_t) ** gamma`` times its cross-entropy,
    where ``p_t`` is the probability given to its true class [1]_.

    Parameters
    ----------
    gamma : float, default=2.0
        Focusing strength. With ``gamma=0`` this is plain cross-entropy.
    weight : torch.Tensor, optional
        One weight per class, applied as in :class:`torch.nn.CrossEntropyLoss`.

    References
    ----------
    .. [1] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
       Focal loss for dense object detection. ICCV.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(input, dim=-1)
        log_p_true = log_p.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        loss = -((1 - log_p_true.exp()) ** self.gamma) * log_p_true
        if self.weight is None:
            return loss.mean()
        weight = self.weight[target]
        return (weight * loss).sum() / weight.sum()


def warmup_cosine(
    epoch: int, *, warmup_epochs: int, max_epochs: int, floor: float = 0.01
) -> float:
    """Return the learning-rate factor of an epoch (the first one is 0).

    The factor rises linearly to 1 over ``warmup_epochs``, then follows a
    cosine down to ``floor`` at ``max_epochs``.
    """
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


_DROPOUT_MODULES = (
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.AlphaDropout,
    torch.nn.FeatureAlphaDropout,
)


def set_dropout(modules, p: float) -> int:
    """Give each dropout among ``modules`` the probability ``p``.

    Dropout modules take ``p``, and so do the attention modules that keep the
    dropout of their attention weights as a number: ``dropout`` of
    :class:`torch.nn.MultiheadAttention`, ``att_drop`` of braindecode's
    ``MultiHeadAttention`` and ``attn_drop`` of LUNA's rotary attention.
    Stochastic depth (``DropPath``) is not dropout and is left as it is.

    Returns the number of modules changed.
    """
    changed = 0
    for module in modules:
        if isinstance(module, _DROPOUT_MODULES):
            module.p = p
        elif isinstance(module, torch.nn.MultiheadAttention):
            module.dropout = p
        else:
            names = [
                name
                for name in ("att_drop", "attn_drop")
                if isinstance(getattr(module, name, None), float)
            ]
            if not names:
                continue
            for name in names:
                setattr(module, name, p)
        changed += 1
    return changed


def adamw_with_head_decay(param_groups, *, head, head_weight_decay, **kwargs):
    """AdamW in which the parameters of the module ``head`` take their own decay.

    ``param_groups`` are those skorch passes to its optimizer; every other
    parameter keeps the ``weight_decay`` of ``kwargs``.
    """
    in_head = {id(parameter) for parameter in head.parameters()}
    groups = []
    for group in param_groups:
        params = list(group["params"])
        groups.append({**group, "params": [p for p in params if id(p) not in in_head]})
        groups.append(
            {
                **group,
                "params": [p for p in params if id(p) in in_head],
                "weight_decay": head_weight_decay,
            }
        )
    return torch.optim.AdamW([g for g in groups if g["params"]], **kwargs)


class _AccumulateGradients:
    """Step the optimizer once every ``accumulate_grad_batches`` batches.

    The step uses the mean gradient of those batches, as one batch that many
    times larger would. Batches left over when training ends are not applied.
    """

    def __init__(self, *args, accumulate_grad_batches: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.accumulate_grad_batches = accumulate_grad_batches

    def initialize(self):
        self.batches_seen_ = 0
        return super().initialize()

    def train_step(self, batch, **fit_params):
        step = self.train_step_single(batch, **fit_params)
        self.batches_seen_ += 1
        if self.batches_seen_ % self.accumulate_grad_batches == 0:
            for _, parameter in self.get_all_learnable_params():
                if parameter.grad is not None:
                    parameter.grad /= self.accumulate_grad_batches
            self.notify(
                "on_grad_computed",
                named_parameters=TeeGenerator(self.get_all_learnable_params()),
                batch=batch,
                training=True,
            )
            self.optimizer_.step()
            self._zero_grad_optimizer()
        return step


class AccumulatingClassifier(_AccumulateGradients, NeuralNetClassifier):
    """A skorch classifier with gradient accumulation."""


class AccumulatingRegressor(_AccumulateGradients, NeuralNetRegressor):
    """A skorch regressor with gradient accumulation."""
