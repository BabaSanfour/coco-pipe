"""
Foundation Model Base Backend
==============================

Abstract base class shared by all EEG/MEG foundation model backends.
Backends can be used directly. Clone-safe adapters wrap them inside sklearn
pipelines and cross-validation.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec, SignalMetadata

_base_logger = logging.getLogger(__name__)

# Architecture-driven LoRA (r, alpha) per model.
# r ≈ 5–16 % of internal attention d_model; alpha = 2*r.
LORA_AUTO_PARAMS: dict[str, tuple[int, int]] = {
    "signaljepa": (4,   8),
    "luna":        (4,   8),
    "cbramod":     (32,  64),
    "biot":        (16,  32),
    "labram":      (16,  32),
    "reve":        (32,  64),
    "bendr":       (8,   16),
    "eegpt":       (16,  32),
}


def resolve_auto_lora_params(model_key: str, model, target_modules, r_cfg, alpha_cfg) -> tuple[int, int]:
    """Resolve "auto" r/alpha to concrete integers.

    Uses the per-model architecture table for known models; falls back to
    inspecting actual linear-layer weight dims for unknown ones.
    alpha defaults to 2*r when "auto".
    """
    if r_cfg != "auto":
        r = int(r_cfg)
    elif model_key in LORA_AUTO_PARAMS:
        r = LORA_AUTO_PARAMS[model_key][0]
        _base_logger.info("[LoRA:%s] auto r=%d (architecture table)", model_key, r)
    else:
        import torch.nn as _nn
        target_list = (
            None if target_modules == "all-linear"
            else ([target_modules] if isinstance(target_modules, str) else list(target_modules))
        )
        dims = [
            min(m.weight.shape)
            for name, m in model.named_modules()
            if isinstance(m, _nn.Linear)
            and (target_list is None or any(t in name for t in target_list))
        ]
        if dims:
            median_dim = sorted(dims)[len(dims) // 2]
            raw = max(1, median_dim // 16)
            r = int(max(4, min(64, 2 ** round(math.log2(raw)))))
        else:
            r = 8
        _base_logger.info(
            "[LoRA:%s] auto r=%d (inspected %d linear layers)", model_key, r, len(dims)
        )

    if alpha_cfg != "auto":
        alpha = int(alpha_cfg)
    elif model_key in LORA_AUTO_PARAMS and r_cfg == "auto":
        alpha = LORA_AUTO_PARAMS[model_key][1]
    else:
        alpha = 2 * r
        _base_logger.info("[LoRA:%s] auto alpha=%d (= 2 × r)", model_key, alpha)

    return r, alpha


def _make_grad_accum_cls(base_cls, accumulate_grad_batches: int):
    """Return a skorch Net subclass that accumulates gradients over N batches.

    Gradients are averaged (loss divided by N before backward) so the effective
    gradient magnitude per update step matches what batch_size*N would give.
    The optimizer only steps every N batches; zero_grad fires at the start of
    each accumulation window.
    """
    from skorch.dataset import unpack_data

    n = accumulate_grad_batches

    class _GradAccumNet(base_cls):
        def initialize(self):
            super().initialize()
            self._accum_step = 0
            return self

        def train_step(self, batch, **fit_params):
            step_acc = self.get_train_step_accumulator()
            is_first = self._accum_step % n == 0
            do_update = self._accum_step % n == n - 1
            self._accum_step += 1

            if is_first:
                self._zero_grad_optimizer()

            self._set_training(True)
            Xi, yi = unpack_data(batch)
            y_pred = self.infer(Xi, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=True)
            (loss / n).backward()  # average gradients over accumulation window

            step_acc.store_step({"loss": loss, "y_pred": y_pred})

            if do_update:
                self.notify(
                    "on_grad_computed",
                    named_parameters=list(self.get_all_learnable_params()),
                    batch=batch,
                    training=True,
                )
                self.optimizer_.step()

            return step_acc.get_step()

    _GradAccumNet.__name__ = f"_GradAccumNet_{base_cls.__name__}"
    return _GradAccumNet


class BackendBase(BaseEstimator, TransformerMixin, ABC):
    """Abstract base for direct foundation-model backend use.

    Training modes
    --------------
    frozen   — backbone weights frozen; only the head trains (head-only fine-tune).
    full     — all weights train end-to-end.
    lora     — peft LoRA adapters injected; only adapter weights train.
    qlora    — 4-bit quantized backbone + LoRA (hugging_face backend only).
    """

    _task: str = "classification"
    _device: str = "cpu"
    _n_outputs: int | None = None
    _expected_n_chans: int | None = None
    signal_metadata_: "SignalMetadata | None" = None
    _net_ = None

    @abstractmethod
    def reset_head(self, n_outputs: int) -> "BackendBase":
        """Replace the classification head without changing backbone weights.

        Parameters
        ----------
        n_outputs : int
            Number of output units for the new head.

        Returns
        -------
        self : BackendBase
            The backend with the new head in place.
        """
        ...

    @abstractmethod
    def get_embedding_info(self) -> "FoundationModelSpec":
        """Return the registry spec for this model instance.

        Returns
        -------
        spec : FoundationModelSpec
            The spec describing this model's provenance and signal expectations.
        """
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check whether all required packages for this backend are installed.

        Returns
        -------
        available : bool
            ``True`` if the backend's dependencies are importable.
        """
        ...

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_key: str,
        metadata: "FoundationModelSpec",
        n_outputs: int | None,
        device: str,
        train_mode: str,
        **backend_kwargs,
    ) -> "BackendBase":
        """Load pretrained weights and return a ready-to-use backend.

        All heavy imports (torch, transformers, braindecode) must happen
        inside this method to keep the module importable without extras.

        Parameters
        ----------
        model_key : str
            Canonical model key from the registry (e.g. ``"cbramod"``).
        metadata : FoundationModelSpec
            Registry spec for this model.
        n_outputs : int or None
            Number of classification/regression outputs. ``None`` = headless.
        device : str
            Resolved device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        train_mode : str
            One of ``"frozen"``, ``"full"``, ``"lora"``, ``"qlora"``.
        **backend_kwargs
            Backend-specific keyword arguments forwarded from
            ``foundation_models.load``.

        Returns
        -------
        backend : BackendBase
            A fully initialised, ready-to-fit backend instance.
        """
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params,
    ) -> "BackendBase":
        """Train the model on (X, y).

        In ``frozen`` mode only the classification head is updated; in
        ``full`` and ``lora``/``qlora`` modes the backbone is also trained.

        ``signal_metadata`` (sfreq + ch_names) must be declared at
        ``load(signal_metadata=...)`` time before fitting.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.
        y : np.ndarray of shape (n_samples,) or None
            Target labels (classification) or continuous values (regression).
        **fit_params
            Recognised extra keys:
            max_epochs : int, default 10
            lr : float, default 1e-3
            batch_size : int, default 32

        Returns
        -------
        self : BackendBase
            The fitted backend instance.
        """
        return self._fit_with_skorch(X, y, **fit_params)

    def get_channel_adaptation(self) -> dict:
        """Return the channel mapping actually applied by the backend."""
        meta = self.signal_metadata_
        channels = list(meta.ch_names) if meta is not None else []
        return {
            "source_channels": channels,
            "target_channels": channels,
            "interpolated_channels": [],
            "zero_filled_channels": [],
            "dropped_channels": [],
        }

    @property
    def device(self) -> str:
        """The torch device string this backend lives on."""
        return self._device

    def checkpoint_components(self) -> dict:
        """Restorable torch modules keyed by name (state-dict layout).

        Backends override this to expose their own modules (e.g.
        ``{"model": ...}`` or ``{"backbone": ..., "head": ...}``). Default:
        nothing to checkpoint.
        """
        return {}

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Extract embeddings from the frozen backbone.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs. Must match the channel and sfreq expectations
            recorded in ``signal_metadata_``.

        Returns
        -------
        embeddings : np.ndarray of shape (n_samples, embedding_dim)
            Backbone feature vectors before the classification head.
        """
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a full forward pass and return predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class indices (classification) or continuous values
            (regression).
        """
        ...

    def configure_peft(self, lora_config: dict) -> "BackendBase":
        """Wrap the backbone with LoRA adapters post-hoc.

        Useful for switching from ``frozen`` to ``lora`` after the initial
        load without reloading pretrained weights.

        Parameters
        ----------
        lora_config : dict
            Keyword arguments forwarded to ``peft.LoraConfig``.

        Returns
        -------
        self : BackendBase
            The backend with LoRA adapters injected.

        Raises
        ------
        NotImplementedError
            If the concrete backend does not support PEFT. Use
            ``train_mode='lora'`` at ``load()`` time instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support configure_peft() / LoRA. "
            "Use train_mode='lora' at load() time instead."
        )

    def _freeze_for_linear_probe(self) -> None:
        """Freeze backbone, unfreeze classification head (lp_ft Phase 1)."""
        for attr in ("_model", "_backbone"):
            mod = getattr(self, attr, None)
            if mod is not None:
                for p in mod.parameters():
                    p.requires_grad = False

        # Unfreeze HuggingFace-style separate head
        hf_head = getattr(self, "_head", None)
        if hf_head is not None:
            for p in hf_head.parameters():
                p.requires_grad = True

        # Unfreeze braindecode-style head embedded in _model
        model = getattr(self, "_model", None)
        if model is not None:
            for container in (model, getattr(model, "base_model", None)):
                if container is None:
                    continue
                for head_name in ("final_layer", "classifier", "head"):
                    head_mod = getattr(container, head_name, None)
                    if head_mod is not None:
                        for p in head_mod.parameters():
                            p.requires_grad = True
                        return

    def _unfreeze_lora_adapters(self) -> None:
        """Unfreeze LoRA params for lp_ft Phase 2 (head already trainable)."""
        for attr in ("_model", "_backbone"):
            mod = getattr(self, attr, None)
            if mod is not None:
                for name, p in mod.named_parameters():
                    if "lora_" in name:
                        p.requires_grad = True

    # ------------------------------------------------------------------
    # Shared skorch training loop
    # ------------------------------------------------------------------

    def _fit_with_skorch(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params,
    ) -> "BackendBase":
        """Shared skorch training loop. Both real backends use this verbatim."""
        from skorch import NeuralNetClassifier, NeuralNetRegressor

        groups = fit_params.pop("groups", None)
        class_weight = fit_params.pop("class_weight", None)
        if self.signal_metadata_ is None:
            raise RuntimeError(
                "fit() requires signal_metadata declared at load() time via "
                "load(signal_metadata=...)."
            )

        out_dim = (
            int(np.unique(y).size)
            if self._task == "classification" and y is not None
            else 1
        )
        if out_dim and self._n_outputs != out_dim:
            self.reset_head(out_dim)
        net_cls = (
            NeuralNetClassifier
            if self._task == "classification"
            else NeuralNetRegressor
        )
        use_focal_loss = fit_params.pop("use_focal_loss", False)
        focal_gamma = float(fit_params.pop("focal_gamma", 2.0))

        net_kwargs = {}
        if net_cls is NeuralNetClassifier:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            weight_tensor = (
                torch.as_tensor(class_weight, dtype=torch.float32).to(self._device)
                if class_weight is not None
                else None
            )

            if use_focal_loss:
                # Define here so torch/nn are in scope.
                class _FocalLoss(nn.Module):
                    """FL(p_t) = -(1-p_t)^gamma * log(p_t), with optional class weights."""

                    def __init__(self, gamma: float = 2.0, weight=None):
                        super().__init__()
                        self.gamma = gamma
                        self.weight = weight

                    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
                        ce = F.cross_entropy(input, target, weight=self.weight, reduction="none")
                        p_t = torch.exp(-ce)
                        return ((1.0 - p_t) ** self.gamma * ce).mean()

                net_kwargs["criterion"] = _FocalLoss
                net_kwargs["criterion__gamma"] = focal_gamma
                if weight_tensor is not None:
                    net_kwargs["criterion__weight"] = weight_tensor
            else:
                net_kwargs["criterion"] = nn.CrossEntropyLoss
                if weight_tensor is not None:
                    net_kwargs["criterion__weight"] = weight_tensor
        validation_fraction = float(fit_params.get("validation_fraction", 0.2))
        if groups is not None and validation_fraction > 0:
            from sklearn.model_selection import (
                GroupShuffleSplit,
                StratifiedGroupKFold,
            )
            from skorch.dataset import Dataset
            from skorch.helper import predefined_split

            groups_arr = np.asarray(groups)
            y_arr = np.asarray(y)
            random_state = fit_params.get("random_state", 42)
            n_splits = max(2, int(round(1.0 / validation_fraction)))
            # Cap folds at the smallest number of distinct groups in any class.
            min_groups_per_class = min(
                np.unique(groups_arr[y_arr == cls]).size for cls in np.unique(y_arr)
            )
            n_splits = min(n_splits, int(min_groups_per_class))
            if n_splits >= 2:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=random_state,
                )
                train_idx, valid_idx = next(splitter.split(X, y, groups_arr))
            else:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=validation_fraction,
                    random_state=random_state,
                )
                train_idx, valid_idx = next(splitter.split(X, y, groups_arr))
            self._training_groups_ = np.unique(groups_arr[train_idx])
            self._validation_groups_ = np.unique(groups_arr[valid_idx])
            # X[fancy_index] copies the entire slice into RAM (up to 2× dataset
            # size when done twice for train and valid).  Use Subset instead:
            # it stores only the index list and defers array access to the
            # DataLoader, so only one batch is materialised at a time.
            import torch.utils.data as _tud
            full_ds = Dataset(X, y_arr)
            train_ds = _tud.Subset(full_ds, train_idx.tolist())
            valid_ds = _tud.Subset(full_ds, valid_idx.tolist())
            net_kwargs["train_split"] = predefined_split(valid_ds)
            X = train_ds
            y = y_arr[train_idx]  # y is tiny (one int per sample); copy is fine

        callbacks = []
        patience = fit_params.get("early_stopping_patience")
        if patience:
            from skorch.callbacks import EarlyStopping

            callbacks.append(
                EarlyStopping(
                    monitor="valid_loss",
                    patience=int(patience),
                    load_best=True,
                )
            )

        from skorch.callbacks import Callback, GradientNormClipping
        import logging as _logging
        _cb_log = _logging.getLogger(__name__)

        class _NaNGradientFilter(Callback):
            def on_grad_computed(self, net, named_parameters, **kwargs):
                import torch
                for _name, p in named_parameters:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        class _LoRAGradientCheck(Callback):
            """Fires once after the first backward pass to verify LoRA params got gradients."""
            def __init__(self):
                self._done = False

            def on_grad_computed(self, net, named_parameters, **kwargs):
                if self._done:
                    return
                self._done = True
                # Check lora_B only — lora_A always has zero gradient on the first
                # batch because B is initialized to 0, so dL/dA = B^T @ grad = 0.
                # That's expected; only lora_B going zero-gradient is a real problem.
                lora_b_params = [(n, p) for n, p in named_parameters if "lora_B" in n]
                if not lora_b_params:
                    all_lora = [(n, p) for n, p in named_parameters if "lora_" in n]
                    if not all_lora:
                        return  # not a LoRA run
                    _cb_log.warning("[LoRA] No lora_B params found — PEFT may not be injected correctly")
                    return
                no_grad = [n for n, p in lora_b_params if p.grad is None or p.grad.abs().max().item() == 0.0]
                has_grad = [n for n, p in lora_b_params if p.grad is not None and p.grad.abs().max().item() > 0.0]
                _cb_log.info(
                    "[LoRA] First-batch gradient check: %d/%d lora_B params have non-zero gradients",
                    len(has_grad), len(lora_b_params),
                )
                if no_grad:
                    _cb_log.warning(
                        "[LoRA] %d lora_B params received NO gradient (dead adapters):", len(no_grad)
                    )
                    for n in no_grad[:10]:
                        _cb_log.warning("  · %s", n)

        lr_warmup_epochs = int(fit_params.get("lr_warmup_epochs", 0)) if fit_params.get("lr_warmup", True) else 0
        max_epochs = int(fit_params.get("max_epochs", 10))

        class _LRWarmupCosineDecay(Callback):
            """Linear LR warmup followed by cosine decay.

            Epoch 1..warmup_epochs: lr scales linearly from 0 → base_lr.
            Epoch warmup_epochs+1..max_epochs: cosine decay from base_lr → min_lr.
            """
            def __init__(self, warmup_epochs_: int, max_epochs_: int, min_lr_ratio: float = 0.01):
                self.warmup_epochs = warmup_epochs_
                self.max_epochs = max_epochs_
                self.min_lr_ratio = min_lr_ratio
                self._base_lr = None
                self._epoch = 0

            def on_train_begin(self, net, **kwargs):
                self._base_lr = net.lr
                self._epoch = 0

            def on_epoch_begin(self, net, **kwargs):
                self._epoch += 1
                if self._base_lr is None or not hasattr(net, "optimizer_"):
                    return
                lr = self._compute_lr(self._epoch)
                for group in net.optimizer_.param_groups:
                    group["lr"] = lr
                _cb_log.debug("[LR] epoch %d/%d: lr=%.3e", self._epoch, self.max_epochs, lr)

            def _compute_lr(self, epoch: int) -> float:
                base = self._base_lr
                min_lr = base * self.min_lr_ratio
                if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                    return base * epoch / self.warmup_epochs
                decay_epochs = max(1, self.max_epochs - self.warmup_epochs)
                progress = (epoch - self.warmup_epochs) / decay_epochs
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
                return min_lr + cosine * (base - min_lr)

        callbacks.append(_NaNGradientFilter())
        callbacks.append(_LoRAGradientCheck())
        if lr_warmup_epochs > 0:
            callbacks.append(_LRWarmupCosineDecay(lr_warmup_epochs, max_epochs))
            _base_logger.info(
                "[LR] warmup+cosine schedule: %d warmup epochs, %d total epochs",
                lr_warmup_epochs, max_epochs,
            )

        grad_clip = fit_params.get("grad_clip_norm", None)
        if grad_clip is not None:
            callbacks.append(GradientNormClipping(gradient_clip_value=grad_clip))

        # Gradient accumulation: simulate larger batches without extra GPU memory.
        # effective_batch = batch_size × accumulate_grad_batches
        accumulate_grad_batches = int(fit_params.get("accumulate_grad_batches", 1))
        if accumulate_grad_batches > 1:
            net_cls = _make_grad_accum_cls(net_cls, accumulate_grad_batches)

        import torch

        training_strategy = fit_params.get("training_strategy", "ft_only")
        lp_epochs = int(fit_params.get("lp_epochs", 5))

        if training_strategy == "lp_ft":
            # ── Phase 1: Linear probe — head only, LoRA frozen ────────────
            # Freeze backbone LoRA params so the head trains alone. Head LoRA
            # params (e.g. luna's final_layer) are intentionally kept trainable:
            # models whose reset_head() is a no-op (luna) never add a fresh
            # post-PEFT linear layer, so their head lives inside the LoRA-wrapped
            # model and its adapters are the only trainable head weights.
            _HEAD_SUBSTRINGS = ("final_layer", "classifier", "head")
            lora_params_frozen: list = []
            for src in (getattr(self, "_model", None), getattr(self, "_backbone", None)):
                if src is not None:
                    for _n, _p in src.named_parameters():
                        if "lora_" in _n and _p.requires_grad:
                            if not any(h in _n for h in _HEAD_SUBSTRINGS):
                                _p.requires_grad = False
                                lora_params_frozen.append(_p)
                    break

            # Temporarily signal "frozen" so the skorch module's train() override
            # keeps the backbone in eval mode (no dropout / BN stat updates).
            _saved_train_mode = self._train_mode
            self._train_mode = "frozen"

            lp_lr = fit_params.get("lp_lr") or fit_params.get("lr", 1e-3)
            lp_net = net_cls(
                module=self._get_skorch_module(),
                module__backend=self,
                module__output_dim=out_dim,
                device=self._device,
                max_epochs=lp_epochs,
                lr=lp_lr,
                batch_size=fit_params.get("batch_size", 32),
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=fit_params.get("weight_decay", 0.01),
                callbacks=[_NaNGradientFilter()],
                **net_kwargs,
            )
            lp_net.fit(X, y)
            _base_logger.info(
                "[lp_ft] Phase 1 complete (%d epochs, lr=%.2e). "
                "Unfreezing %d LoRA params for phase 2.",
                lp_epochs, lp_lr, len(lora_params_frozen),
            )

            # ── Phase 2: LoRA fine-tune — head + LoRA adapters ────────────
            for _p in lora_params_frozen:
                _p.requires_grad = True
            self._train_mode = _saved_train_mode

        self._net_ = net_cls(
            module=self._get_skorch_module(),
            module__backend=self,
            module__output_dim=out_dim,
            device=self._device,
            max_epochs=fit_params.get("max_epochs", 10),
            lr=fit_params.get("lr", 1e-3),
            batch_size=fit_params.get("batch_size", 32),
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=fit_params.get("weight_decay", 0.01),
            callbacks=callbacks,
            **net_kwargs,
        )
        self._net_.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the fitted skorch network."""
        self._validate(X)
        if self._net_ is None:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return np.asarray(self._net_.predict_proba(X))

    def training_history(self) -> list:
        """Return the skorch training history (empty if not trained)."""
        return list(self._net_.history) if self._net_ is not None else []

    def _to_tensor(self, X: np.ndarray):
        import torch

        return torch.from_numpy(X).float().to(self._device)

    @staticmethod
    def _from_tensor(t) -> np.ndarray:
        return t.detach().cpu().numpy()

    @staticmethod
    def _no_grad():
        import torch

        return torch.no_grad()

    def _get_skorch_module(self):
        """Return the nn.Module class (not instance) to use with skorch.

        The returned class must accept ``(backend, output_dim)`` as constructor
        args and override ``train()`` to keep the frozen backbone in eval mode.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_skorch_module()."
        )

    def _set_backbone_eval(self) -> None:
        """Put the backbone in eval mode during frozen-mode head training.

        Called by the skorch module's train() override. No-op by default;
        backends with a backbone override this.
        """

    def _maybe_resample(self, X: np.ndarray) -> np.ndarray:
        """Resample X to the model's pretrained_sfreq when they differ.

        Uses polyphase resampling (scipy.signal.resample_poly) which is exact
        for rational frequency ratios and avoids the edge artefacts of FFT
        resampling.  Returns X unchanged when sfreq already matches.
        """
        meta = self.signal_metadata_
        if meta is None:
            return X
        model_sfreq = self.get_embedding_info().pretrained_sfreq
        if model_sfreq is None or meta.sfreq == model_sfreq:
            return X
        from math import gcd

        from scipy.signal import resample_poly

        data_hz = int(meta.sfreq)
        model_hz = int(model_sfreq)
        g = gcd(model_hz, data_hz)
        return resample_poly(X, up=model_hz // g, down=data_hz // g, axis=-1).astype(
            np.float32
        )

    def _validate(self, X: np.ndarray) -> None:
        """Validate X against stored signal_metadata_ and model constraints.

        sfreq mismatch is silently tolerated when the backend resamples
        automatically (i.e. _maybe_resample is called before this point).
        Channel count mismatch for fixed-channel models → ValueError.
        """
        meta = self.signal_metadata_
        if meta is None:
            raise RuntimeError(
                "Model must be fitted before transform/predict. Call fit() first."
            )
        model_meta = self.get_embedding_info()
        # sfreq mismatch is handled transparently by _maybe_resample()
        expected_n_chans = (
            self._expected_n_chans
            if self._expected_n_chans is not None
            else model_meta.pretrained_n_chans
        )
        if expected_n_chans is not None and X.shape[1] != expected_n_chans:
            raise ValueError(
                f"{model_meta.display_name} expects {expected_n_chans} "
                f"channels, got {X.shape[1]}."
            )
