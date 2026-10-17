"""
Foundation Model Base Backend
==============================

Abstract base class shared by all EEG/MEG foundation model backends.
Backends can be used directly. Clone-safe adapters wrap them inside sklearn
pipelines and cross-validation.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec, SignalMetadata


# Largest absolute value below which microvolt EEG is implausible.
_MICROVOLT_MIN_PLAUSIBLE_PEAK = 1e-2


def check_finite(X: np.ndarray, name: str) -> None:
    """Raise when a window holds NaN or infinite values, naming the windows.

    A model turns such a window into a NaN embedding, which L2 normalisation
    or pooling can hide.
    """
    bad = np.flatnonzero(~np.isfinite(X).all(axis=tuple(range(1, X.ndim))))
    if bad.size:
        raise ValueError(
            f"{name}: {bad.size} of {len(X)} windows hold NaN or infinite values "
            f"(windows {bad[:10].tolist()}{', ...' if bad.size > 10 else ''})."
        )


def resolve_revision(repo_id: str, revision: str | None = None, **hub_kwargs) -> str:
    """Return the commit of ``repo_id`` that ``revision`` points to.

    ``None`` is the latest commit. The commit is read from the snapshot that
    holds the repository's ``config.json``, so it resolves offline from the
    local cache as well.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id, "config.json", revision=revision, **hub_kwargs)
    return Path(path).parent.name


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
    _channel_adapter = None
    _device: str = "cpu"
    _n_outputs: int | None = None
    _expected_n_chans: int | None = None
    signal_metadata_: SignalMetadata | None = None
    _net_ = None
    _model = None
    # The commit loaded from each Hugging Face repository, set by ``load``.
    checkpoint_revisions: Mapping[str, str] = MappingProxyType({})

    def release_memory(self) -> None:
        """Drop references to loaded torch modules so GC can reclaim them.

        Called after a fold is scored to keep peak memory tracking one fitted
        backbone at a time rather than accumulating across the CV loop. The
        backend is unusable afterwards; callers must not predict/transform with
        it once released.
        """
        self._model = None
        self._net_ = None

    def head_module(self) -> Any:
        """Return the output head, the part that a probe phase trains alone."""
        model = getattr(self, "_model", None)
        candidates = [getattr(self, "_head", None)] + [
            getattr(model, name, None) for name in ("final_layer", "classifier", "head")
        ]
        for candidate in candidates:
            if candidate is not None:
                return candidate
        raise NotImplementedError(f"{type(self).__name__} does not expose its head.")

    @abstractmethod
    def reset_head(self, n_outputs: int) -> BackendBase:
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

    @abstractmethod
    def get_embedding_info(self) -> FoundationModelSpec:
        """Return the registry spec for this model instance.

        Returns
        -------
        spec : FoundationModelSpec
            The spec describing this model's provenance and signal expectations.
        """

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Check whether all required packages for this backend are installed.

        Returns
        -------
        available : bool
            ``True`` if the backend's dependencies are importable.
        """

    @classmethod
    @abstractmethod
    def load(
        cls,
        model_key: str,
        metadata: FoundationModelSpec,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        **backend_kwargs,
    ) -> BackendBase:
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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params,
    ) -> BackendBase:
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
            weight_decay : float, default 0.01
            batch_size : int, default 32
            lr_warmup_epochs : int, default 0
            accumulate_grad_batches : int, default 1
            focal_gamma : float or None, default None
            probe_epochs : int, default 0
            probe_lr : float or None, default None

        Returns
        -------
        self : BackendBase
            The fitted backend instance.
        """
        self._validate(X)
        return self._fit_with_skorch(self._native_input(X), y, **fit_params)

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
    def transform(
        self, X: np.ndarray, *, return_tokens: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
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

        When ``return_tokens`` is true, return ``(embeddings, tokens)``. The
        token array is the native backbone feature tensor with only its device
        transport changed (detached, moved to CPU, and converted to NumPy).
        """

    def get_token_output_metadata(self) -> dict[str, object]:
        """Describe the native tensor returned when token output is requested.

        Backends that support native token extraction must override this method
        and explicitly name the tensor source and every axis. No token layout is
        inferred from array rank.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a native token representation."
        )

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

    def configure_peft(self, lora_config: dict) -> BackendBase:
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

    # ------------------------------------------------------------------
    # Shared skorch training loop
    # ------------------------------------------------------------------

    def _fit_with_skorch(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        **fit_params,
    ) -> BackendBase:
        """Shared skorch training loop. Both real backends use this verbatim."""
        from functools import partial

        import torch
        from skorch import NeuralNetClassifier, NeuralNetRegressor

        from . import _training

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
        net_kwargs = {}
        if net_cls is NeuralNetClassifier:
            import torch.nn as nn

            net_kwargs["criterion"] = nn.CrossEntropyLoss
            if fit_params.get("focal_gamma") is not None:
                net_kwargs["criterion"] = _training.FocalLoss
                net_kwargs["criterion__gamma"] = float(fit_params["focal_gamma"])
            if class_weight is not None:
                import torch

                net_kwargs["criterion__weight"] = torch.as_tensor(
                    class_weight, dtype=torch.float32
                ).to(self._device)
        validation_fraction = float(fit_params.get("validation_fraction", 0.2))
        if validation_fraction <= 0:
            net_kwargs["train_split"] = None
        elif groups is None:
            from skorch.dataset import ValidSplit

            net_kwargs["train_split"] = ValidSplit(
                validation_fraction,
                stratified=self._task == "classification",
                random_state=fit_params.get("random_state", 42),
            )
        else:
            from sklearn.model_selection import (
                GroupShuffleSplit,
                StratifiedGroupKFold,
            )
            from skorch.dataset import Dataset
            from skorch.helper import predefined_split
            from torch.utils.data import Subset

            groups_arr = np.asarray(groups)
            y_arr = np.asarray(y)
            random_state = fit_params.get("random_state", 42)
            n_splits = max(2, round(1.0 / validation_fraction))
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
            full_ds = Dataset(X, y_arr)
            valid_ds = Subset(full_ds, valid_idx.tolist())
            X = Subset(full_ds, train_idx.tolist())
            y = y_arr[train_idx]
            net_kwargs["train_split"] = predefined_split(valid_ds)

        callbacks = []
        patience = fit_params.get("early_stopping_patience")
        if patience and validation_fraction <= 0:
            raise ValueError(
                "early_stopping_patience watches the validation loss: it needs "
                "validation_fraction > 0."
            )
        if patience:
            from skorch.callbacks import EarlyStopping

            callbacks.append(
                EarlyStopping(
                    monitor="valid_loss",
                    patience=int(patience),
                    load_best=True,
                )
            )

        max_epochs = int(fit_params.get("max_epochs", 10))
        lr = fit_params.get("lr", 1e-3)
        warmup_epochs = int(fit_params.get("lr_warmup_epochs", 0))
        if warmup_epochs:
            from skorch.callbacks import LRScheduler

            callbacks.append(
                LRScheduler(
                    policy=torch.optim.lr_scheduler.LambdaLR,
                    lr_lambda=partial(
                        _training.warmup_cosine,
                        warmup_epochs=warmup_epochs,
                        max_epochs=max_epochs,
                    ),
                )
            )
        accumulate = int(fit_params.get("accumulate_grad_batches", 1))
        if accumulate > 1:
            net_cls = (
                _training.AccumulatingClassifier
                if self._task == "classification"
                else _training.AccumulatingRegressor
            )
            net_kwargs["accumulate_grad_batches"] = accumulate

        def build_net(max_epochs, lr, callbacks):
            return net_cls(
                module=self._get_skorch_module(),
                module__backend=self,
                module__output_dim=out_dim,
                device=self._device,
                max_epochs=max_epochs,
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=fit_params.get("weight_decay", 0.01),
                lr=lr,
                batch_size=fit_params.get("batch_size", 32),
                # skorch does not shuffle by default, and trials arrive ordered
                # by condition and subject: batches of one class collapse the fit
                iterator_train__shuffle=True,
                callbacks=callbacks,
                **net_kwargs,
            )

        probe_epochs = int(fit_params.get("probe_epochs", 0))
        if probe_epochs:
            # Train the head alone first: everything else that would train is
            # frozen for this phase, then released for the training below.
            probe = build_net(probe_epochs, fit_params.get("probe_lr") or lr, [])
            probe.initialize()
            head = {id(parameter) for parameter in self.head_module().parameters()}
            frozen = [
                parameter
                for parameter in probe.module_.parameters()
                if parameter.requires_grad and id(parameter) not in head
            ]
            for parameter in frozen:
                parameter.requires_grad = False
            try:
                probe.partial_fit(X, y)
            finally:
                for parameter in frozen:
                    parameter.requires_grad = True

        self._net_ = build_net(max_epochs, lr, callbacks)
        self._net_.fit(X, y)
        return self

    def _native_input(self, X: np.ndarray) -> np.ndarray:
        """Bring validated microvolt windows to the input the model was trained on.

        Maps the source montage to the model's layout when the backend has a
        channel adapter, then applies the amplitude convention the spec
        declares (``FoundationModelSpec.input_*``).
        """
        spec = self._metadata
        if self._channel_adapter is not None:
            X = self._channel_adapter(X)
        if spec.input_reference == "average":
            X = X - X.mean(axis=1, keepdims=True)
        if self._train_mode != "frozen" and spec.finetune_input_scale is not None:
            return (X * spec.finetune_input_scale).astype(np.float32)
        X = X * spec.input_scale
        if spec.input_normalization == "quantile95":
            X = X / (np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True) + 1e-8)
        elif spec.input_normalization == "zscore":
            # sample standard deviation, as torch.std in the authors' code
            X = (X - X.mean(axis=-1, keepdims=True)) / (
                X.std(axis=-1, ddof=1, keepdims=True) + 1e-8
            )
        if spec.input_clip is not None:
            X = np.clip(X, -spec.input_clip, spec.input_clip)
        return X.astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities from the fitted skorch network."""
        self._validate(X)
        if self._net_ is None:
            raise RuntimeError("Model must be fitted before predict_proba().")
        return np.asarray(self._net_.predict_proba(self._native_input(X)))

    def training_history(self) -> list:
        """Return the skorch training history (empty if not trained)."""
        return list(self._net_.history) if self._net_ is not None else []

    def _eval_batch_size(self) -> int:
        """Batch size for inference, mirroring the batch size used at fit time.

        ``predict``/``transform`` otherwise run the whole fold through the
        model in one forward pass, which OOMs on large test folds even though
        training (via skorch) and ``predict_proba`` are already batched.
        """
        batch_size = getattr(self._net_, "batch_size", None)
        return int(batch_size) if batch_size else 32

    def _to_tensor(self, X: np.ndarray):
        import torch

        return torch.from_numpy(X).float().to(self._device)

    @staticmethod
    def _from_tensor(t) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _no_grad(self):
        """Enter inference: evaluation mode (a fit can leave dropout on), no grad."""
        import torch

        for module in self.checkpoint_components().values():
            module.eval()
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

    def _validate(self, X: np.ndarray) -> None:
        """Validate X against stored signal_metadata_ and model constraints.

        sfreq mismatch → UserWarning (allows intentional mismatch for research).
        Channel count mismatch for fixed-channel models → ValueError.
        NaN or infinite values → ValueError.
        Volt-scale input for a model that expects microvolts → ValueError.
        """
        meta = self.signal_metadata_
        if meta is None:
            raise RuntimeError(
                "Model must be fitted before transform/predict. Call fit() first."
            )
        model_meta = self.get_embedding_info()
        if meta.sfreq != model_meta.pretrained_sfreq:
            warnings.warn(
                f"{model_meta.display_name}: input sfreq ({meta.sfreq} Hz) != "
                f"pretrained sfreq ({model_meta.pretrained_sfreq} Hz). "
                f"Consider resampling.",
                UserWarning,
                stacklevel=3,
            )
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
        check_finite(X, model_meta.display_name)
        if model_meta.expects_microvolts and X.size:
            peak = max(float(X.max()), -float(X.min()))
            # peak == 0 is all-zero synthetic data, not a units mistake.
            if 0.0 < peak < _MICROVOLT_MIN_PLAUSIBLE_PEAK:
                raise ValueError(
                    f"{model_meta.display_name} expects EEG in microvolts, but the "
                    f"largest absolute value of the input is {peak:.3g}: this looks "
                    "like volts. Load the data in microvolts, e.g. "
                    "`epochs.get_data(units='uV')`."
                )
