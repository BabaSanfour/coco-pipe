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
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec, SignalMetadata


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
        net_kwargs = {}
        if net_cls is NeuralNetClassifier:
            import torch.nn as nn

            net_kwargs["criterion"] = nn.CrossEntropyLoss
            if class_weight is not None:
                import torch

                net_kwargs["criterion__weight"] = torch.as_tensor(
                    class_weight, dtype=torch.float32
                ).to(self._device)
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
            valid_ds = Dataset(X[valid_idx], np.asarray(y)[valid_idx])
            X = X[train_idx]
            y = np.asarray(y)[train_idx]
            net_kwargs["train_split"] = predefined_split(valid_ds)

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

        self._net_ = net_cls(
            module=self._get_skorch_module(),
            module__backend=self,
            module__output_dim=out_dim,
            device=self._device,
            max_epochs=fit_params.get("max_epochs", 10),
            lr=fit_params.get("lr", 1e-3),
            batch_size=fit_params.get("batch_size", 32),
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

    def _validate(self, X: np.ndarray) -> None:
        """Validate X against stored signal_metadata_ and model constraints.

        sfreq mismatch → UserWarning (allows intentional mismatch for research).
        Channel count mismatch for fixed-channel models → ValueError.
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
