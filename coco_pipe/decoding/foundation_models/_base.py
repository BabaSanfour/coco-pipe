"""
Foundation Model Base Backend
==============================

Abstract base class shared by all EEG/MEG foundation model backends.
Backends ARE the sklearn estimator — no wrapping layer is needed.
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
    """Abstract base for all FM backends. The backend IS the sklearn estimator.

    Training modes
    --------------
    frozen   — backbone weights frozen; only the head trains (head-only fine-tune).
    full     — all weights train end-to-end.
    lora     — peft LoRA adapters injected; only adapter weights train.
    qlora    — 4-bit quantized backbone + LoRA (hugging_face backend only).
    """

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

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.
        y : np.ndarray of shape (n_samples,) or None
            Target labels (classification) or continuous values (regression).
        **fit_params
            Recognised extra keys:

            signal_metadata : SignalMetadata
                Binds ``sfreq`` and ``ch_names`` for subsequent validation in
                ``transform`` / ``predict``. Falls back to the model's
                pretrained defaults when omitted.
            max_epochs : int, default 10
            lr : float, default 1e-3
            batch_size : int, default 32

        Returns
        -------
        self : BackendBase
            The fitted backend instance.
        """
        return self._fit_with_skorch(X, y, **fit_params)

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

        from .._specs import SignalMetadata

        signal_metadata = fit_params.pop("signal_metadata", None)
        if signal_metadata is None:
            signal_metadata = SignalMetadata(
                sfreq=self._metadata.pretrained_sfreq,
                ch_names=[f"ch{i}" for i in range(X.shape[1])],
            )
        self.signal_metadata_ = signal_metadata

        out_dim = (
            int(np.unique(y).size)
            if getattr(self, "_task", "classification") == "classification"
            and y is not None
            else 1
        )
        net_cls = (
            NeuralNetClassifier
            if getattr(self, "_task", "classification") == "classification"
            else NeuralNetRegressor
        )
        self._net_ = net_cls(
            module=self._get_skorch_module(),
            module__backend=self,
            module__output_dim=out_dim,
            device=getattr(self, "_device", "cpu"),
            max_epochs=fit_params.get("max_epochs", 10),
            lr=fit_params.get("lr", 1e-3),
            batch_size=fit_params.get("batch_size", 32),
        )
        self._net_.fit(X, y)
        return self

    def _to_tensor(self, X: np.ndarray):
        import torch

        return torch.from_numpy(X).float().to(getattr(self, "_device", "cpu"))

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
        meta: SignalMetadata | None = getattr(self, "signal_metadata_", None)
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
        if (
            model_meta.pretrained_n_chans is not None
            and X.shape[1] != model_meta.pretrained_n_chans
        ):
            raise ValueError(
                f"{model_meta.display_name} expects {model_meta.pretrained_n_chans} "
                f"channels, got {X.shape[1]}."
            )
