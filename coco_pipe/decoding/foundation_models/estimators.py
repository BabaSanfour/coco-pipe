"""Clone-safe sklearn estimators backed by lazily loaded foundation models."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .._cache import make_feature_cache_key
from ._prepare import prepare_backend

_FROZEN_EMBEDDING_CACHE: dict[str, np.ndarray] = {}


def clear_frozen_embedding_cache() -> None:
    """Empty the shared frozen-backbone embedding cache."""
    _FROZEN_EMBEDDING_CACHE.clear()


class FrozenBackboneTransformer(BaseEstimator, TransformerMixin):
    """Target-independent frozen feature extractor suitable for sklearn pipelines."""

    backend_ = None
    prepared_ = None

    def __init__(
        self,
        model_key: str,
        backend: str = "auto",
        device: str = "auto",
        pooling: str = "mean",
        sfreq: float | None = None,
        ch_names: list[str] | None = None,
        cache_embeddings: bool = False,
        backend_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_key = model_key
        self.backend = backend
        self.device = device
        self.pooling = pooling
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.cache_embeddings = cache_embeddings
        self.backend_kwargs = backend_kwargs

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        self.prepared_ = prepare_backend(
            self.model_key,
            backend=self.backend,
            X=X,
            n_outputs=None,
            device=self.device,
            train_mode="frozen",
            pooling=self.pooling,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            backend_kwargs=self.backend_kwargs,
        )
        self.backend_ = self.prepared_.backend
        return self

    def _fingerprint(self) -> str:
        """Stable identity of this deterministic window->embedding mapping."""
        return "|".join(
            str(part)
            for part in (
                self.model_key,
                self.pooling,
                float(getattr(self.prepared_, "target_sfreq", 0.0)),
                tuple(getattr(self.prepared_, "ch_names", ()) or ()),
            )
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        adapted = self.prepared_.adapt(X)
        if not self.cache_embeddings:
            values = np.asarray(self.backend_.transform(adapted))
            return values.reshape(len(values), -1)

        fingerprint = self._fingerprint()
        keys = [
            make_feature_cache_key(
                [hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()],
                [],
                "frozen_backbone",
                fingerprint,
                sort_ids=False,
            )
            for row in adapted
        ]
        missing = [
            i for i, key in enumerate(keys) if key not in _FROZEN_EMBEDDING_CACHE
        ]
        if missing:
            computed = np.asarray(self.backend_.transform(adapted[missing]))
            computed = computed.reshape(len(computed), -1)
            for offset, index in enumerate(missing):
                _FROZEN_EMBEDDING_CACHE[keys[index]] = computed[offset]
        return np.stack([_FROZEN_EMBEDDING_CACHE[key] for key in keys])


class FoundationClassifier(BaseEstimator, ClassifierMixin):
    """Lazy trainable foundation-model classifier with grouped validation."""

    backend_ = None
    prepared_ = None
    checkpoint_path_ = None

    def __init__(
        self,
        model_key: str,
        backend: str = "auto",
        train_mode: str = "full",
        device: str = "auto",
        n_outputs: int | None = None,
        sfreq: float | None = None,
        ch_names: list[str] | None = None,
        trainer: dict[str, Any] | None = None,
        lora: dict[str, Any] | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        checkpoints: dict[str, Any] | None = None,
        class_weight: str | dict[Any, float] | None = "balanced",
        random_state: int = 42,
    ) -> None:
        self.model_key = model_key
        self.backend = backend
        self.train_mode = train_mode
        self.device = device
        self.n_outputs = n_outputs
        self.sfreq = sfreq
        self.ch_names = ch_names
        self.trainer = trainer
        self.lora = lora
        self.backend_kwargs = backend_kwargs
        self.checkpoints = checkpoints
        self.class_weight = class_weight
        self.random_state = random_state

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray | None = None,
    ):
        import torch

        # Seed torch so random_state controls head init, dropout, and batch
        # shuffling — not just the group split.
        torch.manual_seed(self.random_state)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.n_outputs is not None and self.n_outputs != len(self.classes_):
            raise ValueError(
                f"n_outputs={self.n_outputs} must equal the number of classes "
                f"({len(self.classes_)})."
            )
        n_outputs = self.n_outputs or len(self.classes_)
        class_weights = None
        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_class_weight

            class_weights = compute_class_weight("balanced", classes=self.classes_, y=y)
        elif isinstance(self.class_weight, dict):
            class_weights = np.asarray(
                [self.class_weight.get(c, 1.0) for c in self.classes_],
                dtype=float,
            )
        train_mode = "frozen" if self.train_mode == "linear_probe" else self.train_mode
        backend_kwargs = dict(self.backend_kwargs or {})
        if self.lora:
            backend_kwargs.update(
                {
                    "lora_r": self.lora.get("r", 16),
                    "lora_alpha": self.lora.get("alpha", 32),
                    "lora_dropout": self.lora.get("dropout", 0.0),
                    "lora_target_modules": self.lora.get(
                        "target_modules", "all-linear"
                    ),
                }
            )
        self.prepared_ = prepare_backend(
            self.model_key,
            X=X,
            backend=self.backend,
            n_outputs=n_outputs,
            device=self.device,
            train_mode=train_mode,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            backend_kwargs=backend_kwargs,
        )
        self.backend_ = self.prepared_.backend
        trainer = dict(self.trainer or {})
        self.backend_.fit(
            self.prepared_.adapt(X),
            y,
            groups=groups,
            class_weight=class_weights,
            random_state=self.random_state,
            max_epochs=trainer.get("max_epochs", 10),
            batch_size=trainer.get("batch_size", 32),
            validation_fraction=trainer.get("validation_fraction", 0.2),
            lr=trainer.get("lr", 1e-3),
            early_stopping_patience=trainer.get("early_stopping_patience"),
        )
        self.checkpoint_path_ = None
        checkpoint = dict(self.checkpoints or {})
        if checkpoint.get("save", "none") != "none" and checkpoint.get("output_dir"):
            import torch

            output_dir = Path(checkpoint["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint_path_ = output_dir / (
                f"{self.model_key}_{self.train_mode}_{uuid4().hex[:12]}.pt"
            )
            components = self._checkpoint_components()
            if components:
                torch.save(
                    {
                        "format_version": 1,
                        "model_key": self.model_key,
                        "train_mode": self.train_mode,
                        "components": {
                            name: component.state_dict()
                            for name, component in components.items()
                        },
                    },
                    self.checkpoint_path_,
                )
        return self

    def _checkpoint_components(self) -> dict[str, Any]:
        """Return the backend's restorable torch modules (empty if unfitted)."""
        return (
            self.backend_.checkpoint_components() if self.backend_ is not None else {}
        )

    def restore_checkpoint(self, path: str | Path | None = None):
        """Restore every saved backend component into this fitted estimator."""
        import torch

        check_is_fitted(self)
        checkpoint_path = Path(path or self.checkpoint_path_ or "")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        payload = torch.load(
            checkpoint_path,
            map_location=self.backend_.device,
            weights_only=True,
        )
        components = self._checkpoint_components()
        if "components" not in payload:
            if len(components) != 1:
                raise ValueError(
                    "Legacy single-module checkpoint cannot restore a "
                    "multi-component backend."
                )
            next(iter(components.values())).load_state_dict(payload)
            return self
        missing = sorted(set(payload["components"]) - set(components))
        if missing:
            raise ValueError(f"Checkpoint components are unavailable: {missing}")
        for name, state in payload["components"].items():
            components[name].load_state_dict(state)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        indices = np.asarray(self.backend_.predict(self.prepared_.adapt(X)))
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        return np.asarray(self.backend_.predict_proba(self.prepared_.adapt(X)))

    def get_training_history(self) -> list[dict[str, Any]]:
        return self.backend_.training_history() if self.backend_ is not None else []

    def get_checkpoint_manifest(self) -> dict[str, Any]:
        return {
            "checkpoint_path": (
                str(self.checkpoint_path_)
                if self.checkpoint_path_ is not None
                else None
            )
        }

    def get_model_card_info(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "backend": self.backend,
            "train_mode": self.train_mode,
            "sfreq": self.sfreq,
            "ch_names": self.ch_names,
            "trainer": dict(self.trainer or {}),
            "lora": dict(self.lora or {}),
        }

    def get_failure_diagnostics(self) -> dict[str, Any]:
        return {}

    def get_artifact_metadata(self) -> dict[str, Any]:
        return {
            "history": self.get_training_history(),
            "checkpoints": self.get_checkpoint_manifest(),
            "model_card": self.get_model_card_info(),
            "failures": self.get_failure_diagnostics(),
        }
