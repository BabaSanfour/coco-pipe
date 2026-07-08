import os

import numpy as np
import pytest
from sklearn.base import clone

from coco_pipe.decoding import (
    CVConfig,
    Experiment,
    ExperimentConfig,
    FoundationEmbeddingModelConfig,
    FrozenBackboneDecoderConfig,
    NeuralFineTuneConfig,
)
from coco_pipe.decoding.configs import ClassicalModelConfig, TrainerConfig
from coco_pipe.decoding.foundation_models import (
    FoundationClassifier,
    FrozenBackboneTransformer,
    clear_frozen_embedding_cache,
    register_backend,
    unregister_backend,
)
from coco_pipe.decoding.foundation_models._base import BackendBase
from coco_pipe.decoding.foundation_models._braindecode import BrainDecodeBackend
from coco_pipe.decoding.foundation_models._hugging_face import HuggingFaceBackend
from coco_pipe.decoding.foundation_models.validation import (
    validate_real_checkpoints,
    validate_real_training,
)


class FakeFoundationBackend(BackendBase):
    """Deterministic in-test fake implementing the full foundation contract.

    Defined here, not in the shipped package, so coco-pipe ships no test double.
    Lets these tests exercise FoundationClassifier / FrozenBackboneTransformer /
    Experiment end-to-end without real checkpoints or network.
    """

    def __init__(self, metadata, model, n_outputs, device, train_mode):
        self._metadata = metadata
        self._model = model
        self._n_outputs = n_outputs
        self._device = device
        self._train_mode = train_mode
        self._task = "classification"
        self._feat_dim = 4
        self.signal_metadata_ = None

    @classmethod
    def is_available(cls):
        return True

    @classmethod
    def load(cls, model_key, metadata, n_outputs, device, train_mode, **backend_kwargs):
        import torch.nn as nn

        n_chans = len(backend_kwargs.get("electrode_names") or ["ch0", "ch1"])

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Linear(n_chans, 4)
                self.head = nn.Linear(4, n_outputs or 2)

            def features(self, X):
                return self.backbone(X.mean(dim=-1))

            def forward(self, X):
                return self.head(self.features(X))

        model = TinyModel().to(device)
        if train_mode == "frozen":
            for parameter in model.backbone.parameters():
                parameter.requires_grad = False
        return cls(metadata, model, n_outputs, device, train_mode)

    def reset_head(self, n_outputs):
        import torch.nn as nn

        self._model.head = nn.Linear(4, n_outputs).to(self._device)
        self._n_outputs = n_outputs
        return self

    def get_embedding_info(self):
        return self._metadata

    def _get_skorch_module(self):
        import torch.nn as nn

        class Module(nn.Module):
            def __init__(self, backend, output_dim):
                super().__init__()
                self.model = backend._model

            def forward(self, X):
                return self.model(X)

        return Module

    def transform(self, X):
        self._validate(X)
        with self._no_grad():
            return self._from_tensor(self._model.features(self._to_tensor(X)))

    def checkpoint_components(self):
        return {"model": self._model}

    def predict(self, X):
        if self._net_ is not None:
            return np.asarray(self._net_.predict(X))
        with self._no_grad():
            logits = self._model(self._to_tensor(X))
        return self._from_tensor(logits.argmax(dim=-1))


def setup_module():
    register_backend("fake", FakeFoundationBackend, overwrite=True)


def teardown_module():
    unregister_backend("fake")


def _data():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(16, 2, 20)).astype("float32")
    y = np.repeat([0, 1], 8)
    groups = np.array([f"p{index:02d}" for index in range(8) for _ in range(2)])
    return X, y, groups


def test_real_backend_skorch_wrappers_register_torch_parameters():
    import torch.nn as nn

    brain_backend = type("BrainBackend", (), {})()
    brain_backend._model = nn.Linear(4, 2)
    brain_backend._train_mode = "full"
    brain_backend._set_backbone_eval = lambda: None
    brain_module = BrainDecodeBackend._get_skorch_module(brain_backend)(
        brain_backend,
        2,
    )
    assert sum(parameter.numel() for parameter in brain_module.parameters()) > 0

    hf_backend = type("HFBackend", (), {})()
    hf_backend._backbone = nn.Linear(4, 4)
    hf_backend._pos_bank = nn.Linear(4, 4)
    hf_backend._head = nn.Linear(4, 2)
    hf_backend._train_mode = "full"
    hf_backend._reve_forward = lambda X, return_embeddings: hf_backend._head(
        hf_backend._backbone(X)
    )
    hf_module = HuggingFaceBackend._get_skorch_module(hf_backend)(hf_backend, 2)
    assert sum(parameter.numel() for parameter in hf_module.parameters()) > 0


@pytest.mark.parametrize("train_mode", ["linear_probe", "full", "lora"])
def test_clone_safe_transformer_and_classifier(tmp_path, train_mode):
    X, y, groups = _data()
    transformer = FrozenBackboneTransformer(
        "cbramod",
        backend="fake",
        sfreq=200,
        ch_names=["C3", "C4"],
    )
    cloned_transformer = clone(transformer).fit(X, y)
    assert cloned_transformer.transform(X).shape == (16, 4)

    classifier = FoundationClassifier(
        "cbramod",
        backend="fake",
        train_mode=train_mode,
        sfreq=200,
        ch_names=["C3", "C4"],
        trainer={
            "max_epochs": 2,
            "batch_size": 4,
            "validation_fraction": 0.25,
        },
        checkpoints={"save": "best", "output_dir": tmp_path},
    )
    fitted = clone(classifier).fit(X, y, groups=groups)
    probabilities = fitted.predict_proba(X)
    assert probabilities.shape == (16, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert fitted.get_training_history()
    assert fitted.checkpoint_path_.exists()
    import torch

    expected = {
        name: {
            key: value.detach().clone() for key, value in component.state_dict().items()
        }
        for name, component in fitted._checkpoint_components().items()
    }
    with torch.no_grad():
        for component in fitted._checkpoint_components().values():
            for parameter in component.parameters():
                parameter.add_(1)
    fitted.restore_checkpoint()
    for name, component in fitted._checkpoint_components().items():
        for key, value in component.state_dict().items():
            torch.testing.assert_close(value, expected[name][key])
    assert not set(fitted.backend_._training_groups_) & set(
        fitted.backend_._validation_groups_
    )


def test_release_memory_tears_down_fitted_classifier():
    """release_memory drops the fitted backend and its torch modules so the
    garbage collector can reclaim them between CV folds (#2)."""
    X, y, groups = _data()
    classifier = FoundationClassifier(
        "cbramod",
        backend="fake",
        train_mode="linear_probe",
        sfreq=200,
        ch_names=["C3", "C4"],
        trainer={"max_epochs": 1, "batch_size": 4, "validation_fraction": 0.25},
    ).fit(X, y, groups=groups)

    backend = classifier.backend_
    assert backend is not None and backend._model is not None

    classifier.release_memory()

    # Estimator no longer references the backend or the adapted-input prep.
    assert classifier.backend_ is None
    assert classifier.prepared_ is None
    # And the backend itself has dropped its torch modules.
    assert backend._model is None
    assert backend._net_ is None


def test_release_memory_tears_down_frozen_transformer():
    """FrozenBackboneTransformer exposes the same teardown hook."""
    X, y, _ = _data()
    transformer = FrozenBackboneTransformer(
        "cbramod", backend="fake", sfreq=200, ch_names=["C3", "C4"]
    ).fit(X, y)

    backend = transformer.backend_
    assert backend is not None

    transformer.release_memory()
    assert transformer.backend_ is None
    assert backend._model is None


def test_backend_base_release_memory_is_idempotent():
    """BackendBase.release_memory can run on an already-released backend without
    error (teardown must be best-effort)."""
    backend = FakeFoundationBackend.load(
        "cbramod", metadata={}, n_outputs=2, device="cpu", train_mode="frozen"
    )
    backend.release_memory()
    backend.release_memory()  # second call must not raise
    assert backend._model is None and backend._net_ is None


def test_frozen_backbone_embedding_cache_matches_uncached_and_memoizes():
    X, _, _ = _data()
    clear_frozen_embedding_cache()
    cached = FrozenBackboneTransformer(
        "cbramod",
        backend="fake",
        sfreq=200,
        ch_names=["C3", "C4"],
        cache_embeddings=True,
    ).fit(X, None)

    # Ground truth from the SAME fitted backend, bypassing the cache (the fake
    # backbone has random weights, so a second instance would differ).
    expected = np.asarray(cached.backend_.transform(cached.prepared_.adapt(X))).reshape(
        len(X), -1
    )

    call_count = {"n": 0}
    real_transform = cached.backend_.transform

    def counting_transform(batch):
        call_count["n"] += len(batch)
        return real_transform(batch)

    cached.backend_.transform = counting_transform

    first = cached.transform(X)
    assert call_count["n"] == len(X)  # every window is a cache miss the first pass
    np.testing.assert_allclose(first, expected, rtol=1e-6)

    second = cached.transform(X)
    assert call_count["n"] == len(X)  # second pass served entirely from the cache
    np.testing.assert_allclose(second, expected, rtol=1e-6)
    clear_frozen_embedding_cache()


def test_experiment_dispatches_frozen_and_trainable_paths(tmp_path):
    X, y, groups = _data()
    frozen = FrozenBackboneDecoderConfig(
        backbone=FoundationEmbeddingModelConfig(
            model_key="cbramod",
            backend="fake",
            sfreq=200,
            ch_names=["C3", "C4"],
        ),
        head=ClassicalModelConfig(
            estimator="LogisticRegression",
            params={"max_iter": 200},
        ),
    )
    neural = NeuralFineTuneConfig(
        model_key="cbramod",
        backend="fake",
        train_mode="linear_probe",
        sfreq=200,
        ch_names=["C3", "C4"],
        trainer=TrainerConfig(
            max_epochs=1,
            batch_size=4,
            validation_fraction=0.25,
        ),
    )
    config = ExperimentConfig(
        models={"frozen": frozen, "neural": neural},
        cv=CVConfig(
            strategy="stratified_group_kfold",
            n_splits=2,
            group_key="patient_group_id",
        ),
        metrics=["accuracy"],
        n_jobs=1,
    )
    metadata = {
        "subject": groups,
        "session": ["01"] * len(groups),
        "patient_group_id": groups,
    }
    result = Experiment(config).run(
        X,
        y,
        groups=groups,
        sample_metadata=metadata,
    )
    assert set(result.summary().index) == {"frozen", "neural"}


def test_gated_checkpoint_without_token_reports_authentication_required(
    monkeypatch,
):
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)
    records = validate_real_checkpoints(["reve"])
    assert records == [
        {
            "model_key": "reve",
            "status": "authentication_required",
            "expected_embedding_dim": 512,
            "checkpoint": "brain-bzh/reve-base",
            "checkpoint_revision": "fa9a2163a4b7c0a42c8e28b56077ef9c368944dc",
            "checkpoint_filename": None,
            "reason": (
                "Run `hf auth login` or set HF_TOKEN after accepting the "
                "gated-model license."
            ),
        }
    ]


@pytest.mark.real_checkpoints
@pytest.mark.skipif(
    os.environ.get("COCO_PIPE_RUN_REAL_FOUNDATION") != "1",
    reason="requires network access and real foundation checkpoints",
)
def test_real_foundation_checkpoints():
    records = validate_real_checkpoints()
    assert records
    assert all(
        record["status"]
        in {"verified", "authentication_required", "failed", "unsupported"}
        for record in records
    )
    assert all(
        record["status"] == "verified"
        for record in records
        if record["model_key"] != "reve"
    )


@pytest.mark.real_checkpoints
@pytest.mark.skipif(
    os.environ.get("COCO_PIPE_RUN_REAL_FOUNDATION_TRAINING") != "1",
    reason="requires real checkpoints and substantial CPU/GPU time",
)
def test_real_foundation_training():
    records = validate_real_training()
    assert records
    assert all(
        record["status"]
        in {"verified", "authentication_required", "unsupported", "failed"}
        for record in records
    )
