from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from coco_pipe.decoding.fm_hub._factory import build_foundation_model
from coco_pipe.decoding.fm_hub.base import BaseFoundationModel
from coco_pipe.decoding.fm_hub.reve import REVEModel


def test_fm_hub_base_hardening():
    class MockFM(BaseFoundationModel):
        def get_module_cls(self):
            return MagicMock()

        def _get_net_params(self):
            return {}

    fm = MockFM("test-model", train_mode="frozen")
    assert fm.model_name == "test-model"

    with pytest.raises(ValueError, match="train_mode must be one of"):
        MockFM("test-model", train_mode="invalid")

    # Fit/Predict/Transform runtime errors
    with pytest.raises(RuntimeError, match="Model must be fitted before transform"):
        fm.transform(np.zeros((2, 10)))
    with pytest.raises(RuntimeError, match="Model must be fitted before predict"):
        fm.predict(np.zeros((2, 10)))


def test_fm_hub_factory_hardening():
    class MockConfig:
        provider = "reve"
        model_name = "reve-large"
        sfreq = 250.0

    # Mock REVEModel import to avoid torch/transformers issues
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_import.return_value = mock_module
        mock_module.REVEModel = MagicMock()

        build_foundation_model(MockConfig())
        assert mock_module.REVEModel.called

    # Unknown provider
    class BadConfig:
        provider = "unknown"

    with pytest.raises(ValueError, match="Unknown foundation model provider"):
        build_foundation_model(BadConfig())


def test_fm_hub_reve_hardening():
    # We test REVEModel without actually loading it
    fm = REVEModel(model_name="reve-test", sfreq=100.0)
    assert fm.provider == "reve"
    info = fm.get_embedding_info()
    assert info.n_embeddings == 1024
    assert info.model_name == "reve-test"
