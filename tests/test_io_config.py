
import pytest
from pydantic import ValidationError

from coco_pipe.io.config import (
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)


def test_tabular_config_defaults():
    """Test TabularConfig defaults."""
    cfg = TabularConfig(path="dummy.csv")
    assert cfg.mode == "tabular"
    assert cfg.sep == "\t"
    assert cfg.header == 0
    assert cfg.clean is False
    assert isinstance(cfg.clean_kwargs, dict)
    assert isinstance(cfg.select_kwargs, dict)


def test_tabular_config_validation():
    """Test validation errors for TabularConfig."""
    # Missing path
    with pytest.raises(ValidationError):
        TabularConfig()

    # Invalid types
    with pytest.raises(ValidationError):
        TabularConfig(path="data.csv", clean_kwargs="not_a_dict")


def test_bids_config_defaults():
    """Test BIDSConfig defaults."""
    cfg = BIDSConfig(path="/bids_root")
    assert cfg.mode == "bids"
    assert cfg.datatype == "eeg"
    assert cfg.loading_mode == "epochs"
    assert cfg.subjects is None


def test_bids_config_explicit():
    """Test BIDSConfig explicit values."""
    cfg = BIDSConfig(path="/root", task="rest", session=["01", "02"], window_length=2.5)
    assert cfg.task == "rest"
    assert cfg.session == ["01", "02"]
    assert cfg.window_length == 2.5


def test_embedding_config_defaults():
    """Test EmbeddingConfig defaults."""
    cfg = EmbeddingConfig(path="/emb")
    assert cfg.mode == "embedding"
    assert cfg.pattern == "*.pkl"
    assert cfg.dims == ("obs", "feature")


def test_embedding_config_pattern_logic():
    """Test logic around pattern/dimensions."""
    cfg = EmbeddingConfig(path="/emb", dims=("layer", "unit"), pattern="*.npy")
    assert cfg.dims == ("layer", "unit")
    assert cfg.pattern == "*.npy"


def test_dataset_config_discriminator():
    """Test polymorphic deserialization via DatasetConfig."""
    # 1. Tabular
    raw_tab = {"dataset": {"mode": "tabular", "path": "dat.csv", "sep": ","}}
    cfg_tab = DatasetConfig(**raw_tab)
    assert isinstance(cfg_tab.dataset, TabularConfig)
    assert cfg_tab.dataset.sep == ","

    # 2. BIDS
    raw_bids = {"dataset": {"mode": "bids", "path": "/bids", "task": "rest"}}
    cfg_bids = DatasetConfig(**raw_bids)
    assert isinstance(cfg_bids.dataset, BIDSConfig)
    assert cfg_bids.dataset.task == "rest"

    # 3. Embedding
    raw_emb = {"dataset": {"mode": "embedding", "path": "/emb"}}
    cfg_emb = DatasetConfig(**raw_emb)
    assert isinstance(cfg_emb.dataset, EmbeddingConfig)


def test_dataset_config_invalid_mode():
    """Test invalid mode in discriminator."""
    with pytest.raises(ValidationError):
        DatasetConfig(**{"dataset": {"mode": "unknown", "path": "path"}})
