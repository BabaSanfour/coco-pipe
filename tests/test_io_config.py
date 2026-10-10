import pytest
from pydantic import ValidationError

from coco_pipe.io.config import (
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)


def test_configs_mirror_their_loaders():
    tabular = TabularConfig(path="dummy.csv")
    assert (tabular.sep, tabular.header, tabular.clean) == ("\t", 0, False)

    bids = BIDSConfig(path="/root")
    assert (bids.datatype, bids.loading_mode) == ("eeg", "epochs")
    assert (bids.tmin, bids.tmax, bids.baseline) == (-0.2, 0.5, None)
    assert (bids.dtype, bids.n_jobs) == ("float32", 1)
    bids = BIDSConfig(
        path="/root",
        task="rest",
        session=["01", "02"],
        runs=["01"],
        event_id={"go": 1},
        baseline=(None, 0.0),
    )
    assert bids.session == ["01", "02"]
    assert bids.event_id == {"go": 1}
    assert bids.baseline == (None, 0.0)

    embedding = EmbeddingConfig(path="/emb", dims=("layer", "unit"), pattern="*.npy")
    assert (embedding.dims, embedding.pattern) == (("layer", "unit"), "*.npy")
    assert EmbeddingConfig(path="/emb").pattern == "*.pkl"


def test_dataset_config_picks_the_config_by_mode():
    tabular = DatasetConfig(dataset={"mode": "tabular", "path": "d.csv", "sep": ","})
    assert isinstance(tabular.dataset, TabularConfig)
    assert tabular.dataset.sep == ","
    bids = DatasetConfig(dataset={"mode": "bids", "path": "/bids", "task": "rest"})
    assert isinstance(bids.dataset, BIDSConfig)
    embedding = DatasetConfig(dataset={"mode": "embedding", "path": "/emb"})
    assert isinstance(embedding.dataset, EmbeddingConfig)


@pytest.mark.parametrize(
    "build",
    [
        TabularConfig,
        lambda: TabularConfig(path="data.csv", clean_kwargs="not_a_dict"),
        lambda: BIDSConfig(path="/root", n_jobs=0),
        lambda: DatasetConfig(dataset={"mode": "unknown", "path": "path"}),
        lambda: DatasetConfig(dataset={"path": "path"}),
    ],
)
def test_configs_reject_invalid_values(build):
    with pytest.raises(ValidationError):
        build()
