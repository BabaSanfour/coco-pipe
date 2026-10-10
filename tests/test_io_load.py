import importlib
import json
import pickle
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from coco_pipe.io.config import DatasetConfig, TabularConfig
from coco_pipe.io.load import load_data


def _csv(tmp_path):
    path = tmp_path / "feat.csv"
    pd.DataFrame(
        {"id": ["a", "b", "c"], "f0": [1, 2, 3], "f1": [4, 5, 6], "y": [0, 1, 0]}
    ).to_csv(path, index=False)
    return path


def test_load_data_infers_the_mode(tmp_path, write_bids_epochs):
    table = load_data(_csv(tmp_path), mode="auto", sep=",", target_col="y")
    assert table.dims == ("obs", "feature")

    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "rec01_emb.pkl").write_bytes(pickle.dumps(np.ones((1, 4))))
    for path in (blobs, blobs / "rec01_emb.pkl"):
        assert load_data(path, mode="auto", dims=("feature",)).X.shape == (1, 4)

    bids = tmp_path / "bids"
    write_bids_epochs(bids, "01")
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "x"}))
    epochs = load_data(bids, mode="auto", suffix="epo", loading_mode="load_existing")
    assert epochs.dims == ("obs", "channel", "time")


def test_load_data_bids_saved_epochs(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path, "0001", description="base")
    meta = pd.DataFrame({"Study ID": [1], "age": [42], "group": ["case"]})
    container = load_data(
        tmp_path,
        mode="bids",
        task="rest",
        suffix="epo",
        loading_mode="load_existing",
        description="base",
        subject_metadata_df=meta,
        subject_key="Study ID",
        target_col="group",
    )
    assert container.X.shape == (2, 2, 20)
    assert container.ids.tolist() == [
        "sub-0001_task-rest_epoch-0000",
        "sub-0001_task-rest_epoch-0001",
    ]
    assert container.coords["age"].tolist() == [42, 42]
    assert container.y.tolist() == ["case", "case"]

    with pytest.raises(ValueError, match="subject_key must be provided"):
        load_data(tmp_path, mode="bids", suffix="epo", subject_metadata_df=meta)


def test_load_data_config_object_matches_kwargs(tmp_path):
    path = _csv(tmp_path)
    options = {"sep": ",", "target_col": "y", "index_col": "id"}
    via_kwargs = load_data(path, mode="tabular", **options)
    via_config = load_data(config=TabularConfig(path=path, **options))
    via_wrapper = load_data(
        config=DatasetConfig(dataset={"mode": "tabular", "path": str(path), **options})
    )
    for container in (via_kwargs, via_config, via_wrapper):
        assert list(container.coords["feature"]) == ["f0", "f1"]
        assert list(container.coords["obs"]) == ["a", "b", "c"]
        np.testing.assert_array_equal(container.y, [0, 1, 0])


def test_load_data_errors(tmp_path):
    with pytest.raises(ValueError, match="Unknown mode"):
        load_data("dummy", mode="invalid")
    with pytest.raises(ValueError, match="`path` is required"):
        load_data(mode="tabular")
    with pytest.raises(ValidationError):
        load_data(_csv(tmp_path), mode="tabular", clean_kwargs="not_a_dict")


def test_io_import_is_lightweight(monkeypatch):
    module_names = [
        "coco_pipe.io",
        "coco_pipe.io.load",
        "coco_pipe.io.structures",
        "coco_pipe.io.utils",
    ]
    cached_modules = {name: sys.modules.get(name) for name in module_names}
    for module_name in module_names:
        sys.modules.pop(module_name, None)
    monkeypatch.setitem(sys.modules, "mne", None)
    monkeypatch.setitem(sys.modules, "mne_bids", None)
    try:
        io_mod = importlib.import_module("coco_pipe.io")
        assert hasattr(io_mod, "DataContainer")
        assert callable(io_mod.load_data)
    finally:
        for module_name, module_obj in cached_modules.items():
            if module_obj is not None:
                sys.modules[module_name] = module_obj
