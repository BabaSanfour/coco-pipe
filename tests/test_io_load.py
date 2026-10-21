import importlib
import sys

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from coco_pipe.io.config import (
    BIDSConfig,
    DatasetConfig,
    EmbeddingConfig,
    TabularConfig,
)
from coco_pipe.io.embeddings import load_embeddings
from coco_pipe.io.load import load_data
from tests.fixtures.embeddings import extract, write_recording


def _csv(tmp_path):
    path = tmp_path / "feat.csv"
    pd.DataFrame(
        {"id": ["a", "b", "c"], "f0": [1, 2, 3], "f1": [4, 5, 6], "y": [0, 1, 0]}
    ).to_csv(path, index=False)
    return path


def test_load_data_reads_each_config(tmp_path):
    table = load_data(TabularConfig(path=_csv(tmp_path), sep=",", target_col="y"))
    assert table.dims == ("obs", "feature")

    results = {"EC": extract([0, 2]), "EO": extract([1], seed=1)}
    path = write_recording(tmp_path, results)["base", "window"].fpath
    windows = load_data(EmbeddingConfig(path=path))
    np.testing.assert_array_equal(windows.X, load_embeddings(path).X)
    subject = load_data(
        EmbeddingConfig(path=path, conditions="EC", aggregate_by="subject")
    )
    assert subject.ids.tolist() == ["01"]
    np.testing.assert_allclose(
        subject.X[0], results["EC"].window_embeddings.mean(axis=0), rtol=1e-6
    )
    meta = pd.DataFrame({"Study ID": [1], "age": [9]})
    aged = load_data(
        EmbeddingConfig(path=path), subject_metadata_df=meta, subject_key="Study ID"
    )
    assert aged.coords["age"].tolist() == [9, 9, 9]


def test_load_data_bids_saved_epochs(tmp_path, write_bids_epochs):
    write_bids_epochs(tmp_path, "0001", description="base")
    meta = pd.DataFrame({"Study ID": [1], "age": [42], "group": ["case"]})
    config = BIDSConfig(
        path=tmp_path,
        task="rest",
        suffix="epo",
        loading_mode="load_existing",
        description="base",
        target_col="group",
    )
    container = load_data(config, subject_metadata_df=meta, subject_key="Study ID")
    assert container.X.shape == (2, 2, 20)
    assert container.ids.tolist() == [
        "sub-0001_task-rest_epoch-0000",
        "sub-0001_task-rest_epoch-0001",
    ]
    assert container.coords["age"].tolist() == [42, 42]
    assert container.y.tolist() == ["case", "case"]

    with pytest.raises(ValueError, match="subject_key must be provided"):
        load_data(config, subject_metadata_df=meta)


def test_load_data_takes_a_config_or_its_wrapper(tmp_path):
    path = _csv(tmp_path)
    options = {"sep": ",", "target_col": "y", "index_col": "id"}
    via_config = load_data(TabularConfig(path=path, **options))
    via_wrapper = load_data(
        DatasetConfig(dataset={"mode": "tabular", "path": str(path), **options})
    )
    for container in (via_config, via_wrapper):
        assert list(container.coords["feature"]) == ["f0", "f1"]
        assert list(container.coords["obs"]) == ["a", "b", "c"]
        np.testing.assert_array_equal(container.y, [0, 1, 0])


def test_load_data_errors(tmp_path):
    # settings belong to the config, which validates them: a path or a dict, as
    # load_data used to take, is refused with a pointer to the configs
    for not_a_config in (_csv(tmp_path), {"mode": "tabular", "path": _csv(tmp_path)}):
        with pytest.raises(TypeError, match="takes a dataset configuration"):
            load_data(not_a_config)
    with pytest.raises(ValidationError):
        TabularConfig(path=_csv(tmp_path), clean_kwargs="not_a_dict")


def test_io_import_is_lightweight(monkeypatch):
    module_names = [
        name
        for name in sys.modules
        if name == "coco_pipe.io" or name.startswith("coco_pipe.io.")
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


def test_load_data_passes_the_array_type_and_threads(tmp_path):
    import mne
    from mne_bids import BIDSPath, write_raw_bids

    data = np.random.default_rng(0).normal(size=(2, 300)) * 1e-12
    raw = mne.io.RawArray(data, mne.create_info(["M1", "M2"], 100.0, "mag"))
    raw.set_meas_date(0)
    for subject in ("01", "02"):
        write_raw_bids(
            raw,
            BIDSPath(root=tmp_path, subject=subject, task="rest", datatype="meg"),
            allow_preload=True,
            format="FIF",
            verbose=False,
        )
    options = {
        "path": tmp_path,
        "task": "rest",
        "datatype": "meg",
        "window_length": 1.0,
    }

    default = load_data(BIDSConfig(**options))
    wide = load_data(BIDSConfig(dtype="float64", n_jobs=2, **options))

    assert default.X.dtype == np.float32
    assert wide.X.dtype == np.float64
    np.testing.assert_allclose(wide.X, default.X, rtol=1e-6)
    recordings = load_data(BIDSConfig(representation="recording", **options))
    assert recordings.ids.tolist() == [
        "sub-01_task-rest_cond-1",
        "sub-02_task-rest_cond-1",
    ]
