
import numpy as np
import pytest
from pydantic import ValidationError

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.dim_reduction.config import (
    DimReductionConfig,
    EvaluationConfig,
    MDSConfig,
    PacmapConfig,
    PCAConfig,
    TSNEConfig,
    UMAPConfig,
)
from coco_pipe.dim_reduction.evaluation import MethodSelector


def test_umap_config_validation():
    """Test that UMAP configuration validates correctly."""
    # Valid config
    conf = UMAPConfig(method="UMAP", n_components=2, n_neighbors=30)
    assert conf.n_neighbors == 30
    assert conf.method == "UMAP"

    # Invalid config (n_neighbors < 2)
    with pytest.raises(ValidationError):
        UMAPConfig(method="UMAP", n_neighbors=1)


def test_generic_wrapper_config():
    """Test the DimReductionConfig union wrapper."""
    # Wrapping UMAP
    umap_conf = UMAPConfig(method="UMAP", n_neighbors=15)
    wrapper = DimReductionConfig(config=umap_conf)
    assert isinstance(wrapper.config, UMAPConfig)

    # Validating generic loading (simulating Hydra/Dict loading)
    data = {"config": {"method": "PCA", "n_components": 3, "whiten": True}}
    wrapper_loaded = DimReductionConfig(**data)
    assert isinstance(wrapper_loaded.config, PCAConfig)
    assert wrapper_loaded.config.whiten is True


def test_core_initialization_with_config():
    """Test initializing DimReduction with a Config object."""
    # Create config
    umap_conf = UMAPConfig(method="UMAP", n_components=2, n_neighbors=10)
    wrapper = DimReductionConfig(config=umap_conf)

    # Init core
    dr = DimReduction(wrapper)

    # Check attributes
    assert dr.method == "UMAP"
    assert dr.n_components == 2
    # Check if parameters were passed to kwargs
    assert dr.reducer_kwargs["n_neighbors"] == 10

    # Check underlying reducer
    from coco_pipe.dim_reduction.reducers.neighbor import UMAPReducer

    assert isinstance(dr.reducer, UMAPReducer)
    assert dr.reducer.n_components == 2


def test_evaluation_config():
    """Test EvaluationConfig in MethodSelector."""
    eval_conf = EvaluationConfig(k_range=[10, 20])

    # Mock reducers
    reducers = [DimReduction("PCA", n_components=2)]

    # Mock data
    X = np.random.rand(50, 10)

    selector = MethodSelector(reducers, data=X)
    selector.run(k_range=eval_conf)

    # Check results (should have computed for k=10 and k=20)
    res = selector.results_["PCA"]
    assert len(res) == 2
    assert 10 in res["k"].values
    assert 20 in res["k"].values


def test_tsne_config_validation():
    # Valid
    TSNEConfig(method="TSNE", perplexity=30)

    # Invalid perplexity (negative)
    with pytest.raises(ValidationError):
        TSNEConfig(method="TSNE", perplexity=-5)


def test_mds_config_defaults():
    c = MDSConfig(method="MDS")
    assert c.metric is True
    assert c.n_init == 4


def test_pacmap_config():
    c = PacmapConfig(method="PaCMAP", n_neighbors=10, MN_ratio=0.5)
    assert c.n_neighbors == 10
    assert c.MN_ratio == 0.5
    assert c.FP_ratio == 2.0  # default check
