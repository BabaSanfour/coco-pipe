import numpy as np
import pytest
from pydantic import ValidationError

from coco_pipe.dim_reduction import DimReduction
from coco_pipe.dim_reduction.config import (
    DMDConfig,
    EvaluationConfig,
    LLEConfig,
    MDSConfig,
    PacmapConfig,
    PCAConfig,
    TopologicalAEConfig,
    TRCAConfig,
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


def test_reducer_config_is_direct_and_strict():
    """Test direct reducer configs and strict field validation."""
    umap_conf = UMAPConfig(method="UMAP", n_neighbors=15)
    assert umap_conf.method == "UMAP"
    assert umap_conf.n_neighbors == 15

    with pytest.raises(ValidationError):
        PCAConfig(method="PCA", unknown_field=1)


def test_core_initialization_with_config():
    """Test initializing DimReduction with a Config object."""
    umap_conf = UMAPConfig(method="UMAP", n_components=2, n_neighbors=10)

    dr = DimReduction(umap_conf)

    # Check attributes
    assert dr.method == "UMAP"
    assert dr.n_components == 2
    # Check if parameters were passed to kwargs
    assert dr.reducer_kwargs["n_neighbors"] == 10

    # Check underlying reducer
    from coco_pipe.dim_reduction.reducers.neighbor import UMAPReducer

    assert isinstance(dr.reducer, UMAPReducer)
    assert dr.reducer.n_components == 2


def test_lle_config_uses_sklearn_named_parameter():
    """Test LLE config uses a method-aligned parameter name."""
    lle_conf = LLEConfig(method="LLE", n_neighbors=8, lle_method="modified")

    dr = DimReduction(lle_conf)

    assert dr.method == "LLE"
    assert dr.reducer_kwargs["method"] == "modified"
    assert "lle_method" not in dr.reducer_kwargs


def test_evaluation_config():
    """Test EvaluationConfig drives scoring inputs and ranking inputs coherently."""
    eval_conf = EvaluationConfig(
        k_range=[10, 20],
        metrics=["trustworthiness", "continuity"],
        selection_metric="trustworthiness",
        selection_k=10,
        tie_breakers=["continuity"],
    )

    # Mock reducers
    reducers = [DimReduction("PCA", n_components=2)]

    # Mock data
    X = np.random.rand(50, 10)

    reducer = reducers[0]
    embedding = reducer.fit_transform(X)
    reducer.score(
        embedding,
        X=X,
        metrics=eval_conf.metrics,
        k_values=eval_conf.k_range,
        separation_method=eval_conf.separation_method,
    )
    selector = MethodSelector(reducers).collect()

    # Check results (should have computed for k=10 and k=20)
    res = selector.to_frame()
    res = res[(res["method"] == "PCA") & (res["metric"] == "trustworthiness")]
    assert len(res) == 2
    k_values = res["scope_value"].astype(float).to_numpy()
    assert 10 in k_values
    assert 20 in k_values
    assert set(res["scope"]) == {"k"}
    ranked = selector.rank_methods(
        selection_metric=eval_conf.selection_metric,
        selection_k=eval_conf.selection_k,
        tie_breakers=eval_conf.tie_breakers,
    )
    assert list(ranked["method"]) == ["PCA"]

    eval_conf_sep = EvaluationConfig(
        metrics=["trajectory_separation"],
        separation_method="within_between_ratio",
    )
    assert eval_conf_sep.separation_method == "within_between_ratio"


def test_tsne_config_validation():
    # Valid
    cfg = TSNEConfig(method="TSNE", perplexity=30, max_iter=750)
    assert cfg.max_iter == 750

    # Invalid perplexity (negative)
    with pytest.raises(ValidationError):
        TSNEConfig(method="TSNE", perplexity=-5)


def test_mds_config_defaults():
    c = MDSConfig(method="MDS")
    assert c.metric is True
    assert c.n_init == 4


def test_pacmap_config():
    c = PacmapConfig(method="Pacmap", n_neighbors=10, MN_ratio=0.5)
    assert c.n_neighbors == 10
    assert c.MN_ratio == 0.5
    assert c.FP_ratio == 2.0  # default check
    assert c.nn_backend == "faiss"


def test_dmd_trca_config_defaults():
    dmd = DMDConfig(method="DMD")
    assert dmd.force_transpose is False
    assert dmd.tlsq_rank == 0

    trca = TRCAConfig(method="TRCA")
    assert trca.sfreq == 250.0
    assert trca.filterbank is None


def test_topology_config_defaults():
    topo = TopologicalAEConfig(method="TopologicalAE")
    assert topo.device == "auto"
    assert topo.verbose == 0
    assert topo.hidden_dims == [128, 64]


def test_evaluation_config_validation():
    """Test detailed validation logic in EvaluationConfig."""
    # 1. Empty metrics
    with pytest.raises(ValidationError, match="least one metric name"):
        EvaluationConfig(metrics=[])

    # 2. Duplicate metrics
    with pytest.raises(ValidationError, match="not contain duplicate entries"):
        EvaluationConfig(metrics=["trustworthiness", "trustworthiness"])

    # 3. Unknown metrics
    with pytest.raises(ValidationError, match="Unknown evaluation metric"):
        EvaluationConfig(metrics=["non_existent_metric"])

    # 4. Duplicate k_range
    with pytest.raises(ValidationError, match="k_range.*not contain duplicate"):
        EvaluationConfig(metrics=["trustworthiness"], k_range=[5, 5])

    # 5. Non-positive k_range
    with pytest.raises(ValidationError, match="k_range.*positive integers"):
        EvaluationConfig(metrics=["trustworthiness"], k_range=[0, 10])

    # 6. Unknown selection_metric
    with pytest.raises(ValidationError, match="selection_metric.*must be one of"):
        EvaluationConfig(metrics=["trustworthiness"], selection_metric="unknown")

    # 7. selection_metric not in metrics
    with pytest.raises(ValidationError, match="selection_metric.*present in `metrics`"):
        EvaluationConfig(metrics=["trustworthiness"], selection_metric="continuity")

    # 8. Invalid selection_k
    with pytest.raises(ValidationError, match="selection_k.*positive integer"):
        EvaluationConfig(metrics=["trustworthiness"], selection_k=0)

    # 9. Duplicate tie_breakers
    with pytest.raises(ValidationError, match="tie_breakers.*not contain duplicate"):
        EvaluationConfig(
            metrics=["trustworthiness", "continuity"],
            tie_breakers=["trustworthiness", "trustworthiness"],
        )

    # 10. Non-ranking tie_breakers
    with pytest.raises(ValidationError, match="tie_breakers.*ranking metrics only"):
        EvaluationConfig(
            metrics=["trustworthiness"], tie_breakers=["trajectory_separation"]
        )

    # 11. tie_breaker not in metrics
    with pytest.raises(ValidationError, match="tie_breakers.*present in `metrics`"):
        EvaluationConfig(
            metrics=["trustworthiness"],
            tie_breakers=["continuity"],
        )

    # 12. Invalid separation_method
    with pytest.raises(ValidationError, match="separation_method.*must be one of"):
        EvaluationConfig(metrics=["trustworthiness"], separation_method="invalid")


def test_get_reducer_class_errors():
    """Test error paths in get_reducer_class."""
    from coco_pipe.dim_reduction.config import get_reducer_class

    # Unknown method
    with pytest.raises(ValueError, match="Unknown method 'Invalid'"):
        get_reducer_class("Invalid")

    # ImportError for optional method (mocking missing module)
    import sys
    from unittest.mock import patch

    with patch.dict(sys.modules, {"coco_pipe.dim_reduction.reducers.topology": None}):
        with pytest.raises(
            ImportError, match="Could not import reducer 'TopologicalAE'"
        ):
            get_reducer_class("TopologicalAE")

    # Non-optional method failure (e.g., PCA)
    with patch.dict(sys.modules, {"coco_pipe.dim_reduction.reducers.linear": None}):
        with pytest.raises(ImportError):
            get_reducer_class("PCA")


def test_evaluation_config_none_values():
    """Test Validation for None values in EvaluationConfig."""
    conf = EvaluationConfig(metrics=["trustworthiness"], selection_metric=None)
    assert conf.selection_metric is None


def test_lle_config_to_reducer_kwargs_extended():
    """Verify field popping in LLEConfig.to_reducer_kwargs."""
    conf = LLEConfig(method="LLE", n_neighbors=10, lle_method="modified")
    kwargs = conf.to_reducer_kwargs()
    # "lle_method" is popped and renamed to Sklearn-compatible "method"
    assert kwargs["method"] == "modified"
    assert "lle_method" not in kwargs
    assert "n_components" not in kwargs
