import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from coco_pipe.ml import (
    MLPipeline,
    pipeline_baseline,
    pipeline_feature_selection,
    pipeline_HP_search,
    pipeline_feature_selection_HP_search,
    pipeline_unsupervised,
)

# ---------------------------
# Pytest Fixture to Generate Dummy Data
# ---------------------------
@pytest.fixture
def dummy_data():
    """
    Generate a dummy classification dataset and return it as a pandas DataFrame and labels array.
    """
    # Create a synthetic dataset with 50 samples, 10 features (5 informative)
    X, y = make_classification(n_samples=50, n_features=10, n_informative=5, random_state=42)
    # Return X as a DataFrame with column names and y as a numpy array.
    X_df = pd.DataFrame(X, columns=[f"feature{i}" for i in range(10)])
    return X_df, y

# ---------------------------
# Tests for Pipeline Construction
# ---------------------------
def test_pipeline_construction_all(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="all", scoring="accuracy")
    assert isinstance(pipeline.models, dict)
    assert len(pipeline.models) > 0

def test_pipeline_construction_single(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="SVC", scoring="f1-score")
    assert "SVC" in pipeline.models
    assert len(pipeline.models) == 1

def test_pipeline_construction_list_str(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models=["SVC", "Random Forest"], scoring="auc")
    assert "SVC" in pipeline.models and "Random Forest" in pipeline.models
    assert len(pipeline.models) == 2

def test_pipeline_construction_list_int(dummy_data):
    X, y = dummy_data
    # Assuming there are at least 2 models in the full list.
    pipeline = MLPipeline(X, y, models=[0, 1], scoring="accuracy")
    keys = list(pipeline.models.keys())
    assert len(keys) >= 2

# ---------------------------
# Tests for Utility Functions: list and print available models
# ---------------------------
def test_list_print_available_models(dummy_data, capsys):
    X, y = dummy_data
    pipeline = MLPipeline(X, y)
    models_list = pipeline.list_available_models()
    assert isinstance(models_list, list)
    # Capture output of print_available_models
    pipeline.print_available_models(verbose=True)
    captured = capsys.readouterr().out
    assert "Available models:" in captured

# ---------------------------
# Tests for Adding and Updating Models
# ---------------------------
def test_add_update_model(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="all", scoring="accuracy")
    # Import a dummy classifier for testing
    from sklearn.dummy import DummyClassifier
    new_model_name = "DummyClassifier"
    dummy_estimator = DummyClassifier(strategy="most_frequent")
    dummy_params = {"strategy": ["most_frequent", "prior"]}
    pipeline.add_model(new_model_name, dummy_estimator, dummy_params)
    assert new_model_name in pipeline.all_models
    # Update parameters and check
    new_params = {"strategy": ["most_frequent"]}
    pipeline.update_model_params(new_model_name, new_params)
    assert pipeline.all_models[new_model_name]["params"] == new_params

# ---------------------------
# Tests for Pipeline Methods: baseline, feature selection, HP search, combined FS+HP search, unsupervised
# ---------------------------
def test_baseline(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="SVC", scoring="accuracy")
    result = pipeline.baseline()
    for score in result["SVC"].values():
        assert isinstance(score, float)

def test_feature_selection(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="Random Forest", scoring="f1-score")
    num_features = 3
    fs_result = pipeline.feature_selection(num_features)
    # For our single model, "Random Forest" should be present in the dictionary.
    assert "Random Forest" in fs_result
    fs_rf = fs_result["Random Forest"]
    # Check that keys in fs_rf are integers from 1 to num_features
    for k in fs_rf:
        assert isinstance(k, int)
        assert 1 <= k <= num_features

def test_hp_search(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="SVC", scoring="accuracy")
    hp_result = pipeline.hp_search()
    for key in ["best_score", "best_params", "fitted_model"]:
        assert key in hp_result['hp_results']["SVC"]

def test_feature_selection_hp_search(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, models="Random Forest", scoring="f1-score")
    num_features = 3
    fs_hp_result = pipeline.feature_selection_hp_search(num_features)
    assert "combined_results" in fs_hp_result
    rf_results = fs_hp_result["combined_results"]["Random Forest"]
    for k, result in rf_results.items():
        assert isinstance(k, int)
        assert "selected_features" in result
        assert "best_params" in result

def test_unsupervised(dummy_data):
    X, y = dummy_data
    pipeline = MLPipeline(X, y, scoring="accuracy")
    unsup_result = pipeline.unsupervised(n_clusters=2)
    assert "cluster_labels" in unsup_result
    assert "silhouette_score" in unsup_result
    n_samples = X.shape[0]
    assert len(unsup_result["cluster_labels"]) == n_samples

# ---------------------------
# Tests for Wrapper Functions
# ---------------------------
def test_wrapper_functions(dummy_data):
    X, y = dummy_data
    # Baseline wrapper test
    baseline_result = pipeline_baseline(X, y, scoring="accuracy", models="SVC")
    for score in baseline_result["SVC"].values():
        assert isinstance(score, float)

    # Feature selection wrapper test
    fs_result = pipeline_feature_selection(X, y, num_features=2, models="SVC", scoring="f1-score")
    assert "SVC" in fs_result

    # HP search wrapper test
    hp_result = pipeline_HP_search(X, y, models="SVC", scoring="accuracy")
    assert "best_score" in hp_result['hp_results']['SVC']

    # Feature selection HP search wrapper test
    fs_hp_result = pipeline_feature_selection_HP_search(X, y, num_features=2, models="SVC", scoring="accuracy")
    assert "combined_results" in fs_hp_result

    # Unsupervised wrapper test: using pipeline_unsupervised returns tuple (silhouette_score, cluster_labels)
    unsup_score, unsup_labels = pipeline_unsupervised(X, y, n_clusters=2)
    assert isinstance(unsup_score, float)
    assert len(unsup_labels) == X.shape[0]
