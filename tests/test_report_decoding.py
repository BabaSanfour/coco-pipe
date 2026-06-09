from sklearn.datasets import make_classification

from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import CVConfig, LogisticRegressionConfig
from coco_pipe.report.core import Report


def _diagnostic_result():
    X, y = make_classification(
        n_samples=40,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    config = ExperimentConfig(
        task="classification",
        models={"lr": LogisticRegressionConfig(max_iter=200, kind="classical")},
        metrics=["accuracy", "roc_auc", "brier_score"],
        cv=CVConfig(strategy="stratified", n_splits=2, shuffle=True, random_state=7),
        n_jobs=1,
        verbose=False,
    )
    return Experiment(config).run(X, y)


def test_decoding_diagnostics_report_section_renders():
    result = _diagnostic_result()
    report = Report("Diagnostics")

    report.add_decoding_diagnostics(result)
    html = report.render()

    assert "Decoding Diagnostics" in html
    assert "Inference Context" in html
    assert "Fold Scores" in html
    assert "Fit Diagnostics" in html
