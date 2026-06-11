# CoCo Pipe

![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/coco-pipe)
[![Test Status](https://img.shields.io/github/actions/workflow/status/BabaSanfour/coco-pipe/python-tests.yml?branch=main&label=tests)](https://github.com/BabaSanfour/coco-pipe/actions?query=workflow%3Apython-tests)
[![Documentation Status](https://readthedocs.org/projects/cocopipe/badge/?version=latest)](https://cocopipe.readthedocs.io/en/latest/?badge=latest)
[![GitHub Repository](https://img.shields.io/badge/Source%20Code-BabaSanfour%2Fcocopipe-blue)](https://github.com/BabaSanfour/coco-pipe)

CoCo Pipe is a comprehensive Python framework designed for advanced processing and analysis of bio M/EEG data. It seamlessly integrates traditional machine learning, deep learning, and signal processing techniques into a unified pipeline architecture. Key features include:

- **Flexible Data Processing**: Support for various data formats (tabular, M/EEG, embeddings) with automated preprocessing and feature extraction
- **Advanced ML Capabilities**: Integrated classification and regression pipelines with automated feature selection and hyperparameter optimization
- **Modular Design**: Easy-to-extend architecture for adding custom processing steps, models, and analysis methods
- **Experiment Management**: Built-in tools for experiment configuration, reproducibility, and results tracking
- **Visualization & Reporting**: Comprehensive visualization tools and automated report generation for both signal processing and ML results
- **Scientific Workflow**: End-to-end support for neuroimaging research, from raw data processing to publication-ready results

Whether you're conducting clinical research, developing ML models for brain-computer interfaces, or exploring neural signal patterns, CoCo Pipe provides the tools and flexibility to streamline your workflow.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/BabaSanfour/coco-pipe.git
   cd coco-pipe
   ```

2. **(Optional) Create and Activate a Virtual Environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the Package:**

   ```bash
   pip install -e .
   ```

   *Note: This will install all runtime dependencies. for development dependencies, use `pip install -e .[dev]`.*

For detailed development instructions, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Using the Decoding Module

The supported modeling API is `coco_pipe.decoding.Experiment`. It is array-first:
prepare `X` and `y` explicitly, then pass optional sample IDs, groups, feature
names, and time labels when they matter for the analysis.

```python
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    CVConfig,
    FeatureSelectionConfig,
    LogisticRegressionConfig,
    TuningConfig,
)

config = ExperimentConfig(
    task="classification",
    models={"logreg": LogisticRegressionConfig(max_iter=500)},
    metrics=["accuracy", "roc_auc"],
    cv=CVConfig(strategy="stratified", n_splits=5, shuffle=True, random_state=42),
    feature_selection=FeatureSelectionConfig(
        enabled=True,
        method="k_best",
        n_features=20,
        scoring="f_classif",
    ),
    tuning=TuningConfig(
        enabled=True,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="roc_auc",
        cv=CVConfig(strategy="stratified", n_splits=3, shuffle=True, random_state=42),
    ),
    n_jobs=1,
)

result = Experiment(config).run(
    X,
    y,
    groups=subject_ids,
    sample_ids=trial_ids,
    feature_names=feature_names,
)

summary = result.summary()
predictions = result.get_predictions()
splits = result.get_splits()
selected = result.get_selected_features()
```

For grouped EEG studies, make the outer and inner CV decisions explicit:

```python
config = ExperimentConfig(
    task="classification",
    models={"logreg": LogisticRegressionConfig(max_iter=500)},
    metrics=["accuracy"],
    cv=CVConfig(strategy="group_kfold", n_splits=5),
    tuning=TuningConfig(
        enabled=True,
        param_grid={"model__C": [0.1, 1.0, 10.0]},
        scoring="accuracy",
        cv=CVConfig(strategy="group_kfold", n_splits=3),
    ),
)

result = Experiment(config).run(X, y, groups=subject_ids)
```

See the decoding documentation for feature selection, temporal decoding, result
tables, plotting helpers, and report integration. Batch decoding CLIs are not
part of the public surface yet; use the Python API for now.

## Testing

Install the test extras (the suite exercises the optional model backends):

```bash
pip install -e .[full,test]
```

### Standard run (fast, offline)

```bash
pytest
```

This is exactly what CI runs on every push and pull request. It needs **no
network access and no HuggingFace token**. Tests that download real foundation
checkpoints are marked `@pytest.mark.real_checkpoints` and are **deselected by
default** (`addopts = -m 'not real_checkpoints'` in `pyproject.toml`).

> `tests/conftest.py` pins native thread pools (`OMP_NUM_THREADS=1`, etc.) so
> the heavy native backends (faiss, torch, annoy, …) don't segfault when run
> together in one process. Export a higher value before `pytest` if you want
> multi-threaded test runs.

### Running heavy foundation tests

These download gated checkpoints from HuggingFace, so they are opt-in and
require a token. Run them locally with:

```bash
export HF_TOKEN=hf_xxxxxxxx              # do NOT commit this
export COCO_PIPE_RUN_REAL_FOUNDATION=1
pytest -m real_checkpoints
```

The token is resolved (in order) from the `token=` argument, the `HF_TOKEN`
environment variable, or a prior `hf auth login`. The account behind the token
must have **accepted each gated model's license** on huggingface.co, otherwise
downloads come back as `authentication_required`.

The real **training** test is heavier still (needs a GPU); enable it
additionally with `export COCO_PIPE_RUN_REAL_FOUNDATION_TRAINING=1`.

#### In CI

The heavy tests run in a separate `foundation` job that triggers **only** on a
manual *Run workflow* (workflow_dispatch) or a version tag push (`v*`) — never on
ordinary pushes or pull requests, and never on PRs from forks (GitHub does not
expose secrets to those). To enable it, add the token once as a repository
secret:

1. On HuggingFace: create a read token and accept the license for each gated
   model you need.
2. On GitHub: **Settings → Secrets and variables → Actions → New repository
   secret**, name it `HF_TOKEN`, paste the token.

The workflow injects `HF_TOKEN` only into the heavy job's test step, and GitHub
automatically masks it in logs. The token is never needed to clone, push, or
open a PR — only to download checkpoints at test time.

## Documentation

Full documentation for CoCo Pipe is available at:
https://cocopipe.readthedocs.io/en/latest/index.html

## Contributing

Contributions are welcome! If you have suggestions or find any bugs, please open issues or submit pull requests.

### TODO

#### IO Module
- Implement CSV loading and M/EEG data loading functionalities.
- Develop comprehensive unit tests.

#### DL Module
- Define and implement deep learning functionalities.
- Create corresponding unit tests.

#### Visualization Module
- Plan and implement enhancements for visualization features.
- Integrate new visual components and testing.

#### Descriptors Module
- Add a future connectivity descriptor family built on `mne-connectivity`.
- Start that connectivity family with phase-based measures such as `PLV`, with room for later extensions like `ciPLV`, `PLI`, and `wPLI`.
- Add a future wavelet-based descriptor batch built on `PyWavelets`.
- Start that wavelet batch with `sure_entropy`.
- Keep `log_energy_entropy` on the roadmap, but finalize its scientific definition before implementation.

#### Dim reduction:
- Add parallelism

## License

*TODO*
