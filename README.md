# coco-pipe

[![CI](https://img.shields.io/github/actions/workflow/status/BabaSanfour/coco-pipe/ci.yml?branch=main\&label=CI)](https://github.com/BabaSanfour/coco-pipe/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/github/actions/workflow/status/BabaSanfour/coco-pipe/docs.yml?branch=main\&label=docs)](https://babasanfour.github.io/coco-pipe/)
[![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/coco-pipe)](https://codecov.io/gh/BabaSanfour/coco-pipe)
[![PyPI - Version](https://img.shields.io/pypi/v/coco-pipe.svg)](https://pypi.org/project/coco-pipe/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**A modular framework for biosignal analytics, machine learning, deep learning, and foundation-model workflows.**

coco-pipe was originally developed for M/EEG research and provides reusable components for data loading, feature extraction, dimensionality reduction, decoding, visualization, and automated reporting. The framework is designed to support reproducible machine-learning and foundation-model workflows across neuroimaging and biosignal modalities, with planned support for fMRI, fNIRS, and related data types.

---

## Highlights

* **Biosignal-first design** for M/EEG, tabular, and biomedical datasets.
* **Modular architecture** for building reusable analysis pipelines.
* **Machine-learning workflows** for classification, regression, feature selection, and hyperparameter optimization.
* **Foundation-model workflows** for embedding extraction, linear probing, full fine-tuning, parameter-efficient fine-tuning (LoRA/PEFT), and downstream decoding.
* **Dimensionality-reduction tools** for exploratory analysis, embedding comparison, visualization, clustering, and trajectory analysis.
* **Automated reporting** for reproducible experiment summaries and analysis outputs.

---

## Built On

coco-pipe integrates with many widely used scientific Python libraries, including:

* MNE-Python
* Braindecode
* MOABB
* scikit-learn
* PyTorch
* Hugging Face Transformers
* PEFT
* UMAP
* PHATE
* PaCMAP
* Dask

---

## Installation

### With pip

```bash
git clone https://github.com/BabaSanfour/coco-pipe.git
cd coco-pipe
pip install -e .
```

Development installation:

```bash
pip install -e .[dev,test]
```

Full installation (all optional modules):

```bash
pip install -e .[full,test]
```

### With uv

```bash
git clone https://github.com/BabaSanfour/coco-pipe.git
cd coco-pipe

uv venv
source .venv/bin/activate

uv pip install -e .
```

Development installation:

```bash
uv pip install -e ".[dev,test]"
```

Full installation:

```bash
uv pip install -e ".[full,test]"
```

---

## Modules

| Module          | Role                                                         |
| --------------- | ------------------------------------------------------------ |
| `io`            | Dataset loading, validation, and organization.               |
| `descriptors`   | Signal feature extraction and representation building.       |
| `dim_reduction` | Representation learning and dimensionality reduction.        |
| `decoding`      | Classification, regression, and model evaluation workflows.  |
| `viz`           | Exploratory analysis and publication-quality visualizations. |
| `report`        | Automated experiment summaries and reporting.                |

---

## Quick Start

The decoding API centers around the `Experiment` class:

```python
from coco_pipe.decoding import Experiment, ExperimentConfig
from coco_pipe.decoding.configs import (
    CVConfig,
    LogisticRegressionConfig,
)

config = ExperimentConfig(
    task="classification",
    models={
        "logreg": LogisticRegressionConfig(max_iter=500)
    },
    metrics=["accuracy"],
    cv=CVConfig(
        strategy="stratified",
        n_splits=5,
        shuffle=True,
        random_state=42,
    ),
)

result = Experiment(config).run(X, y)

print(result.summary())
```

More advanced workflows, including feature selection, temporal decoding, dimensionality reduction, foundation-model integration, and automated reporting, are covered in the documentation.

---

## Documentation

Documentation, tutorials, examples, and API references are available at:

**https://babasanfour.github.io/coco-pipe/**

---

## Contributing

Contributions are welcome, including:

* Bug reports
* Documentation improvements
* Examples and tutorials
* New analysis modules
* Tests and infrastructure improvements
* Performance optimizations

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## Citation

If you use coco-pipe in academic work, please cite the associated publication(s) when available.

Citation information will be added here as the project matures.

---

## License

coco-pipe is distributed under the terms of the MIT License.
