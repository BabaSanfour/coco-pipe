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

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install the Package in Editable Mode:**

   ```bash
   pip install -e .
   ```

## Using the ML Module

CoCo Pipe provides two main ways to use the ML module:

### 1. Direct Python API Usage

You can use the ML module directly in your Python scripts by importing from `coco_pipe.io` for data loading/feature selection and `coco_pipe.ml` for machine learning pipelines:

```python
from coco_pipe.io import load, select_features
from coco_pipe.ml import MLPipeline

# Load your data
X, y = load(
    type="tabular",  # Supports: 'tabular', 'embeddings', 'meeg'
    data_path="data/your_dataset.csv",
)

# Optionally select specific features
X, y = select_features(
    df=X,  # Your feature DataFrame
    target_columns=y,  # Target variable(s)
    covariates=["age", "sex"],  # Optional demographic/clinical variables
    spatial_units=["left_frontal", "right_frontal"],  # Brain regions/sensors
    feature_names=["alpha", "beta"]  # Features to include
)

# Configure and run ML pipeline
config = {
    "task": "classification",  # or 'regression'
    "analysis_type": "baseline",  # Options: 'baseline', 'feature_selection', 'hp_search', 'hp_search_fs'
    "models": "all",  # or list of specific models
    "metrics": ["accuracy", "f1-score"],
    "cv_strategy": "stratified",
    "n_splits": 5,
    "n_features": 10,  # For feature selection
    "direction": "forward",  # For feature selection
    "search_type": "grid",  # For hyperparameter search
    "n_iter": 100,  # For random search
    "scoring": "accuracy",
    "n_jobs": -1
}

pipeline = MLPipeline(X=X, y=y, config=config)
results = pipeline.run()
```

### 2. Using the CLI Tool

For batch processing or experiment management, use the CLI tool with a YAML configuration file:

```yaml
ID: "experiment_01"
data:
  file: "data/dataset.csv"
  type: "tabular"
  target: "diagnosis"
  features:
    covariates: ["age", "sex", "education"]
    spatial_units: ["left_frontal", "right_frontal", "temporal"]
    names: ["alpha", "beta", "gamma", "connectivity"]

analyses:
  - name: "Baseline Classification"
    task: "classification"
    analysis_type: "baseline"
    models: ["Random Forest", "SVM", "XGBoost"]
    metrics: ["accuracy", "f1-score", "roc_auc"]
    cv_kwargs:
      strategy: "stratified"
      n_splits: 5
    
  - name: "Feature Selection"
    task: "classification"
    analysis_type: "feature_selection"
    models: ["Random Forest"]
    n_features: 10
    direction: "forward"
    scoring: "accuracy"
    
  - name: "Hyperparameter Tuning"
    task: "classification"
    analysis_type: "hp_search"
    models: ["SVM"]
    search_type: "grid"
    n_iter: 100
    scoring: "accuracy"
    
  - name: "Combined FS + HP"
    task: "classification"
    analysis_type: "hp_search_fs"
    models: ["XGBoost"]
    n_features: 15
    direction: "forward"
    search_type: "random"
    n_iter: 50
    scoring: "f1-score"

output:
  save_intermediate: true
  results_dir: "results/experiment_01"
  results_file: "classification_results"
```

Run the analysis using:

```bash
python scripts/run_ml.py --config configs/your_config.yml
```

The pipeline will:
- Load and preprocess your data
- Run all specified analyses
- Save results for each model/analysis
- Generate a combined results file

## Documentation

Full documentation for CoCo Pipe is available at:  
https://cocopipe.readthedocs.io/en/latest/index.html

## Contributing

Contributions are welcome! If you have suggestions or find any bugs, please open issues or submit pull requests.

## License

*TODO*