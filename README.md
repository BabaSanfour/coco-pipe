# CoCo Pipe

![Codecov](https://img.shields.io/codecov/c/github/BabaSanfour/coco-pipe)
[![Test Status](https://img.shields.io/github/actions/workflow/status/BabaSanfour/coco-pipe/python-tests.yml?branch=main&label=tests)](https://github.com/BabaSanfour/coco-pipe/actions?query=workflow%3Apython-tests)
[![Documentation Status](https://readthedocs.org/projects/cocopipe/badge/?version=latest)](https://cocopipe.readthedocs.io/en/latest/?badge=latest)
[![GitHub Repository](https://img.shields.io/badge/Source%20Code-BabaSanfour%cocopipe-blue)](https://github.com/BabaSanfour/coco-pipe)

CoCo Pipe is a modular pipeline framework for processing bio M/EEG data with both traditional machine learning and deep learning components. The project includes tools for configuration, data processing, feature extraction, experiment design, visualization, and reporting, making it easy for users to run end-to-end experiments.

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

## Running the Pipeline

CoCo Pipe provides two primary ways to run analyses:

### 1. Using the Core Module (ml.py)
You can directly import and use the core functionality in your Python scripts or Jupyter notebooks. For example:

```python
import pandas as pd
from coco_pipe.ml import pipeline_baseline

# Load your dataset
df = pd.read_csv("data/your_dataset.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Run a baseline analysis using all available features
results = pipeline_baseline(X, y, scoring="accuracy")
print(results)
```

This method is ideal if you wish to embed CoCo Pipe into a custom workflow.

### 2. Using the CLI Script (run_ml.py)
The CLI tool (located in the `scripts` folder) reads a YAML configuration file and runs one or more analyses as specified.

#### a. Prepare a Configuration File
Place a YAML configuration file (for example, `configs/toy_config.yml`) in your project. Below is a toy configuration example:

```yaml
ID: "toy_example_01"
data:
  file: "data/toy_dataset.csv"
  target: "label"
  features_groups:
    groups: ["sensor1", "sensor2"]
    features: ["feat1", "feat2", "feat3", "feat4", "feat5"]
    global:
      - "sensor1.feat1"
      - "sensor1.feat2"
      - "sensor1.feat3"
      - "sensor2.feat4"
      - "sensor2.feat5"
analysis:
  - name: "Baseline Global"
    type: "baseline"
    subset: "all_features_all_groups"
    models: "all"
    scoring: "accuracy"
  - name: "Baseline Per Group"
    type: "baseline"
    subset: "all_features_per_group"
    models: "all"
    scoring: "accuracy"
  - name: "Single Feature Global"
    type: "baseline"
    subset: "single_feature_all_groups"
    models: "all"
    scoring: "f1-score"
  - name: "Feature Selection Per Group"
    type: "fs"
    subset: "single_feature_per_group"
    num_features: 1
    models: ["Random Forest", "SVC"]
    scoring: "auc"
  - name: "FS + HP Search Global"
    type: "fs_hp"
    subset: "all_features_all_groups"
    num_features: 3
    models: ["Logistic Regression"]
    scoring: "accuracy"
output: "toy_results"
```

#### b. Run the CLI Script
Run the CLI tool by providing the configuration file:

```bash
python scripts/run_ml.py --config configs/toy_config.yml
```

After running, the tool will:
- Execute the analyses defined in the config.
- Save each subanalysis result as a separate pickle file (e.g., `toy_example_01_Baseline Global_baseline_results.pickle`).
- Save a combined YAML file with all results (e.g., `toy_example_01_results.yaml`).

*Note:* Adjust the configuration file as needed to match your dataset.

## Documentation

Full documentation for CoCo Pipe is available at:  
https://cocopipe.readthedocs.io/en/latest/index.html

## Contributing

Contributions are welcome! If you have suggestions or find any bugs, please open issues or submit pull requests.

## License

*TODO*