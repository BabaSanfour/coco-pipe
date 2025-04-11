#!/usr/bin/env python3
"""
create_toy_dataset.py

This script creates a toy dataset for testing cocopipe. It generates a CSV file with randomized data.
The dataset includes columns for two sensors:
  - Sensor1: feat1, feat2, feat3
  - Sensor2: feat4, feat5
and a binary target column called "label".
"""

import os
import numpy as np
import pandas as pd

# Create the data directory if it doesn't exist.
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Number of samples
n_samples = 100

# Generate random feature values
data = {
    "sensor1.feat1": np.random.rand(n_samples),
    "sensor1.feat2": np.random.rand(n_samples),
    "sensor1.feat3": np.random.rand(n_samples),
    "sensor1.feat4": np.random.rand(n_samples),
    "sensor1.feat5": np.random.rand(n_samples),
    "sensor2.feat1": np.random.rand(n_samples),
    "sensor2.feat2": np.random.rand(n_samples),
    "sensor2.feat3": np.random.rand(n_samples),
    "sensor2.feat4": np.random.rand(n_samples),
    "sensor2.feat5": np.random.rand(n_samples),
    # Create a binary target column ("label")
    "label": np.random.choice([0, 1], size=n_samples),
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Save the DataFrame to CSV
output_file = os.path.join(data_dir, "toy_dataset.csv")
df.to_csv(output_file, index=False)

print(f"Toy dataset created at: {output_file}")
