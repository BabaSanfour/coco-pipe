"""
====================================
Basic Feature Descriptors Extraction
====================================

This example demonstrates how to use the ``DescriptorPipeline`` to extract
EEG features (bands, parametric, complexity) from a standard NumPy array.
"""

# %%
# Imports
# -------

from __future__ import annotations

import numpy as np

from coco_pipe.descriptors import DescriptorPipeline

# %%
# 1. Simulate EEG Data
# --------------------
# We will create a dummy dataset with 12 observations, 3 channels, and 256 timepoints.
# We'll also inject some sine waves to simulate alpha (10Hz) and theta (6Hz) activity.

rng = np.random.default_rng(42)
X = rng.normal(size=(12, 3, 256))
t = np.linspace(0, 1, 256, endpoint=False)

# Inject 10Hz signal into the first channel
X[:, 0, :] += np.sin(2 * np.pi * 10 * t)
# Inject 6Hz signal into the second channel
X[:, 1, :] += np.sin(2 * np.pi * 6 * t)

ids = np.asarray([f"obs-{idx:02d}" for idx in range(12)])

# %%
# 2. Configure the Pipeline
# -------------------------
# The descriptor pipeline is configured via a simple dictionary. We can enable
# specific families of descriptors (e.g., spectral bands, aperiodic components,
# and signal complexity measures).

config = {
    "families": {
        "bands": {
            "enabled": True,
            "outputs": ["absolute_power", "corrected_absolute_power"],
        },
        "parametric": {
            "enabled": True,
            "outputs": ["aperiodic"],
        },
        "complexity": {
            "enabled": True,
            "measures": ["sample_entropy", "hjorth_mobility"],
        },
    },
}

# %%
# 3. Extract Features
# -------------------
# We pass our data, configs, sampling frequency, and channel names to the pipeline.

channels = ["Fz", "Cz", "Pz"]
pipe = DescriptorPipeline(config)

result = pipe.extract(
    X=X,
    ids=ids,
    sfreq=256.0,
    channel_names=channels,
)

# %%
# 4. Pooling and Results
# ----------------------
# After extraction, we can pool the channel features together (e.g., averaging
# all frontal or parietal channels). Here, we create an "all" region pool.
# The pipeline natively outputs a ``DataContainer``, which holds all of our labels.

result = pipe.pool_channels(result, {"all": channels})

print("Descriptor matrix shape:", result.X.shape)
print("First five names:\n", result.coords["feature"][:5])

# %%
# 5. Data Aggregation
# -------------------
# Since the output is already a ``DataContainer``, it is perfectly formatted for
# downstream analysis without any extra wrapping. We can easily aggregate the
# descriptors by subject or condition.

# Let's simulate aggregating the descriptors by subjects (assuming 6 epochs per subject)
grouped = result.aggregate(
    by=["sub-01"] * 6 + ["sub-02"] * 6,
    stats=["mean", "std"],
)

print(f"Grouped descriptor shape: {grouped.X.shape}")
print(f"Grouped dims: {grouped.dims}")
print(f"Grouped stats: {grouped.coords['stat'].tolist()}")
