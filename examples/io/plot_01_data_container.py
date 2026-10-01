"""
====================
Data Structures Demo
====================

This example demonstrates the ``DataContainer`` and other core IO structures
used in the coco-pipe package. The ``DataContainer`` is a powerful wrapper
around N-dimensional numpy arrays that keeps track of dimensions, coordinates,
and labels.
"""

# %%
# Imports
# -------
# First, let's import the necessary libraries.

import numpy as np

from coco_pipe.io.structures import DataContainer

# %%
# 1. Tabular Data (2D)
# --------------------
# We can store standard 2D tabular data (Observations x Features).
# The DataContainer will automatically track the coordinates for each dimension.

X_tab = np.random.randn(5, 3)
container_tab = DataContainer(
    X=X_tab,
    dims=("obs", "feature"),
    coords={
        "obs": [f"sub-{i}" for i in range(5)],
        "feature": ["Alpha_Cz", "Alpha_Fz", "Beta_Pz"],
    },
)

print(f"Original container:\n{container_tab}")

# %%
# We can easily select data using wildcards on the coordinates.
# Let's select all features starting with "Alpha":

subset = container_tab.select(feature=["Alpha*"])
print(f"Selected (Alpha*):\n{subset}")


# %%
# 2. EEG Data (3D)
# ----------------
# The DataContainer excels at handling multi-dimensional data like EEG,
# which typically has dimensions (Observations x Channels x Time).
#
# Let's simulate data for 2 subjects, 2 conditions, and 4 epochs each.

n_subs = 2
n_conds = 2
n_epochs = 4
n_obs = n_subs * n_conds * n_epochs
n_chans = 3
n_times = 10

X_eeg = np.random.randn(n_obs, n_chans, n_times)

# Create tracking labels
ids = []
conditions = []
for sub in range(n_subs):
    for cond in ["A", "B"]:
        for ep in range(n_epochs):
            ids.append(f"sub-{sub}_cond-{cond}_ep-{ep}")
            conditions.append(cond)

container_eeg = DataContainer(
    X=X_eeg,
    y=np.array(conditions),
    ids=np.array(ids),
    dims=("obs", "channel", "time"),
    coords={"obs": ids, "channel": ["Fz", "Cz", "Pz"], "time": np.arange(n_times)},
)

print(f"EEG Container:\n{container_eeg}")
print(f"First 5 IDs:\n{container_eeg.ids[:5]}")


# %%
# 3. Flattening Data
# ------------------
# We often need to flatten high-dimensional data into 2D matrices for standard
# machine learning algorithms (like PCA or classifiers), while preserving
# specific dimensions.
#
# **Flatten for TRCA (Spatial)**: Keep Observations and Channels, flatten Time.
# Result: (16, 3, 10) -> (Obs, Chan, Feature=Time)

flat_spatial = container_eeg.flatten(preserve=["obs"])
print(
    f"Flattened (Spatial): {flat_spatial.shape} dims={flat_spatial.dims} | "
    f"Coords: {list(flat_spatial.coords.keys())}"
)

# %%
# **Flatten for Standard ML (2D)**: Keep Observations only.
# Result: (16, 3*10) -> (16, 30) -> (Obs, Feature=Chan*Time)

flat_ml = container_eeg.flatten(preserve=["obs"])
print(f"Flattened (Standard 2D): {flat_ml.shape} dims={flat_ml.dims}")
print(f"Sample Composite Features:\n{flat_ml.coords['feature'][:5]}")


# %%
# 4. Aggregation
# --------------
# You can aggregate data across coordinates or labels. Let's average the
# data across our "Condition" labels (A and B).

agg_cond = container_eeg.aggregate(by=container_eeg.y, stats="mean")
print(f"Aggregated by Condition (A, B): {agg_cond.shape}\nIDs={agg_cond.ids}")


# %%
# 5. Advanced Selection
# ---------------------
# The ``select()`` method is very powerful. It supports wildcards, fuzzy matching,
# mathematical operators, and even custom callables.
#
# **Wildcard Epoch Selection**

subset_epochs = container_eeg.select(obs=["*ep-0", "*ep-1"])
print(f"Selected (*ep-0, *ep-1): {subset_epochs.shape} from {container_eeg.shape}")
print(f"Selected IDs:\n{subset_epochs.ids}")

# %%
# **Case-Insensitive Selection**

subset_fuzzy = container_eeg.select(channel=["fz"], ignore_case=True, fuzzy=False)
print(f"Case-Insensitive 'fz' -> {subset_fuzzy.coords['channel']}")

# %%
# **Operator Selection (e.g., Time >= 5)**

subset_time = container_eeg.select(time={">=": 5})
print(f"Time >= 5 -> {subset_time.coords['time']}")

# %%
# **Filter by Target Label (Y)**

subset_cond = container_eeg.select(y=["B"])
print(f"Select Y='B' -> IDs:\n{subset_cond.ids[:3]}... (Total {subset_cond.shape[0]})")

# %%
# **Stratified Selection via Callable**
# Keep only the first 2 epochs for each unique subject.


def first_n_per_subject(ids_array, n=2):
    """Custom selector: keeps first n occurrences of each unique subject prefix."""
    subjects = [i.split("_")[0] for i in ids_array]

    mask = np.zeros(len(ids_array), dtype=bool)
    counts = {}
    for idx, sub in enumerate(subjects):
        if counts.get(sub, 0) < n:
            mask[idx] = True
            counts[sub] = counts.get(sub, 0) + 1
    return mask


subset_strat = container_eeg.select(ids=lambda x: first_n_per_subject(x, n=2))
print(f"First 2 epochs per subject:\n{subset_strat.ids}")

# %%
# 6. Data Scaling and Normalization
# ---------------------------------
# The container provides built-in methods for data normalization. These operations
# return a new container with the normalized data.

# Z-score normalization (mean=0, std=1) across the time dimension
zscored_eeg = container_eeg.zscore(dim="time")
print(
    f"Z-scored EEG Data:\nMean: {np.mean(zscored_eeg.X):.3f},"
    f"\nStd: {np.std(zscored_eeg.X):.3f}"
)


# %%
# 7. Restructuring Dimensions
# ---------------------------
# You can stack and unstack dimensions to change the shape of your data dynamically.
# Let's stack Observations and Channels into a single "obs_chan" dimension.

stacked = container_eeg.stack(dims=["obs", "channel"], new_dim="obs_chan")
print(f"Stacked (Obs+Chan): {stacked.shape} dims={stacked.dims}")

# %%
# And unstack it back out:

unstacked = stacked.unstack(dim="obs_chan")
print(f"Unstacked back to: {unstacked.shape} dims={unstacked.dims}")


# %%
# 8. Working with Pandas
# ----------------------
# For standard machine learning pipelines or EDA, you might want to export
# your observation metadata to a Pandas DataFrame.

df_obs = container_eeg.observation_frame()
print("Observation DataFrame (First 5 rows):")
print(df_obs.head())
