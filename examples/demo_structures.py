"""
Data Structures Demo
====================

Demonstrates the DataContainer and other core IO structures.
"""
import numpy as np

from coco_pipe.io.structures import DataContainer


def section(title):
    print(f"\n{'='*20} {title} {'='*20}")


# 1. Tabular (2D)
section("1. Tabular Data (2D)")
X_tab = np.random.randn(5, 3)
container_tab = DataContainer(
    X=X_tab,
    dims=("obs", "feature"),
    coords={
        "obs": [f"sub-{i}" for i in range(5)],
        "feature": ["Alpha_Cz", "Alpha_Fz", "Beta_Pz"],
    },
)
print(f"Original: {container_tab}")
# Select
try:
    subset = container_tab.select(feature=["Alpha*"])
    print(f"Selected (Alpha*):\n{subset}")
except Exception as e:
    print(f"Selection failed: {e}")


# 2. EEG (3D) - (Obs, Channel, Time)
# 2. EEG Data (3D) - (Obs, Channel, Time)
section("2. EEG Data (3D)")
# Simulate: 2 Subjects, 2 Conditions, 4 Epochs each = 16 Observations
n_subs = 2
n_conds = 2
n_epochs = 4
n_obs = n_subs * n_conds * n_epochs
n_chans = 3
n_times = 10

X_eeg = np.random.randn(n_obs, n_chans, n_times)

# Create labels
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
print(f"Original: {container_eeg}")
print(f"First 5 IDs: {container_eeg.ids[:5]}")

# Flatten for TRCA (Spatial): Keep Obs + Channel, flatten Time
# Result: (16, 3, 10) -> (Obs, Chan, Feature=Time)
flat_spatial = container_eeg.flatten(preserve=["obs"])
print(
    f"Flattened (Spatial): {flat_spatial.shape} dims={flat_spatial.dims} | Coords: {list(flat_spatial.coords.keys())}"
)

# Flatten for Standard ML (2D): Keep Obs only
# Result: (16, 3*10) -> (16, 30) -> (Obs, Feature=Chan*Time)
flat_ml = container_eeg.flatten(preserve=["obs"])
print(f"Flattened (Standard 2D): {flat_ml.shape} dims={flat_ml.dims}")
print(f"Sample Composite Features: {flat_ml.coords['feature'][:5]}")

# Test Aggregation (Average over Condition)
# We want to average all "A" epochs and all "B" epochs for each subject? No, just globally by condition.
# Let's say we group by 'y' (Conditions A, B)
print("\n--- Aggregation Test ---")
agg_cond = container_eeg.aggregate(by=container_eeg.y, method="mean")
print(f"Aggregated by Condition (A, B): {agg_cond.shape} ids={agg_cond.ids}")

# Test Selection of Specific Epochs (Wildcard)
print("\n--- Wildcard Epoch Selection ---")
# Select first 2 epochs (*ep-0, *ep-1)
try:
    subset_epochs = container_eeg.select(obs=["*ep-0", "*ep-1"])
    print(f"Selected (*ep-0, *ep-1): {subset_epochs.shape} from {container_eeg.shape}")
    print(f"Selected IDs: {subset_epochs.ids}")
except Exception as e:
    print(f"Epoch selection failed: {e}")


# Test Enhanced Selection
print("\n--- Enhanced Selection Test ---")
try:
    # 1. Case-insensitive + Fuzzy
    # 'fz' should match 'Fz'. 'Poz' might match 'Pz' via fuzzy?
    subset_fuzzy = container_eeg.select(channel=["fz"], ignore_case=True, fuzzy=False)
    print(f"Case-Insensitive 'fz' -> {subset_fuzzy.coords['channel']}")

    # 2. Operator on Time
    # Select time >= 5
    subset_time = container_eeg.select(time={">=": 5})
    print(f"Time >= 5 -> {subset_time.coords['time']}")

    # 3. Filter by Y (Condition)
    # Select only Condition 'B'
    subset_cond = container_eeg.select(y=["B"])
    print(
        f"Select Y='B' -> IDs: {subset_cond.ids[:3]}... (Total {subset_cond.shape[0]})"
    )

    # 4. Complex Scenarios Requested by User
    print("\n[User Questions Verification]")

    # Q1: "Select all sensors that have alpha" (assuming flattened 'channel_freq' or separate dims)
    # Our demo has 'channel' and 'time'. Let's simulate selecting all 'z' channels (Fz, Cz, Pz)
    subset_alpha = container_eeg.select(channel="*z")
    print(f"1. All '*z' channels: {subset_alpha.coords['channel']}")

    # Q2: "Subjects 0 to 1" (Range)
    # Note: IDs are strings 'sub-0...', so strictly we can use string comparison or separate int coords.
    # If we had integer subject_ids in coords, we could do {'>=': 0, '<=': 1}.
    # Here illustrating with Wildcard for 'sub-0*' and 'sub-1*' which covers 1 to 6 logic if named uniformly.
    subset_subs = container_eeg.select(ids=["sub-0*"])
    print(
        f"2. Subject 0 Only (Wildcard): {subset_subs.ids[:2]}... (Total {subset_subs.shape[0]})"
    )

    # Q3: "First 2 epochs for each subject" (Stratified Selection via Callable)
    def first_n_per_subject(ids_array, n=2):
        """Custom selector: keeps first n occurrences of each unique subject prefix."""
        # Extract subject part 'sub-X' from 'sub-X_cond-Y_...'
        subjects = [i.split("_")[0] for i in ids_array]

        mask = np.zeros(len(ids_array), dtype=bool)
        counts = {}
        for idx, sub in enumerate(subjects):
            if counts.get(sub, 0) < n:
                mask[idx] = True
                counts[sub] = counts.get(sub, 0) + 1
        return mask

    subset_strat = container_eeg.select(ids=lambda x: first_n_per_subject(x, n=2))
    print(f"3. First 2 epochs per subject: {subset_strat.ids}")

except Exception as e:
    print(f"Enhanced selection failed: {e}")
