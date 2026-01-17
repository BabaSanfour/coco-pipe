import logging

import numpy as np

from coco_pipe.dim_reduction import DimReductionPipeline

# Set up logging to see more details
logging.basicConfig(level=logging.INFO)

# First, let's manually load the data and check input shapes
from coco_pipe.io.meeg import load_meeg_multi_sessions

print("=== LOADING DATA MANUALLY FOR SHAPE INSPECTION ===")
raw_data = load_meeg_multi_sessions(
    bids_root="./test_data/bids_eeg",
    subjects=["pd6"],
    sessions=["on", "off"],
    task="rest",
    verbose=True,
)
print(f"Number of datasets loaded: {len(raw_data)}")
for i, raw in enumerate(raw_data):
    data = raw.get_data()
    print(f"Dataset {i+1} shape: {data.shape} (channels × time points)")
    # Extract session from filename safely
    raw_fname = str(raw.filenames[0])
    if "ses-" in raw_fname:
        session = raw_fname.split("ses-")[1].split("_")[0]
    else:
        session = "unknown"
    print(
        f"Dataset {i+1} info: Subject={raw.info.get('subject_info')}, Session={session}"
    )

# Now run the pipeline
print("\n=== RUNNING PIPELINE WITH AUTO-DETECTED SESSIONS ===")
pipeline = DimReductionPipeline(
    type="eeg",
    method="pca",
    data_path="./test_data/bids_eeg",
    task="rest",
    subjects=["pd6"],  # Using the same subject
    # Auto-detect sessions
    n_components=10,
)

# Execute pipeline
output_path = pipeline.execute()
print(f"Output saved to: {output_path}")

# Analyze output shapes
data = np.load(output_path)
print(f"\nOutput files: {data.files}")
print(f"Reduced data shape: {data['reduced'].shape}")
print(f"Subjects array shape: {data['subjects'].shape}")
print(f"Time segments shape: {data['time_segments'].shape}")

# Count unique subjects
unique_subjects = np.unique(data["subjects"])
print(f"Unique subjects in output: {unique_subjects}")

# Check how many samples per subject
for subj in unique_subjects:
    mask = data["subjects"] == subj
    print(f"Subject {subj}: {np.sum(mask)} samples")
