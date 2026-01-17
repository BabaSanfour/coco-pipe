
import mne
import numpy as np



# Simulated data for multiple subjects
def create_sim_data():
    print("=== CREATING SIMULATED MULTI-SUBJECT DATA ===")
    # Create base info
    info = mne.create_info(
        ch_names=["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"],
        sfreq=100,
        ch_types="eeg",
    )

    # Create three subjects with different data shapes
    subjects = ["sub01", "sub02", "sub03"]
    sessions = ["ses01", "ses02"]

    all_raw_data = []
    total_samples = 0

    print("\nSimulated datasets:")
    for subject_idx, subject in enumerate(subjects):
        for session_idx, session in enumerate(sessions):
            # Create different-sized data for each subject/session
            n_times = 5000 + subject_idx * 1000 + session_idx * 500
            data = np.random.randn(len(info["ch_names"]), n_times)
            raw = mne.io.RawArray(data, info.copy())

            # Set subject info
            raw.info["subject_info"] = {"his_id": subject}

            # Set a filename to simulate BIDS structure
            raw.filenames = [
                f"sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-rest_eeg.edf"
            ]

            all_raw_data.append(raw)
            total_samples += n_times

            print(
                f"  Subject {subject}, Session {session}: shape={data.shape} (channels × time points)"
            )

    # Calculate total input shape
    n_channels = len(info["ch_names"])
    print(
        f"\nTotal input data: {len(all_raw_data)} datasets, {n_channels} channels, {total_samples} time points"
    )
    print(f"Expected shape after stacking: ({total_samples}, {n_channels})")

    return all_raw_data, subjects, sessions


# Create simulated data
raw_data_list, subjects, sessions = create_sim_data()

# Mock our dimension reduction on this data
print("\n=== SIMULATING DIMENSION REDUCTION PIPELINE ===")

# Process the loaded data (similar to what happens in the DimReductionPipeline.execute method)
X_list = []
subjects_list = []

for i, raw in enumerate(raw_data_list):
    # Get the data as a numpy array
    data = raw.get_data()

    # Reshape to 2D (samples, features)
    samples = data.shape[1]
    X_list.append(
        data.reshape(data.shape[0], -1).T
    )  # Transpose to get (samples, channels)

    # Extract subject ID
    subject_id = raw.info["subject_info"].get("his_id", f"subject_{i}")
    subjects_list.extend([subject_id] * samples)

# Combine all data
X = np.vstack(X_list)
subjects_array = np.array(subjects_list)
times_array = np.arange(X.shape[0])

print(f"Input data for PCA: shape={X.shape}")

# Simulate PCA reduction to 10 components
n_components = 10
print(f"After PCA to {n_components} components: shape=({X.shape[0]}, {n_components})")

# Check how samples are distributed across subjects
unique_subjects = np.unique(subjects_array)
print("\nSamples per subject:")
for subj in unique_subjects:
    mask = subjects_array == subj
    print(f"  Subject {subj}: {np.sum(mask)} samples")

print(
    "\nThis simulation shows how the pipeline handles multiple subjects with multiple sessions."
)
print(
    "The dimensionality reduction preserves the temporal structure while reducing feature dimensions."
)
print(f"Input shape: {X.shape} -> Output shape: ({X.shape[0]}, {n_components})")
