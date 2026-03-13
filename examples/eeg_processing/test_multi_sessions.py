from _dimred_example_utils import execute_reduction

output_path = execute_reduction(
    method="pca",
    data_path="./test_data/bids_eeg",
    type="eeg",
    task="rest",
    subjects=["pd6"],
    session=["on", "off"],  # Both "on" and "off" sessions
    n_components=10,
)
print(f"Output saved to: {output_path}")
