from coco_pipe.dim_reduction import DimReductionPipeline

# Initialize pipeline with multiple sessions
pipeline = DimReductionPipeline(
    type="eeg",
    method="pca",
    data_path="./test_data/bids_eeg",
    task="rest",
    subjects=["pd6"],
    session=["on", "off"],  # Both "on" and "off" sessions
    n_components=10,
)

# Execute pipeline
output_path = pipeline.execute()
print(f"Output saved to: {output_path}")
