from coco_pipe.dim_reduction import DimReductionPipeline

# Initialize pipeline with auto-detected sessions
pipeline = DimReductionPipeline(
    type="eeg",
    method="pca",
    data_path="./test_data/bids_eeg",
    task="rest",
    subjects=["pd6"],
    # No session specified - should auto-detect all available sessions
    n_components=10,
)

# Execute pipeline
output_path = pipeline.execute()
print(f"Output saved to: {output_path}")
