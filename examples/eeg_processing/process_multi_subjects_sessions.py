#!/usr/bin/env python
"""
Run dimensionality reduction on multiple subjects and sessions from the EEG Motor Movement/Imagery Dataset.
This demonstrates processing a BIDS dataset with multiple subjects and sessions.
"""
import os
import numpy as np
import logging
from pathlib import Path
from coco_pipe.dim_reduction import DimReductionPipeline

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to the BIDS dataset with multiple subjects
bids_root = './test_data/eegmmidb_bids'

# 1. Single subject with one run per session
logger.info("=== PROCESSING SINGLE SUBJECT WITH SPECIFIC RUNS ===")
single_subject_pipeline = DimReductionPipeline(
    type='eeg',
    method='pca',
    data_path=bids_root,
    task='motor',
    subjects=['01'],  # Single subject
    session='hands',  # Specify one session
    run='03',  # Specify the run number
    n_components=10
)

# Execute pipeline and save output
output_path = single_subject_pipeline.execute()
logger.info(f"Output saved to: {output_path}")

# Display output information
data = np.load(output_path)
logger.info(f"Single subject output shape: {data['reduced'].shape}")

# 2. Multiple subjects with a single session
logger.info("\n=== PROCESSING MULTIPLE SUBJECTS WITH A SINGLE SESSION ===")
multi_subj_single_session_pipeline = DimReductionPipeline(
    type='eeg',
    method='pca',
    data_path=bids_root,
    task='motor',
    subjects=['01', '02', '03', '04', '05'],  # Multiple subjects
    session='hands',  # Single session
    run='03',  # For hands sessions, use run 03
    n_components=10
)

# Execute pipeline and save output
output_path = multi_subj_single_session_pipeline.execute()
logger.info(f"Output saved to: {output_path}")

# Display output information
data = np.load(output_path)
logger.info(f"Multiple subjects, single session output shape: {data['reduced'].shape}")

# 3. Multiple subjects and sessions using separate pipelines and combining results
logger.info("\n=== PROCESSING MULTIPLE SUBJECTS WITH MULTIPLE SESSIONS ===")

# Create a list to store the reduced data from different sessions
all_reduced_data = []
all_subjects_data = []
all_time_segments = []

# Process hands sessions (run 03) for all subjects
hands_pipeline = DimReductionPipeline(
    type='eeg',
    method='pca',
    data_path=bids_root,
    task='motor',
    subjects=['01', '02', '03', '04', '05'],
    session='hands',
    run='03',  # For hands sessions, use run 03
    n_components=10
)

# Execute pipeline and save output
hands_output_path = hands_pipeline.execute()
logger.info(f"Hands session output saved to: {hands_output_path}")

# Process feet sessions (run 04) for all subjects
feet_pipeline = DimReductionPipeline(
    type='eeg',
    method='pca',
    data_path=bids_root,
    task='motor',
    subjects=['01', '02', '03', '04', '05'],
    session='feet',
    run='04',  # For feet sessions, use run 04
    n_components=10
)

# Execute pipeline and save output
feet_output_path = feet_pipeline.execute()
logger.info(f"Feet session output saved to: {feet_output_path}")

# Load and combine data from both sessions
hands_data = np.load(hands_output_path)
feet_data = np.load(feet_output_path)

# Check total samples and dimensions
logger.info(f"Hands session reduced data shape: {hands_data['reduced'].shape}")
logger.info(f"Feet session reduced data shape: {feet_data['reduced'].shape}")

total_samples = hands_data['reduced'].shape[0] + feet_data['reduced'].shape[0]
logger.info(f"Total samples from all sessions: {total_samples}")

# Count samples per subject in the hands session
hands_unique_subjects = np.unique(hands_data['subjects'])
logger.info(f"Subjects in hands session: {hands_unique_subjects}")
for subj in hands_unique_subjects:
    mask = hands_data['subjects'] == subj
    logger.info(f"  Subject {subj} (hands): {np.sum(mask)} samples")

# Count samples per subject in the feet session
feet_unique_subjects = np.unique(feet_data['subjects'])
logger.info(f"Subjects in feet session: {feet_unique_subjects}")
for subj in feet_unique_subjects:
    mask = feet_data['subjects'] == subj
    logger.info(f"  Subject {subj} (feet): {np.sum(mask)} samples")

logger.info("\nAll dimensionality reduction tasks completed!") 