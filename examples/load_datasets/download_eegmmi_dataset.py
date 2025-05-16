#!/usr/bin/env python
"""
Download a subset of the EEG Motor Movement/Imagery Dataset from Physionet and convert it to BIDS format.
Dataset source: https://physionet.org/content/eegmmidb/1.0.0/
"""
import os
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
from mne_bids import BIDSPath, write_raw_bids
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Output directory for the BIDS dataset
bids_root = 'test_data/eegmmidb_bids'
os.makedirs(bids_root, exist_ok=True)

# Download and convert a subset of the dataset (5 subjects, 2 runs each)
subjects = [1, 2, 3, 4, 5]  # Subject numbers from the original dataset - increased to 5
runs = [3, 4]  # Motor execution task: hands vs feet

# Process each subject
for subject_id in subjects:
    logger.info(f"Processing subject {subject_id}")
    
    for run in runs:
        # Download the data for this subject and run
        logger.info(f"  Downloading subject {subject_id}, run {run}")
        file_paths = eegbci.load_data(subject=subject_id, runs=[run], path='test_data/eegmmidb_raw')
        
        # Load the downloaded data
        raw_files = [read_raw_edf(f, preload=True) for f in file_paths]
        raw = concatenate_raws(raw_files)
        
        # Rename channels to standard names
        eegbci.standardize(raw)  # Rename channels to standard names
        
        # Extract session name based on run
        if run == 3:
            session = 'hands'  # Motor execution task: open and close fists
        else:
            session = 'feet'  # Motor execution task: open and close both feet
        
        # Create events and event descriptions
        events, event_id = mne.events_from_annotations(raw)
        
        # Set standard montage for better visualization
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
        
        # Prepare and save to BIDS format
        logger.info(f"  Writing subject {subject_id}, session {session}, run {run} to BIDS format")
        bids_path = BIDSPath(
            subject=f'{subject_id:02d}',
            session=session,
            task='motor',
            run=f'{run:02d}',
            datatype='eeg',
            suffix='eeg',
            root=bids_root
        )
        
        # Add subject metadata
        raw.info['subject_info'] = {
            'his_id': f'sub-{subject_id:02d}',
            'sex': 1 if subject_id % 2 == 0 else 2,  # Just for diversity in the example
            'hand': 1  # Assuming right-handed for simplicity
        }
        
        # Write to BIDS - add allow_preload and format parameters
        write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True, format='BrainVision')

# Create a dataset_description.json file
dataset_json = {
    "Name": "EEG Motor Movement/Imagery Dataset (Subset)",
    "BIDSVersion": "1.9.0",
    "DatasetType": "raw",
    "License": "CC0",
    "Authors": ["Physionet"],
    "Acknowledgements": "Physionet EEG Motor Movement/Imagery Dataset",
    "HowToAcknowledge": "Please cite https://physionet.org/content/eegmmidb/1.0.0/",
    "Funding": ["NIH Grant"],
    "EthicsApprovals": ["Some example ethics protocol"]
}

import json
with open(os.path.join(bids_root, 'dataset_description.json'), 'w') as f:
    json.dump(dataset_json, f, indent=4)

logger.info(f"Dataset download and conversion completed. BIDS dataset saved to {bids_root}") 