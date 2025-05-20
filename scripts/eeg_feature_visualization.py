#!/usr/bin/env python3
"""
EEG Feature Extraction and Visualization with Dimensionality Reduction

This script:
1. Loads EEG data
2. Extracts features using a moving window approach
3. Applies dimensionality reduction techniques
4. Visualizes the results for different motor imagery tasks

Usage:
    # Load preprocessed features (fastest option)
    python scripts/eeg_feature_visualization.py --load-features
    
    # Generate dummy data for testing visualization
    python scripts/eeg_feature_visualization.py --test-mode
    
    # Process raw EEG data (slowest option)
    python scripts/eeg_feature_visualization.py --subjects 1
    
    # Adjust dimensionality reduction methods
    python scripts/eeg_feature_visualization.py --load-features --dim-methods "PCA,UMAP"
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy import signal
from scipy import stats
from scipy.integrate import simpson
import mne
from tqdm import tqdm
import argparse

# Suppress specific sklearn deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Try to import additional reduction methods
PACMAP_AVAILABLE = False
try:
    import pacmap
    PACMAP_AVAILABLE = True
except ImportError:
    warnings.warn("pacmap not found. PaCMAP dimensionality reduction will not be available.")

# Import dimensionality reduction
try:
    from coco_pipe.dim_reduction import DimReducer
except ImportError:
    print("coco_pipe package not found. Please install it or add it to your PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Define directories
DATA_DIR = Path("data/eeg_datasets/eeg_motor_imagery")
FEATURES_DIR = DATA_DIR / "features"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results/eeg_visualization")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Define frequency bands for feature extraction
FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

# Define EEG channels of interest
EEG_CHANNELS = [
    'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
    'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
    'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4', 'Cp6'
]

# Motor imagery event codes
EVENT_CODES = {
    0: "rest",
    1: "left_fist",
    2: "right_fist",
    3: "both_fists",
    4: "both_feet",
    5: "rotation",
    6: "forward_backward",
    7: "horizontal"
}

# Task codes for different runs
TASK_CODES = {
    1: "baseline_eyes_open",
    2: "baseline_eyes_closed",
    3: "task_one",
    4: "motor_imagery_left_right_fist",
    5: "motor_imagery_hands_feet",
    6: "motor_imagery_rotation",
    7: "motor_execution_left_right_fist",
    8: "motor_execution_hands_feet",
    9: "motor_execution_rotation",
    10: "eyes",
    11: "numbers",
    12: "letters",
    13: "navigation",
    14: "subtraction"
}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process EEG data with moving windows and visualize with dimensionality reduction")
    parser.add_argument(
        "--load-features",
        action="store_true",
        help="Load preprocessed features instead of raw EEG data"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with minimal processing for debugging"
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=0.5,
        help="Size of the moving window in seconds (default: 0.5)"
    )
    parser.add_argument(
        "--window-step",
        type=float,
        default=0.1,
        help="Step size for the moving window in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default="all",
        help="Comma-separated list of subject numbers to include or 'all' for all subjects (default: 'all')"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",  # Changed default from "4,6" to "all"
        help="Comma-separated list of task run numbers to include or 'all' for all tasks (default: 'all')"
    )
    parser.add_argument(
        "--dim-methods",
        type=str,
        default="PCA,TSNE,UMAP",
        help="Comma-separated list of dimensionality reduction methods (default: 'PCA,TSNE,UMAP')"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Maximum number of samples to use for visualization (default: 5000)"
    )
    parser.add_argument(
        "--color-by",
        type=str,
        default="task_label",  # Changed default from "label" to "task_label"
        choices=["label", "task", "subject", "task_label"],
        help="Parameter to color the visualization by (default: 'task_label')"
    )
    parser.add_argument(
        "--combined-plot",
        action="store_true",
        default=True,  # Set default to True to create combined plots
        help="Create combined plot with all tasks and conditions"
    )
    parser.add_argument(
        "--num-subjects",
        type=int,
        default=20,
        help="If set and --subjects is 'all', process only the first N subjects"
    )
    return parser.parse_args()

def load_preprocessed_features():
    """
    Load preprocessed features from pickle file.
    
    Returns:
        pd.DataFrame: DataFrame containing features or None if loading fails
    """
    features_file = FEATURES_DIR / "eeg_motor_imagery_features.pkl"
    
    if not features_file.exists():
        logger.error(f"Features file not found at {features_file}")
        logger.info("Please run scripts/process_eeg_features.py first to extract features")
        return None
    
    try:
        logger.info(f"Loading preprocessed features from {features_file}")
        with open(features_file, 'rb') as f:
            df = pickle.load(f)
        return df
    except Exception as e:
        logger.error(f"Error loading features: {e}")
        return None

def extract_features_from_window(window_data, sfreq=160):
    """
    Extract features from a window of EEG data.
    
    Args:
        window_data (np.ndarray): Window of EEG data (channels x samples)
        sfreq (float): Sampling frequency
        
    Returns:
        dict: Dictionary of features
    """
    features = {}
    
    # For each channel
    for ch_idx in range(window_data.shape[0]):
        channel_data = window_data[ch_idx]
        
        # Replace NaN or Inf values
        if np.any(np.isnan(channel_data)) or np.any(np.isinf(channel_data)):
            channel_data = np.nan_to_num(channel_data)
        
        # Calculate frequency band powers
        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            f, psd = signal.welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)))
            
            # Find indices of frequencies within the band
            idx_band = np.logical_and(f >= fmin, f <= fmax)
            
            # Calculate mean power in band
            band_power = np.mean(psd[idx_band]) if np.any(idx_band) else 0
            
            # Calculate band power using Simpson's rule (more accurate area under curve)
            if np.any(idx_band):
                freq_res = f[1] - f[0]
                band_power_simps = simpson(psd[idx_band], dx=freq_res)
                features[f'ch{ch_idx}_{band_name}_power_simps'] = band_power_simps
            
            # Store in features dictionary
            features[f'ch{ch_idx}_{band_name}_power'] = band_power
            
            # Calculate band power normalized by total power
            total_power = np.sum(psd)
            if total_power > 0:
                norm_band_power = np.sum(psd[idx_band]) / total_power
                features[f'ch{ch_idx}_{band_name}_norm_power'] = norm_band_power
        
        # ----- NEW: Relative band power and power ratios -----
        band_powers = [features[f'ch{ch_idx}_{bn}_power'] for bn in FREQ_BANDS.keys()]
        total_power = np.sum(band_powers)
        if total_power == 0:
            total_power = 1e-12  # avoid division by zero
        for bn in FREQ_BANDS.keys():
            rel_key = f'ch{ch_idx}_{bn}_rel_power'
            features[rel_key] = features[f'ch{ch_idx}_{bn}_power'] / total_power
        # Simple ratios commonly used in MI literature
        alpha = features[f'ch{ch_idx}_alpha_power']
        beta = features[f'ch{ch_idx}_beta_power']
        theta = features[f'ch{ch_idx}_theta_power']
        gamma = features[f'ch{ch_idx}_gamma_power']
        delta = features[f'ch{ch_idx}_delta_power']
        # Safe divisions
        features[f'ch{ch_idx}_alpha_beta_ratio'] = alpha / beta if beta != 0 else 0
        features[f'ch{ch_idx}_theta_alpha_ratio'] = theta / alpha if alpha != 0 else 0
        features[f'ch{ch_idx}_beta_gamma_ratio'] = beta / gamma if gamma != 0 else 0
        features[f'ch{ch_idx}_alpha_theta_ratio'] = alpha / theta if theta != 0 else 0
        features[f'ch{ch_idx}_delta_theta_ratio'] = delta / theta if theta != 0 else 0
        features[f'ch{ch_idx}_delta_alpha_ratio'] = delta / alpha if alpha != 0 else 0
        features[f'ch{ch_idx}_theta_beta_ratio'] = theta / beta if beta != 0 else 0
        
        # Calculate statistical features
        features[f'ch{ch_idx}_mean'] = np.mean(channel_data)
        features[f'ch{ch_idx}_var'] = np.var(channel_data)
        features[f'ch{ch_idx}_std'] = np.std(channel_data)
        features[f'ch{ch_idx}_skew'] = stats.skew(channel_data)
        features[f'ch{ch_idx}_kurt'] = stats.kurtosis(channel_data)
        features[f'ch{ch_idx}_max'] = np.max(channel_data)
        features[f'ch{ch_idx}_min'] = np.min(channel_data)
        features[f'ch{ch_idx}_range'] = np.ptp(channel_data)
        features[f'ch{ch_idx}_energy'] = np.sum(channel_data**2)
        features[f'ch{ch_idx}_rms'] = np.sqrt(np.mean(channel_data**2))
        
        # Zero-crossing rate
        zero_crossings = np.where(np.diff(np.signbit(channel_data)))[0]
        features[f'ch{ch_idx}_zero_crossing_rate'] = len(zero_crossings) / len(channel_data)
        
        # ----- NEW: Hjorth parameters -----
        activity, mobility, complexity = _hjorth_parameters(channel_data)
        features[f'ch{ch_idx}_hjorth_activity'] = activity
        features[f'ch{ch_idx}_hjorth_mobility'] = mobility
        features[f'ch{ch_idx}_hjorth_complexity'] = complexity
        
        # ----- NEW: Spectral features -----
        if len(channel_data) > 1:
            # Spectral edge frequency - frequency below which 95% of power resides
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
            cumsum = np.cumsum(psd_norm)
            sef_95_idx = np.argmax(cumsum >= 0.95) if np.any(cumsum >= 0.95) else len(cumsum) - 1
            sef_95 = f[sef_95_idx]
            features[f'ch{ch_idx}_sef_95'] = sef_95
            
            # Spectral entropy
            psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else np.ones_like(psd) / len(psd)
            spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
            features[f'ch{ch_idx}_spectral_entropy'] = spectral_entropy
            
            # Spectral peak features
            peak_freq = f[np.argmax(psd)]
            features[f'ch{ch_idx}_peak_freq'] = peak_freq
            
            # Spectral moment features
            spectral_moment_1 = np.sum(f * psd) / np.sum(psd) if np.sum(psd) > 0 else 0
            features[f'ch{ch_idx}_spectral_moment_1'] = spectral_moment_1
    
    # Calculate connectivity features for selected channels
    # Select a subset of channels to reduce computational complexity
    channel_subset = list(range(0, window_data.shape[0], max(1, window_data.shape[0] // 5)))[:5]
    
    for i in range(len(channel_subset)):
        for j in range(i+1, len(channel_subset)):
            ch_i = channel_subset[i]
            ch_j = channel_subset[j]
            
            # Get channel data
            data_i = window_data[ch_i]
            data_j = window_data[ch_j]
            
            # Replace NaN or Inf values
            data_i = np.nan_to_num(data_i)
            data_j = np.nan_to_num(data_j)
            
            # Calculate coherence
            f, coh = signal.coherence(data_i, data_j, fs=sfreq, nperseg=min(256, len(data_i)))
            
            # Calculate mean coherence in each frequency band
            for band_name, (fmin, fmax) in FREQ_BANDS.items():
                idx_band = np.logical_and(f >= fmin, f <= fmax)
                band_coh = np.mean(coh[idx_band]) if np.any(idx_band) else 0
                
                # Store in features dictionary
                features[f'coh_ch{ch_i}_ch{ch_j}_{band_name}'] = band_coh
            
            # Calculate correlation and covariance between channels
            features[f'corr_ch{ch_i}_ch{ch_j}'] = np.corrcoef(data_i, data_j)[0, 1]
            features[f'cov_ch{ch_i}_ch{ch_j}'] = np.cov(data_i, data_j)[0, 1]
            
            # Calculate phase synchronization (a simple version)
            hilbert_i = np.abs(signal.hilbert(data_i))
            hilbert_j = np.abs(signal.hilbert(data_j))
            if np.sum(hilbert_i) > 0 and np.sum(hilbert_j) > 0:
                features[f'phase_sync_ch{ch_i}_ch{ch_j}'] = np.abs(np.corrcoef(hilbert_i, hilbert_j)[0, 1])
            else:
                features[f'phase_sync_ch{ch_i}_ch{ch_j}'] = 0
    
    return features

def process_raw_eeg_with_windows(args):
    """
    Process raw EEG data using moving windows to extract features.
    
    Args:
        args: Command line arguments
        
    Returns:
        pd.DataFrame: DataFrame containing extracted features
    """
    print("Starting process_raw_eeg_with_windows function")
    
    # Test mode - return dummy data for testing
    if args.test_mode:
        print("Running in TEST MODE - generating dummy data")
        # Create a simple dummy dataset
        n_samples = min(args.n_samples, 500)
        n_features = 50
        dummy_data = []
        
        for i in range(n_samples):
            features = {f'feature_{j}': np.random.random() for j in range(n_features)}
            features['subject_id'] = 1
            features['task_id'] = 4
            features['task_name'] = TASK_CODES.get(4, "unknown")
            features['event_id'] = i % 10
            features['window_id'] = i
            features['label'] = i % 5  # 0-4 matching EVENT_CODES
            features['label_name'] = EVENT_CODES.get(i % 5, "unknown")
            features['window_time'] = (i % 25) / 10.0 - 0.5  # -0.5 to 2.0
            dummy_data.append(features)
        
        print(f"Created {len(dummy_data)} dummy samples")
        return pd.DataFrame(dummy_data)
    
    # Regular processing
    # Parse subject IDs
    try:
        if args.subjects.lower() == 'all':
            # Use all available subjects (may be limited later by num_subjects)
            subject_ids = None  # Will be filled in later
        else:
            subject_ids = [int(s.strip()) for s in args.subjects.split(',')]
        
        # Parse task IDs
        if args.tasks.lower() == 'all':
            # Use all available tasks
            task_ids = list(TASK_CODES.keys())
        else:
            task_ids = [int(t.strip()) for t in args.tasks.split(',')]
        
        print(f"Processing task types: {task_ids}")
        if task_ids:
            task_names = [f"{tid} ({TASK_CODES.get(tid, 'unknown')})" for tid in task_ids]
            logger.info(f"Processing data for tasks: {task_names}")
    except ValueError:
        logger.error("Invalid subject or task list format. Please use comma-separated integers (e.g., '1,2,3').")
        return None
    
    # Find all EDF files in the extracted directory
    edf_dir = DATA_DIR / "temp" / "files" / "files"
    print(f"Looking for EDF files in: {edf_dir}")
    if not edf_dir.exists():
        logger.error(f"EDF directory not found at {edf_dir}")
        logger.info("Please run scripts/process_eeg_features.py first to extract the dataset")
        return None
    
    # If we're using all subjects, find all subject directories
    if subject_ids is None:
        subject_dirs = list(edf_dir.glob("S*"))
        subject_ids = []
        for subj_dir in subject_dirs:
            try:
                # Extract subject ID from directory name (e.g., S001 -> 1)
                subject_id = int(subj_dir.name[1:])
                subject_ids.append(subject_id)
            except (ValueError, IndexError):
                logger.warning(f"Could not parse subject directory: {subj_dir.name}")
                continue
        subject_ids = sorted(subject_ids)
        # If num_subjects is specified, limit to first N subjects
        if args.num_subjects is not None:
            subject_ids = subject_ids[:args.num_subjects]
            logger.info(f"Limiting to first {args.num_subjects} subjects: {subject_ids}")
        logger.info(f"Found {len(subject_ids)} subjects: {subject_ids[:5]}..." + 
                   (f" and {len(subject_ids)-5} more" if len(subject_ids) > 5 else ""))
    
    edf_files = list(edf_dir.glob("**/*.edf"))
    if not edf_files:
        logger.error("No EDF files found in the directory")
        return None
    
    # Filter files by subject ID and task type
    files_to_process = []
    for edf_file in edf_files:
        try:
            # Extract subject ID from filename (e.g., S001R04.edf -> 1)
            file_subject = int(edf_file.name[1:4])
            # Extract run ID from filename (e.g., S001R04.edf -> 4)
            file_task = int(edf_file.name[5:7])
            
            # Only include specified subjects and tasks
            if file_subject in subject_ids and file_task in task_ids:
                files_to_process.append((edf_file, file_subject, file_task))
        except (ValueError, IndexError):
            logger.warning(f"Could not parse filename: {edf_file.name}")
            continue
    
    if not files_to_process:
        logger.error("No matching EDF files found for the specified subjects and tasks")
        return None
    
    logger.info(f"Will process {len(files_to_process)} files")
    
    # Process files and extract features using moving windows
    all_features = []
    
    for edf_file, subject_id, task_id in tqdm(files_to_process, desc="Processing files"):
        # Load EEG data
        try:
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        except Exception as e:
            logger.error(f"Error loading {edf_file}: {e}")
            continue
        
        # Apply basic preprocessing
        try:
            # ----- NEW: Common average reference -----
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except Exception as e:
                logger.warning(f"Could not set average reference for {edf_file}: {e}")
            
            # Bandpass filter (1-45 Hz)
            raw.filter(1, 45, method='iir', verbose=False)
            
            # Apply notch filter for line noise if needed
            if raw.info['sfreq'] > 120:  # Only if Nyquist > 60Hz
                raw.notch_filter(np.array([60]), filter_length='auto', phase='zero', verbose=False)
        except Exception as e:
            logger.error(f"Error preprocessing {edf_file}: {e}")
            continue
        
        # Extract events
        try:
            # For simplicity, use annotations if they exist
            if len(raw.annotations) > 0:
                event_mapping = {'T0': 0, 'T1': 1, 'T2': 2, 'T3': 3, 'T4': 4}
                events, event_ids = mne.events_from_annotations(raw, event_id=event_mapping, verbose=False)
            else:
                # Try to find events in the data
                events = mne.find_events(raw, stim_channel='STI 014', consecutive=True, verbose=False)
                if len(events) == 0:
                    logger.warning(f"No events found in {edf_file}")
                    continue
                
                # Create a simple event_id mapping
                event_ids = {str(i): i for i in np.unique(events[:, 2])}
        except Exception as e:
            logger.error(f"Error extracting events from {edf_file}: {e}")
            continue
        
        # Get EEG data
        try:
            data = raw.get_data()
            sfreq = raw.info['sfreq']
            
            # For each event
            for event_idx, event in enumerate(events):
                event_time = event[0] / sfreq  # Event time in seconds
                event_id = event[2]  # Event ID
                
                # Skip unknown events
                if event_id not in EVENT_CODES:
                    continue
                
                # Define window parameters
                window_size_samples = int(args.window_size * sfreq)
                window_step_samples = int(args.window_step * sfreq)
                
                # Define epoch around the event (-0.5s to 2.0s)
                start_time_sec = event_time - 0.5
                end_time_sec = event_time + 2.0
                
                if start_time_sec < 0 or end_time_sec >= data.shape[1] / sfreq:
                    continue  # Skip if outside data range
                
                # Convert to samples
                start_sample = int(start_time_sec * sfreq)
                end_sample = int(end_time_sec * sfreq)
                
                # Extract epoch data
                epoch_data = data[:, start_sample:end_sample]
                
                # Apply moving window
                window_starts = range(0, epoch_data.shape[1] - window_size_samples, window_step_samples)
                
                for window_idx, window_start in enumerate(window_starts):
                    window_end = window_start + window_size_samples
                    window_data = epoch_data[:, window_start:window_end]
                    
                    # Extract features from window
                    features = extract_features_from_window(window_data, sfreq)
                    
                    # Add metadata
                    features['subject_id'] = subject_id
                    features['task_id'] = task_id
                    features['task_name'] = TASK_CODES.get(task_id, "unknown")
                    features['event_id'] = event_idx
                    features['window_id'] = window_idx
                    features['label'] = event_id
                    features['label_name'] = EVENT_CODES.get(event_id, "unknown")
                    
                    # Relative time of window start from event (in seconds)
                    window_time = (window_start / sfreq) - 0.5  # relative to event onset
                    features['window_time'] = window_time
                    
                    all_features.append(features)
        except Exception as e:
            logger.error(f"Error processing {edf_file}: {e}")
            continue
    
    # Create DataFrame from features
    if not all_features:
        logger.error("No features were extracted")
        return None
    
    logger.info(f"Extracted features for {len(all_features)} windows")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    # Ensure all columns have the same data type where possible
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try to convert to numeric
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    
    return df

def prepare_features_for_dim_reduction(df, args):
    """
    Prepare features for dimensionality reduction.
    
    Args:
        df (pd.DataFrame): DataFrame containing features
        args: Command line arguments
        
    Returns:
        tuple: (feature matrix, labels, label_names, metadata)
    """
    if df is None or df.empty:
        return None, None, None, None
    
    # Check if we have too many samples and need to subsample
    if len(df) > args.n_samples:
        logger.info(f"Subsampling from {len(df)} to {args.n_samples} samples for visualization")
        df = df.sample(n=args.n_samples, random_state=42)
    
    # Collect metadata for visualization - safely handle missing columns
    metadata = {}
    
    # Required columns
    if 'label' not in df.columns:
        logger.error("Required column 'label' not found in data")
        return None, None, None, None
        
    metadata['label'] = df['label'].values
    
    # Handle label names 
    if 'label_name' in df.columns:
        metadata['label_name'] = df['label_name'].values
    else:
        # Create label names from EVENT_CODES if available
        metadata['label_name'] = np.array([EVENT_CODES.get(label, str(label)) for label in metadata['label']])
    
    # Optional columns
    for col in ['subject_id', 'task_id', 'window_time']:
        if col in df.columns:
            metadata[col] = df[col].values
    
    # Add task_name if available
    if 'task_name' in df.columns:
        metadata['task_name'] = df['task_name'].values
    elif 'task_id' in metadata:
        # Create task names from TASK_CODES if available
        metadata['task_name'] = np.array([TASK_CODES.get(tid, str(tid)) for tid in metadata['task_id']])
    
    # Create combined task_label field for coloring
    if 'task_id' in metadata and 'label' in metadata:
        # Create a unique ID for each task-label combination
        task_label_values = [(100 * tid) + lid for tid, lid in zip(metadata['task_id'], metadata['label'])]
        metadata['task_label'] = np.array(task_label_values)
        
        # Create a mapping of these IDs to human-readable names
        task_label_names = []
        for tid, lid in zip(metadata['task_id'], metadata['label']):
            task_name = TASK_CODES.get(tid, f"Task{tid}")
            label_name = EVENT_CODES.get(lid, f"Label{lid}")
            task_label_names.append(f"{task_name}-{label_name}")
        metadata['task_label_name'] = np.array(task_label_names)
    
    # Get feature columns (exclude metadata and label columns)
    metadata_cols = ['subject_id', 'task_id', 'task_name', 'event_id', 'window_id', 
                     'label', 'label_name', 'window_time', 'task_label', 'task_label_name']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    if not feature_cols:
        logger.error("No feature columns found in the data")
        return None, None, None, None
    
    # Extract feature matrix
    X = df[feature_cols].values
    
    # Handle NaN or Inf values
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        logger.warning("Data contains NaN or Inf values, replacing with zeros")
        X = np.nan_to_num(X)
    
    return X, metadata['label'], metadata['label_name'], metadata

def apply_dimensionality_reduction(X, labels, args, metadata=None):
    """
    Apply dimensionality reduction to the feature matrix.
    
    Args:
        X (np.ndarray): Feature matrix
        labels (np.ndarray): Labels
        args: Command line arguments
        metadata (dict): Dictionary of metadata for visualization
        
    Returns:
        dict: Dictionary of dimensionality reduction results
    """
    # Parse methods
    methods = [m.strip().upper() for m in args.dim_methods.split(',')]
    logger.info(f"Applying dimensionality reduction methods: {methods}")
    
    results = {}
    
    # Define parameters for each method
    method_params = {
        "PCA": [
            ({}, "Default"),
            ({"whiten": True}, "Whitened")
        ],
        "TSNE": [
            ({"perplexity": 30, "learning_rate": "auto", "max_iter": 1000}, "perplexity=30"),
            ({"perplexity": 50, "learning_rate": "auto", "max_iter": 1000}, "perplexity=50")
        ],
        "UMAP": [
            ({"n_neighbors": 15, "min_dist": 0.1, "n_jobs": -1}, "n_neighbors=15"),
            ({"n_neighbors": 30, "min_dist": 0.1, "n_jobs": -1}, "n_neighbors=30")
        ],
        "PACMAP": [
            ({"n_neighbors": 10, "MN_ratio": 0.5, "FP_ratio": 2.0}, "n_neighbors=10"),
            ({"n_neighbors": 20, "MN_ratio": 0.5, "FP_ratio": 2.0}, "n_neighbors=20")
        ] if PACMAP_AVAILABLE else []
    }
    
    # Special handling for PaCMAP which is not in DimReducer
    if "PACMAP" in methods and PACMAP_AVAILABLE:
        logger.info("Applying PaCMAP directly")
        pacmap_params = method_params["PACMAP"]
        results["PACMAP"] = []
        
        for params, param_str in pacmap_params:
            try:
                logger.info(f"Applying PACMAP with {param_str}")
                
                # Initialize PaCMAP
                reducer = pacmap.PaCMAP(n_components=2, **params)
                
                # Apply reduction
                start_time = pd.Timestamp.now()
                X_reduced = reducer.fit_transform(X)
                elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
                
                # Store results
                results["PACMAP"].append({
                    'data': X_reduced, 
                    'param_str': param_str, 
                    'elapsed_time': elapsed_time,
                    'explained_variance': None  # PaCMAP doesn't provide explained variance
                })
                
                logger.info(f"PaCMAP with {param_str} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error applying PaCMAP with {param_str}: {e}")
        
        # Remove PaCMAP from the methods list to avoid DimReducer handling it
        methods = [m for m in methods if m != "PACMAP"]
    
    # Apply each method using DimReducer
    for method in methods:
        if method not in method_params:
            logger.warning(f"Unknown method: {method}, skipping")
            continue
        
        results[method] = []
        
        for params, param_str in method_params[method]:
            try:
                logger.info(f"Applying {method} with {param_str}")
                
                # Initialize reducer
                reducer = DimReducer(
                    method=method,
                    n_components=2,  # 2D for visualization
                    **params
                )
                
                # Apply reduction
                start_time = pd.Timestamp.now()
                X_reduced = reducer.fit_transform(X)
                elapsed_time = (pd.Timestamp.now() - start_time).total_seconds()
                
                # Try to get explained variance (mainly for PCA)
                explained_variance = None
                if method == "PCA" and hasattr(reducer, "explained_variance_ratio_"):
                    explained_variance = reducer.explained_variance_ratio_
                elif hasattr(reducer, "model") and hasattr(reducer.model, "explained_variance_ratio_"):
                    explained_variance = reducer.model.explained_variance_ratio_
                
                # Store results
                results[method].append({
                    'data': X_reduced, 
                    'param_str': param_str, 
                    'elapsed_time': elapsed_time,
                    'explained_variance': explained_variance
                })
                
                logger.info(f"{method} with {param_str} completed in {elapsed_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error applying {method} with {param_str}: {e}")
    
    return results

def visualize_results(results, labels, label_names, metadata=None, args=None):
    """
    Create visualizations of dimensionality reduction results.
    
    Args:
        results (dict): Dictionary of dimensionality reduction results
        labels (np.ndarray): Labels
        label_names (np.ndarray): Label names
        metadata (dict): Additional metadata for visualization
        args: Command line arguments
    """
    if not results or not any(results.values()):
        logger.error("No dimensionality reduction results to visualize")
        return
    
    # Color by option
    color_by = 'label' if args is None else args.color_by
    
    # Determine what to color by
    if color_by == 'task_label' and metadata and 'task_label' in metadata:
        color_values = metadata['task_label']
        unique_values = np.unique(color_values)
        # Create task-label name mapping
        value_names = {}
        for i, (task_label, task_label_name) in enumerate(zip(metadata['task_label'], metadata.get('task_label_name', ['unknown']*len(metadata['task_label'])))):
            value_names[task_label] = task_label_name
    elif color_by == 'label':
        color_values = labels
        # Create mapping of unique values to names
        unique_values = np.unique(labels)
        value_names = {label: name for label, name in zip(labels, label_names)}
    elif color_by == 'task' and metadata and 'task_id' in metadata:
        color_values = metadata['task_id']
        unique_values = np.unique(color_values)
        # Create task name mapping
        value_names = {}
        for task_id, task_name in zip(metadata['task_id'], metadata.get('task_name', ['unknown']*len(metadata['task_id']))):
            value_names[task_id] = task_name
    elif color_by == 'subject' and metadata and 'subject_id' in metadata:
        color_values = metadata['subject_id']
        unique_values = np.unique(color_values)
        # Just use subject IDs as names
        value_names = {subj_id: f"Subject {subj_id}" for subj_id in unique_values}
    else:
        # Fallback to label if requested option isn't available
        logger.warning(f"Color by option '{color_by}' not available, falling back to 'label'")
        color_values = labels
        unique_values = np.unique(labels)
        value_names = {label: name for label, name in zip(labels, label_names)}
    
    # Create a grid visualization
    methods = list(results.keys())
    n_methods = len(methods)
    n_params = max(len(results[method]) for method in methods)
    
    # Create figure with a grid of subplots
    fig, axes = plt.subplots(n_params, n_methods, figsize=(n_methods * 5, n_params * 4))
    
    # If only one row or column, make axes 2D
    if n_params == 1 and n_methods == 1:
        axes = np.array([[axes]])
    elif n_params == 1:
        axes = axes.reshape(1, -1)
    elif n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    # Loop through methods (columns)
    for col, method in enumerate(methods):
        method_results = results[method]
        
        # Loop through parameters for this method (rows)
        for row, result in enumerate(method_results):
            ax = axes[row, col]
            
            # Plot with different colors for each class
            scatter = ax.scatter(
                result['data'][:, 0], result['data'][:, 1], 
                c=color_values, cmap='tab20', alpha=0.7, s=5
            )
            
            # Set titles
            if row == 0:
                ax.set_title(f"{method}", fontsize=14, pad=10)
            
            # Add parameter information
            param_text = f"{result['param_str']} ({result['elapsed_time']:.1f}s)"
            
            # Add explained variance if available (mainly for PCA)
            if result['explained_variance'] is not None:
                variance_pct = sum(result['explained_variance']) * 100
                param_text += f" - {variance_pct:.1f}% var"
                
            ax.text(0.5, -0.1, param_text, ha='center', transform=ax.transAxes, fontsize=10)
            
            # Remove ticks for cleaner visualization
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Add legend
    handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
    
    # Create legend labels based on color_by option
    labels_for_legend = []
    for value in unique_values:
        name = value_names.get(value, "unknown")
        if isinstance(name, str):
            labels_for_legend.append(f"{value} ({name})")
        else:
            labels_for_legend.append(f"{value}")
    
    # Position legend at bottom of figure - adjust for many items
    ncols = min(5, len(unique_values))
    if len(unique_values) > 15:
        # Use smaller font for many items
        plt.figlegend(handles, labels_for_legend, loc="upper center", 
                     bbox_to_anchor=(0.5, 0.05), ncol=ncols, fontsize=8)
    else:
        plt.figlegend(handles, labels_for_legend, loc="upper center", 
                     bbox_to_anchor=(0.5, 0.05), ncol=ncols)
    
    # Create title based on coloring
    if color_by == 'label':
        title = "EEG Motor Imagery - Dimensionality Reduction (Colored by Task Type)"
    elif color_by == 'task':
        title = "EEG Motor Imagery - Dimensionality Reduction (Colored by Run Type)"
    elif color_by == 'subject':
        title = "EEG Motor Imagery - Dimensionality Reduction (Colored by Subject)"
    elif color_by == 'task_label':
        title = "EEG Motor Imagery - Dimensionality Reduction (Colored by Task-Condition Combination)"
    else:
        title = "EEG Motor Imagery - Dimensionality Reduction"
        
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    
    # Save figure
    output_path = RESULTS_DIR / f"eeg_dim_reduction_{color_by}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to {output_path}")
    
    # Also save as SVG for high-quality reproduction
    svg_path = RESULTS_DIR / f"eeg_dim_reduction_{color_by}.svg"
    plt.savefig(svg_path, format='svg', bbox_inches='tight')
    
    plt.close()
    
    # Create a special combined visualization for each method with all task-label combinations
    if args and args.combined_plot:
        for method in methods:
            for result in results[method]:
                try:
                    # Create a larger figure for the combined plot
                    plt.figure(figsize=(15, 12))
                    
                    # Create unique markers for different tasks and colors for labels
                    if 'task_id' in metadata and 'label' in metadata:
                        # Get unique tasks and labels
                        unique_tasks = np.unique(metadata['task_id'])
                        unique_labels = np.unique(metadata['label'])
                        
                        # Choose different markers for each task
                        markers = ['o', 's', '^', 'D', 'p', '*', 'X', 'P', 'h', '+']
                        
                        # Plot each task-label combination with a unique marker-color combination
                        for t_idx, task in enumerate(unique_tasks):
                            task_mask = metadata['task_id'] == task
                            task_name = TASK_CODES.get(task, f"Task {task}")
                            
                            # Choose marker (cycle if more tasks than markers)
                            marker = markers[t_idx % len(markers)]
                            
                            for l_idx, label in enumerate(unique_labels):
                                # Combined mask for this task-label combination
                                mask = task_mask & (metadata['label'] == label)
                                
                                if np.sum(mask) == 0:
                                    continue  # Skip if no data for this combination
                                    
                                label_name = EVENT_CODES.get(label, f"Label {label}")
                                
                                # Plot this combination
                                plt.scatter(
                                    result['data'][mask, 0], 
                                    result['data'][mask, 1],
                                    marker=marker, 
                                    s=50,
                                    alpha=0.8,
                                    label=f"{task_name} - {label_name}"
                                )
                        
                        plt.title(f"Combined Tasks and Conditions - {method} {result['param_str']}", fontsize=16)
                        plt.grid(alpha=0.3)
                        
                        # Add legend with smaller font if many combinations
                        if len(unique_tasks) * len(unique_labels) > 10:
                            plt.legend(fontsize=8, loc='upper right', bbox_to_anchor=(1.15, 1))
                        else:
                            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
                        
                        # Save this combined plot
                        combined_path = RESULTS_DIR / f"eeg_{method.lower()}_{result['param_str'].replace('=', '_')}_combined.png"
                        plt.savefig(combined_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        
                except Exception as e:
                    logger.error(f"Error creating combined visualization for {method}: {e}")
    
    # Create separate visualizations for each method with detailed metadata
    for method in methods:
        for result in results[method]:
            try:
                plt.figure(figsize=(12, 10))
                
                # Create scatter plot
                scatter = plt.scatter(
                    result['data'][:, 0], result['data'][:, 1],
                    c=color_values, cmap='tab20', alpha=0.7, s=20
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label(color_by.capitalize())
                
                # Add legend
                handles, _ = scatter.legend_elements(prop="colors", alpha=0.7)
                plt.legend(handles, labels_for_legend, title=f"{color_by.capitalize()}", 
                          loc="upper right", fontsize=8)
                
                # Set title with method details
                param_str = result['param_str']
                plt.title(f"{method} ({param_str}) - EEG Features", fontsize=14)
                
                # Save figure
                method_path = RESULTS_DIR / f"eeg_{method.lower()}_{param_str.replace('=', '_')}_{color_by}.png"
                plt.savefig(method_path, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                logger.error(f"Error creating visualization for {method}: {e}")
    
    # Create time-based visualizations if window_time data is available
    if metadata and 'window_time' in metadata:
        logger.info("Creating time-based visualizations")
        
        for method in methods:
            for result in results[method]:
                try:
                    plt.figure(figsize=(12, 10))
                    
                    # Create a scatter plot with window_time as colors
                    scatter = plt.scatter(
                        result['data'][:, 0], result['data'][:, 1],
                        c=metadata['window_time'], cmap='viridis',
                        alpha=0.7, s=20
                    )
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter)
                    cbar.set_label('Time relative to event (s)')
                    
                    # Set title
                    plt.title(f"{method} - {result['param_str']}: EEG Features by Time Window", fontsize=14)
                    
                    # Save figure
                    output_path = RESULTS_DIR / f"eeg_{method.lower()}_{result['param_str'].replace('=', '_')}_time_based.png"
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Also create a separate plot for each value in color_by
                    plt.figure(figsize=(14, 12))
                    
                    # Plot each value separately
                    for i, value in enumerate(unique_values):
                        mask = color_values == value
                        if np.sum(mask) == 0:
                            continue
                            
                        label_text = labels_for_legend[i] if i < len(labels_for_legend) else f"{value}"
                        plt.scatter(
                            result['data'][mask, 0], result['data'][mask, 1],
                            c=metadata['window_time'][mask], cmap='viridis',
                            alpha=0.7, s=15, label=label_text
                        )
                    
                    cbar = plt.colorbar()
                    cbar.set_label('Time relative to event (s)')
                    plt.title(f"{method} - {result['param_str']}: Time Progression by {color_by.capitalize()}", fontsize=14)
                    plt.legend(fontsize=8)
                    
                    # Save combined figure
                    combined_path = RESULTS_DIR / f"eeg_{method.lower()}_{result['param_str'].replace('=', '_')}_time_{color_by}.png"
                    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    logger.error(f"Error creating time-based visualization for {method}: {e}")
    else:
        logger.info("Skipping time-based visualizations (window_time data not available)")
    
    logger.info("Visualizations completed successfully")

def main():
    """Main function."""
    print("Starting main function")
    try:
        # Parse arguments
        args = parse_arguments()
        print(f"Arguments: {args}")
        
        # Create necessary directories
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Either load preprocessed features or process raw EEG data
        if args.load_features:
            logger.info("Loading preprocessed features")
            df = load_preprocessed_features()
        else:
            print("Will process raw EEG data")
            logger.info("Processing raw EEG data with moving windows")
            df = process_raw_eeg_with_windows(args)
        
        if df is None or df.empty:
            logger.error("No data available for processing")
            return False
        
        # Display summary information
        print("\nDataset Summary:")
        print(f"Total samples: {len(df)}")
        
        if 'subject_id' in df.columns:
            subjects = df['subject_id'].unique()
            print(f"Subjects: {len(subjects)} ({subjects[:5]}{'...' if len(subjects) > 5 else ''})")
        
        if 'task_id' in df.columns:
            tasks = df['task_id'].unique()
            task_names = []
            for task in tasks:
                task_name = TASK_CODES.get(task, "unknown")
                task_names.append(f"{task} ({task_name})")
            print(f"Tasks: {len(tasks)} - {', '.join(task_names)}")
        
        if 'label' in df.columns:
            labels = df['label'].unique()
            label_counts = df['label'].value_counts()
            print("Labels:")
            for label in labels:
                label_name = EVENT_CODES.get(label, "unknown")
                count = label_counts[label]
                print(f"  - {label} ({label_name}): {count} samples")
        
        # Save the processed features
        try:
            features_path = RESULTS_DIR / "eeg_window_features.pkl"
            df.to_pickle(features_path)
            logger.info(f"Saved window features to {features_path}")
            
            # Also save as CSV for easy inspection (sample subset if too large)
            if len(df) > 10000:
                df_sample = df.sample(n=10000, random_state=42)
                df_sample.to_csv(RESULTS_DIR / "eeg_window_features_sample.csv", index=False)
            else:
                df.to_csv(RESULTS_DIR / "eeg_window_features.csv", index=False)
        except Exception as e:
            logger.warning(f"Error saving features: {e}")
        
        # Prepare for dimensionality reduction
        X, labels, label_names, metadata = prepare_features_for_dim_reduction(df, args)
        
        if X is None:
            logger.error("Failed to prepare features for dimensionality reduction")
            return False
        
        # Apply dimensionality reduction
        results = apply_dimensionality_reduction(X, labels, args, metadata)
        
        # Visualize results
        visualize_results(results, labels, label_names, metadata, args)
        
        logger.info("Processing completed successfully")
        return True
    except Exception as e:
        logger.exception(f"Error in main function: {e}")
        print(f"Error in main function: {e}")
        return False

# ------------- New utility functions -------------
# Hjorth parameter calculation (activity, mobility, complexity)

def _hjorth_parameters(sig: np.ndarray):
    """Return Hjorth Activity, Mobility, Complexity for 1-D signal."""
    if sig.size < 2:
        return 0.0, 0.0, 0.0
    diff1 = np.diff(sig)
    diff2 = np.diff(diff1)
    var0 = np.var(sig)
    var_d1 = np.var(diff1)
    var_d2 = np.var(diff2)
    activity = var0
    mobility = np.sqrt(var_d1 / var0) if var0 != 0 else 0.0
    complexity = (np.sqrt(var_d2 / var_d1) / mobility) if var_d1 != 0 and mobility != 0 else 0.0
    return activity, mobility, complexity

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 