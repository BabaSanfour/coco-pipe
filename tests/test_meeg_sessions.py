#!/usr/bin/env python3
"""
Unit tests for handling multiple M/EEG sessions in the DimReductionPipeline.
Tests that the pipeline can correctly handle single, multiple, and auto-detected sessions.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path
import numpy as np
import mne

from coco_pipe.dim_reduction import DimReductionPipeline
from coco_pipe.io.meeg import read_meeg_bids, load_meeg, load_meeg_multi_sessions, detect_sessions, detect_subjects


@pytest.fixture
def mock_eeg_data():
    """Create mock EEG data for testing."""
    # Create a mock raw object
    info = mne.create_info(ch_names=['C3', 'C4', 'Fz', 'Pz'], sfreq=100, ch_types='eeg')
    data = np.random.rand(4, 1000)  # 4 channels, 1000 time points
    mock_raw = mne.io.RawArray(data, info)
    return mock_raw


@pytest.fixture
def mock_bids_structure(tmp_path):
    """Create a mock BIDS directory structure with multiple subjects and sessions."""
    bids_root = tmp_path / "bids_dataset"
    bids_root.mkdir()
    
    # Create subject directories
    for subj in ["01", "02"]:
        subj_dir = bids_root / f"sub-{subj}"
        subj_dir.mkdir()
        
        # Create session directories for each subject
        for sess in ["pre", "post"]:
            sess_dir = subj_dir / f"ses-{sess}"
            sess_dir.mkdir()
            
            # Create eeg directory for each session
            eeg_dir = sess_dir / "eeg"
            eeg_dir.mkdir()
            
            # Create mock EEG files
            eeg_file = eeg_dir / f"sub-{subj}_ses-{sess}_task-rest_eeg.edf"
            eeg_file.touch()
    
    return bids_root


def test_detect_sessions(mock_bids_structure):
    """Test the detect_sessions function."""
    # Should find both 'pre' and 'post' sessions
    sessions = detect_sessions(mock_bids_structure, "01")
    assert len(sessions) == 2
    assert "pre" in sessions
    assert "post" in sessions
    
    # Should return empty list for non-existent subject
    sessions = detect_sessions(mock_bids_structure, "99")
    assert len(sessions) == 0


def test_detect_subjects(mock_bids_structure):
    """Test the detect_subjects function."""
    subjects = detect_subjects(mock_bids_structure)
    assert len(subjects) == 2
    assert "01" in subjects
    assert "02" in subjects


@patch('coco_pipe.io.meeg.read_meeg_bids')
def test_load_meeg_multi_sessions_explicit(mock_read_meeg_bids, mock_eeg_data, mock_bids_structure):
    """Test load_meeg_multi_sessions with explicitly specified subjects and sessions."""
    # Setup mock
    mock_read_meeg_bids.return_value = mock_eeg_data
    
    # Test with explicit subjects and sessions
    raw_data = load_meeg_multi_sessions(
        bids_root=mock_bids_structure,
        subjects=["01", "02"],
        sessions=["pre", "post"],
        task="rest"
    )
    
    # Should load 4 sessions (2 subjects × 2 sessions)
    assert len(raw_data) == 4
    assert mock_read_meeg_bids.call_count == 4


@patch('coco_pipe.io.meeg.read_meeg_bids')
def test_load_meeg_multi_sessions_auto_detect(mock_read_meeg_bids, mock_eeg_data, mock_bids_structure):
    """Test load_meeg_multi_sessions with auto-detection of subjects and sessions."""
    # Setup mock
    mock_read_meeg_bids.return_value = mock_eeg_data
    
    # Test with auto-detection (no subjects or sessions specified)
    raw_data = load_meeg_multi_sessions(
        bids_root=mock_bids_structure,
        task="rest"
    )
    
    # Should load 4 sessions (2 subjects × 2 sessions)
    assert len(raw_data) == 4
    assert mock_read_meeg_bids.call_count == 4


@patch('coco_pipe.io.meeg.load_meeg_multi_sessions')
def test_dim_reduction_with_single_session(mock_load_meeg_multi_sessions, mock_eeg_data, mock_bids_structure):
    """Test dimension reduction pipeline with a single EEG session."""
    # Setup mock
    mock_load_meeg_multi_sessions.return_value = [mock_eeg_data]
    
    # Initialize the dimension reduction pipeline
    pipeline = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=mock_bids_structure,
        task="rest",
        subjects=["01"],
        session="pre",  # Single session
        n_components=2
    )
    
    # Mock the fit_transform method to avoid file loading issues
    with patch.object(pipeline, 'fit_transform', return_value=np.zeros((100, 2))):
        # Mock the save_outputs method to avoid file saving issues
        with patch.object(pipeline, 'save_outputs'):
            # Execute the pipeline
            output_path = pipeline.execute()
            
            # Check that load_meeg_multi_sessions was called with correct parameters
            mock_load_meeg_multi_sessions.assert_called_once()
            args, kwargs = mock_load_meeg_multi_sessions.call_args
            assert kwargs['subjects'] == ["01"]
            assert kwargs['sessions'] == ["pre"]
            
            # Check that the output path includes the session name
            assert "ses-pre" in str(output_path)


@patch('coco_pipe.io.meeg.load_meeg_multi_sessions')
def test_dim_reduction_with_multiple_sessions(mock_load_meeg_multi_sessions, mock_eeg_data, mock_bids_structure):
    """Test dimension reduction pipeline with multiple EEG sessions."""
    # Setup mock
    mock_load_meeg_multi_sessions.return_value = [mock_eeg_data, mock_eeg_data]
    
    # Initialize the dimension reduction pipeline
    pipeline = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=mock_bids_structure,
        task="rest",
        subjects=["01"],
        session=["pre", "post"],  # Multiple sessions as a list
        n_components=2
    )
    
    # Mock the fit_transform method to avoid file loading issues
    with patch.object(pipeline, 'fit_transform', return_value=np.zeros((100, 2))):
        # Mock the save_outputs method to avoid file saving issues
        with patch.object(pipeline, 'save_outputs'):
            # Execute the pipeline
            output_path = pipeline.execute()
            
            # Check that load_meeg_multi_sessions was called with correct parameters
            mock_load_meeg_multi_sessions.assert_called_once()
            args, kwargs = mock_load_meeg_multi_sessions.call_args
            assert kwargs['subjects'] == ["01"]
            assert kwargs['sessions'] == ["pre", "post"]
            
            # Check that the output path indicates multiple sessions
            assert "ses-multi" in str(output_path)
            
            # Verify that both sessions were processed
            assert len(pipeline.sessions) == 2
            assert "pre" in pipeline.sessions
            assert "post" in pipeline.sessions


@patch('coco_pipe.io.meeg.load_meeg_multi_sessions')
def test_dim_reduction_with_auto_detected_sessions(mock_load_meeg_multi_sessions, mock_eeg_data, mock_bids_structure):
    """Test dimension reduction pipeline with auto-detected sessions."""
    # Setup mock
    mock_load_meeg_multi_sessions.return_value = [mock_eeg_data, mock_eeg_data]
    
    # Initialize the dimension reduction pipeline
    pipeline = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=mock_bids_structure,
        task="rest",
        subjects=["01"],
        # No session specified - should auto-detect
        n_components=2
    )
    
    # Mock the fit_transform method to avoid file loading issues
    with patch.object(pipeline, 'fit_transform', return_value=np.zeros((100, 2))):
        # Mock the save_outputs method to avoid file saving issues
        with patch.object(pipeline, 'save_outputs'):
            # Execute the pipeline
            output_path = pipeline.execute()
            
            # Check that load_meeg_multi_sessions was called with correct parameters
            mock_load_meeg_multi_sessions.assert_called_once()
            args, kwargs = mock_load_meeg_multi_sessions.call_args
            assert kwargs['subjects'] == ["01"]
            assert kwargs['sessions'] is None  # None triggers auto-detection
            
            # Check that the output path indicates all sessions were used
            assert "ses-all" in str(output_path)


@patch('coco_pipe.io.meeg.load_meeg_multi_sessions')
def test_dim_reduction_with_multiple_subjects_and_sessions(mock_load_meeg_multi_sessions, mock_eeg_data, mock_bids_structure):
    """Test dimension reduction pipeline with multiple subjects and sessions."""
    # Setup mock - 4 datasets (2 subjects × 2 sessions)
    mock_load_meeg_multi_sessions.return_value = [mock_eeg_data] * 4
    
    # Initialize the dimension reduction pipeline
    pipeline = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=mock_bids_structure,
        task="rest",
        subjects=["01", "02"],  # Multiple subjects
        session=["pre", "post"],  # Multiple sessions
        n_components=2
    )
    
    # Mock the fit_transform method to avoid file loading issues
    with patch.object(pipeline, 'fit_transform', return_value=np.zeros((4000, 2))):
        # Mock the save_outputs method to avoid file saving issues
        with patch.object(pipeline, 'save_outputs'):
            # Execute the pipeline
            output_path = pipeline.execute()
            
            # Check that load_meeg_multi_sessions was called with correct parameters
            mock_load_meeg_multi_sessions.assert_called_once()
            args, kwargs = mock_load_meeg_multi_sessions.call_args
            assert kwargs['subjects'] == ["01", "02"]
            assert kwargs['sessions'] == ["pre", "post"]
            
            # Verify the output
            assert "ses-multi" in str(output_path)


def test_filename_generation_for_different_session_types():
    """Test that the correct filenames are generated for different session types."""
    base_dir = Path("/test/path")
    
    # Test single session
    pipeline1 = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=base_dir,
        task="rest",
        session="baseline"
    )
    assert "ses-baseline" in pipeline1.base
    
    # Test multiple sessions
    pipeline2 = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=base_dir,
        task="rest",
        session=["baseline", "followup"]
    )
    assert "ses-multi" in pipeline2.base
    
    # Test no session specified (auto-detection)
    pipeline3 = DimReductionPipeline(
        type="eeg",
        method="pca",
        data_path=base_dir,
        task="rest",
        session=None
    )
    assert "ses-all" in pipeline3.base 