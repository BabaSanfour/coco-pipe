#!/usr/bin/env python3
"""
Unit tests for coco_pipe.io.meeg module.
Tests the BIDS EEG loading functionality.
"""
import os
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

import mne
import numpy as np
from mne_bids import BIDSPath

from coco_pipe.io.meeg import read_eeg_bids
from coco_pipe.io.load import load


@pytest.fixture
def bids_test_params():
    """Create test parameters for BIDS tests."""
    # Create a temporary directory for mock BIDS dataset
    temp_dir = tempfile.TemporaryDirectory()
    bids_root = Path(temp_dir.name)
    
    # Define common test parameters
    params = {
        "subject": "01",
        "session": "01",
        "task": "test",
        "datatype": "eeg",
        "suffix": "eeg",
        "temp_dir": temp_dir,  # Keep reference to prevent cleanup
        "bids_root": bids_root
    }
    
    yield params
    
    # Cleanup after tests
    temp_dir.cleanup()


@patch('coco_pipe.io.meeg.read_raw_bids')
def test_read_eeg_bids_basic(mock_read_raw_bids, bids_test_params):
    """Test that read_eeg_bids calls read_raw_bids with correct parameters."""
    # Setup mock
    mock_raw = MagicMock(spec=mne.io.Raw)
    mock_read_raw_bids.return_value = mock_raw
    
    # Call the function
    result = read_eeg_bids(
        bids_root=str(bids_test_params["bids_root"]),
        subject=bids_test_params["subject"],
        session=bids_test_params["session"],
        task=bids_test_params["task"]
    )
    
    # Assert function was called correctly
    mock_read_raw_bids.assert_called_once()
    args, kwargs = mock_read_raw_bids.call_args
    
    # Check kwargs contains bids_path
    assert 'bids_path' in kwargs
    
    # Check the BIDSPath was constructed correctly
    bids_path = kwargs['bids_path']
    # Use Path objects for comparison to handle different path representations
    assert Path(bids_path.root) == Path(str(bids_test_params["bids_root"]))
    assert bids_path.subject == bids_test_params["subject"]
    assert bids_path.session == bids_test_params["session"]
    assert bids_path.task == bids_test_params["task"]
    assert bids_path.datatype == bids_test_params["datatype"]
    assert bids_path.suffix == bids_test_params["suffix"]
    
    # Check result is the mock_raw
    assert result == mock_raw


@patch('coco_pipe.io.meeg.read_raw_bids')
def test_read_eeg_bids_extension(mock_read_raw_bids, bids_test_params):
    """Test that read_eeg_bids correctly passes extension parameter."""
    mock_raw = MagicMock(spec=mne.io.Raw)
    mock_read_raw_bids.return_value = mock_raw
    
    # Try with specified extension
    extension = '.edf'
    result = read_eeg_bids(
        bids_root=str(bids_test_params["bids_root"]),
        subject=bids_test_params["subject"],
        session=bids_test_params["session"],
        task=bids_test_params["task"],
        extension=extension
    )
    
    # Check extension was passed to BIDSPath
    args, kwargs = mock_read_raw_bids.call_args
    bids_path = kwargs['bids_path']
    assert bids_path.extension == extension


@patch('coco_pipe.io.meeg.read_raw_bids')
def test_read_eeg_bids_verbose(mock_read_raw_bids, bids_test_params):
    """Test that read_eeg_bids correctly passes verbose parameter."""
    mock_raw = MagicMock(spec=mne.io.Raw)
    mock_read_raw_bids.return_value = mock_raw
    
    # Call with verbose=True
    result = read_eeg_bids(
        bids_root=str(bids_test_params["bids_root"]),
        subject=bids_test_params["subject"],
        session=bids_test_params["session"],
        task=bids_test_params["task"],
        verbose=True
    )
    
    # Check verbose was passed to read_raw_bids
    args, kwargs = mock_read_raw_bids.call_args
    assert kwargs['verbose'] is True


@patch('coco_pipe.io.meeg.read_raw_bids')
def test_load_with_eeg_type(mock_read_raw_bids, bids_test_params):
    """Test integration with the load function for 'eeg' type."""
    mock_raw = MagicMock(spec=mne.io.Raw)
    mock_read_raw_bids.return_value = mock_raw
    
    # Call the load function with type="eeg"
    with patch('coco_pipe.io.load.read_eeg_bids', side_effect=read_eeg_bids) as mock_load_read_eeg_bids:
        result = load(
            type="eeg",
            data_path=str(bids_test_params["bids_root"]),
            subjects=bids_test_params["subject"],
            session=bids_test_params["session"],
            task=bids_test_params["task"]
        )
        
        # Check read_eeg_bids was called via the load function
        mock_load_read_eeg_bids.assert_called_once()
        
        # Verify parameters passed
        args, kwargs = mock_load_read_eeg_bids.call_args
        assert kwargs['bids_root'] == str(bids_test_params["bids_root"])
        assert kwargs['subject'] == bids_test_params["subject"]
        assert kwargs['session'] == bids_test_params["session"]
        assert kwargs['task'] == bids_test_params["task"]
        
        # Check result is the mock_raw
        assert result == mock_raw


@patch('coco_pipe.io.meeg.read_raw_bids')
def test_load_with_meeg_type(mock_read_raw_bids, bids_test_params):
    """Test integration with the load function for 'meeg' type."""
    mock_raw = MagicMock(spec=mne.io.Raw)
    mock_read_raw_bids.return_value = mock_raw
    
    # Call the load function with type="meeg"
    with patch('coco_pipe.io.load.read_eeg_bids', side_effect=read_eeg_bids) as mock_load_read_eeg_bids:
        result = load(
            type="meeg",
            data_path=str(bids_test_params["bids_root"]),
            subjects=bids_test_params["subject"],
            session=bids_test_params["session"],
            task=bids_test_params["task"]
        )
        
        # Check read_eeg_bids was called via the load function
        mock_load_read_eeg_bids.assert_called_once()
        
        # Check result is the mock_raw
        assert result == mock_raw


@patch('coco_pipe.io.meeg.read_raw_bids')
def test_subjects_list_handling(mock_read_raw_bids, bids_test_params):
    """Test that the load function correctly handles subjects as a list."""
    mock_raw = MagicMock(spec=mne.io.Raw)
    mock_read_raw_bids.return_value = mock_raw
    
    # Call with subjects as a list with one element
    with patch('coco_pipe.io.load.read_eeg_bids', side_effect=read_eeg_bids) as mock_load_read_eeg_bids:
        result = load(
            type="eeg",
            data_path=str(bids_test_params["bids_root"]),
            subjects=[bids_test_params["subject"]],  # As a list
            session=bids_test_params["session"],
            task=bids_test_params["task"]
        )
        
        # Verify read_eeg_bids was called with subject as string
        args, kwargs = mock_load_read_eeg_bids.call_args
        assert kwargs['subject'] == bids_test_params["subject"]


def test_subjects_list_multiple_error(bids_test_params):
    """Test that load raises an error when subjects list has multiple items."""
    with pytest.raises(ValueError):
        result = load(
            type="eeg",
            data_path=str(bids_test_params["bids_root"]),
            subjects=["01", "02"],  # Multiple subjects
            session=bids_test_params["session"],
            task=bids_test_params["task"]
        )


def test_missing_subject_error(bids_test_params):
    """Test that load raises an error when subjects is None."""
    with pytest.raises(ValueError):
        result = load(
            type="eeg",
            data_path=str(bids_test_params["bids_root"]),
            subjects=None,  # No subject
            session=bids_test_params["session"],
            task=bids_test_params["task"]
        ) 