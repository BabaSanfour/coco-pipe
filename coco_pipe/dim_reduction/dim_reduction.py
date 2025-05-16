#!/usr/bin/env python3
"""
run_dim_reduction.py

DimReductionPipeline: A class to orchestrate dimensionality reduction 
process for M/EEG, CSV, or Embedding data.
"""
import json
import logging
from pathlib import Path
from typing import Union, List, Optional

import numpy as np

from coco_pipe.dim_reduction.config import METHODS, METHODS_DICT
from coco_pipe.dim_reduction.reducers.base import BaseReducer
from coco_pipe.io.load import load
from coco_pipe.io.meeg import detect_subjects, detect_sessions, load_meeg_multi_sessions

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


class DimReductionPipeline:
    """
    DimReductionPipeline: A class to orchestrate the dimensionality reduction
    process for M/EEG, CSV, or Embedding data.
    """

    def __init__(
        self,
        type: str,
        method: str,
        data_path: Path,
        task: str = None,
        run: str = None,
        processing: str = None,
        subjects: Union[str, List[str]] = None,
        max_seg: int = None,
        flatten: bool = False,
        sensorwise: bool = False,
        n_components: int = 2,
        reducer_kwargs: dict = None,
        session: Union[str, List[str]] = None,  # Can be a single session or a list of sessions
    ):
        self.type = type.lower()
        if self.type not in {"embeddings", "csv", "meg", "eeg", "meeg"}:
            raise ValueError(f"Unknown data '{type}', choose from 'embeddings', 'csv', 'meg', 'eeg', 'meeg'")

        self.method = method.upper()
        if self.method not in METHODS:
            raise ValueError(f"Unknown method '{method}', choose from {METHODS}")

        ReducerCls = METHODS_DICT[self.method]
        self.reducer: BaseReducer = ReducerCls(
            n_components=n_components,
            **(reducer_kwargs or {})
        )

        self.data_path = Path(data_path)
        self.task = task
        self.run = run
        self.processing = processing
        
        # Normalize subjects to a list
        if subjects is None:
            self.subjects = None
        elif isinstance(subjects, (list, tuple)):
            self.subjects = list(subjects)
        else:
            self.subjects = [subjects]
            
        # Normalize sessions to a list or None
        if session is None:
            self.sessions = None
        elif isinstance(session, (list, tuple)):
            self.sessions = list(session)
        else:
            self.sessions = [session]
            
        self.max_seg = max_seg
        self.flatten = flatten
        self.sensorwise = sensorwise
        self.n_components = n_components
        self.reducer_kwargs = reducer_kwargs or {}

        # Prepare output paths
        self.output_dir = Path(data_path).parent / "dim_reduction"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Include session information in the base filename
        filename_parts = []
        filename_parts.append(f"{self.type}_dimred-{self.method}_{self.n_components}d")
        
        if self.task:
            filename_parts.append(f"task-{self.task}")
        
        if self.sessions and len(self.sessions) == 1:
            filename_parts.append(f"ses-{self.sessions[0]}")
        elif self.sessions and len(self.sessions) > 1:
            # Multiple sessions specified
            filename_parts.append("ses-multi")
        elif self.type in ['eeg', 'meg', 'meeg']:
            # When no session is specified for M/EEG, indicate this in the filename
            filename_parts.append("ses-all")
            
        if self.run:
            filename_parts.append(f"run-{self.run}")
            
        if self.processing:
            filename_parts.append(f"proc-{self.processing}")
            
        self.base = "_".join(filename_parts)
        
        self.reducer_path = self.output_dir / f"{self.base}_reducer.joblib"
        self.meta_path = self.output_dir / f"{self.base}_meta.json"

    def fit(self, X: np.ndarray, y: np.ndarray = None, save: bool = True) -> None:
        """
        Fit the reducer on the data.
        """
        if self.reducer_path.exists():
            logger.info(f"Loading reducer from {self.reducer_path}")
            self.reducer = BaseReducer.load(str(self.reducer_path))
            return
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        self.reducer.fit(X, y=y)
        if save:
            logger.info(f"Saving fitted reducer to {self.reducer_path}")
            self.reducer.save(str(self.reducer_path))

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, save: bool = True) -> np.ndarray:
        # Load existing reducer if present
        if self.reducer_path.exists():
            logger.info(f"Loading reducer from {self.reducer_path}")
            self.reducer = BaseReducer.load(str(self.reducer_path))
            return self.reducer.transform(X)

        # Fit and save reducer
        logger.info(f"Fitting {self.method} on data shape {X.shape}")
        reduced = self.reducer.fit_transform(X, y=y)
        if save:
            logger.info(f"Saving fitted reducer to {self.reducer_path}")
            self.reducer.save(str(self.reducer_path))
        return reduced

    def save_outputs(self, reduced: np.ndarray, subjects: np.ndarray, times: np.ndarray):
        logger.info(f"Saving reduced data to {self.output_dir}")
        np.savez_compressed(
            self.output_dir / f"{self.base}.npz",
            reduced=reduced,
            subjects=subjects,
            time_segments=times,
        )
        meta = {
            "type":             self.type,
            "method":           self.method,
            "task":             self.task,
            "run":              self.run,
            "processing":       self.processing,
            "sessions":         self.sessions,  # Store sessions as a list in metadata
            "subjects":         self.subjects,
            "max_seg":          self.max_seg,
            "sensorwise":       self.sensorwise,
            "n_components":     self.n_components,
            "reducer_kwargs":   self.reducer_kwargs,
            "reducer_file":     self.reducer_path.name,
            "output_file":      self.base + ".npz",
        }
        logger.info(f"Writing metadata to {self.meta_path}")
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def execute(self) -> Path:
        # Handle M/EEG data types
        if self.type in ['eeg', 'meg', 'meeg']:
            logger.info(f"Loading {self.type.upper()} data from {self.data_path}")
            
            # Use the load_meeg_multi_sessions function to handle subjects and sessions
            raw_data = load_meeg_multi_sessions(
                bids_root=self.data_path,
                subjects=self.subjects,
                sessions=self.sessions,
                task=self.task,
                run=self.run,
                datatype=self.type if self.type != 'meeg' else 'meg',
                suffix=self.type if self.type != 'meeg' else 'meg',
                verbose=True
            )
            
            # Process the loaded data
            if isinstance(raw_data, list):
                logger.info(f"Processing {len(raw_data)} datasets")
                # For each dataset, get the data and concatenate
                X_list = []
                subjects_list = []
                
                for i, raw in enumerate(raw_data):
                    # Get the EEG data as a numpy array
                    data = raw.get_data()
                    # Reshape to 2D (samples, features)
                    samples = data.shape[1]
                    X_list.append(data.reshape(data.shape[0], -1).T)  # Transpose to get (samples, channels)
                    
                    # Extract subject ID from raw if possible
                    if hasattr(raw, 'info') and 'subject_info' in raw.info and raw.info['subject_info'] is not None:
                        subject_id = raw.info['subject_info'].get('his_id', f"subject_{i}")
                    elif self.subjects and i < len(self.subjects):
                        subject_id = self.subjects[i]
                    else:
                        subject_id = f"subject_{i}"
                    subjects_list.extend([subject_id] * samples)
                
                # Combine all subjects
                X = np.vstack(X_list)
                subjects_array = np.array(subjects_list)
                times_array = np.arange(X.shape[0])  # Simple index as time
            else:
                # Single subject/dataset case
                data = raw_data.get_data()
                X = data.reshape(data.shape[0], -1).T  # Transpose to get (samples, channels)
                subject_id = self.subjects[0] if isinstance(self.subjects, list) else self.subjects
                subjects_array = np.array([subject_id] * X.shape[0])
                times_array = np.arange(X.shape[0])  # Simple index as time
        else:
            # For other data types, use the standard load function
            X, subjects_array, times_array = load(
                type=self.type, 
                data_path=str(self.data_path),
                task=self.task, 
                run=self.run, 
                processing=self.processing, 
                subjects=self.subjects, 
                max_seg=self.max_seg, 
                flatten=self.flatten, 
                sensorwise=self.sensorwise
            )
        
        logger.info(f"Data loaded with shape {X.shape}")
        reduced = self.fit_transform(X)
        self.save_outputs(reduced, subjects_array, times_array)
        logger.info("Pipeline complete.")
        # Return the output file path where reduced data was saved
        return self.output_dir / f"{self.base}.npz"