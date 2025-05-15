import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple
import mne

# Import read_eeg_bids from meeg
from coco_pipe.io.meeg import read_eeg_bids

def load(
    type: str,
    data_path: str,
    task: Optional[str] = None,
    run: Optional[str] = None,  # 'run' in BIDS corresponds to 'session' in some contexts or can be part of BIDSPath
    processing: Optional[str] = None, # Potentially useful for BIDSPath if it's part of filename
    subjects: Optional[Union[str, List[str]]] = None, # read_eeg_bids expects a single subject string
    max_seg: Optional[int] = None, # Not directly used by read_eeg_bids, but kept for consistency
    flatten: bool = False, # Not directly used by read_eeg_bids for MEEG
    sensorwise: bool = False, # Not directly used by read_eeg_bids for MEEG
    target_col: Optional[str] = None,
    header: Union[int, None] = 0,
    index_col: Optional[Union[int, str]] = None,
    sheet_name: Optional[str] = None,
    sep: Optional[str] = None,
    # BIDS specific parameters that might be needed for read_eeg_bids
    session: Optional[str] = None, # Explicitly adding session for BIDS
    datatype: Optional[str] = 'eeg', # Defaulting to eeg for M/EEG types
    suffix: Optional[str] = 'eeg',     # Defaulting to eeg for M/EEG types
    extension: Optional[str] = None, # For read_eeg_bids
    verbose: bool = False # For read_eeg_bids
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame, mne.io.Raw]:
    if type == "embeddings":
        from coco_pipe.io.embeddings import load_embeddings, flatten_embeddings
        emb, subj_array, times = load_embeddings(
            embeddings_root=data_path,
            task=task,
            run=run, # This 'run' is for embeddings, BIDSPath might use it as 'run' entity
            processing=processing,
            subjects=subjects if isinstance(subjects, list) or subjects is None else [subjects], # load_embeddings can take list
            max_seg=max_seg,
        )
        emb = emb.astype(np.float32)
        if flatten:
            emb = flatten_embeddings(emb, sensorwise=sensorwise)
        # Ensuring consistent return type with other branches if possible,
        # though embeddings are typically (data, subjects, segments)
        return emb, subj_array, times
    elif type in ["meeg", "meg", "eeg"]:
        # Assuming 'data_path' is bids_root
        # 'run' from load() could map to BIDS 'run' or 'session'.
        # For read_eeg_bids, 'session' is a distinct parameter.
        # We need to clarify how 'run' in the load() signature maps to BIDSPath entities.
        # If 'run' in load() is meant to be the BIDS 'run' entity:
        bids_run_entity = run
        bids_session_entity = session # Use the new 'session' parameter

        if subjects is None:
            # read_raw_bids typically processes one subject at a time, or needs specific handling for multiple.
            # MNE-BIDS BIDSPath usually targets a single subject file.
            # Let's assume for now 'subjects' if provided is a single subject_id string, or error if list.
            raise ValueError("Subject ID must be provided for M/EEG BIDS loading via this function.")
        
        if isinstance(subjects, list):
            if len(subjects) > 1:
                # Or loop through them if read_eeg_bids is adapted, or MNE-BIDS allows multiple subjects directly
                raise ValueError("Currently, only a single subject ID string is supported for M/EEG BIDS loading.")
            subject_id = subjects[0]
        else: #  isinstance(subjects, str)
            subject_id = subjects

        # The 'task' parameter aligns well.
        # 'datatype' and 'suffix' are now parameters to load(), defaulting to 'eeg'.
        return read_eeg_bids(
            bids_root=data_path,
            subject=subject_id,
            session=bids_session_entity, # This should be the BIDS session
            task=task,
            datatype=datatype, # Pass through
            suffix=suffix, # Pass through
            extension=extension, # Pass through
            verbose=verbose # Pass through
        )
        # Note: read_eeg_bids returns mne.io.Raw. The return type hint needs mne.
        # For simplicity with current Union, I'll not add mne.io.Raw yet, but it should be.
    elif type in ["tabular", "csv", "excel", "tsv"]:
        from coco_pipe.io.tabular import load_tabular
        return load_tabular(
            data_path=data_path,
            target_col=target_col,
            header=header,
            index_col=index_col,
            sheet_name=sheet_name,
            sep=sep,
        )
    else:
        raise ValueError(f"Unknown data type '{type}', choose from 'embeddings', 'csv', 'meg', 'eeg', or 'meeg'")