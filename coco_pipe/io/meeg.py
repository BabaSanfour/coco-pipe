from mne_bids import BIDSPath, read_raw_bids
import mne
from typing import Union, List, Optional
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

def read_meeg_bids(bids_root: str, subject: str, session: Optional[str], task: Optional[str], run: Optional[str] = None, datatype: str = 'eeg', suffix: str = 'eeg', extension: Optional[str] = None, verbose: bool = False):
    """
    Reads M/EEG data from a BIDS-compliant dataset.

    Parameters
    ----------
    bids_root : str
        The root directory of the BIDS dataset.
    subject : str
        The subject identifier (e.g., '01').
    session : str | None
        The session identifier (e.g., '01'). Optional if not present in the dataset.
    task : str | None
        The task identifier (e.g., 'rest'). Optional if not present in the dataset.
    run : str | None, optional
        The run identifier (e.g., '01'). Optional. If provided, it will be passed to ``BIDSPath``.
    datatype : str, optional
        The type of data to read (e.g., 'eeg', 'meg'). Defaults to 'eeg'.
    suffix : str, optional
        The suffix of the data file. Defaults to 'eeg'.
        For EEG/MEG raw data, this is typically the same as the datatype.
    extension : str | None, optional
        The file extension (e.g., '.edf', '.bdf'). If None, MNE-BIDS will
        attempt to auto-detect it. Defaults to None.
    verbose : bool, optional
        Whether to print verbose output. Defaults to False.

    Returns
    -------
    mne.io.Raw
        The loaded raw M/EEG data.
    """
    bids_path = BIDSPath(
        root=bids_root,
        subject=subject,
        session=session,
        run=run,
        task=task,
        datatype=datatype,
        suffix=suffix,
        extension=extension
    )
    
    raw = read_raw_bids(bids_path=bids_path, verbose=verbose)
    return raw


def load_meeg(
    bids_root: str, 
    subjects: Union[str, List[str]], 
    session: Optional[str] = None, 
    task: Optional[str] = None, 
    run: Optional[str] = None,
    datatype: str = 'eeg', 
    suffix: str = 'eeg', 
    extension: Optional[str] = None, 
    verbose: bool = False
) -> Union[mne.io.Raw, List[mne.io.Raw]]:
    """
    Load M/EEG data for single or multiple subjects from a BIDS-compliant dataset.

    Parameters
    ----------
    bids_root : str
        The root directory of the BIDS dataset.
    subjects : str or list of str
        Subject identifier(s) (e.g., '01' or ['01', '02', '03']).
    session : str, optional
        The session identifier (e.g., '01'). Optional if not present in the dataset.
    task : str, optional
        The task identifier (e.g., 'rest'). Optional if not present in the dataset.
    run : str, optional
        The run identifier (e.g., '01'). Optional if not present in the dataset.
    datatype : str, optional
        The type of data to read (e.g., 'eeg', 'meg'). Defaults to 'eeg'.
    suffix : str, optional
        The suffix of the data file. Defaults to 'eeg'.
    extension : str, optional
        The file extension (e.g., '.edf', '.bdf'). If None, MNE-BIDS will
        attempt to auto-detect it. Defaults to None.
    verbose : bool, optional
        Whether to print verbose output. Defaults to False.

    Returns
    -------
    mne.io.Raw or list of mne.io.Raw
        If a single subject is provided, returns a single Raw object.
        If multiple subjects are provided, returns a list of Raw objects.
    """
    if isinstance(subjects, list):
        return [read_meeg_bids(
            bids_root=bids_root,
            subject=subj,
            session=session,
            task=task,
            run=run,
            datatype=datatype,
            suffix=suffix,
            extension=extension,
            verbose=verbose
        ) for subj in subjects]
    else:
        return read_meeg_bids(
            bids_root=bids_root,
            subject=subjects,
            session=session,
            task=task,
            run=run,
            datatype=datatype,
            suffix=suffix,
            extension=extension,
            verbose=verbose
        )


def detect_sessions(bids_root: Union[str, Path], subject: str) -> List[str]:
    """
    Detect available sessions for a specific subject in a BIDS dataset.
    
    Parameters
    ----------
    bids_root : str or Path
        The root directory of the BIDS dataset.
    subject : str
        The subject identifier (e.g., '01').
        
    Returns
    -------
    list of str
        List of session identifiers found for the subject.
    """
    bids_root = Path(bids_root)
    subject_path = bids_root / f"sub-{subject}"
    
    if not subject_path.exists():
        logger.warning(f"Subject directory not found: {subject_path}")
        return []
        
    sessions = [d.name.replace('ses-', '') for d in subject_path.glob('ses-*') if d.is_dir()]
    return sessions


def detect_subjects(bids_root: Union[str, Path]) -> List[str]:
    """
    Detect all available subjects in a BIDS dataset.
    
    Parameters
    ----------
    bids_root : str or Path
        The root directory of the BIDS dataset.
        
    Returns
    -------
    list of str
        List of subject identifiers found in the dataset.
    """
    bids_root = Path(bids_root)
    subjects = [d.name.replace('sub-', '') for d in bids_root.glob('sub-*') if d.is_dir()]
    return subjects


def load_meeg_multi_sessions(
    bids_root: Union[str, Path],
    subjects: Optional[Union[str, List[str]]] = None,
    sessions: Optional[Union[str, List[str]]] = None,
    task: Optional[str] = None,
    run: Optional[str] = None,
    datatype: str = 'eeg',
    suffix: str = 'eeg',
    extension: Optional[str] = None,
    verbose: bool = False
) -> List[mne.io.Raw]:
    """
    Load M/EEG data for multiple subjects and/or sessions from a BIDS-compliant dataset.
    
    This function provides flexibility in loading data, with auto-detection capabilities:
    - If subjects is None, all subjects in the BIDS directory will be loaded
    - If sessions is None, all sessions for each subject will be loaded
    
    Parameters
    ----------
    bids_root : str or Path
        The root directory of the BIDS dataset.
    subjects : str, list of str, or None
        Subject identifier(s) (e.g., '01' or ['01', '02', '03']).
        If None, all subjects in the dataset will be loaded.
    sessions : str, list of str, or None
        Session identifier(s) (e.g., 'baseline' or ['baseline', 'follow-up']).
        If None, all sessions for each subject will be loaded.
    task : str, optional
        The task identifier (e.g., 'rest').
    run : str, optional
        The run identifier (e.g., '01').
    datatype : str, optional
        The type of data to read (e.g., 'eeg', 'meg'). Defaults to 'eeg'.
    suffix : str, optional
        The suffix of the data file. Defaults to 'eeg'.
    extension : str, optional
        The file extension (e.g., '.edf', '.bdf'). If None, MNE-BIDS will
        attempt to auto-detect it.
    verbose : bool, optional
        Whether to print verbose output. Defaults to False.
        
    Returns
    -------
    list of mne.io.Raw
        List of loaded Raw objects, one for each subject-session combination.
    """
    bids_root = Path(bids_root)
    raw_data_list = []
    
    # Handle subjects (auto-detect if None)
    if subjects is None:
        subjects_to_load = detect_subjects(bids_root)
        if not subjects_to_load:
            raise ValueError(f"No subjects found in {bids_root}")
        logger.info(f"Auto-detected subjects: {subjects_to_load}")
    elif isinstance(subjects, (list, tuple)):
        subjects_to_load = list(subjects)
    else:
        subjects_to_load = [subjects]
    
    # Process each subject
    for subject in subjects_to_load:
        # Handle sessions for this subject (auto-detect if None)
        if sessions is None:
            available_sessions = detect_sessions(bids_root, subject)
            if not available_sessions:
                logger.warning(f"No sessions found for subject {subject}")
                continue
            logger.info(f"Found sessions: {available_sessions} for subject {subject}")
            sessions_to_load = available_sessions
        elif isinstance(sessions, (list, tuple)):
            sessions_to_load = list(sessions)
        else:
            sessions_to_load = [sessions]
        
        # Load each session for this subject
        for session in sessions_to_load:
            try:
                logger.info(f"Loading data for subject {subject}, session {session}")
                session_data = read_meeg_bids(
                    bids_root=str(bids_root),
                    subject=subject,
                    session=session,
                    task=task,
                    run=run,
                    datatype=datatype,
                    suffix=suffix,
                    extension=extension,
                    verbose=verbose
                )
                raw_data_list.append(session_data)
            except Exception as e:
                logger.warning(f"Could not load data for subject {subject}, session {session}: {e}")
    
    # Check if we loaded any data
    if not raw_data_list:
        raise ValueError("No data could be loaded from any subject/session")
    
    logger.info(f"Successfully loaded {len(raw_data_list)} datasets across subjects/sessions")
    return raw_data_list