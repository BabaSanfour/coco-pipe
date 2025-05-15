from mne_bids import BIDSPath, read_raw_bids
import mne

def read_eeg_bids(bids_root: str, subject: str, session: str | None, task: str | None, run: str | None = None, datatype: str = 'eeg', suffix: str = 'eeg', extension: str | None = None, verbose: bool = False):
    """
    Reads EEG data from a BIDS-compliant dataset.

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
        The loaded raw EEG data.
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

# Example usage (assuming you have a BIDS dataset):
# if __name__ == '__main__':
#     bids_root_path = '/path/to/your/bids_dataset'
#     subject_id = '01'
#     session_id = '01'
#     task_name = 'taskname'
#
#     # Example 1: Auto-detect extension
#     # raw_data = read_eeg_bids(bids_root_path, subject_id, session_id, task_name)
#
#     # Example 2: Specify extension (e.g., for .bdf files)
#     # raw_data_bdf = read_eeg_bids(bids_root_path, subject_id, session_id, task_name, extension='.bdf')
#
#     # print(raw_data.info)
#     # raw_data.plot(show_scrollbars=True, show_scalebars=True)