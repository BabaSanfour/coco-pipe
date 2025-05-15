import numpy as np
import pandas as pd
from typing import Union, Optional, List, Tuple
import mne
from typing import overload, Literal

# Import read_eeg_bids from meeg
from coco_pipe.io.meeg import read_eeg_bids

# -----------------------------------------------------------------------------
# Typing helpers & overloads for `load`
# -----------------------------------------------------------------------------

# Return-type aliases
EmbeddingsReturn = Tuple[np.ndarray, np.ndarray, np.ndarray]
TabularReturnWithTarget = Tuple[pd.DataFrame, pd.Series]
TabularReturn = pd.DataFrame
MeegReturnSingle = mne.io.Raw
MeegReturnMultiple = List[mne.io.Raw]


# Overload signatures mapping `type` argument to concrete return type.

@overload
def load(
    *,
    type: Literal["embeddings"],
    data_path: str,
    task: Optional[str] = ...,  # other params omitted for brevity
    **kwargs,
) -> EmbeddingsReturn: ...


@overload
def load(
    *,
    type: Literal["eeg", "meg", "meeg"],
    data_path: str,
    subjects: Union[str, List[str]],
    **kwargs,
) -> Union[MeegReturnSingle, MeegReturnMultiple]: ...


@overload
def load(
    *,
    type: Literal["tabular", "csv", "excel", "tsv"],
    data_path: str,
    target_col: str,
    **kwargs,
) -> TabularReturnWithTarget: ...


@overload
def load(
    *,
    type: Literal["tabular", "csv", "excel", "tsv"],
    data_path: str,
    target_col: None = ...,  # explicitly None to return full DataFrame
    **kwargs,
) -> TabularReturn: ...

# The concrete implementation follows below.

def load(
    type: str,
    data_path: str,
    task: Optional[str] = None,
    run: Optional[str] = None,
    processing: Optional[str] = None,
    subjects: Optional[Union[str, List[str]]] = None,
    max_seg: Optional[int] = None,
    flatten: bool = False,
    sensorwise: bool = False,
    target_col: Optional[str] = None,
    header: Union[int, None] = 0,
    index_col: Optional[Union[int, str]] = None,
    sheet_name: Optional[str] = None,
    sep: Optional[str] = None,
    session: Optional[str] = None,
    datatype: str = 'eeg',
    suffix: str = 'eeg',
    extension: Optional[str] = None,
    verbose: bool = False
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray, np.ndarray], pd.DataFrame, mne.io.Raw, List[mne.io.Raw]]:
    """Generic data loader for COCO-Pipe.

    Parameters
    ----------
    type : {'embeddings', 'eeg', 'meg', 'meeg', 'csv', 'tsv', 'excel', 'tabular'}
        The modality / file-type to load.
    data_path : str
        Root directory or file path, depending on *type*.
    task, run, processing : str | None
        BIDS entities used by some loaders.  *run* maps to the BIDS *run* entity;
        *session* (see below) maps to the BIDS *session* entity.
    subjects : str | list[str] | None
        Subject identifier(s).  For M/EEG loaders a list is accepted and returns
        a list of ``mne.io.Raw`` objects; for other loaders either a single
        subject or *None* (ignored) can be given.
    max_seg, flatten, sensorwise : optional
        Additional options used by the embeddings loader.
    target_col : str | None
        Target/label column for tabular data.  If *None*, the full DataFrame is
        returned without target separation.
    header, index_col, sheet_name, sep
        Standard pandas read options for tabular loaders.
    session : str | None
        BIDS *session* entity for M/EEG files.
    datatype, suffix, extension : str
        Extra BIDSPath components forwarded to ``read_eeg_bids``.
    verbose : bool
        Passed through to MNE-BIDS for verbose output.

    Returns
    -------
    Depending on *type*:

    * 'embeddings' → (embeddings, subjects, times) :tuple[np.ndarray, …]
    * M/EEG types  → ``mne.io.Raw`` or list thereof
    * Tabular with *target_col* → (X, y) :tuple[pd.DataFrame, pd.Series]
    * Tabular without *target_col* → pd.DataFrame
    """
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
        # Clarify BIDS entities: use 'session' for BIDS session and 'run' for BIDS run.
        bids_session_entity = session  # May be None if dataset has no session level.

        if subjects is None:
            raise ValueError("'subjects' must be provided for M/EEG BIDS loading.")
        
        # Support single subject string or list of subjects
        def _load_single(subj: str):
            return read_eeg_bids(
                bids_root=data_path,
                subject=subj,
                session=bids_session_entity,
                run=run,
                task=task,
                datatype=datatype,
                suffix=suffix,
                extension=extension,
                verbose=verbose,
            )

        if isinstance(subjects, list):
            return [_load_single(subj) for subj in subjects]
        else:
            return _load_single(subjects)
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