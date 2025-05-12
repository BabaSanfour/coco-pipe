import numpy as np

def load(type: str, data_path: str, task: str, run: str, processing: str, subjects: list, max_seg: int, flatten: bool, sensorwise: bool):
    if type == "embeddings":
        from coco_pipe.io.embeddings import load_embeddings, flatten_embeddings
        emb, subj, times = load_embeddings(
            embeddings_root=data_path,
            task=task,
            run=run,
            processing=processing,
            subjects=subjects,
            max_seg=max_seg,
        )
        emb = emb.astype(np.float32)
        if flatten:
            emb = flatten_embeddings(emb, sensorwise=sensorwise)
        return emb, subj, times

    elif type in ["meeg", "meg", "eeg"]:
        # Import the M/EEG loader function
        from coco_pipe.io.meeg import load_meeg
        data, subj, times = load_meeg(
            data_path=data_path,
            task=task,
            run=run,
            processing=processing,
            subjects=subjects,
            max_seg=max_seg,
        )
        data = data.astype(np.float32)
        return data, subj, times

    # CSV loader for a single CSV file with all data using load_tabular in io.tabular
    elif type == "csv":
        from coco_pipe.io.tabular import load_tabular
        data, subj, times = load_tabular(
            data_path=data_path,
            task=task,
            run=run,
            processing=processing,
            subjects=subjects,
            max_seg=max_seg,
            sensorwise=sensorwise,
        )
        data = data.astype(np.float32)
        return data, subj, times

    else:
        raise ValueError(f"Unknown data type '{type}', choose from 'embeddings', 'csv', 'meg', 'eeg', or 'meeg'")