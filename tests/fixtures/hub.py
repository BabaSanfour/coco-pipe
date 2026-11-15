"""Braindecode foundation models built offline, with random weights.

Only the download is replaced: each model is built from its class instead of
being fetched from the Hugging Face Hub.
"""

import pytest

# The commit a repository resolves to when no revision is given.
LATEST = "0123456789abcdef0123456789abcdef01234567"
# Settings a checkpoint's config would otherwise provide.
CHECKPOINT_DEFAULTS = {
    "Labram": {"n_outputs": 1},
    "BIOT": {"n_times": 2000, "sfreq": 200.0, "n_outputs": 1},
    "BENDR": {"n_outputs": 1},
}


@pytest.fixture
def offline(monkeypatch, tmp_path):
    """Build models from their classes; return the kwargs each load received."""
    import braindecode.models as bm
    import huggingface_hub
    import mne
    import torch

    from coco_pipe.decoding.foundation_models.validation import DEFAULT_CHANNELS

    calls = []

    def build(cls, repo_id, **kwargs):
        calls.append(kwargs)
        params = {
            key: value
            for key, value in kwargs.items()
            if key not in {"revision", "filename", "token"}
        }
        for key, value in CHECKPOINT_DEFAULTS.get(cls.__name__, {}).items():
            params.setdefault(key, value)
        return cls(**params)

    classes = ("BENDR", "BIOT", "CBraMod", "Labram", "LUNA")
    for name in (*classes, "SignalJEPA_Contextual"):
        if hasattr(bm, name):
            monkeypatch.setattr(
                getattr(bm, name), "from_pretrained", classmethod(build)
            )

    # EEGPT reads its weights file itself: give it a randomly initialised one.
    info = mne.create_info(list(DEFAULT_CHANNELS), 250.0, "eeg")
    info.set_montage("colin27_1020")
    eegpt = bm.EEGPT(
        chs_info=info["chs"],
        n_chans=len(DEFAULT_CHANNELS),
        n_times=500,
        sfreq=250.0,
        n_outputs=1,
        chan_proj_type="none",
    )
    weights = tmp_path / "eegpt.pt"
    torch.save(eegpt.state_dict(), weights)

    def download(repo_id, filename, revision=None, **kwargs):
        if filename == "config.json":
            # a cached snapshot folder is named by its commit
            return str(tmp_path / "snapshots" / (revision or LATEST) / filename)
        return str(weights)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls
