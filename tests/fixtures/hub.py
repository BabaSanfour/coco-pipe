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

    for name in (
        "BENDR",
        "BIOT",
        "CBraMod",
        "EEGPT",
        "Labram",
        "LUNA",
        "SignalJEPA_Contextual",
    ):
        monkeypatch.setattr(getattr(bm, name), "from_pretrained", classmethod(build))

    def download(repo_id, filename, revision=None, **kwargs):
        # a cached snapshot folder is named by its commit
        return str(tmp_path / "snapshots" / (revision or LATEST) / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return calls
