"""Foundation-model embeddings for io tests, from the real extractor."""

from __future__ import annotations

import numpy as np
from mne_bids import BIDSPath

from coco_pipe.decoding._specs import SignalMetadata
from coco_pipe.decoding.foundation_models import FoundationEmbeddingExtractor
from coco_pipe.decoding.registry import get_foundation_model_spec
from coco_pipe.io import write_recording_embeddings


class TokenModel:
    """Stands in for CBraMod: tokens are (window, channel, time_patch, feature)."""

    def transform(self, X, *, return_tokens=False):
        tokens = np.stack([X[..., :5], X[..., 5:]], axis=2)
        pooled = tokens.mean(axis=(1, 2))
        return (pooled, tokens) if return_tokens else pooled

    def pool_tokens(self, tokens, pooling):
        if pooling != "max":
            raise ValueError(f"TokenModel pools by 'max', got {pooling!r}.")
        return tokens.max(axis=(1, 2))

    def get_token_output_metadata(self):
        return {
            "token_source": "test_output",
            "token_axes": ["window", "channel", "time_patch", "feature"],
            "token_observation_axes": ["channel", "time_patch"],
            "token_feature_axis": "feature",
        }

    def get_embedding_info(self):
        return get_foundation_model_spec("cbramod")


def extract(epochs, *, seed=0, channels=("T3", "Cz"), store_tokens=True, **settings):
    """The extractor's result for windows numbered ``epochs``, 10 samples each."""
    X = np.random.default_rng(seed).normal(size=(len(epochs), len(channels), 10))
    extractor = FoundationEmbeddingExtractor(
        "cbramod",
        model=TokenModel(),
        resample=False,
        store_tokens=store_tokens,
        **settings,
    )
    starts = np.asarray(epochs) * 10
    return extractor.extract(
        X,
        signal_metadata=SignalMetadata(sfreq=200.0, ch_names=list(channels)),
        window_start=starts,
        window_stop=starts + 10,
        window_index=epochs,
    )


def source_path(tmp_path, **entities):
    """The epochs file a recording's embeddings come from."""
    entities = {
        "subject": "01",
        "session": "02",
        "task": "rest",
        "run": "03",
        **entities,
    }
    return BIDSPath(
        root=tmp_path / "preproc",
        description="base",
        suffix="epo",
        extension=".fif",
        datatype="eeg",
        check=False,
        **entities,
    )


def write_recording(
    tmp_path, results, *, subject="01", label="base", bad_channels=(), **entities
):
    """Write a recording's embeddings to ``tmp_path / "out"``."""
    return write_recording_embeddings(
        results,
        tmp_path / "out",
        label=label,
        source=source_path(tmp_path, subject=subject, **entities),
        source_dataset="preproc",
        bad_channels=bad_channels,
    )
