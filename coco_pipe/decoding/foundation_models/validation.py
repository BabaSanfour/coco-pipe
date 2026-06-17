"""Opt-in real-checkpoint validation for foundation-model registry entries."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .._specs import SignalMetadata
from ..registry import get_foundation_model_spec
from .estimators import FoundationClassifier
from .extraction import FoundationEmbeddingExtractor

DEFAULT_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "O1",
    "O2",
]


def _resolve_hf_token(token: str | None) -> str | None:
    """Resolve an HF token from arg, HF_TOKEN env, or `hf auth login`."""
    if token:
        return token
    env = os.environ.get("HF_TOKEN")
    if env:
        return env
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def validate_real_checkpoints(
    model_keys: Sequence[str] = ("cbramod", "labram", "reve", "luna"),
    device: str = "cpu",
    token: str | None = None,
    ch_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Download each checkpoint and run one headless forward pass.

    Parameters
    ----------
    ch_names : sequence of str, optional
        Montage to validate against. Defaults to :data:`DEFAULT_CHANNELS`
        (a representative 19-channel 10-20 set). Pass the study's actual
        channel list to exercise the exact channel adaptation (e.g. the real
        ``128 x N`` LaBraM interpolation) the cohort run will use.
    """
    records = []
    channels = list(ch_names) if ch_names else list(DEFAULT_CHANNELS)
    token = _resolve_hf_token(token)
    for model_key in model_keys:
        spec = get_foundation_model_spec(model_key)
        if spec.requires_auth and not token:
            records.append(
                {
                    "model_key": model_key,
                    "status": "authentication_required",
                    "expected_embedding_dim": spec.embedding_dim,
                    "checkpoint": spec.hub_repo,
                    "checkpoint_revision": spec.checkpoint_revision,
                    "checkpoint_filename": spec.checkpoint_filename,
                    "reason": (
                        "Run `hf auth login` or set HF_TOKEN after accepting "
                        "the gated-model license."
                    ),
                }
            )
            continue
        n_times = spec.pretrained_n_times or int(round(spec.pretrained_sfreq * 2))
        backend_kwargs: dict[str, Any] = {}
        if model_key == "labram":
            backend_kwargs["interpolate_channels"] = True
        if token is not None:
            backend_kwargs["token"] = token
        try:
            extractor = FoundationEmbeddingExtractor(
                model_key,
                device=device,
                normalize_embeddings=False,
                backend_kwargs=backend_kwargs,
            )
            result = extractor.extract(
                np.zeros((1, len(channels), n_times), dtype=np.float32),
                signal_metadata=SignalMetadata(
                    sfreq=spec.pretrained_sfreq,
                    ch_names=channels,
                ),
            )
            observed = int(result.window_embeddings.shape[1])
            records.append(
                {
                    "model_key": model_key,
                    "status": "verified"
                    if observed == spec.embedding_dim
                    else "dimension_mismatch",
                    "expected_embedding_dim": spec.embedding_dim,
                    "observed_embedding_dim": observed,
                    "checkpoint": spec.hub_repo,
                    "checkpoint_revision": spec.checkpoint_revision,
                    "checkpoint_filename": spec.checkpoint_filename,
                    "channel_adaptation": result.metadata["channel_adaptation"],
                }
            )
        except Exception as exc:
            records.append(
                {
                    "model_key": model_key,
                    "status": "failed",
                    "expected_embedding_dim": spec.embedding_dim,
                    "checkpoint": spec.hub_repo,
                    "checkpoint_revision": spec.checkpoint_revision,
                    "checkpoint_filename": spec.checkpoint_filename,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    return records


def validate_real_training(
    model_keys: Sequence[str] = ("cbramod", "labram", "reve", "luna"),
    train_modes: Sequence[str] = ("linear_probe", "full", "lora"),
    device: str = "cpu",
    token: str | None = None,
    max_epochs: int = 1,
    ch_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Run tiny grouped training and checkpoint-reload smoke tests.

    ``ch_names`` defaults to :data:`DEFAULT_CHANNELS`; pass the study montage
    to train against the exact channel adaptation used in cohort runs.
    """
    channels = list(ch_names) if ch_names else list(DEFAULT_CHANNELS)
    token = _resolve_hf_token(token)
    records: list[dict[str, Any]] = []
    for model_key in model_keys:
        spec = get_foundation_model_spec(model_key)
        if spec.requires_auth and not token:
            for train_mode in train_modes:
                records.append(
                    {
                        "model_key": model_key,
                        "train_mode": train_mode,
                        "status": "authentication_required",
                        "reason": (
                            "Run `hf auth login` or set HF_TOKEN after accepting "
                            "the gated-model license."
                        ),
                    }
                )
            continue
        n_times = spec.pretrained_n_times or int(round(spec.pretrained_sfreq * 2))
        rng = np.random.default_rng(42)
        X = rng.normal(size=(8, len(channels), n_times)).astype(np.float32)
        y = np.repeat([0, 1], 4)
        groups = np.asarray([f"group-{index}" for index in range(4) for _ in range(2)])
        backend_kwargs: dict[str, Any] = {}
        if model_key == "labram":
            backend_kwargs["interpolate_channels"] = True
        if token is not None:
            backend_kwargs["token"] = token
        for train_mode in train_modes:
            registry_mode = "frozen" if train_mode == "linear_probe" else train_mode
            if registry_mode not in spec.supported_train_modes:
                records.append(
                    {
                        "model_key": model_key,
                        "train_mode": train_mode,
                        "status": "unsupported",
                    }
                )
                continue
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"coco-{model_key}-{train_mode}-"
                ) as checkpoint_dir:
                    classifier = FoundationClassifier(
                        model_key,
                        train_mode=train_mode,
                        device=device,
                        sfreq=spec.pretrained_sfreq,
                        ch_names=channels,
                        trainer={
                            "max_epochs": max_epochs,
                            "batch_size": 2,
                            "validation_fraction": 0.25,
                        },
                        lora={"target_modules": "all-linear"},
                        backend_kwargs=backend_kwargs,
                        checkpoints={
                            "save": "best",
                            "output_dir": checkpoint_dir,
                        },
                    ).fit(X, y, groups=groups)
                    probabilities = classifier.predict_proba(X[:2])
                    checkpoint_path = classifier.checkpoint_path_
                    if checkpoint_path is None or not checkpoint_path.exists():
                        raise RuntimeError("Training did not produce a checkpoint.")
                    classifier.restore_checkpoint()
                    components = classifier._checkpoint_components()
                    trainable = sum(
                        parameter.numel()
                        for component in components.values()
                        for parameter in component.parameters()
                        if parameter.requires_grad
                    )
                    records.append(
                        {
                            "model_key": model_key,
                            "train_mode": train_mode,
                            "status": "verified",
                            "checkpoint_reload": True,
                            "probability_shape": list(probabilities.shape),
                            "trainable_parameters": int(trainable),
                            "training_history_rows": len(
                                classifier.get_training_history()
                            ),
                        }
                    )
            except Exception as exc:
                records.append(
                    {
                        "model_key": model_key,
                        "train_mode": train_mode,
                        "status": "failed",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cbramod", "labram", "reve", "luna"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help="Montage to validate against (default: representative 19-ch 10-20). "
        "Pass the study's real channel list to exercise the exact adaptation.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--training",
        action="store_true",
        help="Run grouped training/checkpoint smoke tests instead of extraction.",
    )
    parser.add_argument(
        "--train-modes",
        nargs="+",
        default=["linear_probe", "full", "lora"],
    )
    args = parser.parse_args()
    records = (
        validate_real_training(
            args.models,
            args.train_modes,
            device=args.device,
            token=args.token,
            ch_names=args.channels,
        )
        if args.training
        else validate_real_checkpoints(
            args.models,
            device=args.device,
            token=args.token,
            ch_names=args.channels,
        )
    )
    payload = json.dumps(records, indent=2, default=str)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    if any(
        record["status"] not in {"verified", "authentication_required", "unsupported"}
        for record in records
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
