"""Continued self-supervised pretraining of CBraMod, LUNA and REVE.

Each model goes on with the masked reconstruction it was pretrained with, on
unlabelled windows, and the weights are saved to a folder that the backends
load through ``backend_kwargs={"checkpoint_dir": ...}`` in place of the
checkpoint's.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from ._base import seeded
from ._channels import normalize_channel_names

logger = logging.getLogger(__name__)

ENCODER_FILE = "encoder.pt"
INFO_FILE = "pretraining.json"

_MASK_RATIO = {"cbramod": 0.5, "luna": 0.5, "reve": 0.55}
_OBJECTIVE = {
    "cbramod": "Bernoulli mask of patches, MSE of proj_out on the masked patches",
    "luna": (
        "random mask of the patches of each window, smooth L1 on the masked "
        "patches + 0.05 x on the others + 0.8 x query specialization"
    ),
    "reve": (
        "approximation: random mask of tokens, encoder on the visible tokens, "
        "new 2-layer decoder, L1 on the masked patches"
    ),
}
# Modules that only reconstruction uses, saved but absent from a classifier
_RECONSTRUCTION_ONLY = ("decoder_head.", "channel_emb.")
# The vocabulary of LUNA's channel embeddings: the sorted union of the SEED,
# TUEG and Siena channels (pulp-bio/BioFoundation,
# models/modules/channel_embeddings.py).
_LUNA_CHANNELS = [
    "A1-T3", "AF3", "AF4", "C1", "C2", "C3", "C3-CZ", "C3-P3", "C4", "C4-P4",
    "C4-T4", "C5", "C6", "CB1", "CB2", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6",
    "CPZ", "CZ", "CZ-C4", "F1", "F10", "F2", "F3", "F3-C3", "F4", "F4-C4", "F5",
    "F6", "F7", "F7-T3", "F8", "F8-T4", "F9", "FC1", "FC2", "FC3", "FC4", "FC5",
    "FC6", "FCZ", "FP1", "FP1-F3", "FP1-F7", "FP2", "FP2-F4", "FP2-F8", "FPZ",
    "FT7", "FT8", "FZ", "O1", "O2", "OZ", "P1", "P2", "P3", "P3-O1", "P4",
    "P4-O2", "P5", "P6", "P7", "P8", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8",
    "POZ", "PZ", "T3", "T3-C3", "T3-T5", "T4", "T4-A2", "T4-T6", "T5", "T5-O1",
    "T6", "T6-O2", "T7", "T8", "TP7", "TP8",
]  # fmt: skip


def _random_mask(shape, ratio, x, generator):
    """Mask ``round(ratio * n)`` of the ``n`` tokens of each window at random."""
    import torch

    n_tokens = math.prod(shape[1:])
    order = torch.rand(shape[0], n_tokens, device=x.device, generator=generator)
    order = order.argsort(dim=1)
    mask = torch.zeros(shape[0], n_tokens, dtype=torch.bool, device=x.device)
    mask.scatter_(1, order[:, : round(ratio * n_tokens)], True)
    return mask.view(shape)


def _objective(model_key: str, backend, mask_ratio: float):
    """Return the module whose ``forward(x, generator)`` gives the loss to train
    on and the reconstruction loss of the masked tokens."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class CBraMod(nn.Module):
        """CBraMod's pretraining (wjq-learning/CBraMod, pretrain_trainer.py)."""

        def __init__(self) -> None:
            super().__init__()
            # loaded again, with proj_out, whatever the backend keeps of it
            net = backend._model
            hub_repo = backend._metadata.hub_repo
            self.encoder = type(net).from_pretrained(
                hub_repo,
                revision=backend.checkpoint_revisions[hub_repo],
                n_chans=net.n_chans,
                chs_info=net.chs_info,
                n_times=net.n_times,
                sfreq=net.sfreq,
                return_encoder_output=True,
            )

        def forward(self, x, generator=None):
            model = self.encoder
            patches = model.rearrange(x)  # (batch, channel, patch, sample)
            shape = patches.shape[:3]
            mask = torch.rand(shape, device=x.device, generator=generator) < mask_ratio
            tokens = model.encoder(model.patch_embedding(patches, mask))
            # proj_out is the head that reconstructs the patches
            loss = F.mse_loss(model.proj_out(tokens)[mask].float(), patches[mask])
            return loss, loss

    class LUNA(nn.Module):
        """LUNA's pretraining (pulp-bio/BioFoundation, tasks/pretrain_task_LUNA.py,
        criterion/, config/experiment/LUNA_pretrain.yaml)."""

        def __init__(self) -> None:
            super().__init__()
            net = backend._model
            hub_repo = backend._metadata.hub_repo
            # n_outputs=0 builds the decoder and channel embeddings
            self.encoder = type(net).from_pretrained(
                hub_repo,
                revision=backend.checkpoint_revisions[hub_repo],
                filename=backend._metadata.checkpoint_filename,
                n_outputs=0,
                n_chans=net.n_chans,
                chs_info=net.chs_info,
                n_times=net.n_times,
                sfreq=net.sfreq,
                embed_dim=net.embed_dim,
                num_queries=net.num_queries,
                depth=net.depth,
                on_non_divisible="error",
            )
            names = [name.upper() for name in backend.signal_metadata_.ch_names]
            unknown = sorted(set(names) - set(_LUNA_CHANNELS))
            if unknown:
                raise ValueError(
                    f"{unknown} are not among the channels of LUNA's decoder."
                )
            self.register_buffer(
                "channels", torch.tensor([_LUNA_CHANNELS.index(n) for n in names])
            )
            self._scores = []
            self.encoder.cross_attn.register_forward_hook(
                lambda _, __, out: self._scores.append(out[1])
            )

        def forward(self, x, generator=None):
            size = self.encoder.patch_size
            shape = (len(x), x.shape[1], x.shape[2] // size)
            mask = _random_mask(shape, mask_ratio, x, generator)
            mask = mask.repeat_interleave(size, dim=-1)
            self._scores.clear()
            out = self.encoder(
                x,
                mask=mask.float(),
                channel_names=self.channels.expand(len(x), -1),
            ).float()
            masked = F.smooth_l1_loss(out[mask], x[mask])
            unmasked = F.smooth_l1_loss(out[~mask], x[~mask])
            # the attention of the queries over the channels, kept apart
            scores = self._scores[0].float()
            overlap = scores @ scores.transpose(1, 2)
            overlap = overlap * (1 - torch.eye(overlap.shape[1], device=x.device))
            return masked + 0.05 * unmasked + 0.8 * overlap.pow(2).mean(), masked

    class REVE(nn.Module):
        """A masked autoencoder on REVE's encoder, after El Ouahidi et al.
        (2025), whose decoder and pretraining code are not released."""

        def __init__(self) -> None:
            super().__init__()
            self.encoder = backend._backbone
            for parameter in self.encoder.parameters():
                parameter.requires_grad = True
            self.position_bank = backend._pos_bank
            self.names = list(backend._electrode_names)
            dim = self.encoder.embed_dim
            self.mask_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                dim,
                max(1, dim // 64),
                dim_feedforward=4 * dim,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(dim)
            self.to_patch = nn.Linear(dim, self.encoder.patch_size)

        def forward(self, x, generator=None):
            # REVE's forward, but only the visible tokens enter the encoder
            model = self.encoder
            size = model.patch_size
            patches = x.unfold(2, size, size - model.overlap_size).flatten(1, 2)
            n_patches = patches.shape[1] // x.shape[1]
            pos = self.position_bank(self.names).expand(len(x), -1, -1)
            pos = type(model.fourier4d).add_time_patch(pos, n_patches)
            pos_embed = model.ln(model.fourier4d(pos) + model.mlp4d(pos))
            tokens = model.to_patch_embedding(patches) + pos_embed
            mask = _random_mask(patches.shape[:2], mask_ratio, x, generator)
            n_visible = int((~mask[0]).sum())
            visible = (~mask).nonzero()[:, 1].view(len(x), n_visible, 1)
            dim = tokens.shape[-1]
            encoded = model.transformer(tokens.gather(1, visible.expand(-1, -1, dim)))
            full = self.mask_token.expand_as(tokens).to(encoded.dtype)
            full = full.scatter(1, visible.expand(-1, -1, dim), encoded)
            out = self.to_patch(self.norm(self.decoder(full + pos_embed)))
            loss = F.l1_loss(out[mask].float(), patches[mask])
            return loss, loss

    return {"cbramod": CBraMod, "luna": LUNA, "reve": REVE}[model_key]()


def continue_pretraining(
    model_key: str,
    X,
    ch_names: Sequence[str],
    sfreq: float,
    *,
    output_dir: str | Path,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 64,
    mask_ratio: float | None = None,
    weight_decay: float = 0.05,
    warmup_epochs: float = 1.0,
    held_out: float = 0.05,
    groups: Sequence | None = None,
    device: str = "auto",
    seed: int = 0,
    backend_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Go on with the self-supervised pretraining of a model on unlabelled EEG.

    The model is loaded as for embedding extraction, its input is prepared the
    same way (resampled to the model's rate, scaled and normalised as in its
    authors' pretraining), and it is trained on masked reconstruction:

    - ``"cbramod"``: a Bernoulli mask of the patches, and the MSE between the
      masked patches and their reconstruction by ``proj_out`` [1]_.
    - ``"luna"``: the patches of each window masked at random, the smooth L1
      loss on the masked patches plus 0.05 times that on the others, plus 0.8
      times the query-specialization loss [2]_.
    - ``"reve"``: an approximation of REVE's masked autoencoder [3]_, whose
      decoder and pretraining code are not released: tokens masked at random
      (not in spatio-temporal blocks, without channel dropout), the encoder on
      the visible tokens, a new two-layer transformer decoder, and the L1 loss
      on the masked patches, without the secondary loss on the pooled token.
      The position bank is not trained.

    Training uses AdamW, a linear warmup then a cosine decay of the learning
    rate by step, gradients clipped to a norm of 1, and bfloat16 autocast on
    CUDA. After each epoch, the reconstruction loss of the masked tokens on the
    training windows and on the held-out windows (with the same masks each
    epoch) is logged, and the weights are saved to ``output_dir``:
    ``encoder.pt``, the state dict of the network without its task head (with
    the reconstruction head of CBraMod and LUNA, not REVE's decoder), and
    ``pretraining.json``, this function's settings and losses. Load them with
    ``backend_kwargs={"checkpoint_dir": output_dir}``.

    Parameters
    ----------
    model_key : {"cbramod", "luna", "reve"}
        The model.
    X : array-like of shape (n_windows, n_channels, n_times)
        Windows in microvolts, e.g. a ``np.memmap``: batches are read from it
        and prepared each epoch, so windows at the model's rate (256 Hz for
        LUNA) are not resampled again and again.
    ch_names : sequence of str
        The channels of ``X``.
    sfreq : float
        The sampling rate of ``X``.
    output_dir : str or Path
        Where the weights are saved.
    epochs : int, default=10
        Passes over the training windows.
    lr : float, default=1e-4
        Peak learning rate.
    batch_size : int, default=64
        Windows per step.
    mask_ratio : float, optional
        Share of the tokens masked. Defaults to the authors': 0.5 for CBraMod
        and LUNA, 0.55 for REVE.
    weight_decay : float, default=0.05
        AdamW weight decay.
    warmup_epochs : float, default=1.0
        Epochs of linear warmup.
    held_out : float, default=0.05
        Share of the windows held out from training to watch the loss.
    groups : sequence, optional
        The recording (or subject) of each window: held-out windows then come
        from held-out groups.
    device : str, default="auto"
        Torch device.
    seed : int, default=0
        Seeds the split, the batches, the masks and new weights.
    backend_kwargs : dict, optional
        Passed to the backend's loader (e.g. ``revision``).

    Returns
    -------
    info : dict
        The content of ``pretraining.json``.

    References
    ----------
    .. [1] Wang, J., et al. (2025). CBraMod: A criss-cross brain foundation
       model for EEG decoding. ICLR.
    .. [2] Döner, B., et al. (2025). LUNA: Efficient and topology-agnostic
       foundation model for EEG signal analysis. NeurIPS.
    .. [3] El Ouahidi, Y., et al. (2025). REVE: A foundation model for EEG.
       NeurIPS.
    """
    import torch

    from ._prepare import prepare_backend
    from ._training import warmup_cosine

    if model_key not in _MASK_RATIO:
        raise ValueError(
            f"Continued pretraining supports {sorted(_MASK_RATIO)}, got {model_key!r}."
        )
    mask_ratio = _MASK_RATIO[model_key] if mask_ratio is None else float(mask_ratio)
    if not 0 < mask_ratio < 1:
        raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}.")
    if not 0 < held_out < 1:
        raise ValueError(f"held_out must be in (0, 1), got {held_out}.")
    # the input of the frozen backbone is that of the authors' pretraining
    prepared = prepare_backend(
        model_key,
        X[:1],
        device=device,
        train_mode="frozen",
        sfreq=sfreq,
        ch_names=normalize_channel_names(ch_names),
        backend_kwargs=backend_kwargs,
    )
    backend = prepared.backend
    spec = prepared.spec
    with seeded(seed):
        objective = _objective(model_key, backend, mask_ratio).to(backend.device)
    # a braindecode network was loaded again into the objective
    backend.release_memory()

    rng = np.random.default_rng(seed)
    if groups is None:
        order = rng.permutation(len(X))
        n_held_out = max(1, round(held_out * len(X)))
        held_out_rows, train_rows = order[:n_held_out], order[n_held_out:]
    else:
        from sklearn.model_selection import GroupShuffleSplit

        split = GroupShuffleSplit(1, test_size=held_out, random_state=seed)
        train_rows, held_out_rows = next(split.split(np.zeros(len(X)), groups=groups))
    n_steps = math.ceil(len(train_rows) / batch_size)

    params = [p for p in objective.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # warmup_cosine counted in steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            warmup_cosine,
            warmup_epochs=round(warmup_epochs * n_steps),
            max_epochs=epochs * n_steps,
        ),
    )
    cuda = str(backend.device).startswith("cuda")

    def batches(rows):
        for start in range(0, len(rows), batch_size):
            # sorted, for a memmap
            x = prepared.adapt(np.asarray(X[np.sort(rows[start : start + batch_size])]))
            backend._validate(x)
            yield backend._to_tensor(backend._native_input(x))

    def autocast():
        if cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "model_key": model_key,
        "base_repo": spec.hub_repo,
        "base_revision": backend.checkpoint_revisions[spec.hub_repo],
        "objective": _OBJECTIVE[model_key],
        "mask_ratio": mask_ratio,
        "epochs": 0,
        "lr": lr,
        "batch_size": batch_size,
        "weight_decay": weight_decay,
        "warmup_epochs": warmup_epochs,
        "seed": seed,
        "sfreq": prepared.target_sfreq,
        "n_times": prepared.model_n_times,
        "ch_names": prepared.ch_names,
        "n_windows": {"train": len(train_rows), "held_out": len(held_out_rows)},
        "train_loss": [],
        "held_out_loss": [],
    }
    for epoch in range(epochs):
        objective.train()
        total = 0.0
        for x in batches(rng.permutation(train_rows)):
            with autocast():
                loss, reconstruction = objective(x)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            total += reconstruction.item() * len(x)
        objective.eval()
        held_out_total = 0.0
        generator = torch.Generator(backend.device).manual_seed(seed)
        with torch.no_grad():
            for x in batches(held_out_rows):
                with autocast():
                    _, reconstruction = objective(x, generator)
                held_out_total += reconstruction.item() * len(x)
        info["epochs"] = epoch + 1
        info["train_loss"].append(total / len(train_rows))
        info["held_out_loss"].append(held_out_total / len(held_out_rows))
        logger.info(
            "%s epoch %d: reconstruction loss %.4g, held out %.4g",
            model_key,
            epoch + 1,
            info["train_loss"][-1],
            info["held_out_loss"][-1],
        )
        state = {
            key: value
            for key, value in objective.encoder.state_dict().items()
            if not key.startswith("final_layer.")
        }
        torch.save(state, output_dir / ENCODER_FILE)
        (output_dir / INFO_FILE).write_text(json.dumps(info, indent=2))
    return info


def load_continued_weights(model, checkpoint_dir, model_key: str, revision: str):
    """Load the weights of ``continue_pretraining`` into a loaded network.

    The checkpoint must come from the same model and base revision. Returns
    ``{checkpoint_dir: sha256 of encoder.pt}``, for the provenance of the
    weights.
    """
    import torch

    path = Path(checkpoint_dir)
    info = json.loads((path / INFO_FILE).read_text())
    if (info["model_key"], info["base_revision"]) != (model_key, revision):
        raise ValueError(
            f"{path} continues {info['model_key']} at revision "
            f"{info['base_revision']}, not {model_key} at revision {revision}."
        )
    state = torch.load(path / ENCODER_FILE, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if not k.startswith("final_layer.")]
    unexpected = [
        k for k in result.unexpected_keys if not k.startswith(_RECONSTRUCTION_ONLY)
    ]
    if missing or unexpected:
        raise ValueError(
            f"{path} does not fit the network: missing {missing[:5]}, "
            f"unexpected {unexpected[:5]}."
        )
    digest = hashlib.sha256((path / ENCODER_FILE).read_bytes()).hexdigest()
    return {str(path): digest}
