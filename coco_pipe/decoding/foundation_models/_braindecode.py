"""
BrainDecode Backend
===================

Backend that loads pretrained EEG/MEG models from the braindecode library
via ``Model.from_pretrained()``. Supports CBraMod, BIOT, LaBraM, EEGPT,
SignalJEPA, BENDR, LUNA and ST-EEGFormer (large, largeV2), and builds CTNet,
Deep4Net, EEGNet and ShallowFBCSPNet, which have no checkpoint, from random weights.

To add a new braindecode model: add one entry to ``_BD_MODEL_MAP`` and one
``FoundationModelSpec`` entry in ``coco_pipe/decoding/_specs.py``.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import warnings
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

from .._specs import SignalMetadata
from . import _montages
from ._base import BackendBase, check_weights, resolve_revision, seeded

if TYPE_CHECKING:
    from .._specs import FoundationModelSpec

_BD_MODEL_MAP: dict[str, tuple[str, str]] = {
    "cbramod": ("CBraMod", "braindecode.models"),
    "biot": ("BIOT", "braindecode.models"),
    "labram": ("Labram", "braindecode.models"),
    "eegpt": ("EEGPT", "braindecode.models"),
    "signaljepa": ("SignalJEPA_Contextual", "braindecode.models"),
    "bendr": ("BENDR", "braindecode.models"),
    "luna": ("LUNA", "braindecode.models"),
    "steegformer_large": ("STEEGFormer", "braindecode.models"),
    "steegformer_largev2": ("STEEGFormer", "braindecode.models"),
    "ctnet": ("CTNet", "braindecode.models"),
    "deep4net": ("Deep4Net", "braindecode.models"),
    "eegnet": ("EEGNet", "braindecode.models"),
    "shallowfbcspnet": ("ShallowFBCSPNet", "braindecode.models"),
}


class BrainDecodeBackend(BackendBase):
    """Backend that delegates to braindecode.models.<Model>.from_pretrained().

    Covers CBraMod, BIOT, LaBraM, EEGPT, SignalJEPA, BENDR, LUNA, ST-EEGFormer,
    and CTNet, Deep4Net, EEGNet, ShallowFBCSPNet from random weights.

    Training modes: frozen, full, lora.
    QLoRA is not supported (requires HF quantization in from_pretrained).

    BrainDecode's dict output is unwrapped in ``_forward_features`` and
    ``_forward_logits`` — no dict leaks outside those methods.
    """

    _channel_plan = None

    def __init__(
        self,
        metadata: FoundationModelSpec,
        model,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        task: str,
    ) -> None:
        self._metadata = metadata
        self._model = model
        self._n_outputs = n_outputs
        self._device = device
        self._train_mode = train_mode
        self._task = task
        self.signal_metadata_: SignalMetadata | None = None

    @classmethod
    def is_available(cls) -> bool:
        """Check whether braindecode and torch are installed.

        Returns
        -------
        available : bool
            ``True`` if both ``braindecode`` and ``torch`` are importable.
        """
        return (
            importlib.util.find_spec("braindecode") is not None
            and importlib.util.find_spec("torch") is not None
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_key: str,
        metadata: FoundationModelSpec,
        n_outputs: int | None,
        device: str,
        train_mode: str,
        task: str = "classification",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: str | list[str] = "all-linear",
        lora_dropout: float = 0.05,
        pretrained: bool = True,
        seed: int | None = None,
        **kw,
    ) -> BrainDecodeBackend:
        """Load a pretrained braindecode model and return an initialised backend.

        Parameters
        ----------
        model_key : str
            Registry key for the model (e.g. ``"cbramod"``). Must be present
            in ``_BD_MODEL_MAP``.
        metadata : FoundationModelSpec
            Registry spec providing ``hub_repo`` and signal metadata.
        n_outputs : int or None
            Number of output units. ``None`` = headless feature extractor.
        device : str
            Resolved device string (``"cpu"``, ``"cuda"``, ``"mps"``).
        train_mode : str
            One of ``"frozen"``, ``"full"``, ``"lora"``. ``"qlora"`` raises
            ``NotImplementedError``.
        task : str, default ``"classification"``
            ``"classification"`` or ``"regression"``.
        lora_r : int, default 16
            LoRA rank. Ignored unless ``train_mode="lora"``.
        lora_alpha : int, default 32
            LoRA scaling factor.
        lora_target_modules : str or list of str, default ``"all-linear"``
            Which modules to inject LoRA adapters into.
        lora_dropout : float, default 0.05
            Dropout probability inside LoRA adapters.
        pretrained : bool, default True
            ``False`` keeps the checkpoint's architecture and draws new weights
            from ``seed``.
        seed : int or None
            The seed of random weights; only with ``pretrained=False``.
        **kw
            Extra keyword arguments forwarded to ``Model.from_pretrained``.

        Returns
        -------
        backend : BrainDecodeBackend
            A fully initialised backend instance.

        Raises
        ------
        ImportError
            If braindecode or torch is not installed.
        ValueError
            If ``model_key`` is not in ``_BD_MODEL_MAP``, if the model does not
            support ``train_mode``, or if pretrained weights are asked of a
            model without a checkpoint.
        NotImplementedError
            If ``train_mode="qlora"``.
        """
        if not cls.is_available():
            raise ImportError(
                f"Model '{model_key}' requires braindecode and torch.\n"
                'To install: pip install "coco-pipe[braindecode]"'
            )
        if model_key not in _BD_MODEL_MAP:
            raise ValueError(
                f"BrainDecodeBackend does not support '{model_key}'. "
                f"Supported: {sorted(_BD_MODEL_MAP)}"
            )
        check_weights(pretrained, seed)
        if train_mode == "qlora":
            raise NotImplementedError(
                "QLoRA for BrainDecodeBackend requires HuggingFace quantization "
                "support which is not available via braindecode.from_pretrained. "
                "Use train_mode='lora' instead, or switch to hugging_face (REVE)."
            )
        if train_mode not in metadata.supported_train_modes:
            raise ValueError(
                f"{metadata.display_name} supports train_mode in "
                f"{metadata.supported_train_modes}, got {train_mode!r} "
                "('frozen' is the linear probe)."
            )
        if metadata.hub_repo is None and pretrained:
            raise ValueError(
                f"{metadata.display_name} has no pretrained weights: train it from "
                "random weights, with pretrained=False and a seed."
            )

        class_name, module_path = _BD_MODEL_MAP[model_key]
        models_module = importlib.import_module(module_path)
        electrode_names = kw.pop("electrode_names", None)
        pooling = kw.pop("pooling", "mean")
        if pooling != "mean":
            raise ValueError(
                f"braindecode models support pooling='mean', got {pooling!r}."
            )
        sfreq = float(kw.pop("sfreq", metadata.pretrained_sfreq))
        n_times = int(kw.pop("n_times", round(sfreq * 2)))
        revision = kw.pop("revision", metadata.checkpoint_revision)
        if metadata.hub_repo is not None:
            revision = resolve_revision(
                metadata.hub_repo, revision, token=kw.get("token")
            )
        filename = kw.pop("filename", metadata.checkpoint_filename)
        # Faithful fixed-montage construction (BIOT bipolar derivation, BENDR
        # reorder + computed SCALE, EEGPT name-indexed subset). The plan is the
        # shared source of truth with the capability gate (see extraction.py);
        # an unservable montage raises here and the extraction job records it.
        channel_plan = None
        if _montages.is_special(model_key):
            fill_missing = bool(kw.pop("fill_missing_channels", False))
            channel_plan = _montages.plan_channels(
                model_key,
                electrode_names or [],
                fill_missing=fill_missing,
                sfreq=sfreq,
            )
            if channel_plan.status != "available":
                raise ValueError(
                    f"{model_key} cannot be served on this montage: "
                    f"{channel_plan.reason}"
                )
        model_cls = getattr(models_module, class_name)

        # Montage-dependent models need ``chs_info``. LaBraM, SignalJEPA (with its
        # pretrained channel embeddings) and ST-EEGFormer find channels by name
        # (ST-EEGFormer places a name outside its vocabulary at the nearest 10-05
        # site); LUNA, and SignalJEPA with channel_embedding="scratch", embed their
        # positions, where braindecode gives made-up positions to channels that
        # have none.
        if electrode_names and (
            model_key in {"cbramod", "labram", "luna", "signaljepa"}
            or model_key.startswith("steegformer")
        ):
            import mne

            kw.setdefault("n_chans", len(electrode_names))
            kw.setdefault("sfreq", sfreq)
            needs_positions = model_key == "luna" or (
                model_key == "signaljepa" and kw.get("channel_embedding") == "scratch"
            )
            if "chs_info" not in kw:
                info = mne.create_info(electrode_names, sfreq=sfreq, ch_types="eeg")
                info.set_montage(
                    "colin27_1020",
                    match_case=False,
                    on_missing="raise" if needs_positions else "ignore",
                )
                kw["chs_info"] = info["chs"]
        if model_key == "luna":
            kw.setdefault("n_times", n_times)
            kw.setdefault("embed_dim", 64)
            kw.setdefault("num_queries", 4)
            kw.setdefault("depth", 8)
            # braindecode>=2 pads windows that do not divide into patches.
            kw.setdefault("on_non_divisible", "error")
        elif model_key == "cbramod":
            kw.setdefault("n_times", n_times)
        elif model_key == "labram":
            # One pretrained time embedding per 1 s patch, 16 in the checkpoint.
            kw.setdefault("n_times", n_times)
            kw.setdefault("on_non_divisible", "error")
        elif model_key == "signaljepa":
            # Load the full 62-row pretrained channel-embedding matrix and index
            # the rows matching the input ``chs_info`` by name (braindecode#991),
            # so any montage that is a subset of the pretraining set works.
            kw.setdefault("channel_embedding", "pretrain_aligned")
            # braindecode builds a classification head either way, sized from
            # the window length.
            kw.setdefault("n_times", n_times)
        elif model_key == "eegpt":
            # EEGPT finds each channel's embedding by name (chans_id), so it
            # loads on a subset of its 62 pretraining channels, with no channel
            # projection, which the checkpoint does not hold.
            import mne

            names = channel_plan.model_ch_names
            info = mne.create_info(names, metadata.pretrained_sfreq, "eeg")
            kw.update(
                chs_info=info["chs"],
                n_chans=len(names),
                n_times=n_times,
                chan_proj_type="none",
            )
        elif model_key == "biot":
            # Plain BIOT on the 16-channel TCP bipolar montage built by exact
            # subtraction (see channel_plan.adapter).
            kw["n_chans"] = channel_plan.model_n_chans
            kw.pop("chs_info", None)
        elif model_key == "bendr":
            # Plain BENDR on 20 channels (19 reordered EEG + computed SCALE);
            # no chs_info → skip BENDR's exact-name check (the adapter already
            # reorders into BENDR_CHANNEL_ORDER).
            kw["n_chans"] = channel_plan.model_n_chans
            kw.pop("chs_info", None)
        elif model_key.startswith("steegformer"):
            kw.setdefault("n_times", n_times)
            kw.setdefault("on_non_divisible", "error")
        elif metadata.hub_repo is None:
            kw.update(n_chans=len(electrode_names), n_times=n_times, sfreq=sfreq)
            if model_key == "eegnet":
                # the max-norm dense head of the paper; braindecode deprecates False
                kw.setdefault("final_layer_with_constraint", True)

        load_kwargs = dict(kw)
        load_kwargs["revision"] = revision
        if filename is not None:
            load_kwargs["filename"] = filename
        effective_outputs = (
            1
            if model_key in {"cbramod", "luna", "eegpt", "signaljepa"}
            and n_outputs is None
            else n_outputs
        )
        if effective_outputs is not None:
            load_kwargs["n_outputs"] = effective_outputs
        if metadata.hub_repo is not None:
            with warnings.catch_warnings():
                # braindecode asks for the channel names at forward: given below
                warnings.filterwarnings("ignore", "Labram chs_info does not match")
                # LaBraM beyond 16 s: the extra time embeddings would be untrained
                warnings.filterwarnings("error", "The checkpoint holds")
                # ST-EEGFormer without its channel slots (input channel i -> slot i)
                warnings.filterwarnings("error", "STEEGFormer: .*identity")
                model = model_cls.from_pretrained(metadata.hub_repo, **load_kwargs)
        # backbone, LoRA and head all drawn from the seed, on the CPU
        with contextlib.nullcontext() if pretrained else seeded(seed):
            if metadata.hub_repo is None:
                model = model_cls(n_outputs=n_outputs, **kw)
            elif not pretrained:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Modifying chs_info argument")
                    warnings.filterwarnings("ignore", "Labram chs_info does not match")
                    model = type(model).from_config(model.get_config())
            if model_key == "labram" and electrode_names:
                # As in the authors' code, each channel takes the position
                # embedding of its name, so a subset of the pretraining channels
                # is fed as is.
                model.forward = partial(model.forward, ch_names=list(electrode_names))

            if train_mode == "frozen":
                for param in model.parameters():
                    param.requires_grad = False
                model.eval()
            elif train_mode == "lora":
                from peft import LoraConfig, get_peft_model

                lora_cfg = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=lora_target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                model = get_peft_model(model, lora_cfg)
            # full: no changes — all params trainable

            if n_outputs is not None and metadata.hub_repo is not None:
                model.reset_head(n_outputs)
            if train_mode == "frozen" and n_outputs is not None:
                for head_name in ("final_layer", "classifier", "head"):
                    head = getattr(model, head_name, None)
                    if head is not None:
                        for parameter in head.parameters():
                            parameter.requires_grad = True
                        break
        model = model.to(device)

        backend = cls(
            metadata=metadata,
            model=model,
            n_outputs=n_outputs,
            device=device,
            train_mode=train_mode,
            task=task,
        )
        if electrode_names:
            backend._expected_n_chans = len(electrode_names)
        if metadata.hub_repo is not None:
            backend.checkpoint_revisions = {metadata.hub_repo: revision}
        backend.weights_seed = None if pretrained else int(seed)
        backend._channel_adapter = (
            channel_plan.adapter if channel_plan is not None else None
        )
        backend._channel_plan = channel_plan
        return backend

    @classmethod
    def _forward_logits(cls, model, tensor, model_key: str):
        if model_key == "bendr":
            # The head reads the features of ``transform``, not BENDR's start
            # token (see ``_forward_features``).
            features = cls._forward_features(model, tensor, model_key).mean(dim=1)
            return model.final_layer(features)
        out = model(tensor)
        return out["logits"] if isinstance(out, dict) else out

    @staticmethod
    def _forward_features(model, tensor, model_key: str):
        if model_key in {"ctnet", "deep4net", "eegnet", "shallowfbcspnet"}:
            # what their final layer reads, flattened: its last axis is time
            out = model(tensor, return_features=True)
            return out["features"].flatten(start_dim=1)
        if model_key == "bendr":
            # BENDR's own features are the output at the start token, which
            # pretraining never trained (Kostas et al. 2021), so they barely
            # vary between inputs. Return the contextualized time tokens instead.
            context = model.contextualizer(model.encoder(tensor))
            return context[:, :, 1:].transpose(1, 2)
        out = model(tensor, return_features=True)
        if not isinstance(out, dict):
            return out
        # BIOT's features are the mean of its tokens
        return out["tokens"] if model_key == "biot" else out["features"]

    def _get_skorch_module(self):
        import torch.nn as nn

        class _BDSklearnModule(nn.Module):
            def __init__(self, backend, output_dim: int) -> None:
                super().__init__()
                self._backend = backend
                self._output_dim = output_dim
                self.model = backend._model

            def forward(self, X, sample_weight=None):
                # sample_weight goes to the loss (_training._SampleWeightedLoss)
                backend = self._backend
                return backend._forward_logits(self.model, X, backend._metadata.name)

            def train(self, mode: bool = True):
                super().train(mode)
                if mode and self._backend._train_mode == "frozen":
                    self._backend._set_backbone_eval()
                return self

        return _BDSklearnModule

    def _set_backbone_eval(self) -> None:
        self._model.eval()
        for module in self._model.modules():
            if any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=False)
            ):
                module.train()

    def transform(
        self, X: np.ndarray, *, return_tokens: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Extract backbone embeddings without running the classification head.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.

        Returns
        -------
        embeddings : np.ndarray of shape (n_samples, embedding_dim)
            Backbone feature vectors.
        """
        self._validate(X)
        X = self._native_input(X)
        batch_size = self._eval_batch_size()
        embedding_chunks = []
        token_chunks = []
        with self._no_grad():
            for start in range(0, len(X), batch_size):
                tensor = self._to_tensor(X[start : start + batch_size])
                tokens = self._forward_features(
                    self._model, tensor, self._metadata.name
                )
                if tokens.ndim < 3:
                    pooled = tokens
                else:
                    pooled = tokens.flatten(start_dim=1, end_dim=-2).mean(dim=1)
                embedding_chunks.append(self._from_tensor(pooled))
                if return_tokens:
                    if tokens.ndim < 3:
                        raise NotImplementedError(
                            f"{self._metadata.display_name} exposes only a pooled "
                            "feature tensor; native token extraction is unavailable."
                        )
                    if self._metadata.name == "labram":
                        # one token per channel and time patch, channel-major
                        tokens = tokens.unflatten(1, (tensor.shape[1], -1))
                    token_chunks.append(self._from_tensor(tokens))
        embeddings = np.concatenate(embedding_chunks, axis=0)
        if return_tokens:
            return embeddings, np.concatenate(token_chunks, axis=0)
        return embeddings

    def get_token_output_metadata(self) -> dict[str, object]:
        """Describe the upstream feature tensor.

        LaBraM's tokens, one per channel and time patch in a flat axis, are split
        into those two axes, unless the model interpolates channels: its tokens
        then cover its target channels, not the input ones, and stay flat.
        """
        layouts = {
            "cbramod": {
                "token_source": "return_features.features",
                "token_axes": ["window", "channel", "time_patch", "feature"],
                "token_observation_axes": ["channel", "time_patch"],
                "token_feature_axis": "feature",
            },
            "biot": {
                "token_source": "return_features.tokens",
                "token_axes": ["window", "channel", "time_patch", "feature"],
                "token_observation_axes": ["channel", "time_patch"],
                "token_feature_axis": "feature",
            },
            "labram": {
                "token_source": "return_features.features",
                "token_axes": ["window", "channel", "time_patch", "feature"],
                "token_observation_axes": ["channel", "time_patch"],
                "token_feature_axis": "feature",
            },
            "luna": {
                "token_source": "return_features.features",
                "token_axes": ["window", "time_patch", "feature"],
                "token_observation_axes": ["time_patch"],
                "token_feature_axis": "feature",
            },
            "signaljepa": {
                "token_source": "return_features.features",
                "token_axes": ["window", "context_token", "feature"],
                "token_observation_axes": ["context_token"],
                "token_feature_axis": "feature",
            },
            "eegpt": {
                "token_source": "return_features.features",
                "token_axes": ["window", "time_patch", "summary_feature"],
                "token_observation_axes": ["time_patch"],
                "token_feature_axis": "summary_feature",
            },
            "bendr": {
                "token_source": "contextualizer_output",
                "token_axes": ["window", "context_token", "feature"],
                "token_observation_axes": ["context_token"],
                "token_feature_axis": "feature",
            },
        }
        metadata = layouts.get(self._metadata.name)
        if metadata is None:
            raise NotImplementedError(
                f"{self._metadata.display_name} does not expose a documented "
                "native token representation."
            )
        return metadata

    def checkpoint_components(self) -> dict:
        """BrainDecode keeps one model (LoRA-wrapped when applicable)."""
        return {"model": self._model}

    def get_channel_adaptation(self) -> dict:
        """Describe how the input channels reach the model."""
        meta = self.signal_metadata_
        source = list(meta.ch_names) if meta is not None else []
        if self._channel_plan is not None:
            plan = self._channel_plan
            provenance = dict(plan.provenance)
            return {
                "source_channels": source,
                "target_channels": list(plan.model_ch_names or []),
                "zero_filled_channels": [],
                **provenance,
                # ``filled_channels`` (provenance) are interpolated unipolar
                # electrodes; surface them under the legacy key too.
                "interpolated_channels": list(provenance.get("filled_channels", [])),
            }
        return {
            "source_channels": source,
            "target_channels": source,
            "interpolated_channels": [],
            "zero_filled_channels": [],
            "dropped_channels": [],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run a full forward pass and return predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_channels, n_times)
            EEG/MEG epochs.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class indices (classification) or continuous values
            (regression).
        """
        self._validate(X)
        X = self._native_input(X)
        batch_size = self._eval_batch_size()
        is_regression = getattr(self, "_task", "classification") == "regression"
        chunks = []
        with self._no_grad():
            for start in range(0, len(X), batch_size):
                tensor = self._to_tensor(X[start : start + batch_size])
                logits = self._forward_logits(self._model, tensor, self._metadata.name)
                if is_regression:
                    chunks.append(self._from_tensor(logits).squeeze(-1))
                else:
                    chunks.append(self._from_tensor(logits.argmax(dim=-1)))
        return np.concatenate(chunks, axis=0)

    def reset_head(self, n_outputs: int) -> BrainDecodeBackend:
        """Replace the classification head without modifying backbone weights.

        Parameters
        ----------
        n_outputs : int
            Number of output units for the new head.

        Returns
        -------
        self : BrainDecodeBackend
            The backend with the updated head.
        """
        self._model.reset_head(n_outputs)
        self._n_outputs = n_outputs
        return self

    def configure_peft(self, lora_config: dict) -> BrainDecodeBackend:
        """Apply LoRA adapters to the backbone post-hoc.

        Parameters
        ----------
        lora_config : dict
            Keyword arguments forwarded to ``peft.LoraConfig``.

        Returns
        -------
        self : BrainDecodeBackend
            The backend with LoRA adapters injected and ``train_mode`` set
            to ``"lora"``.
        """
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(**lora_config)
        self._model = get_peft_model(self._model, cfg)
        self._train_mode = "lora"
        return self

    def get_embedding_info(self) -> FoundationModelSpec:
        """Return the registry spec for this model instance.

        Returns
        -------
        spec : FoundationModelSpec
            The spec describing this model's provenance and signal expectations.
        """
        return self._metadata
