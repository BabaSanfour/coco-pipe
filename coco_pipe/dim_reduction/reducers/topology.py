"""
Topological dimensionality reduction reducers.

This module provides topology-aware neural reducers that preserve geometric
structure through reconstruction and topological regularization. Heavy
dependencies such as `torch`, `gudhi`, and `skorch` remain lazy at import time
so the base package can be imported without deep-learning backends installed.

Classes
-------
TopologicalSignatureDistance
    Persistent-homology-inspired loss term used by the topological autoencoder.
TopologicalAEReducer
    Topological autoencoder wrapper built on PyTorch and Skorch.

References
----------
.. [1] Moor, M., Horn, M., Rieck, B., and Borgwardt, K. (2020).
       "Topological Autoencoders". Proceedings of the 37th International
       Conference on Machine Learning.
.. [2] Gudhi documentation:
       https://gudhi.inria.fr/python/latest/

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
"""

from typing import Any, Optional

import numpy as np

from ...utils import import_optional_dependency
from .base import ArrayLike, BaseReducer

__all__ = ["TopologicalSignatureDistance", "TopologicalAEReducer"]

_SKORCH_ALLOWED_PARAMS = {"callbacks", "dataset", "optimizer", "train_split"}


def _load_torch():
    """Lazily import `torch`."""
    return import_optional_dependency(
        lambda: __import__("torch"),
        feature="topology reducers",
        dependency="torch",
        install_hint="pip install coco-pipe[topology]",
    )


def _load_torch_nn():
    """Lazily import `torch.nn`."""
    return import_optional_dependency(
        lambda: __import__("torch.nn", fromlist=["Module"]),
        feature="topology reducers",
        dependency="torch",
        install_hint="pip install coco-pipe[topology]",
    )


def _load_gudhi():
    """Lazily import `gudhi`."""
    return import_optional_dependency(
        lambda: __import__("gudhi"),
        feature="topological loss",
        dependency="gudhi",
        install_hint="pip install coco-pipe[topology]",
    )


def _resolve_device(device: str) -> str:
    """
    Resolve the execution device for topology models.

    Parameters
    ----------
    device : str
        Requested device identifier. If set to `"auto"`, available backends
        are probed in the order CUDA, MPS, CPU.

    Returns
    -------
    str
        Concrete device identifier to use during training and inference.
    """
    if device != "auto":
        return device

    try:
        torch = _load_torch()
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _filter_skorch_params(params: dict) -> dict:
    """
    Keep only advanced Skorch parameters that are intentionally supported.

    Parameters
    ----------
    params : dict
        Extra reducer parameters captured on initialization.

    Returns
    -------
    dict
        Filtered parameter mapping containing only allowed Skorch kwargs.

    Notes
    -----
    Parameters using the `name__subparam` convention are preserved so callers
    can still customize module, criterion, optimizer, iterator, and callback
    internals without exposing every Skorch argument directly on the reducer.
    """
    return {
        key: value
        for key, value in params.items()
        if "__" in key or key in _SKORCH_ALLOWED_PARAMS
    }


def _as_2d_float32(X: ArrayLike, feature_name: str) -> np.ndarray:
    """
    Cast input to a 2D float32 array for topology reducers.

    Parameters
    ----------
    X : ArrayLike
        Input data expected to have shape `(n_samples, n_features)`.
    feature_name : str
        Name used in validation error messages.

    Returns
    -------
    np.ndarray
        Float32 array with shape `(n_samples, n_features)`.

    Raises
    ------
    ValueError
        If `X` is not 2-dimensional.
    """
    X_cast = np.asarray(X, dtype=np.float32)
    if X_cast.ndim != 2:
        raise ValueError(f"{feature_name} expects 2D input: (n_samples, n_features)")
    return X_cast


class TopologicalSignatureDistance:
    """
    Topological signature distance used as a regularization loss.

    Parameters
    ----------
    match_edges : str, default="symmetric"
        Matching strategy placeholder retained for API compatibility.
    p : int, default=2
        Norm used when computing pairwise distances with `torch.cdist`.

    Notes
    -----
    This implementation currently compares active edge lengths derived from
    persistence pairs in the input and latent spaces.
    """

    def __init__(self, match_edges: str = "symmetric", p: int = 2):
        self.match_edges = match_edges
        self.p = p

    def _get_active_pairs(
        self, dist_matrix: Any, dim: int = 1
    ) -> list[tuple[int, int]]:
        gd = _load_gudhi()

        d_mat_np = dist_matrix.detach().cpu().numpy()
        rips = gd.RipsComplex(distance_matrix=d_mat_np)
        st = rips.create_simplex_tree(max_dimension=dim + 1)
        st.persistence()
        pairs = st.persistence_pairs()

        critical_edges = []
        for birth_simplex, death_simplex in pairs:
            if dim == 0 and len(death_simplex) == 2:
                critical_edges.append(tuple(sorted(death_simplex)))
            if dim == 1 and len(birth_simplex) == 2:
                critical_edges.append(tuple(sorted(birth_simplex)))
        return critical_edges

    def _compute_distance_matrix(self, x: Any, p: int = 2) -> Any:
        torch = _load_torch()
        return torch.cdist(x, x, p=p)

    def forward(self, x: Any, z: Any) -> Any:
        """
        Compute the topological signature distance.

        Parameters
        ----------
        x : torch.Tensor
            Input-space samples.
        z : torch.Tensor
            Latent-space samples.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        torch = _load_torch()

        dx = self._compute_distance_matrix(x, p=self.p)
        dz = self._compute_distance_matrix(z, p=self.p)
        dx = dx / (dx.mean() + 1e-8)
        dz = dz / (dz.mean() + 1e-8)

        pairs_x_0 = self._get_active_pairs(dx, dim=0)
        pairs_z_0 = self._get_active_pairs(dz, dim=0)

        loss = torch.tensor(0.0, device=x.device)
        all_edges = set(pairs_x_0) | set(pairs_z_0)
        if not all_edges:
            return loss

        idx_list = list(all_edges)
        rows = [row for row, _ in idx_list]
        cols = [col for _, col in idx_list]

        vals_x = dx[rows, cols]
        vals_z = dz[rows, cols]
        loss += torch.mean((vals_x - vals_z) ** 2)
        return loss

    __call__ = forward


def _build_topology_training_classes():
    """
    Build PyTorch module classes used during topology autoencoder training.

    Returns
    -------
    tuple
        Tuple `(torch_module, autoencoder_cls, criterion_cls)`.
    """
    torch = _load_torch()
    nn = _load_torch_nn()

    class _TopologicalAE(nn.Module):
        def __init__(
            self,
            input_dim: int = 10,
            latent_dim: int = 2,
            hidden_dims: Optional[list[int]] = None,
        ):
            super().__init__()

            if hidden_dims is None:
                hidden_dims = [128, 64]

            encoder_layers = []
            in_dim = input_dim
            for hidden_dim in hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden_dim))
                encoder_layers.append(nn.ReLU())
                in_dim = hidden_dim
            encoder_layers.append(nn.Linear(in_dim, latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

            decoder_layers = []
            in_dim = latent_dim
            for hidden_dim in hidden_dims[::-1]:
                decoder_layers.append(nn.Linear(in_dim, hidden_dim))
                decoder_layers.append(nn.ReLU())
                in_dim = hidden_dim
            decoder_layers.append(nn.Linear(in_dim, input_dim))
            self.decoder = nn.Sequential(*decoder_layers)

        def forward(self, x):
            z = self.encoder(x)
            recon = self.decoder(z)
            return z, recon

    class _TopologicalLossCriterion(nn.Module):
        def __init__(self, lam: float = 0.0):
            super().__init__()
            self.lam = lam
            self.mse = nn.MSELoss()
            self.topo_loss = TopologicalSignatureDistance()

        def forward(self, y_pred, y_true):
            z, recon = y_pred
            loss = self.mse(recon, y_true)
            if self.lam > 0:
                loss += self.lam * self.topo_loss(y_true, z)
            return loss

    return torch, _TopologicalAE, _TopologicalLossCriterion


class TopologicalAEReducer(BaseReducer):
    """
    Topological autoencoder reducer.

    This reducer trains an autoencoder with an optional topological
    regularization term. The low-dimensional embedding is obtained from the
    encoder output and supports out-of-sample transformation after fitting.

    Parameters
    ----------
    n_components : int, default=2
        Latent dimensionality of the embedding.
    hidden_dims : list of int, optional
        Hidden layer sizes for the encoder. The decoder mirrors this sequence.
        If omitted, `[128, 64]` is used.
    lam : float, default=0.0
        Regularization strength applied to the topological loss term.
    lr : float, default=1e-3
        Optimizer learning rate.
    batch_size : int, default=64
        Training batch size.
    epochs : int, default=50
        Number of training epochs.
    device : str, default="auto"
        Training device. If `"auto"`, the reducer selects CUDA, then MPS, then
        CPU depending on availability.
    verbose : int, default=0
        Verbosity forwarded to Skorch.
    **kwargs : dict
        Additional advanced Skorch parameters. Keys using the
        `name__subparam` convention are forwarded, as are a small number of
        top-level Skorch options such as `callbacks` and `train_split`.

    Attributes
    ----------
    model : skorch.NeuralNetRegressor or None
        Fitted Skorch wrapper after `fit`.

    See Also
    --------
    IVISReducer : Neural reducer based on triplet loss.
    ParametricUMAPReducer : Neural graph-based reducer with transform support.
    PHATEReducer : Nonlinear diffusion-based reducer for smooth trajectories.
    PCAReducer : Linear baseline for tabular inputs.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction import TopologicalAEReducer
    >>> X = np.random.rand(20, 10).astype(np.float32)
    >>> reducer = TopologicalAEReducer(
    ...     n_components=2,
    ...     epochs=2,
    ...     batch_size=10,
    ...     device="cpu",
    ... )
    >>> reducer.fit(X)  # doctest: +SKIP
    TopologicalAEReducer(...)
    >>> reducer.transform(X).shape  # doctest: +SKIP
    (20, 2)
    """

    @property
    def capabilities(self) -> dict:
        """
        Return capability metadata for the topology autoencoder.

        Returns
        -------
        dict
            Capability mapping describing the reducer as a stochastic nonlinear
            model with transform support and loss-history diagnostics.
        """
        return self._merge_capabilities(
            super().capabilities,
            has_transform=True,
            has_components=False,
            supported_diagnostics=("loss_history_",),
            supported_metadata=(
                "lam",
                "lr",
                "batch_size",
                "epochs",
                "device",
                "input_dim_",
            ),
            is_linear=False,
            is_stochastic=True,
        )

    def __init__(
        self,
        n_components: int = 2,
        hidden_dims: Optional[list[int]] = None,
        lam: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        device: str = "auto",
        verbose: int = 0,
        **kwargs,
    ):
        """
        Initialize the topology autoencoder reducer.

        Parameters
        ----------
        n_components : int, default=2
            Latent dimensionality of the embedding.
        hidden_dims : list of int, optional
            Hidden layer sizes for the encoder.
        lam : float, default=0.0
            Topological regularization strength.
        lr : float, default=1e-3
            Learning rate.
        batch_size : int, default=64
            Training batch size.
        epochs : int, default=50
            Number of training epochs.
        device : str, default="auto"
            Execution device.
        verbose : int, default=0
            Skorch verbosity level.
        **kwargs : dict
            Additional advanced Skorch keyword arguments.
        """
        super().__init__(n_components=n_components, **kwargs)
        self.hidden_dims = hidden_dims if hidden_dims is not None else [128, 64]
        self.lam = lam
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.requested_device = device
        self.device = "cpu" if device == "auto" else device
        self.verbose = verbose
        self.input_dim_ = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "TopologicalAEReducer":
        """
        Fit the topology autoencoder on the input data.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : ArrayLike, optional
            Ignored. Present for API compatibility.

        Returns
        -------
        TopologicalAEReducer
            Fitted reducer instance.

        Raises
        ------
        ValueError
            If `X` is not 2-dimensional.
        """
        X_cast = _as_2d_float32(X, "TopologicalAEReducer")

        torch, topology_ae_cls, topology_loss_cls = _build_topology_training_classes()
        NeuralNetRegressor = import_optional_dependency(
            lambda: (
                __import__("skorch", fromlist=["NeuralNetRegressor"]).NeuralNetRegressor
            ),
            feature="TopologicalAEReducer",
            dependency="skorch",
            install_hint="pip install coco-pipe[topology]",
        )

        self.device = _resolve_device(self.requested_device)
        self.input_dim_ = int(X_cast.shape[1])
        estimator_kwargs = {
            **_filter_skorch_params(self.params),
            "module": topology_ae_cls,
            "module__input_dim": self.input_dim_,
            "module__latent_dim": self.n_components,
            "module__hidden_dims": self.hidden_dims,
            "criterion": topology_loss_cls,
            "criterion__lam": self.lam,
            "optimizer": torch.optim.Adam,
            "lr": self.lr,
            "max_epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "verbose": self.verbose,
            "train_split": None,
            "iterator_train__shuffle": True,
        }
        self.model = NeuralNetRegressor(**estimator_kwargs)
        self.model.fit(X_cast, X_cast)
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Encode new samples with the fitted topology autoencoder.

        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Samples to encode.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components)
            Latent embedding produced by the encoder.
        """
        self._require_fitted()
        X_cast = _as_2d_float32(X, "TopologicalAEReducer")

        torch = _load_torch()
        self.model.module_.eval()
        X_tensor = torch.tensor(X_cast).to(self.device)
        with torch.no_grad():
            z, _ = self.model.module_(X_tensor)
        return z.cpu().numpy()

    @property
    def loss_history_(self) -> list[float]:
        """
        Return the recorded training loss history.

        Returns
        -------
        list of float
            Training loss values. Returns an empty list if no fitted history is
            available.
        """
        if self.model is None or not hasattr(self.model, "history_"):
            return []
        return list(self.model.history_[:, "train_loss"])

    def get_pytorch_module(self) -> Optional[Any]:
        """
        Return the fitted underlying PyTorch module.

        Returns
        -------
        torch.nn.Module or None
            Fitted encoder-decoder module, or ``None`` if unavailable.
        """
        if self.model is not None and hasattr(self.model, "module_"):
            return self.model.module_
        return None
