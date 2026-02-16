"""
Topological Reducers
====================

This module implements dimensionality reduction techniques that preserve
topological features (persistent homology) of the data.

It features a custom Topological Autoencoder (TopologicalAE) implemented in
PyTorch and wrapped with Skorch for Scikit-Learn compatibility.

Classes
-------
TopologicalAEReducer
    Autoencoder with topological regularization.

References
----------
.. [1] Moor, M., et al. (2020). Topological autoencoders.
       International Conference on Machine Learning (ICML).
.. [2] Gudhi: https://gudhi.inria.fr/python/latest/

Author: Hamza Abdelhedi (hamza.abdelhedi@umontreal.ca)
Date: 2026-01-08
"""

import logging
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

    # Define dummy for inheritance to avoid IndentationError and import failures
    class MockModule:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            raise ImportError("torch is required to use this model.")

    class nn:
        Module = MockModule
        MSELoss = MockModule

from .base import ArrayLike, BaseReducer

logger = logging.getLogger(__name__)


# --- PyTorch Module ---


class TopologicalAE(nn.Module):
    """
    Simple Autoencoder Network.

    Encoder: MLP with ReLU activations.
    Decoder: MLP with ReLU activations (mirror of encoder).
    """

    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 2,
        hidden_dims: List[int] = [128, 64],
    ):
        if not HAS_TORCH:
            raise ImportError("torch and nn are required for TopologicalAE.")
        super().__init__()

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        reversed_hidden = hidden_dims[::-1]
        for h_dim in reversed_hidden:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


class TopologicalSignatureDistance(nn.Module):
    """
    Topological Signature Distance (Loss).
    """

    def __init__(self, match_edges: str = "symmetric", p: int = 2):
        if not HAS_TORCH:
            raise ImportError("torch is required for TopologicalSignatureDistance.")
        super().__init__()
        self.match_edges = match_edges
        self.p = p

    def _get_active_pairs(
        self, dist_matrix: Any, dim: int = 1
    ) -> List[Tuple[int, int]]:
        try:
            import gudhi as gd
        except ImportError:
            raise ImportError(
                "gudhi is required for topological loss. "
                "Install it with 'pip install gudhi'."
            )

        # Same Gudhi logic as before
        d_mat_np = dist_matrix.detach().cpu().numpy()
        rips = gd.RipsComplex(distance_matrix=d_mat_np)
        st = rips.create_simplex_tree(max_dimension=dim + 1)
        st.persistence()
        pairs = st.persistence_pairs()

        critical_edges = []
        for birth_simplex, death_simplex in pairs:
            # Simple heuristic for 0-dim (H0) death and 1-dim (H1) birth
            if dim == 0 and len(death_simplex) == 2:
                critical_edges.append(tuple(sorted(death_simplex)))
            if dim == 1 and len(birth_simplex) == 2:
                critical_edges.append(tuple(sorted(birth_simplex)))
        return critical_edges

    def _compute_distance_matrix(self, x: Any, p: int = 2) -> Any:
        return torch.cdist(x, x, p=p)

    def forward(self, x: Any, z: Any) -> Any:
        # Distance matrices (normalized)
        dx = self._compute_distance_matrix(x)
        dz = self._compute_distance_matrix(z)
        dx = dx / (dx.mean() + 1e-8)
        dz = dz / (dz.mean() + 1e-8)

        # Get critical edges for H0 (O-dim)
        pairs_x_0 = self._get_active_pairs(dx, dim=0)
        pairs_z_0 = self._get_active_pairs(dz, dim=0)

        loss = torch.tensor(0.0, device=x.device)
        all_edges = set(pairs_x_0) | set(pairs_z_0)

        if not all_edges:
            return loss

        idx_list = list(all_edges)
        rows = [r for r, c in idx_list]
        cols = [c for r, c in idx_list]

        vals_x = dx[rows, cols]
        vals_z = dz[rows, cols]

        loss += torch.mean((vals_x - vals_z) ** 2)
        return loss


class TopologicalLossCriterion(nn.Module):
    """
    Custom Criterion for Skorch.
    Computes Reconstruction Loss + Lambda * Topological Loss.

    Accepts y_pred=(z, recon) and y_true=X.
    """

    def __init__(self, lam: float = 0.0):
        if not HAS_TORCH:
            raise ImportError("torch is required for TopologicalLossCriterion.")
        super().__init__()
        self.lam = lam
        self.mse = nn.MSELoss()
        self.topo_loss = TopologicalSignatureDistance()

    def forward(self, y_pred, y_true):
        z, recon = y_pred
        X = y_true

        # Reconstruction Loss
        loss = self.mse(recon, X)

        # Topological Loss
        if self.lam > 0:
            loss += self.lam * self.topo_loss(X, z)

        return loss


# --- Reducer Class (Skorch Wrapper) ---


class TopologicalAEReducer(BaseReducer):
    """
    Topological Autoencoder Reducer (via Skorch).

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the latent space.
    hidden_dims : list of int, default=[128, 64]
        Architecture of the encoder/decoder hidden layers.
    lam : float, default=0.0
        Regularization strength for topological loss.
    lr : float, default=1e-3
        Learning rate.
    batch_size : int, default=64
        Batch size.
    epochs : int, default=50
        Number of training epochs.
    device : str, default='cpu'
        Device to train on ('cpu', 'cuda', 'mps').

    Attributes
    ----------
    model : skorch.NeuralNetRegressor
        The fitted Skorch estimator.
    """

    @property
    def capabilities(self) -> dict:
        """Capabilities of TopologicalAEReducer."""
        caps = super().capabilities
        caps.update({
            "supported_diagnostics": ["loss_history_"],
        })
        return caps

    def __init__(
        self,
        n_components: int = 2,
        hidden_dims: List[int] = [128, 64],
        lam: float = 0.0,
        lr: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 50,
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(n_components=n_components, **kwargs)
        self.hidden_dims = hidden_dims
        self.lam = lam
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for TopologicalAEReducer. "
                "Install it with 'pip install torch'."
            )

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = None

    def fit(
        self, X: ArrayLike, y: Optional[ArrayLike] = None
    ) -> "TopologicalAEReducer":
        """
        Fit the model using Skorch.
        """
        try:
            import torch
            from skorch import NeuralNetRegressor
        except ImportError:
            raise ImportError(
                "torch and skorch are required for TopologicalAEReducer. "
                "Install them with 'pip install torch skorch'."
            )

        input_dim = X.shape[1]

        # Initialize Skorch Net
        self.model = NeuralNetRegressor(
            module=TopologicalAE,
            module__input_dim=input_dim,
            module__latent_dim=self.n_components,
            module__hidden_dims=self.hidden_dims,
            criterion=TopologicalLossCriterion,
            criterion__lam=self.lam,
            optimizer=torch.optim.Adam,
            lr=self.lr,
            max_epochs=self.epochs,
            batch_size=self.batch_size,
            device=self.device,
            # Simple logging
            verbose=1,
            train_split=None,
            iterator_train__shuffle=True,
        )

        X_cast = X.astype(np.float32)
        self.model.fit(X_cast, X_cast)

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        if self.model is None:
            raise RuntimeError(
                "TopologicalAEReducer must be fitted before calling transform()."
            )

        X_cast = X.astype(np.float32)
        # Method 1: Use the module directly (safest for extracting latent)
        self.model.module_.eval()
        X_tensor = torch.tensor(X_cast).to(self.device)
        with torch.no_grad():
            z, _ = self.model.module_(X_tensor)

        return z.cpu().numpy()

    @property
    def loss_history_(self) -> List[float]:
        """Return training loss history from Skorch history."""
        if self.model is None or not hasattr(self.model, "history_"):
            return []
        return self.model.history_[:, "train_loss"]

    def get_pytorch_module(self) -> Optional[Any]:
        """
        Return the underlying PyTorch module for analysis.

        Exposes the module_ attribute from the Skorch wrapper if fitted.
        """
        if self.model is not None and hasattr(self.model, "module_"):
            return self.model.module_
        return None
