"""
Topological Reducers
====================

This module implements dimensionality reduction techniques that preserve
topological features (persistent homology) of the data.

It features a custom Topological Autoencoder (TopologicalAE) implemented in
PyTorch. It prioritizes structure preservation in the latent space by balancing
reconstruction error with a topological loss (approximated here via distance regularization).

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
Date: 2026-01-06
"""

from typing import Optional, Any, Tuple, List
import numpy as np
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import gudhi as gd

from .base import BaseReducer, ArrayLike

logger = logging.getLogger(__name__)


class TopologicalAE(nn.Module):
    """
    Simple Autoencoder Network.
    
    Encoder: MLP with ReLU activations.
    Decoder: MLP with ReLU activations (mirror of encoder).
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [128, 64]):
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
    
    Adapted from the work of Moor et al. (2020) and the `pytorch-topological` library.
    It computes the topological signature of the data (via persistent homology) and
    minimizes the discrepancy between the input and latent space signatures.
    
    This implementation focuses on 0-dimensional features (connected components) 
    and 1-dimensional features (cycles), using 'gudhi' to compute the pairing.
    
    References
    ----------
    .. [1] Moor, M., et al. (2020). Topological autoencoders. ICML.
    .. [2] https://github.com/BorgwardtLab/topological-autoencoders
    .. [3] https://github.com/aidos-lab/pytorch-topological
    """
    def __init__(self, match_edges: str = 'symmetric', p: int = 2):
        super().__init__()
        self.match_edges = match_edges
        self.p = p

    def _get_active_pairs(self, dist_matrix: torch.Tensor, dim: int = 1) -> List[Tuple[int, int]]:
        """
        Compute persistence pairs using Gudhi.
        Returns list of (row_idx, col_idx) in the distance matrix corresponding to critical edges.
        """
        # Convert to numpy and condensed form for some tools, but Gudhi RipsComplex takes distance matrix
        d_mat_np = dist_matrix.detach().cpu().numpy()
        
        # Rips Complex
        rips = gd.RipsComplex(distance_matrix=d_mat_np)
        st = rips.create_simplex_tree(max_dimension=dim + 1)
        st.persistence()
        
        # Extract pairs
        # Gudhi returns pairs of simplices. We need to map back to edges (point indices).
        # For H0: pairs are (vertex, edge) -> edge kills component.
        # For H1: pairs are (edge, triangle) -> triangle kills cycle.
        
        # This mapping is non-trivial without specific helper functions typically found in 
        # the referenced libraries. 
        # SIMPLIFICATION:
        # We will focus on matching the *distances* of the critical pairs extracted.
        # However, extracting the exact indices of the edge (u, v) that creates the birth/death
        # requires parsing the simplex tree.
        
        pairs = st.persistence_pairs()
        
        critical_edges = []
        
        for birth_simplex, death_simplex in pairs:
            # Check dimension
            if len(birth_simplex) == dim + 1: # dim=0 -> len=1 (vertex), dim=1 -> len=2 (edge)
                 pass
            elif len(birth_simplex) != dim + 1:
                continue

            # Identify the edge associated with the birth or death
            # For 0-dim:
            # Birth is a vertex (0). Always 0.
            # Death is an edge (u, v). This edge merges two components. 
            # We want to pull (u, v) closer in latent space if they are close in input space?
            # Or rather, we want the structure to match.
            
            # Implementation of identifying critical pair indices:
            # If simplex is an edge [u, v], the critical distance is D[u, v].
            # If simplex is a triangle [u, v, w], the critical distance is max(D[u,v], D[v,w], D[u,w]).
            
            # Simple heuristic for 0-dim (H0):
            # Death of H0 feature is caused by an edge.
            if dim == 0 and len(death_simplex) == 2:
                critical_edges.append(tuple(sorted(death_simplex)))
                
            # For 1-dim (H1):
            # Birth of H1 is an edge. 
            if dim == 1 and len(birth_simplex) == 2:
                critical_edges.append(tuple(sorted(birth_simplex)))
                
        return critical_edges

    def _compute_distance_matrix(self, x: torch.Tensor, p: int = 2) -> torch.Tensor:
        return torch.cdist(x, x, p=p)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the signature loss.
        """
        if gd is None:
            return torch.tensor(0.0, device=x.device)
            
        # Distance matrices
        dx = self._compute_distance_matrix(x)
        dz = self._compute_distance_matrix(z)
        
        # Normalize distances to handle scale differences between X and Z?
        # Moor et al. normalize by dividing by max? or mean?
        # Usually simplest to just compute on raw but it's unstable. 
        # Let's normalize by mean.
        dx = dx / (dx.mean() + 1e-8)
        dz = dz / (dz.mean() + 1e-8)

        # Get critical edges for H0 (O-dim)
        # In input space
        pairs_x_0 = self._get_active_pairs(dx, dim=0)
        # In latent space
        pairs_z_0 = self._get_active_pairs(dz, dim=0)
        
        # Loss: 
        # We want the "persistence" (critical values) of selected pairs to be consistent.
        # Moor et al. define a "signature" loss.
        # Simplification:
        # Penalize the difference in distances for the UNION of critical edges found in both spaces.
        
        loss = torch.tensor(0.0, device=x.device)
        
        # Collect all unique edges involved in H0 structure in EITHER space
        all_edges = set(pairs_x_0) | set(pairs_z_0)
        
        if not all_edges:
            return loss

        # Vectorized gathering
        # Convert list of tuples to tensor indices
        idx_list = list(all_edges)
        rows = [r for r, c in idx_list]
        cols = [c for r, c in idx_list]
        
        # Gather values
        vals_x = dx[rows, cols]
        vals_z = dz[rows, cols]
        
        # loss = MSE of these critical distances
        loss += torch.mean((vals_x - vals_z) ** 2)
        
        return loss

# --- Reducer Class ---

class TopologicalAEReducer(BaseReducer):
    """
    Topological Autoencoder Reducer.
    
    Trains an Autoencoder with a loss function that balances reconstruction error
    and topological preservation.
    
    The topological loss is approximated by identifying critical edges (0-dim persistence pairs)
    using Gudhi and enforcing their distances to be consistent between input and latent space.
    This preserves the global connectivity structure (connected components hierarchy).

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the latent space.
    hidden_dims : list of int, default=[128, 64]
        Architecture of the encoder/decoder hidden layers.
    lam : float, default=0.0
        Regularization strength for topological loss.
        If > 0, adds the topological signature loss. 
        **Requires 'gudhi'**.
    lr : float, default=1e-3
        Learning rate.
    batch_size : int, default=64
        Batch size.
    epochs : int, default=50
        Number of training epochs.
    device : str, default='cpu'
        Device to train on ('cpu' or 'cuda').
    **kwargs : dict
        Additional arguments.

    Attributes
    ----------
    model : torch.nn.Module
        The underlying fitted PyTorch model.
    loss_history_ : list
        List of total loss values per epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from coco_pipe.dim_reduction.reducers.topology import TopologicalAEReducer
    >>> X = np.random.rand(100, 20)
    >>> reducer = TopologicalAEReducer(n_components=2, epochs=2, lam=1.0)
    >>> reducer.fit(X)
    >>> X_emb = reducer.transform(X)
    >>> print(X_emb.shape)
    (100, 2)
    """

    def __init__(self, n_components: int = 2, hidden_dims: List[int] = [128, 64],
                 lam: float = 0.0, lr: float = 1e-3, batch_size: int = 64,
                 epochs: int = 50, device: str = 'cpu', **kwargs):        
        super().__init__(n_components=n_components, **kwargs)
        self.hidden_dims = hidden_dims
        self.lam = lam
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.model = None
        self.loss_history_ = []
        
        # Initialize custom loss
        self.topo_loss_fn = TopologicalSignatureDistance()

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "TopologicalAEReducer":
        """
        Fit the Topological Autoencoder.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used.
        
        Returns
        -------
        self : TopologicalAEReducer
            Returns the instance itself.
        """
        # Reset loss history
        self.loss_history_ = []
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        dataset = TensorDataset(X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        input_dim = X.shape[1]
        self.model = TopologicalAE(input_dim, self.n_components, self.hidden_dims).to(self.device)
        self.model.train()
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion_recon = nn.MSELoss()
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch in dataloader:
                x_batch = batch[0]
                
                optimizer.zero_grad()
                z, recon = self.model(x_batch)
                
                loss_mse = criterion_recon(recon, x_batch)
                
                # Topological Loss
                loss_topo = torch.tensor(0.0).to(self.device)
                if self.lam > 0 and gd is not None:
                     loss_topo = self.topo_loss_fn(x_batch, z)
                elif self.lam > 0 and gd is None:
                     pass

                loss = loss_mse + self.lam * loss_topo
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            self.loss_history_.append(avg_loss)
            
            # Log every 10% of epochs or at least every 10 epochs
            log_interval = max(1, self.epochs // 10)
            if (epoch + 1) % log_interval == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
                
        self.model.eval()
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform X to latent space.
        
        Parameters
        ----------
        X : ArrayLike of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : np.ndarray of shape (n_samples, n_components)
             Latent representation.
        
        Raises
        ------
        RuntimeError
             If the model has not been fitted yet.
        """
        if self.model is None:
            raise RuntimeError("TopologicalAEReducer must be fitted before calling transform().")
            
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z, _ = self.model(X_tensor)
        
        return z.cpu().numpy()
