"""PyTorch modules of the topological autoencoder (imports torch)."""

from torch import nn

from .topology import TopologicalSignatureDistance


class TopologicalAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 10,
        latent_dim: int = 2,
        hidden_dims: list[int] | None = None,
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


class TopologicalLossCriterion(nn.Module):
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
