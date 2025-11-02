"""
VAE v2.6.1 - 5 features with RSC color + MSP

Features:
- GRA bulk density
- MS magnetic susceptibility (loop sensor)
- NGR total counts
- RSC Reflectance L* (lightness)
- RSC Reflectance a* (red-green)
- RSC Reflectance b* (blue-yellow)
- MSP magnetic susceptibility (point sensor)

Architecture: Same as v2.6 (distribution-aware scaling, β annealing)
Expected: ~341,000 samples from 484 boreholes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler

class DistributionAwareScaler:
    """
    Feature-specific scaling based on observed distributions.

    For v2.6.1 (7 features):
    - GRA: Gaussian → StandardScaler
    - MS, NGR: Poisson/Bimodal → sign(x)·log(|x|+1) + StandardScaler
    - RSC L*: Roughly uniform 0-100 → StandardScaler
    - RSC a*, b*: Roughly Gaussian around 0 → StandardScaler
    - MSP: Similar to MS (Poisson) → sign(x)·log(|x|+1) + StandardScaler
    """

    def __init__(self):
        self.scaler = StandardScaler()
        # Indices needing signed log transform (MS, NGR, MSP)
        self.signed_log_indices = [1, 2, 6]  # MS, NGR, MSP
        # All others use standard scaling only

    def fit_transform(self, X):
        X = X.copy()

        # Apply signed log transform to specified features
        for idx in self.signed_log_indices:
            X[:, idx] = np.sign(X[:, idx]) * np.log1p(np.abs(X[:, idx]))

        # Fit and transform with StandardScaler
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled

    def transform(self, X):
        X = X.copy()

        # Apply signed log transform
        for idx in self.signed_log_indices:
            X[:, idx] = np.sign(X[:, idx]) * np.log1p(np.abs(X[:, idx]))

        # Transform with fitted StandardScaler
        X_scaled = self.scaler.transform(X)
        return X_scaled

    def inverse_transform(self, X_scaled):
        # Inverse StandardScaler
        X = self.scaler.inverse_transform(X_scaled)

        # Inverse signed log transform
        for idx in self.signed_log_indices:
            X[:, idx] = np.sign(X[:, idx]) * (np.exp(np.abs(X[:, idx])) - 1)

        return X


class VAE(nn.Module):
    """
    VAE for 7-feature input (GRA, MS, NGR, L*, a*, b*, MSP)
    Same architecture as v2.6, scaled for 7 features
    """

    def __init__(self, input_dim=7, latent_dim=8, hidden_dims=[32, 16]):
        super(VAE, self).__init__()

        # Encoder: 7 → 32 → 16 → 8
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(0.1)

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder: 8 → 16 → 32 → 7
        self.fc3 = nn.Linear(latent_dim, hidden_dims[1])
        self.fc4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.fc5 = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout2(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss with β parameter for KL weighting.

    Args:
        recon_x: Reconstructed data
        x: Original data
        mu: Latent mean
        logvar: Latent log variance
        beta: Weight for KL divergence term
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl_loss, recon_loss, kl_loss
