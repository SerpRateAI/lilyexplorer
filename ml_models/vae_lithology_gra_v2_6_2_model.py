"""
VAE GRA v2.6.2: Two-Stage Training with Transfer Learning

Stage 1: Pre-train on physical properties (GRA+MS+NGR, 524 boreholes)
Stage 2: Fine-tune with RGB color (GRA+MS+NGR+RGB, 296 boreholes)

Architecture supports both 3D and 6D inputs for progressive training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class DistributionAwareScaler:
    """
    Custom scaler that applies distribution-specific transforms before StandardScaler.

    - GRA (Gaussian) → StandardScaler only
    - MS, NGR (Poisson/Bimodal) → sign(x)·log(|x|+1) + StandardScaler
    - R, G, B (Log-normal) → log(x+1) + StandardScaler
    """
    def __init__(self, input_dim=3):
        """
        Args:
            input_dim: 3 for GRA+MS+NGR, 6 for GRA+MS+NGR+RGB
        """
        self.scaler = StandardScaler()
        self.input_dim = input_dim

        # Indices for signed log transform (MS, NGR)
        self.signed_log_indices = [1, 2]  # MS, NGR for both 3D and 6D

        # Indices for regular log transform (RGB)
        self.log_indices = [3, 4, 5] if input_dim == 6 else []

    def fit_transform(self, X):
        X_transformed = X.copy()

        # Apply signed log to MS, NGR
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = np.sign(X[:, idx]) * np.log1p(np.abs(X[:, idx]))

        # Apply regular log to RGB (if 6D input)
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # StandardScaler on all transformed features
        X_scaled = self.scaler.fit_transform(X_transformed)
        return X_scaled

    def transform(self, X):
        X_transformed = X.copy()

        # Apply signed log to MS, NGR
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = np.sign(X[:, idx]) * np.log1p(np.abs(X[:, idx]))

        # Apply regular log to RGB (if 6D input)
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # StandardScaler on all transformed features
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled


class VAE(nn.Module):
    """
    Variational Autoencoder with progressive architecture (3D → 6D).

    Architecture:
        Encoder: input_dim → 32 → 16 → latent_dim
        Decoder: latent_dim → 16 → 32 → output_dim
    """
    def __init__(self, input_dim=3, latent_dim=8, hidden_dims=[32, 16], output_dim=None):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[1])
        self.fc4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.bn4 = nn.BatchNorm1d(hidden_dims[0])
        self.fc5 = nn.Linear(hidden_dims[0], self.output_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = F.relu(self.bn4(self.fc4(h)))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def expand_model_3d_to_6d(model_3d, device='cpu'):
    """
    Expand a 3D model (GRA+MS+NGR) to 6D (GRA+MS+NGR+RGB) by:
    1. Creating new 6D model
    2. Transferring weights for GRA/MS/NGR pathways
    3. Initializing RGB-related weights

    Args:
        model_3d: Trained VAE with input_dim=3
        device: Device to place new model on

    Returns:
        model_6d: New VAE with input_dim=6, partially initialized
    """
    model_6d = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16], output_dim=6).to(device)

    with torch.no_grad():
        # Transfer encoder first layer (3→32 becomes 6→32)
        # Copy weights for first 3 inputs (GRA, MS, NGR)
        model_6d.fc1.weight[:, :3] = model_3d.fc1.weight.clone()
        model_6d.fc1.bias[:] = model_3d.fc1.bias.clone()

        # Initialize RGB input weights with small random values
        nn.init.xavier_uniform_(model_6d.fc1.weight[:, 3:])

        # Transfer encoder second layer (32→16, full transfer)
        model_6d.fc2.weight[:] = model_3d.fc2.weight.clone()
        model_6d.fc2.bias[:] = model_3d.fc2.bias.clone()

        # Transfer batch norms
        model_6d.bn1.weight[:] = model_3d.bn1.weight.clone()
        model_6d.bn1.bias[:] = model_3d.bn1.bias.clone()
        model_6d.bn1.running_mean[:] = model_3d.bn1.running_mean.clone()
        model_6d.bn1.running_var[:] = model_3d.bn1.running_var.clone()

        model_6d.bn2.weight[:] = model_3d.bn2.weight.clone()
        model_6d.bn2.bias[:] = model_3d.bn2.bias.clone()
        model_6d.bn2.running_mean[:] = model_3d.bn2.running_mean.clone()
        model_6d.bn2.running_var[:] = model_3d.bn2.running_var.clone()

        # Transfer latent space layers (16→8, full transfer)
        model_6d.fc_mu.weight[:] = model_3d.fc_mu.weight.clone()
        model_6d.fc_mu.bias[:] = model_3d.fc_mu.bias.clone()
        model_6d.fc_logvar.weight[:] = model_3d.fc_logvar.weight.clone()
        model_6d.fc_logvar.bias[:] = model_3d.fc_logvar.bias.clone()

        # Transfer decoder layers (8→16→32, full transfer)
        model_6d.fc3.weight[:] = model_3d.fc3.weight.clone()
        model_6d.fc3.bias[:] = model_3d.fc3.bias.clone()
        model_6d.fc4.weight[:] = model_3d.fc4.weight.clone()
        model_6d.fc4.bias[:] = model_3d.fc4.bias.clone()

        model_6d.bn3.weight[:] = model_3d.bn3.weight.clone()
        model_6d.bn3.bias[:] = model_3d.bn3.bias.clone()
        model_6d.bn3.running_mean[:] = model_3d.bn3.running_mean.clone()
        model_6d.bn3.running_var[:] = model_3d.bn3.running_var.clone()

        model_6d.bn4.weight[:] = model_3d.bn4.weight.clone()
        model_6d.bn4.bias[:] = model_3d.bn4.bias.clone()
        model_6d.bn4.running_mean[:] = model_3d.bn4.running_mean.clone()
        model_6d.bn4.running_var[:] = model_3d.bn4.running_var.clone()

        # Transfer decoder output layer (32→3 becomes 32→6)
        # Copy weights for first 3 outputs (GRA, MS, NGR)
        model_6d.fc5.weight[:3, :] = model_3d.fc5.weight.clone()
        model_6d.fc5.bias[:3] = model_3d.fc5.bias.clone()

        # Initialize RGB output weights with small random values
        nn.init.xavier_uniform_(model_6d.fc5.weight[3:, :])
        nn.init.zeros_(model_6d.fc5.bias[3:])

    print("Model expansion complete:")
    print(f"  Transferred: GRA/MS/NGR encoding/decoding pathways")
    print(f"  Initialized: RGB input (fc1[:, 3:]) and output (fc5[3:, :]) weights")

    return model_6d


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss function with configurable β parameter.

    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Latent mean
        logvar: Latent log variance
        beta: KL divergence weight (default 1.0)

    Returns:
        loss, recon_loss, kl_loss
    """
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss with β weighting
    loss = recon_loss + beta * kl_loss

    return loss, recon_loss, kl_loss


def train_vae_with_annealing(model, train_loader, val_loader, epochs=100,
                             learning_rate=1e-3, device='cpu',
                             beta_start=0.001, beta_end=0.5, anneal_epochs=50,
                             patience=20):
    """
    Train VAE with β annealing schedule.

    Args:
        model: VAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Maximum number of epochs
        learning_rate: Learning rate for Adam optimizer
        device: Device to train on
        beta_start: Starting β value
        beta_end: Final β value
        anneal_epochs: Number of epochs to anneal β over
        patience: Early stopping patience

    Returns:
        model: Trained model
        history: Training history dict
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
        'beta': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # Calculate current β
        if epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        history['beta'].append(current_beta)

        # Training
        model.train()
        train_loss, train_recon, train_kl = 0, 0, 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=current_beta)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        # Average over batches
        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_recon, val_kl = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=current_beta)

                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()

        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)

        print(f'Epoch {epoch+1:3d}/{epochs} | β={current_beta:.4f} | '
              f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | '
              f'Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return model, history
