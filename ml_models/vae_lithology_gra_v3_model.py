"""
VAE Lithology Model v3 - Dual Encoder Architecture

This model uses separate encoding streams for physical vs visual features:
- Physical encoder: [GRA, MS, NGR] → latent_physical (4D)
- Visual encoder: [R, G, B] → latent_visual (4D)
- Fusion: Simple concatenation → 8D
- Split decoders: Physical decoder (8D → 3D) + Visual decoder (8D → 3D)

Key innovation: Respects fundamental differences between measurement types.

Dataset: 238K+ samples from 296 boreholes
Features: 6D input with distribution-aware scaling (same as v2.1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
try:
    import umap
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available, will use PCA only for 8D visualization")
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import json
import pickle

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class LithologyDataset(Dataset):
    """Dataset for VAE training."""

    def __init__(self, features, lithology_labels):
        self.features = torch.FloatTensor(features)
        self.lithology_labels = lithology_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.lithology_labels[idx]

class DualEncoderVAE(nn.Module):
    """VAE with separate encoders for physical and visual features."""

    def __init__(self, latent_dim=8, hidden_dims_phys=[16, 8], hidden_dims_vis=[16, 8]):
        super(DualEncoderVAE, self).__init__()

        self.latent_dim = latent_dim
        # Each encoder produces half the latent dimensions
        self.latent_dim_phys = latent_dim // 2  # 4D for physical
        self.latent_dim_vis = latent_dim // 2   # 4D for visual

        # Physical encoder (GRA, MS, NGR) - 3D input
        phys_encoder_layers = []
        prev_dim = 3
        for h_dim in hidden_dims_phys:
            phys_encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.physical_encoder = nn.Sequential(*phys_encoder_layers)

        # Visual encoder (R, G, B) - 3D input
        vis_encoder_layers = []
        prev_dim = 3
        for h_dim in hidden_dims_vis:
            vis_encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.visual_encoder = nn.Sequential(*vis_encoder_layers)

        # Latent space parameters (separate for each stream)
        self.fc_mu_phys = nn.Linear(hidden_dims_phys[-1], self.latent_dim_phys)
        self.fc_logvar_phys = nn.Linear(hidden_dims_phys[-1], self.latent_dim_phys)

        self.fc_mu_vis = nn.Linear(hidden_dims_vis[-1], self.latent_dim_vis)
        self.fc_logvar_vis = nn.Linear(hidden_dims_vis[-1], self.latent_dim_vis)

        # Split decoders (both take full 8D latent)
        # Physical decoder: 8D → 3D (GRA, MS, NGR)
        phys_decoder_layers = []
        prev_dim = latent_dim  # Full 8D
        for h_dim in reversed(hidden_dims_phys):
            phys_decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        phys_decoder_layers.append(nn.Linear(hidden_dims_phys[0], 3))
        self.physical_decoder = nn.Sequential(*phys_decoder_layers)

        # Visual decoder: 8D → 3D (R, G, B)
        vis_decoder_layers = []
        prev_dim = latent_dim  # Full 8D
        for h_dim in reversed(hidden_dims_vis):
            vis_decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        vis_decoder_layers.append(nn.Linear(hidden_dims_vis[0], 3))
        self.visual_decoder = nn.Sequential(*vis_decoder_layers)

    def encode(self, x):
        """
        Encode with dual streams.

        Args:
            x: (batch, 6) - [GRA, MS, NGR, R, G, B]

        Returns:
            mu: (batch, 8) - concatenated latent means
            logvar: (batch, 8) - concatenated latent log variances
        """
        # Split input
        x_phys = x[:, :3]  # GRA, MS, NGR
        x_vis = x[:, 3:]   # R, G, B

        # Encode separately
        h_phys = self.physical_encoder(x_phys)
        mu_phys = self.fc_mu_phys(h_phys)
        logvar_phys = self.fc_logvar_phys(h_phys)

        h_vis = self.visual_encoder(x_vis)
        mu_vis = self.fc_mu_vis(h_vis)
        logvar_vis = self.fc_logvar_vis(h_vis)

        # Concatenate (simple fusion)
        mu = torch.cat([mu_phys, mu_vis], dim=1)
        logvar = torch.cat([logvar_phys, logvar_vis], dim=1)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decode with split decoders.

        Args:
            z: (batch, 8) - full latent representation

        Returns:
            recon_x: (batch, 6) - [GRA, MS, NGR, R, G, B]
        """
        # Both decoders see full latent
        recon_phys = self.physical_decoder(z)  # (batch, 3)
        recon_vis = self.visual_decoder(z)     # (batch, 3)

        # Concatenate outputs
        recon_x = torch.cat([recon_phys, recon_vis], dim=1)

        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence."""
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

class DistributionAwareScaler:
    """Custom scaler that applies distribution-specific transformations."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR can be negative
        self.log_indices = [3, 4, 5]  # R, G, B are always positive

    def signed_log_transform(self, x):
        """Log transform that preserves sign for data with negative values."""
        return np.sign(x) * np.log1p(np.abs(x))

    def inverse_signed_log_transform(self, x):
        """Inverse of signed log transform."""
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

    def fit_transform(self, X):
        """Apply distribution-specific transforms, then standard scale."""
        X_transformed = X.copy()

        # Apply signed log(|x| + 1) to features that can be negative (MS, NGR)
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply log(x + 1) to features that are always positive (RGB)
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Standard scale all features
        X_scaled = self.scaler.fit_transform(X_transformed)
        return X_scaled

    def transform(self, X):
        """Transform new data using fitted scaler."""
        X_transformed = X.copy()

        # Apply signed log to MS, NGR
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply regular log to RGB
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Standard scale
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled

    def inverse_transform(self, X_scaled):
        """Inverse transform back to original scale."""
        # First inverse standard scaling
        X_transformed = self.scaler.inverse_transform(X_scaled)

        # Then inverse log transforms
        X_original = X_transformed.copy()

        # Inverse signed log for MS, NGR
        for idx in self.signed_log_indices:
            X_original[:, idx] = self.inverse_signed_log_transform(X_transformed[:, idx])

        # Inverse regular log for RGB
        for idx in self.log_indices:
            X_original[:, idx] = np.expm1(X_transformed[:, idx])

        return X_original

def load_and_prepare_data(data_path):
    """Load and prepare data with distribution-aware scaling."""
    import sys
    sys.stdout.write("Loading data...\n")
    sys.stdout.flush()

    # Feature columns
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    # Only load necessary columns
    cols_to_read = feature_cols + ['Principal', 'Borehole_ID']
    df = pd.read_csv(data_path, usecols=cols_to_read)

    sys.stdout.write(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes\n")
    sys.stdout.flush()

    X = df[feature_cols].values
    lithology_labels = df['Principal'].values
    borehole_ids = df['Borehole_ID'].values

    sys.stdout.write(f"\nFeature shape: {X.shape}\n")
    sys.stdout.write(f"Unique lithologies: {len(np.unique(lithology_labels))}\n")
    sys.stdout.flush()

    return X, lithology_labels, borehole_ids

def split_by_borehole(X, lithology_labels, borehole_ids, train_size=0.7, val_size=0.15):
    """Split data by borehole to avoid data leakage."""
    print("\nSplitting data by borehole...")

    unique_boreholes = np.unique(borehole_ids)
    n_boreholes = len(unique_boreholes)

    # Split boreholes
    train_holes, temp_holes = train_test_split(
        unique_boreholes,
        train_size=train_size,
        random_state=42
    )

    val_ratio = val_size / (1 - train_size)
    val_holes, test_holes = train_test_split(
        temp_holes,
        train_size=val_ratio,
        random_state=42
    )

    # Create masks
    train_mask = np.isin(borehole_ids, train_holes)
    val_mask = np.isin(borehole_ids, val_holes)
    test_mask = np.isin(borehole_ids, test_holes)

    # Split data
    X_train = X[train_mask]
    lith_train = lithology_labels[train_mask]

    X_val = X[val_mask]
    lith_val = lithology_labels[val_mask]

    X_test = X[test_mask]
    lith_test = lithology_labels[test_mask]

    print(f"  Train: {len(train_holes)} boreholes, {len(X_train):,} samples")
    print(f"  Val:   {len(val_holes)} boreholes, {len(X_val):,} samples")
    print(f"  Test:  {len(test_holes)} boreholes, {len(X_test):,} samples")

    return (X_train, lith_train), (X_val, lith_val), (X_test, lith_test)

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu'):
    """Train VAE model with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_recon': [],
        'val_kl': []
    }

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    print("\nTraining VAE...")
    start_time = time.time()

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        train_recons = []
        train_kls = []

        for batch_features, _ in train_loader:
            batch_features = batch_features.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_features)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_features, mu, logvar)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item() / len(batch_features))
            train_recons.append(recon_loss.item() / len(batch_features))
            train_kls.append(kl_loss.item() / len(batch_features))

        # Validation
        model.eval()
        val_losses = []
        val_recons = []
        val_kls = []

        with torch.no_grad():
            for batch_features, _ in val_loader:
                batch_features = batch_features.to(device)

                recon_batch, mu, logvar = model(batch_features)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_features, mu, logvar)

                val_losses.append(loss.item() / len(batch_features))
                val_recons.append(recon_loss.item() / len(batch_features))
                val_kls.append(kl_loss.item() / len(batch_features))

        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_recon'].append(np.mean(train_recons))
        history['train_kl'].append(np.mean(train_kls))
        history['val_recon'].append(np.mean(val_recons))
        history['val_kl'].append(np.mean(val_kls))

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train={avg_train_loss:.3f}, Val={avg_val_loss:.3f}, "
                  f"Recon={np.mean(val_recons):.3f}, KL={np.mean(val_kls):.3f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best model
    model.load_state_dict(best_model_state)

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds ({epoch+1} epochs)")
    print(f"Best validation loss: {best_val_loss:.3f}")

    return model, history

def main():
    import sys
    sys.stdout.write("="*80 + "\n")
    sys.stdout.write("VAE GRA v3 - Dual Encoder Architecture\n")
    sys.stdout.write("="*80 + "\n")
    sys.stdout.flush()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sys.stdout.write(f"Using device: {device}\n")
    sys.stdout.flush()

    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    output_dir = Path('/home/utig5/johna/bhai/vae_v3_outputs')
    output_dir.mkdir(exist_ok=True)

    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    sys.stdout.write("About to load data...\n")
    sys.stdout.flush()

    # Load and prepare data
    X, lithology_labels, borehole_ids = load_and_prepare_data(data_path)

    sys.stdout.write("Data loaded successfully!\n")
    sys.stdout.flush()

    # Split by borehole
    train_data, val_data, test_data = split_by_borehole(
        X, lithology_labels, borehole_ids
    )
    X_train, lith_train = train_data
    X_val, lith_val = val_data
    X_test, lith_test = test_data

    # Scale features
    print("\nApplying distribution-aware scaling...")
    scaler = DistributionAwareScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Encode lithology labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([lith_train, lith_val, lith_test]))

    # Create datasets
    train_dataset = LithologyDataset(X_train_scaled, lith_train)
    val_dataset = LithologyDataset(X_val_scaled, lith_val)
    test_dataset = LithologyDataset(X_test_scaled, lith_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Train models
    latent_dims = [2, 8]

    for latent_dim in latent_dims:
        print(f"\n{'='*80}")
        print(f"Training {latent_dim}D Latent Model")
        print(f"{'='*80}")

        model = DualEncoderVAE(
            latent_dim=latent_dim,
            hidden_dims_phys=[16, 8],
            hidden_dims_vis=[16, 8]
        ).to(device)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Physical encoder: {sum(p.numel() for p in model.physical_encoder.parameters()):,}")
        print(f"  Visual encoder: {sum(p.numel() for p in model.visual_encoder.parameters()):,}")
        print(f"  Physical decoder: {sum(p.numel() for p in model.physical_decoder.parameters()):,}")
        print(f"  Visual decoder: {sum(p.numel() for p in model.visual_decoder.parameters()):,}")

        model, history = train_model(
            model, train_loader, val_loader,
            epochs=100, lr=0.001, device=device
        )

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'vae_gra_v3_latent{latent_dim}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'label_encoder': label_encoder,
            'history': history,
            'latent_dim': latent_dim,
            'version': 'v3',
            'architecture': {
                'type': 'dual_encoder',
                'latent_dim': latent_dim,
                'latent_dim_phys': latent_dim // 2,
                'latent_dim_vis': latent_dim // 2,
                'hidden_dims_phys': [16, 8],
                'hidden_dims_vis': [16, 8],
                'fusion': 'concatenation',
                'split_decoders': True
            }
        }, checkpoint_path)

        print(f"\n✓ Saved checkpoint to {checkpoint_path}")

        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'VAE v3 Loss (Latent Dim={latent_dim})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(history['train_recon'], label='Reconstruction')
        axes[1].plot(history['train_kl'], label='KL Divergence')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss Component')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'training_history_v3_latent{latent_dim}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved training history plot")

    print("\n" + "="*80)
    print("VAE GRA v3 Training Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
