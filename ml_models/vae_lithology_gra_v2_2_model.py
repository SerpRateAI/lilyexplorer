"""
VAE Lithology Model v2.2 - Adding Spatial Context

This model extends v2.1 by adding local stratigraphic context:
- Input: 18D (6 features × 3 positions: above, current, below)
- Output: 6D (only reconstructs current position)
- Uses distribution-aware scaling from v2.1

Key innovation: Spatial context provides gradient information without absolute depth.

Dataset: 238K+ samples from 296 boreholes
Architecture: 18D input → [hidden] → 8D latent → [hidden] → 6D output
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
    """Dataset for VAE training with spatial context."""

    def __init__(self, features_18d, targets_6d, lithology_labels):
        """
        Args:
            features_18d: 18D input features (6 × 3 positions)
            targets_6d: 6D target features (current position only)
            lithology_labels: Lithology labels for evaluation
        """
        self.features = torch.FloatTensor(features_18d)
        self.targets = torch.FloatTensor(targets_6d)
        self.lithology_labels = lithology_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx], self.lithology_labels[idx]

class VAE(nn.Module):
    """Variational Autoencoder with spatial context."""

    def __init__(self, input_dim=18, output_dim=6, latent_dim=8, hidden_dims=[32, 16]):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # Encoder (18D → latent)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (latent → 6D)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, target_x, mu, logvar, beta=1.0):
    """
    VAE loss function.

    Args:
        recon_x: Reconstructed current bin (6D)
        target_x: Actual current bin values (6D)
        mu, logvar: Latent distribution parameters
        beta: Weight for KL divergence
    """
    # Reconstruction loss (MSE on 6D current values only)
    recon_loss = nn.functional.mse_loss(recon_x, target_x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

class DistributionAwareScaler:
    """Custom scaler for 18D input with spatial context."""

    def __init__(self):
        self.scaler = StandardScaler()

        # Feature indices (0-5 = current, 6-11 = above, 12-17 = below)
        # For each position, indices are: [GRA, MS, NGR, R, G, B]

        # MS and NGR can be negative (indices 1, 2, 7, 8, 13, 14)
        self.signed_log_indices = [1, 2, 7, 8, 13, 14]

        # R, G, B are always positive (indices 3, 4, 5, 9, 10, 11, 15, 16, 17)
        self.log_indices = [3, 4, 5, 9, 10, 11, 15, 16, 17]

    def signed_log_transform(self, x):
        """Log transform that preserves sign for data with negative values."""
        return np.sign(x) * np.log1p(np.abs(x))

    def inverse_signed_log_transform(self, x):
        """Inverse of signed log transform."""
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

    def fit_transform(self, X):
        """Apply distribution-specific transforms, then standard scale."""
        X_transformed = X.copy()

        # Apply signed log to MS, NGR (all positions)
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply log to RGB (all positions)
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
        """Inverse transform back to original scale (for 6D outputs)."""
        # First inverse standard scaling
        X_transformed = self.scaler.inverse_transform(X_scaled)

        # Then inverse log transforms (only for first 6 features - current position)
        X_original = X_transformed.copy()

        # Inverse signed log for MS, NGR (indices 1, 2)
        for idx in [1, 2]:
            X_original[:, idx] = self.inverse_signed_log_transform(X_transformed[:, idx])

        # Inverse regular log for RGB (indices 3, 4, 5)
        for idx in [3, 4, 5]:
            X_original[:, idx] = np.expm1(X_transformed[:, idx])

        return X_original

def load_and_prepare_data(data_path):
    """Load and prepare data with spatial context."""
    print("Loading data...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

    # Feature columns
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    # Current position features (target for reconstruction)
    X_current = df[feature_cols].values

    # Above position features
    X_above = df[[f'{col}_above' for col in feature_cols]].values

    # Below position features
    X_below = df[[f'{col}_below' for col in feature_cols]].values

    # Combine into 18D input: [current, above, below]
    X_18d = np.concatenate([X_current, X_above, X_below], axis=1)

    # Target is only current position (6D)
    y_6d = X_current

    # Lithology labels
    lithology_labels = df['Principal'].values

    # Borehole IDs for proper splitting
    borehole_ids = df['Borehole_ID'].values

    print(f"\nFeature shapes:")
    print(f"  Input (18D): {X_18d.shape}")
    print(f"  Target (6D): {y_6d.shape}")
    print(f"  Unique lithologies: {len(np.unique(lithology_labels))}")

    return X_18d, y_6d, lithology_labels, borehole_ids

def split_by_borehole(X_18d, y_6d, lithology_labels, borehole_ids, train_size=0.7, val_size=0.15):
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
    X_train = X_18d[train_mask]
    y_train = y_6d[train_mask]
    lith_train = lithology_labels[train_mask]

    X_val = X_18d[val_mask]
    y_val = y_6d[val_mask]
    lith_val = lithology_labels[val_mask]

    X_test = X_18d[test_mask]
    y_test = y_6d[test_mask]
    lith_test = lithology_labels[test_mask]

    print(f"  Train: {len(train_holes)} boreholes, {len(X_train):,} samples")
    print(f"  Val:   {len(val_holes)} boreholes, {len(X_val):,} samples")
    print(f"  Test:  {len(test_holes)} boreholes, {len(X_test):,} samples")

    return (X_train, y_train, lith_train), (X_val, y_val, lith_val), (X_test, y_test, lith_test)

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

        for batch_features, batch_targets, _ in train_loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_features)
            loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_targets, mu, logvar)

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
            for batch_features, batch_targets, _ in val_loader:
                batch_features = batch_features.to(device)
                batch_targets = batch_targets.to(device)

                recon_batch, mu, logvar = model(batch_features)
                loss, recon_loss, kl_loss = vae_loss(recon_batch, batch_targets, mu, logvar)

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
    print("="*80)
    print("VAE GRA v2.2 - Spatial Context Model")
    print("="*80)

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_2_20cm.csv')
    output_dir = Path('/home/utig5/johna/bhai/vae_v2_2_outputs')
    output_dir.mkdir(exist_ok=True)

    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Load and prepare data
    X_18d, y_6d, lithology_labels, borehole_ids = load_and_prepare_data(data_path)

    # Split by borehole
    train_data, val_data, test_data = split_by_borehole(
        X_18d, y_6d, lithology_labels, borehole_ids
    )
    X_train, y_train, lith_train = train_data
    X_val, y_val, lith_val = val_data
    X_test, y_test, lith_test = test_data

    # Scale features
    print("\nApplying distribution-aware scaling...")
    scaler = DistributionAwareScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Also need to scale targets (6D current values)
    # Use only the first 6 columns of scaler for target scaling
    target_scaler = StandardScaler()

    # Apply same transforms to targets as we did to current position in input
    y_train_transformed = y_train.copy()
    y_train_transformed[:, 1] = scaler.signed_log_transform(y_train[:, 1])  # MS
    y_train_transformed[:, 2] = scaler.signed_log_transform(y_train[:, 2])  # NGR
    y_train_transformed[:, 3:6] = np.log1p(y_train[:, 3:6])  # RGB
    y_train_scaled = target_scaler.fit_transform(y_train_transformed)

    y_val_transformed = y_val.copy()
    y_val_transformed[:, 1] = scaler.signed_log_transform(y_val[:, 1])
    y_val_transformed[:, 2] = scaler.signed_log_transform(y_val[:, 2])
    y_val_transformed[:, 3:6] = np.log1p(y_val[:, 3:6])
    y_val_scaled = target_scaler.transform(y_val_transformed)

    y_test_transformed = y_test.copy()
    y_test_transformed[:, 1] = scaler.signed_log_transform(y_test[:, 1])
    y_test_transformed[:, 2] = scaler.signed_log_transform(y_test[:, 2])
    y_test_transformed[:, 3:6] = np.log1p(y_test[:, 3:6])
    y_test_scaled = target_scaler.transform(y_test_transformed)

    # Encode lithology labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([lith_train, lith_val, lith_test]))

    # Create datasets
    train_dataset = LithologyDataset(X_train_scaled, y_train_scaled, lith_train)
    val_dataset = LithologyDataset(X_val_scaled, y_val_scaled, lith_val)
    test_dataset = LithologyDataset(X_test_scaled, y_test_scaled, lith_test)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Train models
    latent_dims = [2, 8]

    for latent_dim in latent_dims:
        print(f"\n{'='*80}")
        print(f"Training {latent_dim}D Latent Model")
        print(f"{'='*80}")

        model = VAE(input_dim=18, output_dim=6, latent_dim=latent_dim, hidden_dims=[32, 16]).to(device)

        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

        model, history = train_model(
            model, train_loader, val_loader,
            epochs=100, lr=0.001, device=device
        )

        # Save checkpoint
        checkpoint_path = checkpoint_dir / f'vae_gra_v2_2_latent{latent_dim}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'target_scaler': target_scaler,
            'label_encoder': label_encoder,
            'history': history,
            'latent_dim': latent_dim,
            'version': 'v2.2',
            'input_dim': 18,
            'output_dim': 6,
            'architecture': {
                'input_dim': 18,
                'output_dim': 6,
                'latent_dim': latent_dim,
                'hidden_dims': [32, 16],
                'spatial_context': True
            }
        }, checkpoint_path)

        print(f"\n✓ Saved checkpoint to {checkpoint_path}")

        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'VAE v2.2 Loss (Latent Dim={latent_dim})')
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
        plt.savefig(output_dir / f'training_history_v2_2_latent{latent_dim}.png', dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved training history plot")

    print("\n" + "="*80)
    print("VAE GRA v2.2 Training Complete!")
    print("="*80)

if __name__ == "__main__":
    main()
