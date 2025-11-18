#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare VAE v2.6.7 vs v2.13 Models

Systematically verify that the two models are different and analyze
why their UMAP projections appear so similar.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from pathlib import Path
import hashlib

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class DistributionAwareScaler:
    """Custom scaler that applies distribution-specific transformations."""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR can be negative
        self.log_indices = [3, 4, 5]  # R, G, B are always positive

    def signed_log_transform(self, x):
        """Log transform that preserves sign for data with negative values."""
        return np.sign(x) * np.log1p(np.abs(x))

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


class VAE_v267(nn.Module):
    """VAE v2.6.7 Architecture - Single Decoder"""

    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


class MultiDecoderVAE_v213(nn.Module):
    """VAE v2.13 Architecture - Multi-Decoder"""

    def __init__(self, input_dim=6, latent_dim=10, encoder_dims=[32, 16], decoder_dims=[16, 32]):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_names = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']

        # Shared encoder
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Separate decoder per feature
        self.decoders = nn.ModuleDict()
        for name in self.feature_names:
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in decoder_dims:
                decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(decoder_dims[-1], 1))
            self.decoders[name] = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


def check_file_info(checkpoint_path):
    """Get file metadata"""
    import os
    import datetime

    path = Path(checkpoint_path)
    size_mb = path.stat().st_size / (1024 * 1024)
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime)

    # Calculate MD5 hash
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)

    return {
        'size_mb': size_mb,
        'modified': mtime,
        'md5': md5.hexdigest()
    }


def load_model_267():
    """Load VAE v2.6.7 model"""
    import sys

    model = VAE_v267(input_dim=6, latent_dim=10, hidden_dims=[32, 16])

    class FakeModule:
        VAE = VAE_v267
        DistributionAwareScaler = DistributionAwareScaler

    sys.modules['vae_lithology_gra_v2_5_model'] = FakeModule()

    checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                           map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_model_213():
    """Load VAE v2.13 model"""
    import sys

    model = MultiDecoderVAE_v213(input_dim=6, latent_dim=10,
                                  encoder_dims=[32, 16], decoder_dims=[16, 32])

    class FakeModule:
        MultiDecoderVAE = MultiDecoderVAE_v213
        DistributionAwareScaler = DistributionAwareScaler

    sys.modules['vae_lithology_gra_v2_5_model'] = FakeModule()

    checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_13_final.pth',
                           map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def load_data():
    """Load training data"""
    df = pd.read_csv('vae_training_data_v2_20cm.csv')
    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                   'NGR total counts (cps)', 'R', 'G', 'B']
    X = df[feature_cols].values
    return X, df


def generate_latent_representations(model, X):
    """Generate latent space representations"""
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        z = mu.numpy()

    return z


def compare_latent_spaces(z_267, z_213):
    """Compare latent space coordinates"""
    print("\n" + "="*70)
    print("LATENT SPACE COMPARISON")
    print("="*70)

    # Basic statistics
    print("\nv2.6.7 latent space:")
    print(f"  Shape: {z_267.shape}")
    print(f"  Mean: {np.mean(z_267, axis=0)}")
    print(f"  Std:  {np.std(z_267, axis=0)}")

    print("\nv2.13 latent space:")
    print(f"  Shape: {z_213.shape}")
    print(f"  Mean: {np.mean(z_213, axis=0)}")
    print(f"  Std:  {np.std(z_213, axis=0)}")

    # Correlation between corresponding dimensions
    print("\nCorrelation between corresponding dimensions:")
    for i in range(10):
        corr, pval = pearsonr(z_267[:, i], z_213[:, i])
        print(f"  Dimension {i}: r = {corr:7.4f} (p={pval:.2e})")

    # Overall similarity
    print("\nOverall similarity:")
    # Flatten and correlate all values
    corr_overall, _ = pearsonr(z_267.flatten(), z_213.flatten())
    print(f"  Pearson correlation (all values): r = {corr_overall:.4f}")

    # L2 distance
    l2_dist = np.linalg.norm(z_267 - z_213)
    print(f"  L2 distance: {l2_dist:.2f}")

    # Mean absolute difference
    mae = np.mean(np.abs(z_267 - z_213))
    print(f"  Mean absolute difference: {mae:.4f}")

    return corr_overall, mae


def plot_dimension_comparison(z_267, z_213):
    """Create side-by-side histogram comparison"""
    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    axes = axes.flatten()

    for i in range(10):
        # v2.6.7
        ax = axes[i*2]
        ax.hist(z_267[:, i], bins=100, alpha=0.7, color='blue', edgecolor='black')
        ax.set_title(f'v2.6.7 Dim {i}\n(μ={np.mean(z_267[:, i]):.2f}, σ={np.std(z_267[:, i]):.2f})',
                    fontsize=10, weight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)

        # v2.13
        ax = axes[i*2 + 1]
        ax.hist(z_213[:, i], bins=100, alpha=0.7, color='red', edgecolor='black')
        ax.set_title(f'v2.13 Dim {i}\n(μ={np.mean(z_213[:, i]):.2f}, σ={np.std(z_213[:, i]):.2f})',
                    fontsize=10, weight='bold')
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v2_6_7_vs_v2_13_dimension_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Dimension comparison saved as 'v2_6_7_vs_v2_13_dimension_comparison.png'")
    plt.close()


def plot_scatter_comparison(z_267, z_213):
    """Create scatter plots comparing corresponding dimensions"""
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()

    for i in range(10):
        ax = axes[i]

        # Subsample for visibility
        n_sample = min(10000, len(z_267))
        idx = np.random.choice(len(z_267), n_sample, replace=False)

        ax.scatter(z_267[idx, i], z_213[idx, i], s=1, alpha=0.3, rasterized=True)

        # Add correlation
        corr, _ = pearsonr(z_267[:, i], z_213[:, i])

        ax.set_xlabel(f'v2.6.7 Dim {i}', fontsize=10, weight='bold')
        ax.set_ylabel(f'v2.13 Dim {i}', fontsize=10, weight='bold')
        ax.set_title(f'Dimension {i} (r={corr:.4f})', fontsize=11, weight='bold')
        ax.grid(True, alpha=0.3)

        # Add diagonal line
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=1)

    plt.tight_layout()
    plt.savefig('v2_6_7_vs_v2_13_scatter_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Scatter comparison saved as 'v2_6_7_vs_v2_13_scatter_comparison.png'")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("VAE v2.6.7 vs v2.13 Comparison")
    print("="*70)

    # Check checkpoint files
    print("\nCheckpoint file information:")
    print("\nv2.6.7:")
    info_267 = check_file_info('ml_models/checkpoints/vae_gra_v2_6_7_final.pth')
    print(f"  Size: {info_267['size_mb']:.2f} MB")
    print(f"  Modified: {info_267['modified']}")
    print(f"  MD5: {info_267['md5']}")

    print("\nv2.13:")
    info_213 = check_file_info('ml_models/checkpoints/vae_gra_v2_13_final.pth')
    print(f"  Size: {info_213['size_mb']:.2f} MB")
    print(f"  Modified: {info_213['modified']}")
    print(f"  MD5: {info_213['md5']}")

    if info_267['md5'] == info_213['md5']:
        print("\n⚠️  WARNING: Checkpoint files have identical MD5 hashes!")
    else:
        print("\n✓ Checkpoint files are different (different MD5 hashes)")

    # Load data
    print("\nLoading data...")
    X, df = load_data()
    print(f"Loaded {len(X):,} samples")

    # Load models
    print("\nLoading v2.6.7 model...")
    model_267 = load_model_267()
    print("✓ v2.6.7 loaded")

    print("\nLoading v2.13 model...")
    model_213 = load_model_213()
    print("✓ v2.13 loaded")

    # Generate latent representations
    print("\nGenerating latent representations...")
    z_267 = generate_latent_representations(model_267, X)
    z_213 = generate_latent_representations(model_213, X)
    print("✓ Latent spaces generated")

    # Compare
    corr, mae = compare_latent_spaces(z_267, z_213)

    # Create visualizations
    print("\nCreating visualizations...")
    plot_dimension_comparison(z_267, z_213)
    plot_scatter_comparison(z_267, z_213)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Checkpoint files: {'IDENTICAL' if info_267['md5'] == info_213['md5'] else 'DIFFERENT'}")
    print(f"Overall latent space correlation: r = {corr:.4f}")
    print(f"Mean absolute difference: {mae:.4f}")

    if corr > 0.9:
        print("\n⚠️  Latent spaces are highly correlated (r > 0.9)")
        print("   This explains why UMAP projections look similar")
    elif corr > 0.5:
        print("\n⚠️  Latent spaces are moderately correlated")
        print("   UMAP may find similar structure in both spaces")
    else:
        print("\n✓ Latent spaces are different (r < 0.5)")
        print("   UMAP should show different projections")

    print("="*70)
