#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UMAP Projection for VAE v2.13 Multi-Decoder Model

Creates UMAP visualization of the 10D latent space with GMM cluster labels
and dominant lithology annotations at cluster centroids.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import umap
from collections import Counter

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


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


class MultiDecoderVAE(nn.Module):
    """VAE v2.13 Multi-Decoder Architecture"""

    def __init__(self, input_dim=6, latent_dim=10, encoder_dims=[32, 16],
                 decoder_dims=[16, 32]):
        super().__init__()

        self.input_dim = input_dim
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

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        outputs = []
        for name in self.feature_names:
            outputs.append(self.decoders[name](z))
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def load_model_and_data():
    """Load trained model and data"""
    print("Loading data...")
    df = pd.read_csv('vae_training_data_v2_20cm.csv')
    print(f"Loaded {len(df):,} samples")

    # Extract features and lithologies
    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                   'NGR total counts (cps)', 'R', 'G', 'B']
    X = df[feature_cols].values
    lithologies = df['Principal'].values

    # Load model
    print("\nLoading VAE v2.13 model...")
    model = MultiDecoderVAE(input_dim=6, latent_dim=10, encoder_dims=[32, 16], decoder_dims=[16, 32])

    # Load checkpoint with pickle
    import pickle
    import sys

    # Temporarily add a fake module to handle the old import
    class FakeModule:
        MultiDecoderVAE = MultiDecoderVAE
        DistributionAwareScaler = DistributionAwareScaler

    sys.modules['vae_lithology_gra_v2_5_model'] = FakeModule()

    checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_13_final.pth',
                           map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Model loaded successfully")

    return model, X, lithologies, df


def generate_latent_representations(model, X):
    """Generate latent space representations"""
    print("\nGenerating latent representations...")
    X_tensor = torch.FloatTensor(X)

    with torch.no_grad():
        mu, logvar = model.encode(X_tensor)
        z = mu.numpy()  # Use mean of latent distribution

    print(f"Latent space shape: {z.shape}")
    return z


def perform_gmm_clustering(z, lithologies, n_components=18):
    """Perform GMM clustering and evaluate"""
    print(f"\nPerforming GMM clustering (k={n_components})...")

    # Check for dimension collapse
    z_std = np.std(z, axis=0)
    active_dims = np.sum(z_std > 0.01)
    print(f"  Active dimensions (std > 0.01): {active_dims}/10")
    print(f"  Dimension stds: {z_std}")

    # Standardize latent space (important for numerical stability)
    z_scaled = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-8)
    print(f"  Latent space standardized for GMM")

    # Use diagonal covariance for numerical stability
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',  # More stable than 'full'
        random_state=42,
        n_init=10,
        max_iter=200,
        reg_covar=1e-3  # Strong regularization for numerical stability
    )

    cluster_labels = gmm.fit_predict(z_scaled)

    # Calculate ARI
    ari = adjusted_rand_score(lithologies, cluster_labels)
    print(f"✓ GMM clustering complete")
    print(f"  Adjusted Rand Index: {ari:.4f}")

    return cluster_labels, gmm, ari


def find_dominant_lithologies(cluster_labels, lithologies, n_clusters):
    """Find dominant lithology for each cluster"""
    dominant_lithologies = []

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_liths = lithologies[cluster_mask]

        if len(cluster_liths) > 0:
            # Find most common lithology
            lith_counts = Counter(cluster_liths)
            dominant_lith = lith_counts.most_common(1)[0][0]
            dominant_lithologies.append(dominant_lith)
        else:
            dominant_lithologies.append("empty")

    return dominant_lithologies


def create_umap_projection(z, cluster_labels, lithologies, gmm, ari):
    """Create UMAP projection with labeled centroids"""
    print("\nCreating UMAP projection...")

    # Run UMAP
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    z_umap = reducer.fit_transform(z)
    print(f"✓ UMAP projection complete: {z_umap.shape}")

    # Find dominant lithology for each cluster
    n_clusters = len(np.unique(cluster_labels))
    dominant_liths = find_dominant_lithologies(cluster_labels, lithologies, n_clusters)

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 12))

    # Define colormap
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))

    # Plot each cluster
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = z_umap[cluster_mask]

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=[colors[cluster_id]],
            s=1,
            alpha=0.3,
            rasterized=True,
            label=f'Cluster {cluster_id}'
        )

    # Calculate and label cluster centroids
    print("\nCluster centroids and dominant lithologies:")
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_points = z_umap[cluster_mask]

        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            dominant_lith = dominant_liths[cluster_id]
            n_points = len(cluster_points)

            # Truncate long lithology names
            if len(dominant_lith) > 25:
                label = dominant_lith[:22] + "..."
            else:
                label = dominant_lith

            # Add label at centroid
            ax.annotate(
                label,
                xy=centroid,
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                weight='bold',
                color=colors[cluster_id],
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor=colors[cluster_id],
                         alpha=0.8)
            )

            # Mark centroid
            ax.scatter(
                centroid[0],
                centroid[1],
                c=[colors[cluster_id]],
                s=200,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                zorder=100
            )

            print(f"  Cluster {cluster_id:2d} (n={n_points:6,}): {dominant_lith}")

    ax.set_xlabel('UMAP Dimension 1', fontsize=14, weight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=14, weight='bold')
    ax.set_title(
        f'VAE v2.13 Multi-Decoder: UMAP Projection of 10D Latent Space\n'
        f'GMM Clustering (k={n_clusters}) with Dominant Lithology Labels\n'
        f'Adjusted Rand Index = {ari:.4f}',
        fontsize=16,
        weight='bold',
        pad=20
    )

    # Remove legend (too many clusters)
    # ax.legend(loc='upper right', fontsize=8, ncol=2)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'v2_13_umap_projection.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ UMAP plot saved as '{output_file}'")
    plt.close()

    return z_umap


def create_lithology_colored_umap(z_umap, lithologies):
    """Create additional UMAP plot colored by true lithology (top 10)"""
    print("\nCreating lithology-colored UMAP...")

    # Find top 10 most common lithologies
    lith_counts = Counter(lithologies)
    top_10_liths = [lith for lith, _ in lith_counts.most_common(10)]

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plot top 10 lithologies
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, lith in enumerate(top_10_liths):
        mask = lithologies == lith
        points = z_umap[mask]
        n_points = len(points)

        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=[colors[i]],
            s=2,
            alpha=0.4,
            rasterized=True,
            label=f'{lith} (n={n_points:,})'
        )

    # Plot "other" lithologies in gray
    other_mask = ~np.isin(lithologies, top_10_liths)
    other_points = z_umap[other_mask]
    n_other = len(other_points)

    ax.scatter(
        other_points[:, 0],
        other_points[:, 1],
        c='lightgray',
        s=1,
        alpha=0.2,
        rasterized=True,
        label=f'Other lithologies (n={n_other:,})'
    )

    # Add labeled centroids for each lithology
    print("\nLithology centroids:")
    for i, lith in enumerate(top_10_liths):
        mask = lithologies == lith
        lith_points = z_umap[mask]

        if len(lith_points) > 0:
            centroid = lith_points.mean(axis=0)
            n_points = len(lith_points)

            # Truncate long lithology names
            if len(lith) > 25:
                label = lith[:22] + "..."
            else:
                label = lith

            # Add label at centroid with lithology-colored outline and black text
            ax.annotate(
                label,
                xy=centroid,
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                weight='bold',
                color='black',  # Black text
                bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='white',
                         edgecolor=colors[i],  # Lithology-colored outline
                         linewidth=2,
                         alpha=0.9)
            )

            # Mark centroid
            ax.scatter(
                centroid[0],
                centroid[1],
                c=[colors[i]],
                s=200,
                marker='*',
                edgecolors='black',
                linewidths=1.5,
                zorder=100
            )

            print(f"  {lith}: n={n_points:,}")

    ax.set_xlabel('UMAP Dimension 1', fontsize=14, weight='bold')
    ax.set_ylabel('UMAP Dimension 2', fontsize=14, weight='bold')
    ax.set_title(
        'VAE v2.13 Multi-Decoder: UMAP Projection Colored by True Lithology\n'
        '(Top 10 Most Common Lithologies with Centroids)',
        fontsize=16,
        weight='bold',
        pad=20
    )

    ax.legend(loc='upper right', fontsize=10, ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = 'v2_13_umap_by_lithology.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Lithology-colored UMAP saved as '{output_file}'")
    plt.close()


if __name__ == "__main__":
    print("="*70)
    print("VAE v2.13 Multi-Decoder: UMAP Projection Analysis")
    print("="*70)

    # Load model and data
    model, X, lithologies, df = load_model_and_data()

    # Generate latent representations
    z = generate_latent_representations(model, X)

    # Perform GMM clustering
    cluster_labels, gmm, ari = perform_gmm_clustering(z, lithologies, n_components=18)

    # Create UMAP projections
    z_umap = create_umap_projection(z, cluster_labels, lithologies, gmm, ari)
    create_lithology_colored_umap(z_umap, lithologies)

    print("\n" + "="*70)
    print("UMAP Analysis Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print("  - v2_13_umap_projection.png (GMM clusters with lithology labels)")
    print("  - v2_13_umap_by_lithology.png (colored by true lithology)")
    print(f"\nAdjusted Rand Index: {ari:.4f}")
    print("="*70)
