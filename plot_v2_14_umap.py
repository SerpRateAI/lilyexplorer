#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate UMAP Projections for Semi-Supervised VAE v2.14

Creates UMAP visualizations of the 10D latent space with:
1. Cluster-based coloring (GMM k=18)
2. Lithology-based coloring with labeled centroids
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from umap import UMAP
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)


class DistributionAwareScaler:
    """Custom scaler matching v2.6.7 preprocessing"""
    def __init__(self):
        self.median = None
        self.iqr = None
        self.signed_log_indices = [1, 2]  # MS, NGR
        self.log_indices = [3, 4, 5]      # R, G, B

    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit_transform(self, X):
        X_transformed = X.copy()

        # Apply feature-specific transforms
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Median-IQR scaling
        self.median = np.median(X_transformed, axis=0)
        q75 = np.percentile(X_transformed, 75, axis=0)
        q25 = np.percentile(X_transformed, 25, axis=0)
        self.iqr = q75 - q25
        self.iqr[self.iqr == 0] = 1.0

        X_scaled = (X_transformed - self.median) / self.iqr
        return X_scaled

    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        X_scaled = (X_transformed - self.median) / self.iqr
        return X_scaled


class SemiSupervisedVAE(nn.Module):
    """Semi-supervised VAE with classification head"""
    def __init__(self, input_dim=6, latent_dim=10, n_classes=139,
                 encoder_dims=[32, 16], classifier_hidden=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Encoder: 6D → [32, 16] → 10D latent
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder: 10D → [16, 32] → 6D (symmetric)
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        # Classification head: 10D → [32, ReLU, Dropout] → 139 classes
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits


def load_data():
    """Load test dataset"""
    print("Loading dataset...")
    df = pd.read_csv('vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                   'NGR total counts (cps)', 'R', 'G', 'B']
    X = df[feature_cols].values
    lithologies = df['Principal'].values

    # Encode lithologies
    le = LabelEncoder()
    y = le.fit_transform(lithologies)

    # Scale features
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Borehole-level split (80/10/10)
    boreholes = df['Borehole_ID'].values
    unique_boreholes = np.unique(boreholes)
    np.random.shuffle(unique_boreholes)

    n_train = int(0.8 * len(unique_boreholes))
    n_val = int(0.1 * len(unique_boreholes))

    test_boreholes = unique_boreholes[n_train+n_val:]
    test_mask = np.isin(boreholes, test_boreholes)

    X_test = X_scaled[test_mask]
    lith_test = lithologies[test_mask]

    print(f"Test samples: {len(X_test):,}")
    print(f"Unique lithologies: {len(np.unique(lith_test))}")

    return X_test, lith_test, len(le.classes_)


def generate_latent_representations(model, X_test, device):
    """Generate latent representations from test set"""
    print("\nGenerating latent representations...")
    model.eval()

    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        z = mu.cpu().numpy()

    # Check dimension collapse
    z_std = np.std(z, axis=0)
    active_dims = np.sum(z_std > 0.01)
    print(f"Active dimensions: {active_dims}/10")
    print(f"Dimension stds: {z_std}")

    return z


def cluster_with_gmm(z, lith_test, k=18):
    """Cluster latent space with GMM"""
    print(f"\nClustering with GMM (k={k})...")

    # Standardize latent space
    z_scaled = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-8)

    gmm = GaussianMixture(
        n_components=k,
        covariance_type='diag',
        random_state=42,
        n_init=10,
        max_iter=200,
        reg_covar=1e-3
    )

    cluster_labels = gmm.fit_predict(z_scaled)
    ari = adjusted_rand_score(lith_test, cluster_labels)

    print(f"✓ Adjusted Rand Index: {ari:.4f}")

    return cluster_labels, ari


def compute_umap(z):
    """Compute UMAP projection"""
    print("\nComputing UMAP projection...")

    reducer = UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        metric='euclidean',
        random_state=42
    )

    z_umap = reducer.fit_transform(z)
    print(f"✓ UMAP shape: {z_umap.shape}")

    return z_umap


def plot_umap_clusters(z_umap, cluster_labels, ari):
    """Plot UMAP colored by GMM clusters"""
    print("\nCreating cluster-based UMAP plot...")

    fig, ax = plt.subplots(figsize=(12, 10))

    scatter = ax.scatter(
        z_umap[:, 0], z_umap[:, 1],
        c=cluster_labels,
        cmap='tab20',
        s=1,
        alpha=0.6,
        rasterized=True
    )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(
        f'Semi-Supervised VAE v2.14 (α=0.1) - UMAP Projection (GMM k=18, ARI={ari:.4f})',
        fontsize=14, fontweight='bold'
    )

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster', fontsize=12)

    plt.tight_layout()
    plt.savefig('v2_14_umap_projection.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: v2_14_umap_projection.png")
    plt.close()


def plot_umap_lithology(z_umap, lith_test):
    """Plot UMAP colored by lithology with labeled centroids"""
    print("\nCreating lithology-based UMAP plot...")

    # Get top 15 most common lithologies
    unique_lith, counts = np.unique(lith_test, return_counts=True)
    top_indices = np.argsort(counts)[-15:][::-1]
    top_lithologies = unique_lith[top_indices]

    # Create color map
    colors_list = plt.cm.tab20(np.linspace(0, 1, 15))
    color_map = {lith: colors_list[i] for i, lith in enumerate(top_lithologies)}

    # Assign colors
    point_colors = np.array([
        color_map.get(lith, [0.8, 0.8, 0.8, 0.3]) for lith in lith_test
    ])

    fig, ax = plt.subplots(figsize=(14, 12))

    # Plot all points
    ax.scatter(
        z_umap[:, 0], z_umap[:, 1],
        c=point_colors,
        s=1,
        alpha=0.6,
        rasterized=True
    )

    # Add labeled centroids for top lithologies
    for i, lith in enumerate(top_lithologies):
        mask = lith_test == lith
        if np.sum(mask) < 10:
            continue

        centroid = np.mean(z_umap[mask], axis=0)

        # Shorten label if too long
        label = lith if len(lith) <= 20 else lith[:17] + '...'

        ax.annotate(
            label,
            xy=centroid,
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            weight='bold',
            color='black',  # Black text
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=colors_list[i],  # Lithology-colored outline
                linewidth=2,
                alpha=0.9
            ),
            zorder=100
        )

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title(
        'Semi-Supervised VAE v2.14 (α=0.1) - UMAP by Lithology (Top 15)',
        fontsize=14, fontweight='bold'
    )

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                  markerfacecolor=colors_list[i], markersize=8, label=lith)
        for i, lith in enumerate(top_lithologies)
    ]
    ax.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        framealpha=0.9
    )

    plt.tight_layout()
    plt.savefig('v2_14_umap_by_lithology.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: v2_14_umap_by_lithology.png")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("SEMI-SUPERVISED VAE v2.14 - UMAP VISUALIZATION")
    print("="*80)

    # Load data
    X_test, lith_test, n_classes = load_data()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    checkpoint_path = Path('ml_models/checkpoints/semisup_vae_alpha0.1.pth')
    print(f"Loading checkpoint: {checkpoint_path}")

    model = SemiSupervisedVAE(
        input_dim=6,
        latent_dim=10,
        n_classes=n_classes,
        encoder_dims=[32, 16],
        classifier_hidden=32
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")

    # Generate latent representations
    z = generate_latent_representations(model, X_test, device)

    # Cluster with GMM
    cluster_labels, ari = cluster_with_gmm(z, lith_test, k=18)

    # Compute UMAP
    z_umap = compute_umap(z)

    # Create plots
    plot_umap_clusters(z_umap, cluster_labels, ari)
    plot_umap_lithology(z_umap, lith_test)

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Generated:")
    print(f"  - v2_14_umap_projection.png (cluster-based)")
    print(f"  - v2_14_umap_by_lithology.png (lithology-based with labels)")
    print(f"\nAdjusted Rand Index: {ari:.4f}")
