#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize Semi-Supervised VAE v2.14 Reconstruction Quality

Generates scatter plots showing input vs reconstructed values for all 6 features,
along with reconstruction metrics (R², RMSE).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
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

    def inverse_signed_log(self, x):
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

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

    def inverse_transform(self, X_scaled):
        """Transform back to original scale"""
        X_transformed = X_scaled * self.iqr + self.median

        X_original = X_transformed.copy()
        for idx in self.signed_log_indices:
            X_original[:, idx] = self.inverse_signed_log(X_transformed[:, idx])
        for idx in self.log_indices:
            X_original[:, idx] = np.expm1(X_transformed[:, idx])

        return X_original


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

    X_test_raw = X[test_mask]
    X_test_scaled = X_scaled[test_mask]
    lith_test = lithologies[test_mask]

    print(f"Test samples: {len(X_test_raw):,}")

    return X_test_raw, X_test_scaled, lith_test, scaler, feature_cols, len(le.classes_)


def reconstruct_features(model, X_test_scaled, scaler, device):
    """Generate reconstructions and transform back to original scale"""
    print("\nGenerating reconstructions...")
    model.eval()

    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    with torch.no_grad():
        recon_scaled, _, _, _ = model(X_tensor)
        recon_scaled = recon_scaled.cpu().numpy()

    # Transform back to original scale
    X_recon = scaler.inverse_transform(recon_scaled)

    return X_recon


def compute_metrics(X_true, X_recon, feature_names):
    """Compute reconstruction metrics for each feature"""
    print("\nComputing reconstruction metrics:")
    print("="*60)

    metrics = []
    for i, name in enumerate(feature_names):
        r2 = r2_score(X_true[:, i], X_recon[:, i])
        rmse = np.sqrt(mean_squared_error(X_true[:, i], X_recon[:, i]))
        mean_val = np.mean(X_true[:, i])
        rmse_pct = (rmse / mean_val) * 100 if mean_val != 0 else 0

        metrics.append({
            'feature': name,
            'r2': r2,
            'rmse': rmse,
            'rmse_pct': rmse_pct,
            'mean': mean_val
        })

        print(f"{name:40s} R²={r2:6.3f}  RMSE={rmse:8.2f} ({rmse_pct:5.1f}% of mean)")

    print("="*60)
    return pd.DataFrame(metrics)


def plot_reconstruction_scatter(X_true, X_recon, feature_names, metrics_df):
    """Create 2×3 scatter plot grid showing reconstruction quality"""
    print("\nCreating reconstruction scatter plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        true_vals = X_true[:, i]
        recon_vals = X_recon[:, i]

        # Subsample for plotting (10K points max)
        if len(true_vals) > 10000:
            indices = np.random.choice(len(true_vals), 10000, replace=False)
            true_plot = true_vals[indices]
            recon_plot = recon_vals[indices]
        else:
            true_plot = true_vals
            recon_plot = recon_vals

        # Scatter plot in dark red
        ax.scatter(true_plot, recon_plot, s=1, alpha=0.4, c='darkred', rasterized=True)

        # Perfect reconstruction line
        min_val = min(true_plot.min(), recon_plot.min())
        max_val = max(true_plot.max(), recon_plot.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.7, label='Perfect')

        # Metrics annotation
        r2 = metrics_df.iloc[i]['r2']
        rmse = metrics_df.iloc[i]['rmse']
        rmse_pct = metrics_df.iloc[i]['rmse_pct']

        ax.text(
            0.05, 0.95,
            f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\n({rmse_pct:.1f}% of mean)',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_xlabel(f'True {name}', fontsize=10)
        ax.set_ylabel(f'Reconstructed {name}', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Semi-Supervised VAE v2.14 (α=0.1) - Reconstruction Quality',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig('v2_14_reconstruction_scatter.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: v2_14_reconstruction_scatter.png")
    plt.close()


def plot_residuals(X_true, X_recon, feature_names):
    """Create residual plots for each feature"""
    print("\nCreating residual plots...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        true_vals = X_true[:, i]
        recon_vals = X_recon[:, i]
        residuals = recon_vals - true_vals

        # Subsample for plotting
        if len(residuals) > 10000:
            indices = np.random.choice(len(residuals), 10000, replace=False)
            true_plot = true_vals[indices]
            resid_plot = residuals[indices]
        else:
            true_plot = true_vals
            resid_plot = residuals

        # Residual scatter
        ax.scatter(true_plot, resid_plot, s=1, alpha=0.3, rasterized=True)
        ax.axhline(0, color='r', linestyle='--', lw=2, alpha=0.7)

        # Statistics
        mean_resid = np.mean(residuals)
        std_resid = np.std(residuals)

        ax.text(
            0.05, 0.95,
            f'Mean: {mean_resid:.2f}\nStd: {std_resid:.2f}',
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ax.set_xlabel(f'True {name}', fontsize=10)
        ax.set_ylabel('Residual (Recon - True)', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        'Semi-Supervised VAE v2.14 (α=0.1) - Reconstruction Residuals',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout()
    plt.savefig('v2_14_reconstruction_residuals.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: v2_14_reconstruction_residuals.png")
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("SEMI-SUPERVISED VAE v2.14 - RECONSTRUCTION QUALITY ANALYSIS")
    print("="*80)

    # Load data
    X_test_raw, X_test_scaled, lith_test, scaler, feature_names, n_classes = load_data()

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

    # Generate reconstructions
    X_recon = reconstruct_features(model, X_test_scaled, scaler, device)

    # Compute metrics
    metrics_df = compute_metrics(X_test_raw, X_recon, feature_names)

    # Create visualizations
    plot_reconstruction_scatter(X_test_raw, X_recon, feature_names, metrics_df)
    plot_residuals(X_test_raw, X_recon, feature_names)

    # Save metrics
    metrics_df.to_csv('v2_14_reconstruction_metrics.csv', index=False)
    print("\n✓ Saved: v2_14_reconstruction_metrics.csv")

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"Generated:")
    print(f"  - v2_14_reconstruction_scatter.png (input vs reconstructed)")
    print(f"  - v2_14_reconstruction_residuals.png (residual analysis)")
    print(f"  - v2_14_reconstruction_metrics.csv (R², RMSE)")
    print(f"\nOverall R² scores:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['feature']:40s} R²={row['r2']:6.3f}")
