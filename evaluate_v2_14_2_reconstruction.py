#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate v2.14.2 reconstruction quality with R² scores
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load the trained model architecture
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


class SemiSupervisedVAE_Masked(nn.Module):
    """Semi-Supervised VAE with Random Feature Masking"""

    def __init__(self, input_dim=6, latent_dim=10, n_classes=209,
                 encoder_dims=[32, 16], classifier_hidden=32, mask_prob=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.mask_prob = mask_prob

        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def apply_mask(self, x):
        """No masking during evaluation"""
        return x, torch.ones_like(x)

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
        x_masked, mask = self.apply_mask(x)
        mu, logvar = self.encode(x_masked)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        logits = self.classifier(mu)
        return x_recon, mu, logvar, logits, mask


print("=" * 80)
print("VAE v2.14.2 RECONSTRUCTION EVALUATION")
print("=" * 80)

# Load data
print("\nLoading dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')
print(f"Total samples: {len(df):,}")

# Prepare features
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values

# Train/val split (same as training)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

# Load checkpoint
print("\nLoading model checkpoint...")
checkpoint = torch.load('ml_models/checkpoints/vae_v2_14_2_best.pth', weights_only=False)
scaler = checkpoint['scaler']

# Scale data
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

n_classes = 139  # From training log
model = SemiSupervisedVAE_Masked(
    input_dim=6,
    latent_dim=10,
    n_classes=n_classes,
    encoder_dims=[32, 16],
    classifier_hidden=32,
    mask_prob=0.3
).to(device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\nModel loaded from epoch {checkpoint['epoch']}")
print(f"Best validation ARI: {checkpoint['ari']:.4f}")

# Evaluate reconstruction on validation set
print("\n" + "=" * 80)
print("RECONSTRUCTION QUALITY EVALUATION")
print("=" * 80)

X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)

with torch.no_grad():
    x_recon, mu, logvar, logits, _ = model(X_val_tensor)
    x_recon_np = x_recon.cpu().numpy()

# Compute R² for each feature
feature_names = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']
print("\nPer-feature reconstruction R² scores:")
print("-" * 60)
print(f"{'Feature':<15} {'R²':<10} {'MSE':<15}")
print("-" * 60)

r2_scores = []
mse_scores = []
for i, name in enumerate(feature_names):
    y_true = X_val_scaled[:, i]
    y_pred = x_recon_np[:, i]

    r2 = r2_score(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)

    r2_scores.append(r2)
    mse_scores.append(mse)

    print(f"{name:<15} {r2:>7.4f}    {mse:>12.6f}")

print("-" * 60)
print(f"{'MEAN':<15} {np.mean(r2_scores):>7.4f}    {np.mean(mse_scores):>12.6f}")
print(f"{'MIN':<15} {np.min(r2_scores):>7.4f}    {np.min(mse_scores):>12.6f}")
print(f"{'MAX':<15} {np.max(r2_scores):>7.4f}    {np.max(mse_scores):>12.6f}")
print()

# Overall R² (multivariate)
y_true_flat = X_val_scaled.flatten()
y_pred_flat = x_recon_np.flatten()
overall_r2 = r2_score(y_true_flat, y_pred_flat)
overall_mse = np.mean((y_true_flat - y_pred_flat) ** 2)

print(f"\nOverall reconstruction quality:")
print(f"  R² (all features): {overall_r2:.4f}")
print(f"  MSE (all features): {overall_mse:.6f}")

# Compare masked vs unmasked reconstruction
print("\n" + "=" * 80)
print("MASKED RECONSTRUCTION ANALYSIS")
print("=" * 80)

# Create artificially masked data (30% per feature)
mask = (torch.rand_like(X_val_tensor) > 0.3).float()
X_val_masked = X_val_tensor * mask

with torch.no_grad():
    # Encode masked input
    mu_masked, logvar_masked = model.encode(X_val_masked)
    z_masked = model.reparameterize(mu_masked, logvar_masked)
    x_recon_masked = model.decode(z_masked)
    x_recon_masked_np = x_recon_masked.cpu().numpy()

# Compute R² on masked features only
mask_np = mask.cpu().numpy()
masked_indices = (mask_np == 0)

print("\nReconstruction R² on masked features (30% missing):")
print("-" * 60)
print(f"{'Feature':<15} {'R² (masked)':<15} {'Samples masked':<15}")
print("-" * 60)

for i, name in enumerate(feature_names):
    feature_mask = masked_indices[:, i]
    if feature_mask.sum() > 0:
        y_true = X_val_scaled[feature_mask, i]
        y_pred = x_recon_masked_np[feature_mask, i]
        r2_masked = r2_score(y_true, y_pred)
        print(f"{name:<15} {r2_masked:>12.4f}    {feature_mask.sum():>12,}")
    else:
        print(f"{name:<15} {'N/A':>12}    {0:>12,}")

print("-" * 60)

print("\n✓ Evaluation complete!")
