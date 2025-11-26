#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Adaptive VAE v2.14.1

Transformer-based VAE that handles variable feature inputs.
Can train on ALL boreholes regardless of missing measurements.

Training strategy:
- β annealing: 1e-10 → 0.75 (same as v2.14)
- α weighting: 0.1 for classification loss
- Batch processing with variable-length sequences
- GMM clustering evaluation on complete 6-feature subset
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import time
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

import sys
sys.path.append('ml_models')
from adaptive_vae_v2_14_1_model import AdaptiveVAE, adaptive_vae_loss


class AdaptiveVAEDataset(Dataset):
    """Dataset for adaptive VAE with variable-length inputs"""
    def __init__(self, df, scaler):
        self.df = df
        self.scaler = scaler

        # Feature columns
        self.feature_cols = [
            'Bulk density (GRA)',
            'Magnetic susceptibility (instr. units)',
            'NGR total counts (cps)',
            'R', 'G', 'B'
        ]

        # Get lithology labels
        self.lithologies = sorted(df['Principal'].unique())
        self.litho_to_idx = {litho: i for i, litho in enumerate(self.lithologies)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Get feature mask
        mask = np.array([row[f'mask_{i}'] for i in range(6)], dtype=bool)

        # Get present features and their indices
        present_indices = np.where(mask)[0]
        present_values = np.array([row[col] for col in self.feature_cols])[present_indices]

        # Get lithology label
        label = self.litho_to_idx[row['Principal']]

        return {
            'feature_ids': torch.LongTensor(present_indices),
            'feature_values': torch.FloatTensor(present_values).unsqueeze(-1),
            'label': torch.LongTensor([label]),
            'n_features': len(present_indices)
        }


def collate_variable_length(batch):
    """Collate function for variable-length sequences"""
    # Find max sequence length in batch
    max_len = max(item['n_features'] for item in batch)
    batch_size = len(batch)

    # Pre-allocate tensors
    feature_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    feature_values = torch.zeros(batch_size, max_len, 1)
    masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for i, item in enumerate(batch):
        n_feat = item['n_features']
        feature_ids[i, :n_feat] = item['feature_ids']
        feature_values[i, :n_feat] = item['feature_values']
        masks[i, :n_feat] = True
        labels[i] = item['label']

    return feature_ids, feature_values, masks, labels


class DistributionAwareScaler:
    """Distribution-aware scaler (placeholder - loaded from dataset)"""
    def __init__(self):
        self.median = None
        self.iqr = None
        self.signed_log_indices = [1, 2]
        self.log_indices = [3, 4, 5]

    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit_transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        self.median = np.median(X_transformed, axis=0)
        q75 = np.percentile(X_transformed, 75, axis=0)
        q25 = np.percentile(X_transformed, 25, axis=0)
        self.iqr = q75 - q25
        self.iqr[self.iqr == 0] = 1.0

        X_scaled = (X_transformed - self.median) / self.iqr
        return X_scaled


def train_epoch(model, dataloader, optimizer, device, beta, alpha, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_class = 0

    for feature_ids, feature_values, masks, labels in dataloader:
        feature_ids = feature_ids.to(device)
        feature_values = feature_values.to(device)
        masks = masks.to(device)
        labels = labels.to(device)

        # Forward pass
        recons, mu, logvar, logits = model(feature_ids, feature_values, masks)

        # Loss (only on present features)
        loss, recon_loss, kl_loss, class_loss = adaptive_vae_loss(
            recons, feature_values, mu, logvar, logits, labels,
            masks, beta=beta, alpha=alpha
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_class += class_loss.item()

    n_batches = len(dataloader)
    return total_loss/n_batches, total_recon/n_batches, total_kl/n_batches, total_class/n_batches


def evaluate_clustering(model, df_complete, device):
    """Evaluate clustering on complete 6-feature samples"""
    # Filter to complete samples
    df_eval = df_complete[df_complete['n_features'] == 6].copy()

    if len(df_eval) == 0:
        return 0.0

    # Prepare features
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]
    X = df_eval[feature_cols].values

    # Scale
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Get embeddings
    model.eval()
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # All 6 features present
    feature_ids = torch.LongTensor([[0, 1, 2, 3, 4, 5]]).repeat(len(X), 1).to(device)
    feature_values = X_tensor.unsqueeze(-1)
    masks = torch.ones(len(X), 6, dtype=torch.bool).to(device)

    with torch.no_grad():
        embeddings = model.get_embedding(feature_ids, feature_values, masks).cpu().numpy()

    # GMM clustering
    gmm = GaussianMixture(n_components=18, covariance_type='full', random_state=42)
    clusters = gmm.fit_predict(embeddings)

    # Get lithology labels
    lithologies = sorted(df_eval['Principal'].unique())
    litho_to_idx = {litho: i for i, litho in enumerate(lithologies)}
    labels = np.array([litho_to_idx[l] for l in df_eval['Principal']])

    # ARI
    ari = adjusted_rand_score(labels, clusters)

    return ari


print("=" * 80)
print("ADAPTIVE VAE v2.14.1 TRAINING")
print("=" * 80)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('adaptive_vae_training_data.csv')
print(f"Total samples: {len(df):,}")
print(f"Unique boreholes: {df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {df['Principal'].nunique()}")

# Feature coverage
print("\nFeature coverage:")
for i, name in enumerate(['GRA', 'MS', 'NGR', 'R', 'G', 'B']):
    n_present = df[f'mask_{i}'].sum()
    pct = 100 * n_present / len(df)
    print(f"  {name}: {n_present:,} ({pct:.1f}%)")

print(f"\nComplete 6-feature samples: {(df['n_features'] == 6).sum():,}")

# Create scaler (dummy, features already scaled in dataset)
scaler = DistributionAwareScaler()

# Create dataset and dataloader
print("\nPreparing data loader...")
dataset = AdaptiveVAEDataset(df, scaler)
dataloader = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True,
    collate_fn=collate_variable_length,
    num_workers=4,
    pin_memory=True
)

# Model
print("\nInitializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

n_classes = len(dataset.lithologies)
model = AdaptiveVAE(
    n_features=6,
    embed_dim=16,
    n_heads=4,
    n_layers=2,
    latent_dim=10,
    n_classes=n_classes,
    decoder_hidden=32
)
model = model.to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training settings
n_epochs = 100
beta_start = 1e-10
beta_end = 0.75
alpha = 0.1

print(f"\nTraining settings:")
print(f"  Epochs: {n_epochs}")
print(f"  β annealing: {beta_start} → {beta_end}")
print(f"  α (classification): {alpha}")
print(f"  Batch size: 512")
print(f"  Learning rate: 1e-3")

# Training loop
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_ari = 0.0

for epoch in range(1, n_epochs + 1):
    # β annealing
    beta = beta_start + (beta_end - beta_start) * (epoch - 1) / (n_epochs - 1)

    start_time = time.time()

    # Train
    loss, recon, kl, class_loss = train_epoch(model, dataloader, optimizer, device, beta, alpha, epoch)

    # Evaluate clustering (every 2 epochs)
    if epoch % 2 == 0 or epoch == n_epochs:
        ari = evaluate_clustering(model, df, device)
        if ari > best_ari:
            best_ari = ari
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ari': ari,
            }, 'ml_models/checkpoints/adaptive_vae_v2_14_1_best.pth')
    else:
        ari = None

    elapsed = time.time() - start_time

    # Print progress
    print(f"Epoch {epoch:3d}/{n_epochs} | "
          f"β={beta:.2e} | "
          f"Loss={loss:.4f} (Recon={recon:.4f}, KL={kl:.4f}, Class={class_loss:.4f}) | "
          f"{elapsed:.1f}s" +
          (f" | ARI={ari:.4f}" if ari is not None else ""))

# Save final model
torch.save({
    'epoch': n_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'ml_models/checkpoints/adaptive_vae_v2_14_1_final.pth')

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Best ARI: {best_ari:.4f}")
print(f"Final model: ml_models/checkpoints/adaptive_vae_v2_14_1_final.pth")
print(f"Best model: ml_models/checkpoints/adaptive_vae_v2_14_1_best.pth")
