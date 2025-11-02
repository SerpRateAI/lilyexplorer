"""
Train VAE models with different β values to find optimal clustering performance.

Tests: β = 0.001, 0.005, 0.01 (baseline), 0.05, 0.1, 0.5, 1.0 (v2.1 baseline)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler, train_vae
from torch.utils.data import DataLoader, TensorDataset

# Beta values to test
BETAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

print("="*80)
print("VAE β PARAMETER SWEEP FOR OPTIMAL CLUSTERING")
print("="*80)
print()

# Load and prepare data
print("Loading data...", flush=True)
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

print(f"Train: {len(train_boreholes)} boreholes, {len(df_train):,} samples")
print(f"Val:   {len(val_boreholes)} boreholes, {len(df_val):,} samples")
print(f"Test:  {len(test_boreholes)} boreholes, {len(df_test):,} samples")
print()

# Prepare features
X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

# Scale features
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(X_train_scaled)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(X_val_scaled)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

# Train models for each β
results = []

for beta in BETAS:
    print("="*80)
    print(f"Training with β={beta}")
    print("="*80)
    print()

    start_time = time.time()

    # Create model
    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train
    model, history = train_vae(
        model, train_loader, val_loader,
        epochs=100, device=device, beta=beta
    )

    train_time = time.time() - start_time

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    # Cluster and evaluate
    beta_results = {'beta': beta, 'train_time': train_time}

    for k in [10, 12, 15, 20]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent)
        ari = adjusted_rand_score(y_test, labels)
        beta_results[f'ari_k{k}'] = ari

    results.append(beta_results)

    # Save checkpoint
    checkpoint_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_5_beta{beta}_latent8.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'beta': beta,
        'history': history
    }, checkpoint_path)

    print(f"\nResults for β={beta}:")
    print(f"  k=10: ARI={beta_results['ari_k10']:.3f}")
    print(f"  k=12: ARI={beta_results['ari_k12']:.3f}")
    print(f"  k=15: ARI={beta_results['ari_k15']:.3f}")
    print(f"  k=20: ARI={beta_results['ari_k20']:.3f}")
    print(f"  Training time: {train_time:.1f}s")
    print()

# Summary table
print("="*80)
print("SUMMARY: β PARAMETER SWEEP")
print("="*80)
print()

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# Find optimal β for each k
print("Optimal β for each k:")
for k in [10, 12, 15, 20]:
    col = f'ari_k{k}'
    best_idx = df_results[col].idxmax()
    best_beta = df_results.loc[best_idx, 'beta']
    best_ari = df_results.loc[best_idx, col]
    print(f"  k={k}: β={best_beta} (ARI={best_ari:.3f})")

print()
print("="*80)
print("RECOMMENDATION")
print("="*80)
print()

# Weighted average ARI across k values
df_results['avg_ari'] = df_results[[f'ari_k{k}' for k in [10, 12, 15, 20]]].mean(axis=1)
best_overall = df_results.loc[df_results['avg_ari'].idxmax()]

print(f"Best overall β: {best_overall['beta']}")
print(f"Average ARI across k=[10,12,15,20]: {best_overall['avg_ari']:.3f}")
print()
print("Use this β value for production VAE GRA v2.5 model.")
print("="*80)

# Save results
df_results.to_csv('beta_sweep_results.csv', index=False)
print("\nResults saved to: beta_sweep_results.csv")
