"""
Find optimal k for VAE GRA v2.6 with best β annealing schedule.

Previous experiments tested k∈{10,12,15,20} with k=12 performing best.
This script comprehensively tests k∈[5,30] to find the true optimal number of clusters.

Uses best annealing schedule from previous experiments: β=0.001→0.5 over 50 epochs.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def train_vae_with_annealing(model, train_loader, val_loader, epochs=100, device='cpu',
                              beta_start=0.001, beta_end=0.5, anneal_epochs=50):
    """Train VAE with β annealing."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'train_kl': [],
        'beta_schedule': []
    }

    for epoch in range(epochs):
        # Compute current β
        if epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        history['beta_schedule'].append(current_beta)

        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)

            recon_loss = F.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)

            loss = recon_loss + current_beta * kl_loss

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)

                recon_loss = F.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)

                loss = recon_loss + current_beta * kl_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)

        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: β={current_beta:.4f}, Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f}",
                  flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    return model, history

print("="*80)
print("VAE GRA v2.6: OPTIMAL k SELECTION")
print("="*80)
print()
print("Testing k∈[5,30] with best β annealing schedule (0.001→0.5 over 50 epochs)")
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

print(f"Train: {len(train_boreholes)} boreholes, {len(df_train):,} samples", flush=True)
print(f"Val:   {len(val_boreholes)} boreholes, {len(df_val):,} samples", flush=True)
print(f"Test:  {len(test_boreholes)} boreholes, {len(df_test):,} samples", flush=True)
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

print("="*80)
print("Training VAE with β annealing: 0.001→0.5 over 50 epochs")
print("="*80)
print()

start_time = time.time()

# Create and train model
model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
model.to(device)

model, history = train_vae_with_annealing(
    model, train_loader, val_loader,
    epochs=100, device=device,
    beta_start=0.001,
    beta_end=0.5,
    anneal_epochs=50
)

train_time = time.time() - start_time

print(f"\nTraining completed in {train_time:.1f}s", flush=True)
print(f"Epochs trained: {len(history['train_loss'])}", flush=True)
print(f"Final β: {history['beta_schedule'][-1]:.4f}", flush=True)
print(f"Final train loss: {history['train_loss'][-1]:.4f}", flush=True)
print(f"Final val loss: {history['val_loss'][-1]:.4f}", flush=True)
print()

# Get latent representations
print("Extracting latent representations...", flush=True)
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    mu, _ = model.encode(X_tensor)
    latent = mu.cpu().numpy()

print(f"Latent shape: {latent.shape}", flush=True)
print()

# Test comprehensive range of k values
print("="*80)
print("Testing k∈[5,30]")
print("="*80)
print()

k_values = list(range(5, 31))
results = []

for k in k_values:
    print(f"k={k:2d}: ", end='', flush=True)

    # Cluster
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent)

    # Metrics
    ari = adjusted_rand_score(y_test, labels)
    silhouette = silhouette_score(latent, labels, sample_size=10000)

    results.append({
        'k': k,
        'ari': ari,
        'silhouette': silhouette
    })

    print(f"ARI={ari:.4f}, Silhouette={silhouette:.4f}", flush=True)

print()

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Find optimal k
best_ari_idx = df_results['ari'].idxmax()
best_ari_k = df_results.loc[best_ari_idx, 'k']
best_ari_value = df_results.loc[best_ari_idx, 'ari']

best_sil_idx = df_results['silhouette'].idxmax()
best_sil_k = df_results.loc[best_sil_idx, 'k']
best_sil_value = df_results.loc[best_sil_idx, 'silhouette']

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()

print(f"Best ARI:        k={best_ari_k:2d}, ARI={best_ari_value:.4f}")
print(f"Best Silhouette: k={best_sil_k:2d}, Silhouette={best_sil_value:.4f}")
print()

# Show top 5 by ARI
print("Top 5 by ARI:")
top_ari = df_results.nlargest(5, 'ari')
for idx, row in top_ari.iterrows():
    print(f"  k={row['k']:2d}: ARI={row['ari']:.4f}, Silhouette={row['silhouette']:.4f}")
print()

# Show top 5 by Silhouette
print("Top 5 by Silhouette:")
top_sil = df_results.nlargest(5, 'silhouette')
for idx, row in top_sil.iterrows():
    print(f"  k={row['k']:2d}: ARI={row['ari']:.4f}, Silhouette={row['silhouette']:.4f}")
print()

# Compare to previously tested k values
print("Previously tested k values:")
for k in [10, 12, 15, 20]:
    row = df_results[df_results['k'] == k].iloc[0]
    print(f"  k={k:2d}: ARI={row['ari']:.4f}, Silhouette={row['silhouette']:.4f}")
print()

print("="*80)
print("RECOMMENDATION")
print("="*80)
print()

if best_ari_k == 12:
    print(f"k=12 confirmed as optimal choice (ARI={best_ari_value:.4f})")
else:
    improvement = (best_ari_value - df_results[df_results['k']==12].iloc[0]['ari']) / df_results[df_results['k']==12].iloc[0]['ari'] * 100
    print(f"k={best_ari_k} is better than k=12 by {improvement:+.1f}%")
    print(f"  k={best_ari_k}: ARI={best_ari_value:.4f}")
    print(f"  k=12: ARI={df_results[df_results['k']==12].iloc[0]['ari']:.4f}")

print()

# Save results
output_path = 'vae_v2_6_optimal_k_results.csv'
df_results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Save model
checkpoint_path = '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_optimal_k.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'k_results': df_results.to_dict('records'),
    'best_k_ari': int(best_ari_k),
    'best_ari': float(best_ari_value),
    'training_params': {
        'beta_start': 0.001,
        'beta_end': 0.5,
        'anneal_epochs': 50
    }
}, checkpoint_path)
print(f"Model saved to: {checkpoint_path}")
print()
