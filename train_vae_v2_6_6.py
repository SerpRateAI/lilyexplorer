"""
Train VAE GRA v2.6.6 with latent_dim=10 to confirm improvement.

Based on latent dimensionality experiments showing:
- latent_dim=8 (v2.6): GMM ARI=0.2662
- latent_dim=10: GMM ARI=0.2771 (+4.1%)

This script trains v2.6.6 with latent_dim=10 to confirm the result.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def train_vae(model, train_loader, val_loader, epochs=100, device='cpu',
              beta_start=0.001, beta_end=0.5, anneal_epochs=50):
    """Train VAE with β annealing."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # Compute current β
        if epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        # Training
        model.train()
        train_loss = 0

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

        train_loss /= len(train_loader)

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

        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch+1:3d}: β={current_beta:.4f}, Train={train_loss:.4f}, Val={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model

print("="*100)
print("VAE GRA v2.6.6: latent_dim=10 with β annealing")
print("="*100)
print()

# Load data
print("Loading data...")
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

print(f"Train: {len(df_train):,} samples from {len(train_boreholes)} boreholes")
print(f"Val:   {len(df_val):,} samples from {len(val_boreholes)} boreholes")
print(f"Test:  {len(df_test):,} samples from {len(test_boreholes)} boreholes")
print()

# Prepare features
X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

# Scale
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Data loaders
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

# Create and train model with latent_dim=10
print("="*100)
print("Training VAE with latent_dim=10")
print("="*100)
print()

start_time = time.time()

model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16])
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

model = train_vae(model, train_loader, val_loader, epochs=100, device=device,
                  beta_start=0.001, beta_end=0.5, anneal_epochs=50)

train_time = time.time() - start_time
print(f"\nTraining completed in {train_time:.1f}s")
print()

# Save model
checkpoint_path = 'ml_models/checkpoints/vae_gra_v2_6_6_latent10.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'latent_dim': 10,
    'hidden_dims': [32, 16],
    'scaler': scaler
}, checkpoint_path)
print(f"Model saved to: {checkpoint_path}")
print()

# Extract latent representations
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    mu, logvar = model.encode(X_tensor)
    latent = mu.cpu().numpy()

print(f"Latent shape: {latent.shape}")
print()

# Analyze latent space
latent_stds = latent.std(axis=0)
collapsed_dims = (latent_stds < 0.1).sum()
effective_dim = (latent_stds >= 0.1).sum()

print("="*100)
print("LATENT SPACE ANALYSIS")
print("="*100)
print(f"Latent std per dimension: {latent_stds}")
print(f"Collapsed dimensions: {collapsed_dims}/10")
print(f"Effective dimensionality: {effective_dim}")
print(f"Utilization: {100 * effective_dim / 10:.1f}%")
print()

# Clustering evaluation
print("="*100)
print("CLUSTERING EVALUATION")
print("="*100)
print()

results = []

# K-Means with different k values
print("K-Means clustering:")
for k in [10, 12, 15, 18, 20]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(latent)
    ari_km = adjusted_rand_score(y_test, labels_km)
    sil_km = silhouette_score(latent, labels_km, sample_size=10000)

    results.append({
        'method': 'K-Means',
        'k': k,
        'ARI': ari_km,
        'Silhouette': sil_km
    })

    print(f"  k={k:2d}: ARI={ari_km:.4f}, Silhouette={sil_km:.4f}")

print()

# GMM with different k values
print("GMM clustering (full covariance):")
for k in [10, 12, 15, 18, 20]:
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    labels_gmm = gmm.fit_predict(latent)
    ari_gmm = adjusted_rand_score(y_test, labels_gmm)
    sil_gmm = silhouette_score(latent, labels_gmm, sample_size=10000)

    results.append({
        'method': 'GMM',
        'k': k,
        'ARI': ari_gmm,
        'Silhouette': sil_gmm
    })

    print(f"  k={k:2d}: ARI={ari_gmm:.4f}, Silhouette={sil_gmm:.4f}")

print()

# Summary
df_results = pd.DataFrame(results)
df_results.to_csv('vae_v2_6_6_clustering_results.csv', index=False)

print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()
print(df_results.to_string(index=False))
print()

# Best results
best_kmeans = df_results[df_results['method'] == 'K-Means'].sort_values('ARI', ascending=False).iloc[0]
best_gmm = df_results[df_results['method'] == 'GMM'].sort_values('ARI', ascending=False).iloc[0]

print("BEST RESULTS:")
print("-"*100)
print(f"K-Means: k={int(best_kmeans['k'])}, ARI={best_kmeans['ARI']:.4f}, Silhouette={best_kmeans['Silhouette']:.4f}")
print(f"GMM:     k={int(best_gmm['k'])}, ARI={best_gmm['ARI']:.4f}, Silhouette={best_gmm['Silhouette']:.4f}")
print()

# Compare to v2.6 baseline
print("="*100)
print("COMPARISON TO v2.6 BASELINE")
print("="*100)
print()
print("v2.6 (latent_dim=8):")
print("  K-Means (k=12): ARI=0.2579")
print("  GMM (k=18):     ARI=0.2662")
print()
print(f"v2.6.6 (latent_dim=10):")
kmeans_12 = df_results[(df_results['method'] == 'K-Means') & (df_results['k'] == 12)].iloc[0]
gmm_18 = df_results[(df_results['method'] == 'GMM') & (df_results['k'] == 18)].iloc[0]
print(f"  K-Means (k=12): ARI={kmeans_12['ARI']:.4f} ({(kmeans_12['ARI']/0.2579 - 1)*100:+.1f}%)")
print(f"  GMM (k=18):     ARI={gmm_18['ARI']:.4f} ({(gmm_18['ARI']/0.2662 - 1)*100:+.1f}%)")
print()

if gmm_18['ARI'] > 0.2662:
    improvement = (gmm_18['ARI'] / 0.2662 - 1) * 100
    print(f"✓ Confirmed: latent_dim=10 improves GMM ARI by {improvement:.1f}%")
else:
    print("⚠ Result not confirmed - v2.6 (8D) still better")

print("="*100)
