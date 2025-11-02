"""
Test VAE performance with different latent dimensionalities.

Question: Does 4D latent space work better than 8D given that 4/8 dimensions collapsed?

Tests: latent_dim ∈ {2, 4, 6, 8, 10, 12}
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

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model

print("="*100)
print("TESTING LATENT DIMENSIONALITY")
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

print(f"Train: {len(df_train):,} samples")
print(f"Val:   {len(df_val):,} samples")
print(f"Test:  {len(df_test):,} samples")
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

# Test different latent dimensions
latent_dims = [2, 4, 6, 8, 10, 12]
results = []

for latent_dim in latent_dims:
    print("="*100)
    print(f"Testing latent_dim = {latent_dim}")
    print("="*100)
    print()

    start_time = time.time()

    # Create and train model
    model = VAE(input_dim=6, latent_dim=latent_dim, hidden_dims=[32, 16])
    model = train_vae(model, train_loader, val_loader, epochs=100, device=device,
                      beta_start=0.001, beta_end=0.5, anneal_epochs=50)

    train_time = time.time() - start_time
    print(f"\nTraining completed in {train_time:.1f}s")

    # Extract latent representations
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        mu, logvar = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    # Analyze latent space
    latent_stds = latent.std(axis=0)
    collapsed_dims = (latent_stds < 0.1).sum()
    effective_dim = (latent_stds >= 0.1).sum()

    print(f"Latent std per dimension: {latent_stds}")
    print(f"Collapsed dimensions: {collapsed_dims}/{latent_dim}")
    print(f"Effective dimensionality: {effective_dim}")
    print()

    # Clustering performance
    print("Testing clustering methods...")

    # K-Means at k=12
    kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
    labels_km = kmeans.fit_predict(latent)
    ari_km = adjusted_rand_score(y_test, labels_km)
    sil_km = silhouette_score(latent, labels_km, sample_size=10000)

    # GMM (full) at k=18
    gmm = GaussianMixture(n_components=18, covariance_type='full', random_state=42)
    labels_gmm = gmm.fit_predict(latent)
    ari_gmm = adjusted_rand_score(y_test, labels_gmm)
    sil_gmm = silhouette_score(latent, labels_gmm, sample_size=10000)

    results.append({
        'latent_dim': latent_dim,
        'train_time': train_time,
        'collapsed_dims': collapsed_dims,
        'effective_dim': effective_dim,
        'utilization_pct': 100 * effective_dim / latent_dim,
        'kmeans_ari': ari_km,
        'kmeans_sil': sil_km,
        'gmm_ari': ari_gmm,
        'gmm_sil': sil_gmm
    })

    print(f"K-Means (k=12): ARI={ari_km:.4f}, Sil={sil_km:.4f}")
    print(f"GMM (k=18):     ARI={ari_gmm:.4f}, Sil={sil_gmm:.4f}")
    print()

# Summary
df_results = pd.DataFrame(results)

print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()
print(df_results.to_string(index=False))
print()

# Find optimal
best_kmeans_idx = df_results['kmeans_ari'].idxmax()
best_gmm_idx = df_results['gmm_ari'].idxmax()

print("BEST CONFIGURATIONS:")
print("-"*100)
print(f"K-Means: latent_dim={int(df_results.loc[best_kmeans_idx, 'latent_dim'])}, "
      f"ARI={df_results.loc[best_kmeans_idx, 'kmeans_ari']:.4f}, "
      f"Effective dim={int(df_results.loc[best_kmeans_idx, 'effective_dim'])}")

print(f"GMM:     latent_dim={int(df_results.loc[best_gmm_idx, 'latent_dim'])}, "
      f"ARI={df_results.loc[best_gmm_idx, 'gmm_ari']:.4f}, "
      f"Effective dim={int(df_results.loc[best_gmm_idx, 'effective_dim'])}")
print()

# Analysis
print("="*100)
print("KEY INSIGHTS")
print("="*100)
print()

# Does dimensionality affect collapse?
print("1. Posterior Collapse vs Latent Dimensionality:")
for _, row in df_results.iterrows():
    print(f"   latent_dim={int(row['latent_dim']):2d}: "
          f"{int(row['collapsed_dims']):2d} collapsed, "
          f"{int(row['effective_dim']):2d} active "
          f"({row['utilization_pct']:.1f}% utilization)")
print()

# Does performance scale with dimensionality?
print("2. Clustering Performance vs Latent Dimensionality:")
print("   (Does more dimensions = better clustering?)")
for _, row in df_results.iterrows():
    print(f"   latent_dim={int(row['latent_dim']):2d}: "
          f"K-Means ARI={row['kmeans_ari']:.4f}, "
          f"GMM ARI={row['gmm_ari']:.4f}")
print()

# Is there an optimal effective dimensionality?
print("3. Performance vs Effective Dimensionality:")
df_results_sorted = df_results.sort_values('effective_dim')
for _, row in df_results_sorted.iterrows():
    print(f"   Effective dim={int(row['effective_dim']):2d} "
          f"(from latent_dim={int(row['latent_dim']):2d}): "
          f"GMM ARI={row['gmm_ari']:.4f}")
print()

# Save
df_results.to_csv('latent_dimensionality_comparison.csv', index=False)
print("Results saved to: latent_dimensionality_comparison.csv")
print()

print("="*100)
print("CONCLUSION")
print("="*100)
print()

baseline_8d = df_results[df_results['latent_dim'] == 8].iloc[0]
best_result = df_results.loc[best_gmm_idx]

if best_result['latent_dim'] == 8:
    print("✓ Current 8D latent space is OPTIMAL")
    print(f"  Even though {int(baseline_8d['collapsed_dims'])} dimensions collapsed,")
    print(f"  the configuration with latent_dim=8 achieves best performance.")
    print()
    print("  This suggests:")
    print("  - The collapse pattern is beneficial (selective dimensionality)")
    print("  - Extra capacity helps optimization even if not all dimensions used")
    print("  - β=0.5 allows model to find optimal effective dimensionality")
elif best_result['latent_dim'] == 4:
    improvement = (best_result['gmm_ari'] - baseline_8d['gmm_ari']) / baseline_8d['gmm_ari'] * 100
    print(f"✓ 4D latent space is BETTER than 8D (+{improvement:.1f}%)")
    print(f"  GMM ARI: {best_result['gmm_ari']:.4f} vs {baseline_8d['gmm_ari']:.4f}")
    print()
    print("  This suggests:")
    print("  - Forcing 4D is more efficient than letting 8D collapse to ~4D")
    print("  - Smaller bottleneck prevents overfitting")
    print("  - Should use latent_dim=4 in production")
else:
    improvement = (best_result['gmm_ari'] - baseline_8d['gmm_ari']) / baseline_8d['gmm_ari'] * 100
    print(f"✓ latent_dim={int(best_result['latent_dim'])} is BETTER than 8D (+{improvement:.1f}%)")
    print(f"  GMM ARI: {best_result['gmm_ari']:.4f} vs {baseline_8d['gmm_ari']:.4f}")
    print(f"  Effective dim: {int(best_result['effective_dim'])} vs {int(baseline_8d['effective_dim'])}")

print("="*100)
