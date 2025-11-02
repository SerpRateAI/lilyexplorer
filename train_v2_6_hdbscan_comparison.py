"""
Test HDBSCAN (Hierarchical Density-Based Spatial Clustering) for VAE clustering.

HDBSCAN advantages:
- No need to specify k (finds optimal number of clusters automatically)
- Can find arbitrary-shaped clusters (not just spherical/elliptical)
- Identifies noise/outliers (assigns label -1)
- Density-based: clusters dense regions, flexible shapes
- Hierarchical: builds cluster tree, picks best level

Key parameters:
- min_cluster_size: Minimum samples in a cluster (smaller = more clusters)
- min_samples: How conservative clustering is (higher = more noise points)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
import sys
import time

# Install hdbscan if needed
try:
    import hdbscan
except ImportError:
    print("Installing hdbscan...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "hdbscan"])
    import hdbscan

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
print("VAE GRA v2.6: HDBSCAN COMPARISON")
print("="*80)
print()
print("Testing HDBSCAN (density-based, automatic k selection)")
print("Comparing to K-Means and GMM baselines")
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

# Test HDBSCAN with different min_cluster_size values
min_cluster_sizes = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
min_samples_values = [10, 20, 50]

all_results = []

print("="*80)
print("Testing HDBSCAN with different parameters")
print("="*80)
print()

for min_samples in min_samples_values:
    print(f"\nmin_samples = {min_samples}")
    print("-" * 80)

    for min_cluster_size in min_cluster_sizes:
        print(f"min_cluster_size={min_cluster_size:4d}: ", end='', flush=True)

        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                cluster_selection_method='eom',  # excess of mass
                prediction_data=True
            )

            labels = clusterer.fit_predict(latent)

            # Count clusters (excluding noise label -1)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            noise_pct = 100 * n_noise / len(labels)

            # Only compute metrics if we have valid clusters and non-noise points
            if n_clusters > 1 and n_noise < len(labels):
                # For ARI: compare all points including noise
                ari = adjusted_rand_score(y_test, labels)

                # For Silhouette: exclude noise points (label -1)
                non_noise_mask = labels != -1
                if non_noise_mask.sum() > n_clusters:  # Need more points than clusters
                    silhouette = silhouette_score(
                        latent[non_noise_mask],
                        labels[non_noise_mask],
                        sample_size=min(10000, non_noise_mask.sum())
                    )
                else:
                    silhouette = None

                all_results.append({
                    'method': 'HDBSCAN',
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_pct': noise_pct,
                    'ari': ari,
                    'silhouette': silhouette
                })

                sil_str = f"{silhouette:.4f}" if silhouette is not None else "N/A"
                print(f"k={n_clusters:2d}, ARI={ari:.4f}, Sil={sil_str}, Noise={noise_pct:.1f}%", flush=True)
            else:
                all_results.append({
                    'method': 'HDBSCAN',
                    'min_cluster_size': min_cluster_size,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'noise_pct': noise_pct,
                    'ari': None,
                    'silhouette': None
                })
                print(f"k={n_clusters:2d} (insufficient clusters/data)", flush=True)

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            all_results.append({
                'method': 'HDBSCAN',
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'n_clusters': None,
                'n_noise': None,
                'noise_pct': None,
                'ari': None,
                'silhouette': None
            })

print()

# Add baseline results from K-Means and GMM (from previous runs)
print("="*80)
print("Running baseline comparisons (K-Means, GMM)")
print("="*80)
print()

# K-Means at k=12 (previous optimal)
print("K-Means (k=12): ", end='', flush=True)
kmeans = KMeans(n_clusters=12, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(latent)
ari_km = adjusted_rand_score(y_test, labels_km)
sil_km = silhouette_score(latent, labels_km, sample_size=10000)
print(f"ARI={ari_km:.4f}, Sil={sil_km:.4f}")

all_results.append({
    'method': 'K-Means',
    'min_cluster_size': None,
    'min_samples': None,
    'n_clusters': 12,
    'n_noise': 0,
    'noise_pct': 0.0,
    'ari': ari_km,
    'silhouette': sil_km
})

# GMM (full) at k=18 (previous optimal)
print("GMM (full, k=18): ", end='', flush=True)
gmm = GaussianMixture(n_components=18, covariance_type='full', random_state=42)
labels_gmm = gmm.fit_predict(latent)
ari_gmm = adjusted_rand_score(y_test, labels_gmm)
sil_gmm = silhouette_score(latent, labels_gmm, sample_size=10000)
print(f"ARI={ari_gmm:.4f}, Sil={sil_gmm:.4f}")

all_results.append({
    'method': 'GMM (full)',
    'min_cluster_size': None,
    'min_samples': None,
    'n_clusters': 18,
    'n_noise': 0,
    'noise_pct': 0.0,
    'ari': ari_gmm,
    'silhouette': sil_gmm
})

print()

# Convert to DataFrame
df_results = pd.DataFrame(all_results)

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()

# Best HDBSCAN result
hdbscan_results = df_results[df_results['method'] == 'HDBSCAN'].dropna(subset=['ari'])

if len(hdbscan_results) > 0:
    best_hdb_idx = hdbscan_results['ari'].idxmax()
    best_hdb = hdbscan_results.loc[best_hdb_idx]

    print("BEST HDBSCAN:")
    print(f"  min_cluster_size = {int(best_hdb['min_cluster_size'])}")
    print(f"  min_samples = {int(best_hdb['min_samples'])}")
    print(f"  n_clusters = {int(best_hdb['n_clusters'])}")
    print(f"  noise = {best_hdb['noise_pct']:.1f}%")
    print(f"  ARI = {best_hdb['ari']:.4f}")
    sil_str = f"{best_hdb['silhouette']:.4f}" if pd.notna(best_hdb['silhouette']) else "N/A"
    print(f"  Silhouette = {sil_str}")
    print()

    # Top 5 HDBSCAN configurations
    print("Top 5 HDBSCAN configurations by ARI:")
    top5_hdb = hdbscan_results.nlargest(5, 'ari')
    for idx, row in top5_hdb.iterrows():
        sil_str = f"{row['silhouette']:.4f}" if pd.notna(row['silhouette']) else "N/A"
        print(f"  min_size={int(row['min_cluster_size']):4d}, min_samp={int(row['min_samples']):2d}: "
              f"k={int(row['n_clusters']):2d}, ARI={row['ari']:.4f}, Sil={sil_str}, "
              f"Noise={row['noise_pct']:.1f}%")
    print()
else:
    print("No valid HDBSCAN results")
    print()

# Overall comparison
print("="*80)
print("METHOD COMPARISON")
print("="*80)
print()

comparison = []
comparison.append({
    'Method': 'K-Means (k=12)',
    'ARI': ari_km,
    'Silhouette': sil_km,
    'Clusters': 12,
    'Noise %': 0.0
})
comparison.append({
    'Method': 'GMM (full, k=18)',
    'ARI': ari_gmm,
    'Silhouette': sil_gmm,
    'Clusters': 18,
    'Noise %': 0.0
})

if len(hdbscan_results) > 0:
    comparison.append({
        'Method': f'HDBSCAN (best)',
        'ARI': best_hdb['ari'],
        'Silhouette': best_hdb['silhouette'] if pd.notna(best_hdb['silhouette']) else None,
        'Clusters': int(best_hdb['n_clusters']),
        'Noise %': best_hdb['noise_pct']
    })

df_comp = pd.DataFrame(comparison).sort_values('ARI', ascending=False)
print(df_comp.to_string(index=False))
print()

# Save results
output_path = 'vae_v2_6_hdbscan_comparison_results.csv'
df_results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Save model
checkpoint_path = '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_hdbscan_comparison.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'hdbscan_results': df_results.to_dict('records'),
    'best_hdbscan': best_hdb.to_dict() if len(hdbscan_results) > 0 else None,
    'kmeans_k12': {'ari': ari_km, 'silhouette': sil_km},
    'gmm_k18': {'ari': ari_gmm, 'silhouette': sil_gmm}
}, checkpoint_path)
print(f"Model saved to: {checkpoint_path}")
print()

print("="*80)
print("KEY INSIGHTS")
print("="*80)
print()
print("HDBSCAN advantages:")
print("  - Automatic k selection (no need to specify)")
print("  - Finds arbitrary-shaped clusters")
print("  - Identifies noise/outliers")
print("  - Density-based approach")
print()
print("Trade-offs:")
print("  - Slower than K-Means")
print("  - Sensitive to min_cluster_size parameter")
print("  - May classify many points as noise")
print("="*80)
