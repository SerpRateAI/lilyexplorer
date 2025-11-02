"""
Compare Gaussian Mixture Models (GMM) vs K-Means for VAE clustering.

GMM advantages over K-Means:
- Elliptical clusters (not just circular)
- Soft probabilistic assignments
- Different cluster densities/variances
- More flexible for real geological data

Tests multiple covariance types:
- 'full': Each cluster has its own general covariance matrix (most flexible)
- 'tied': All clusters share the same covariance matrix
- 'diag': Diagonal covariance (axes-aligned ellipses)
- 'spherical': Spherical clusters (like K-Means but probabilistic)
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
print("VAE GRA v2.6: GMM vs K-MEANS COMPARISON")
print("="*80)
print()
print("Testing Gaussian Mixture Models with different covariance types")
print("Comparing to K-Means baseline")
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

# Test k values focusing on optimal range
k_values = list(range(5, 21))  # Focus on 5-20 based on previous results

# Test different clustering methods
methods = [
    {'name': 'K-Means', 'type': 'kmeans'},
    {'name': 'GMM (full)', 'type': 'gmm', 'cov_type': 'full'},
    {'name': 'GMM (tied)', 'type': 'gmm', 'cov_type': 'tied'},
    {'name': 'GMM (diag)', 'type': 'gmm', 'cov_type': 'diag'},
    {'name': 'GMM (spherical)', 'type': 'gmm', 'cov_type': 'spherical'},
]

all_results = []

for method in methods:
    print("="*80)
    print(f"Testing: {method['name']}")
    print("="*80)
    print()

    method_start = time.time()

    for k in k_values:
        print(f"k={k:2d}: ", end='', flush=True)

        try:
            if method['type'] == 'kmeans':
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clusterer.fit_predict(latent)
            else:
                # GMM
                clusterer = GaussianMixture(
                    n_components=k,
                    covariance_type=method['cov_type'],
                    random_state=42,
                    max_iter=200,
                    n_init=5
                )
                labels = clusterer.fit_predict(latent)

            # Metrics
            ari = adjusted_rand_score(y_test, labels)
            silhouette = silhouette_score(latent, labels, sample_size=10000)

            # BIC/AIC for GMM
            bic = clusterer.bic(latent) if method['type'] == 'gmm' else None
            aic = clusterer.aic(latent) if method['type'] == 'gmm' else None

            all_results.append({
                'method': method['name'],
                'k': k,
                'ari': ari,
                'silhouette': silhouette,
                'bic': bic,
                'aic': aic
            })

            if method['type'] == 'gmm':
                print(f"ARI={ari:.4f}, Sil={silhouette:.4f}, BIC={bic:.0f}, AIC={aic:.0f}", flush=True)
            else:
                print(f"ARI={ari:.4f}, Sil={silhouette:.4f}", flush=True)

        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            all_results.append({
                'method': method['name'],
                'k': k,
                'ari': None,
                'silhouette': None,
                'bic': None,
                'aic': None
            })

    method_time = time.time() - method_start
    print(f"\n{method['name']} completed in {method_time:.1f}s")
    print()

# Convert to DataFrame
df_results = pd.DataFrame(all_results)

print("="*80)
print("RESULTS SUMMARY")
print("="*80)
print()

# Find best for each method
for method_name in df_results['method'].unique():
    method_df = df_results[df_results['method'] == method_name].copy()
    method_df = method_df.dropna(subset=['ari'])  # Remove failed runs

    if len(method_df) == 0:
        print(f"{method_name}: All runs failed")
        continue

    best_ari_idx = method_df['ari'].idxmax()
    best_row = method_df.loc[best_ari_idx]

    print(f"{method_name}:")
    print(f"  Best k={int(best_row['k'])}: ARI={best_row['ari']:.4f}, Silhouette={best_row['silhouette']:.4f}")

    # Top 3
    top3 = method_df.nlargest(3, 'ari')
    print(f"  Top 3 k values: {list(top3['k'].astype(int))}")
    print()

# Overall best
df_valid = df_results.dropna(subset=['ari'])
overall_best_idx = df_valid['ari'].idxmax()
overall_best = df_valid.loc[overall_best_idx]

print("="*80)
print("BEST OVERALL")
print("="*80)
print()
print(f"Method: {overall_best['method']}")
print(f"k = {int(overall_best['k'])}")
print(f"ARI = {overall_best['ari']:.4f}")
print(f"Silhouette = {overall_best['silhouette']:.4f}")
if overall_best['bic'] is not None:
    print(f"BIC = {overall_best['bic']:.0f}")
    print(f"AIC = {overall_best['aic']:.0f}")
print()

# Comparison at k=12 (previous optimal)
print("="*80)
print("COMPARISON AT k=12")
print("="*80)
print()

k12_results = df_results[df_results['k'] == 12].copy()
k12_results = k12_results.sort_values('ari', ascending=False)

print(k12_results[['method', 'ari', 'silhouette']].to_string(index=False))
print()

# Best improvement over K-Means
kmeans_best = df_results[df_results['method'] == 'K-Means']['ari'].max()
print(f"K-Means best ARI: {kmeans_best:.4f}")
print(f"Overall best ARI: {overall_best['ari']:.4f}")
improvement = (overall_best['ari'] - kmeans_best) / kmeans_best * 100
print(f"Improvement: {improvement:+.1f}%")
print()

# Save results
output_path = 'vae_v2_6_gmm_comparison_results.csv'
df_results.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Save model
checkpoint_path = '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_gmm_comparison.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'gmm_results': df_results.to_dict('records'),
    'best_method': overall_best['method'],
    'best_k': int(overall_best['k']),
    'best_ari': float(overall_best['ari'])
}, checkpoint_path)
print(f"Model saved to: {checkpoint_path}")
print()

print("="*80)
print("KEY FINDINGS")
print("="*80)
print()
print("GMM Advantages:")
print("  - 'full': Most flexible, each cluster has its own covariance")
print("  - 'tied': Moderate flexibility, shared covariance across clusters")
print("  - 'diag': Axes-aligned ellipses")
print("  - 'spherical': Similar to K-Means but probabilistic")
print()
print("Use BIC/AIC to select k automatically (lower is better)")
print("ARI measures alignment with true lithology labels")
print("Silhouette measures cluster separation quality")
print("="*80)
