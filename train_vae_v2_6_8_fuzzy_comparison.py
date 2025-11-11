"""
VAE v2.6.8 - Fuzzy Matching Tolerance Comparison

Tests multiple fuzzy matching tolerances to find optimal data/performance trade-off.

Approach:
- Same architecture as v2.6.7 (10D latent, β annealing 1e-10→0.75)
- Train on datasets with different fuzzy matching tolerances
- Compare ARI performance vs sample count
- Find optimal tolerance

Tolerances to test: ±10cm, ±20cm (baseline), ±50cm, ±1m, ±2m
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import sys
import time
import argparse

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler

print("="*100)
print("VAE v2.6.8 - FUZZY MATCHING TOLERANCE COMPARISON")
print("="*100)
print()

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--tolerance_cm', type=int, required=True,
                   help='Fuzzy matching tolerance in cm (e.g., 10, 20, 50, 100, 200)')
parser.add_argument('--gpu_id', type=int, default=0,
                   help='GPU ID to use (0-3 on cotopaxi)')
parser.add_argument('--epochs', type=int, default=100,
                   help='Maximum number of epochs')
args = parser.parse_args()

tolerance_cm = args.tolerance_cm
gpu_id = args.gpu_id

print(f"Configuration:")
print(f"  Tolerance: ± {tolerance_cm} cm")
print(f"  GPU ID: {gpu_id}")
print(f"  Max epochs: {args.epochs}")
print()

# Load fuzzy-matched dataset
dataset_file = f'vae_training_data_fuzzy_{tolerance_cm}cm.csv'
print(f"Loading fuzzy ±{tolerance_cm}cm dataset: {dataset_file}")

df = pd.read_csv(dataset_file)
print(f"✓ Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")
print()

# Load lithology hierarchy
hierarchy_df = pd.read_csv('lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'],
                             hierarchy_df['Lithology_Group']))
df['Lithology_Group'] = df['Principal'].map(principal_to_group)

# Remove samples with NaN lithology
df = df.dropna(subset=['Lithology_Group'])

# Features and labels
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
y = df['Lithology_Group'].values

print(f"Dataset statistics:")
print(f"  Samples: {len(X):,}")
print(f"  Boreholes: {df['Borehole_ID'].nunique()}")
print(f"  Lithology groups: {len(np.unique(y))}")
print()

# Scale features
print("Scaling features (distribution-aware)...")
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X)
print("✓ Features scaled")
print()

# Convert to tensors
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

X_tensor = torch.FloatTensor(X_scaled).to(device)

# Initialize VAE (same architecture as v2.6.7)
latent_dim = 10
hidden_dims = [32, 16]
vae_model = VAE(input_dim=6, latent_dim=latent_dim, hidden_dims=hidden_dims).to(device)

print(f"VAE architecture:")
print(f"  Input: 6D (GRA, MS, NGR, RGB)")
print(f"  Hidden: {hidden_dims}")
print(f"  Latent: {latent_dim}D")
print(f"  Parameters: {sum(p.numel() for p in vae_model.parameters()):,}")
print()

# β annealing schedule (same as v2.6.7)
beta_start = 1e-10
beta_end = 0.75
anneal_epochs = 50

def get_beta(epoch):
    if epoch < anneal_epochs:
        return beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
    else:
        return beta_end

# Training
optimizer = torch.optim.Adam(vae_model.parameters(), lr=1e-3)
batch_size = 256
n_samples = len(X_tensor)

print("="*100)
print("TRAINING")
print("="*100)
print()
print(f"β annealing: {beta_start:.2e} → {beta_end} over {anneal_epochs} epochs")
print(f"Batch size: {batch_size}")
print(f"Learning rate: 1e-3")
print()

best_loss = float('inf')
patience = 15
patience_counter = 0

start_time = time.time()

for epoch in range(args.epochs):
    vae_model.train()

    # Shuffle data
    perm = torch.randperm(n_samples)
    X_shuffled = X_tensor[perm]

    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_kl_loss = 0
    n_batches = 0

    beta = get_beta(epoch)

    for i in range(0, n_samples, batch_size):
        batch = X_shuffled[i:i+batch_size]

        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, logvar = vae_model(batch)

        # Loss
        recon_loss = nn.functional.mse_loss(x_recon, batch, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_loss + beta * kl_loss

        # Backward pass
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_recon_loss += recon_loss.item()
        epoch_kl_loss += kl_loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    avg_recon = epoch_recon_loss / n_batches
    avg_kl = epoch_kl_loss / n_batches

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, "
              f"KL={avg_kl:.4f}, β={beta:.4f}")

    # Early stopping
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': vae_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler': scaler,
            'tolerance_cm': tolerance_cm,
            'loss': avg_loss
        }, f'ml_models/checkpoints/vae_gra_v2_6_8_fuzzy_{tolerance_cm}cm.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

train_time = time.time() - start_time
print()
print(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
print(f"Best loss: {best_loss:.4f}")
print()

# Load best model
checkpoint = torch.load(f'ml_models/checkpoints/vae_gra_v2_6_8_fuzzy_{tolerance_cm}cm.pth',
                       map_location=device, weights_only=False)
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()

# Extract embeddings
print("Extracting embeddings...")
with torch.no_grad():
    mu, _ = vae_model.encode(X_tensor)
    embeddings = mu.cpu().numpy()

print(f"✓ Embeddings extracted: {embeddings.shape}")
print()

# Clustering evaluation (multiple k values)
print("="*100)
print("CLUSTERING EVALUATION")
print("="*100)
print()

k_values = [10, 12, 15, 20]
results = []

for k in k_values:
    print(f"GMM clustering with k={k}...")

    gmm = GaussianMixture(n_components=k, covariance_type='diag',
                         random_state=42, n_init=10, reg_covar=1e-5)
    cluster_labels = gmm.fit_predict(embeddings)

    # ARI (requires lithology labels)
    ari = adjusted_rand_score(y, cluster_labels)

    # Silhouette score (unsupervised quality)
    sil = silhouette_score(embeddings, cluster_labels)

    results.append({
        'tolerance_cm': tolerance_cm,
        'n_samples': len(X),
        'k': k,
        'ARI': ari,
        'Silhouette': sil
    })

    print(f"  k={k:2d}: ARI={ari:.4f}, Silhouette={sil:.4f}")

print()

# Save results
results_df = pd.DataFrame(results)
results_file = f'vae_v2_6_8_fuzzy_{tolerance_cm}cm_results.csv'
results_df.to_csv(results_file, index=False)
print(f"✓ Results saved to {results_file}")
print()

# Summary
print("="*100)
print("SUMMARY")
print("="*100)
print()
print(f"Tolerance: ± {tolerance_cm} cm")
print(f"Samples: {len(X):,}")
print(f"Training time: {train_time:.1f}s ({train_time/60:.1f} min)")
print()
print("Clustering performance:")
for _, row in results_df.iterrows():
    print(f"  k={int(row['k']):2d}: ARI={row['ARI']:.4f}, Silhouette={row['Silhouette']:.4f}")
print()

# Compare to v2.6.7 baseline
baseline_ari_k12 = 0.196  # v2.6.7 average from 5-fold CV at k=12
best_ari = results_df['ARI'].max()
best_k = results_df.loc[results_df['ARI'].idxmax(), 'k']

print(f"Best ARI: {best_ari:.4f} at k={int(best_k)}")
print(f"v2.6.7 baseline (±20cm): 0.196 (5-fold CV average)")

if tolerance_cm != 20:
    pct_change = (best_ari - baseline_ari_k12) / baseline_ari_k12 * 100
    sample_change = (len(X) - 238506) / 238506 * 100

    if pct_change > 0:
        print(f"✓ Improvement: {pct_change:+.1f}% ARI with {sample_change:+.1f}% samples")
    else:
        print(f"✗ Degradation: {pct_change:+.1f}% ARI despite {sample_change:+.1f}% samples")
else:
    print("(This is the baseline)")

print()
print("="*100)
