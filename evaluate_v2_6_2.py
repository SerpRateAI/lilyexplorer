"""
Evaluate VAE GRA v2.6.2 clustering performance and compare to v2.6.
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from vae_lithology_gra_v2_6_2_model import VAE

print("="*80)
print("VAE GRA v2.6.2 Clustering Evaluation")
print("="*80)
print()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load v2.6.2 (Stage 2) model
print("Loading v2.6.2 model...")
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_2_stage2.pth', map_location=device, weights_only=False)

model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16], output_dim=6).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = checkpoint['scaler']
print("Model loaded")
print()

# Load test data
print("Loading v2.6 dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'R', 'G', 'B'
]

X = df[feature_cols].values
lithology = df['Principal'].values
borehole_ids = df['Borehole_ID'].values

print(f"Dataset: {len(X):,} samples")
print(f"Lithologies: {len(np.unique(lithology))}")
print()

# Scale data
print("Scaling data...")
X_scaled = scaler.transform(X)
print()

# Extract latent representations
print("Extracting latent representations...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    mu, logvar = model.encode(X_tensor)
    latent = mu.cpu().numpy()

print(f"Latent shape: {latent.shape}")
print()

# Use subset for clustering evaluation (silhouette score is expensive on large datasets)
print("Using 50K sample subset for clustering evaluation...")
np.random.seed(42)
subset_indices = np.random.choice(len(latent), size=min(50000, len(latent)), replace=False)
latent_subset = latent[subset_indices]
lithology_subset = lithology[subset_indices]
print(f"Subset size: {len(latent_subset):,} samples")
print()

# Evaluate clustering for different k values
print("="*80)
print("Clustering Performance (v2.6.2)")
print("="*80)
print()

k_values = [10, 12, 15, 20]
results_v2_6_2 = []

for k in k_values:
    print(f"k={k} clusters:")

    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_subset)

    # Compute metrics
    ari = adjusted_rand_score(lithology_subset, cluster_labels)
    silhouette = silhouette_score(latent_subset, cluster_labels)

    results_v2_6_2.append({
        'k': k,
        'ARI': ari,
        'Silhouette': silhouette
    })

    print(f"  ARI: {ari:.4f}")
    print(f"  Silhouette: {silhouette:.4f}")
    print()

# Compute averages
avg_ari = np.mean([r['ARI'] for r in results_v2_6_2])
avg_silhouette = np.mean([r['Silhouette'] for r in results_v2_6_2])

print(f"Average ARI: {avg_ari:.4f}")
print(f"Average Silhouette: {avg_silhouette:.4f}")
print()

# Load v2.6 results for comparison
print("="*80)
print("Comparison: v2.6.2 (Transfer Learning) vs v2.6 (Baseline)")
print("="*80)
print()

# v2.6 reference results
results_v2_6 = [
    {'k': 10, 'ARI': 0.238, 'Silhouette': 0.428},
    {'k': 12, 'ARI': 0.258, 'Silhouette': 0.409},
    {'k': 15, 'ARI': 0.237, 'Silhouette': 0.392},
    {'k': 20, 'ARI': 0.237, 'Silhouette': 0.379}
]

print(f"{'k':<5} {'v2.6.2 ARI':<15} {'v2.6 ARI':<15} {'Δ ARI':<15} {'v2.6.2 Sil':<15} {'v2.6 Sil':<15} {'Δ Sil':<15}")
print("-"*90)

for i, k in enumerate(k_values):
    ari_v2_6_2 = results_v2_6_2[i]['ARI']
    ari_v2_6 = results_v2_6[i]['ARI']
    delta_ari = ari_v2_6_2 - ari_v2_6
    pct_ari = 100 * delta_ari / ari_v2_6 if ari_v2_6 > 0 else 0

    sil_v2_6_2 = results_v2_6_2[i]['Silhouette']
    sil_v2_6 = results_v2_6[i]['Silhouette']
    delta_sil = sil_v2_6_2 - sil_v2_6
    pct_sil = 100 * delta_sil / sil_v2_6 if sil_v2_6 > 0 else 0

    print(f"{k:<5} {ari_v2_6_2:<15.4f} {ari_v2_6:<15.4f} {delta_ari:+.4f} ({pct_ari:+.1f}%)  "
          f"{sil_v2_6_2:<15.4f} {sil_v2_6:<15.4f} {delta_sil:+.4f} ({pct_sil:+.1f}%)")

print()

# Average comparison
avg_ari_v2_6 = np.mean([r['ARI'] for r in results_v2_6])
avg_sil_v2_6 = np.mean([r['Silhouette'] for r in results_v2_6])

delta_avg_ari = avg_ari - avg_ari_v2_6
pct_avg_ari = 100 * delta_avg_ari / avg_ari_v2_6 if avg_ari_v2_6 > 0 else 0

delta_avg_sil = avg_silhouette - avg_sil_v2_6
pct_avg_sil = 100 * delta_avg_sil / avg_sil_v2_6 if avg_sil_v2_6 > 0 else 0

print(f"{'Avg':<5} {avg_ari:<15.4f} {avg_ari_v2_6:<15.4f} {delta_avg_ari:+.4f} ({pct_avg_ari:+.1f}%)  "
      f"{avg_silhouette:<15.4f} {avg_sil_v2_6:<15.4f} {delta_avg_sil:+.4f} ({pct_avg_sil:+.1f}%)")
print()

# Summary
print("="*80)
print("Summary")
print("="*80)
print()

print("v2.6.2 Training Strategy:")
print("  Stage 1: Pre-train on GRA+MS+NGR (403K samples, 524 boreholes)")
print("  Stage 2: Fine-tune with RGB (239K samples, 296 boreholes)")
print()

print(f"v2.6.2 Performance:")
print(f"  Average ARI: {avg_ari:.4f}")
print(f"  Best ARI: {max([r['ARI'] for r in results_v2_6_2]):.4f} (k={[r['k'] for r in results_v2_6_2][np.argmax([r['ARI'] for r in results_v2_6_2])]:.0f})")
print(f"  Average Silhouette: {avg_silhouette:.4f}")
print()

print(f"v2.6 Performance (baseline):")
print(f"  Average ARI: {avg_ari_v2_6:.4f}")
print(f"  Best ARI: {max([r['ARI'] for r in results_v2_6]):.4f} (k=12)")
print(f"  Average Silhouette: {avg_sil_v2_6:.4f}")
print()

if delta_avg_ari > 0:
    print(f"✓ Transfer learning IMPROVES performance by {pct_avg_ari:+.1f}%")
    print(f"  Pre-training on 524 boreholes captures useful physical property patterns")
    print(f"  that improve RGB-based lithology clustering.")
elif abs(delta_avg_ari) < 0.01:
    print(f"≈ Transfer learning performs SIMILARLY to baseline ({pct_avg_ari:+.1f}%)")
    print(f"  Pre-training on physical properties doesn't significantly change performance.")
    print(f"  RGB features may dominate clustering signal regardless of initialization.")
else:
    print(f"✗ Transfer learning DEGRADES performance by {pct_avg_ari:.1f}%")
    print(f"  Pre-trained physical property representations may interfere with RGB learning.")
    print(f"  Training from scratch (v2.6) may be better for this dataset.")
print()

# Save results
results_df = pd.DataFrame(results_v2_6_2)
results_df.to_csv('vae_v2_6_2_clustering_results.csv', index=False)
print("Results saved to: vae_v2_6_2_clustering_results.csv")
