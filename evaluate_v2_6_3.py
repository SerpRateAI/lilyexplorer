"""
Evaluate VAE GRA v2.6.3 (RGB-only) clustering performance.

Compare to:
  - v2.6 (GRA+MS+NGR+RGB, 6D) - best model
  - v1 (GRA+MS+NGR, 3D) - physical properties only
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from vae_lithology_gra_v2_5_model import VAE

print("="*80)
print("VAE GRA v2.6.3 (RGB-Only) Clustering Evaluation")
print("="*80)
print()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load v2.6.3 (RGB-only) model
print("Loading v2.6.3 model (RGB-only)...")
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_3_rgb_only.pth', map_location=device, weights_only=False)

model = VAE(input_dim=3, latent_dim=8, hidden_dims=[32, 16]).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = checkpoint['scaler']
print("Model loaded")
print()

# Load test data
print("Loading v2.6 dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = ['R', 'G', 'B']

X = df[feature_cols].values
lithology = df['Principal'].values
borehole_ids = df['Borehole_ID'].values

print(f"Dataset: {len(X):,} samples")
print(f"Features: 3D (R, G, B)")
print(f"Lithologies: {len(np.unique(lithology))}")
print()

# Scale data
print("Scaling data...")
X_log = np.log1p(X)
X_scaled = scaler.transform(X_log)
print()

# Extract latent representations
print("Extracting latent representations...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    mu, logvar = model.encode(X_tensor)
    latent = mu.cpu().numpy()

print(f"Latent shape: {latent.shape}")
print()

# Use subset for clustering evaluation
print("Using 50K sample subset for clustering evaluation...")
np.random.seed(42)
subset_indices = np.random.choice(len(latent), size=min(50000, len(latent)), replace=False)
latent_subset = latent[subset_indices]
lithology_subset = lithology[subset_indices]
print(f"Subset size: {len(latent_subset):,} samples")
print()

# Evaluate clustering for different k values
print("="*80)
print("Clustering Performance (v2.6.3 - RGB Only)")
print("="*80)
print()

k_values = [10, 12, 15, 20]
results_v2_6_3 = []

for k in k_values:
    print(f"k={k} clusters:")

    # K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_subset)

    # Compute metrics
    ari = adjusted_rand_score(lithology_subset, cluster_labels)
    silhouette = silhouette_score(latent_subset, cluster_labels)

    results_v2_6_3.append({
        'k': k,
        'ARI': ari,
        'Silhouette': silhouette
    })

    print(f"  ARI: {ari:.4f}")
    print(f"  Silhouette: {silhouette:.4f}")
    print()

# Compute averages
avg_ari = np.mean([r['ARI'] for r in results_v2_6_3])
avg_silhouette = np.mean([r['Silhouette'] for r in results_v2_6_3])

print(f"Average ARI: {avg_ari:.4f}")
print(f"Average Silhouette: {avg_silhouette:.4f}")
print()

# Comparison to v2.6 and v1
print("="*80)
print("Comparison: RGB-Only vs Full Features vs Physical-Only")
print("="*80)
print()

# Reference results
results_v2_6 = [
    {'k': 10, 'ARI': 0.238, 'Silhouette': 0.428},
    {'k': 12, 'ARI': 0.258, 'Silhouette': 0.409},
    {'k': 15, 'ARI': 0.237, 'Silhouette': 0.392},
    {'k': 20, 'ARI': 0.237, 'Silhouette': 0.379}
]

results_v1 = [
    {'k': 10, 'ARI': 0.080, 'Silhouette': 0.413},
    {'k': 12, 'ARI': 0.084, 'Silhouette': 0.395},
    {'k': 15, 'ARI': 0.087, 'Silhouette': 0.381},
    {'k': 20, 'ARI': 0.099, 'Silhouette': 0.369}
]

print(f"{'k':<5} {'RGB-only':<12} {'v2.6 (full)':<12} {'v1 (phys)':<12} {'RGB vs Full':<15} {'RGB vs Phys':<15}")
print(f"{'':5} {'(v2.6.3)':<12} {'(6D)':<12} {'(3D)':<12}")
print("-"*90)

for i, k in enumerate(k_values):
    ari_rgb = results_v2_6_3[i]['ARI']
    ari_full = results_v2_6[i]['ARI']
    ari_phys = results_v1[i]['ARI']

    delta_full = ari_rgb - ari_full
    pct_full = 100 * delta_full / ari_full if ari_full > 0 else 0

    delta_phys = ari_rgb - ari_phys
    pct_phys = 100 * delta_phys / ari_phys if ari_phys > 0 else 0

    print(f"{k:<5} {ari_rgb:<12.4f} {ari_full:<12.4f} {ari_phys:<12.4f} {delta_full:+.4f} ({pct_full:+.1f}%)   {delta_phys:+.4f} ({pct_phys:+.1f}%)")

print()

# Average comparison
avg_ari_v2_6 = np.mean([r['ARI'] for r in results_v2_6])
avg_ari_v1 = np.mean([r['ARI'] for r in results_v1])

delta_avg_full = avg_ari - avg_ari_v2_6
pct_avg_full = 100 * delta_avg_full / avg_ari_v2_6 if avg_ari_v2_6 > 0 else 0

delta_avg_phys = avg_ari - avg_ari_v1
pct_avg_phys = 100 * delta_avg_phys / avg_ari_v1 if avg_ari_v1 > 0 else 0

print(f"{'Avg':<5} {avg_ari:<12.4f} {avg_ari_v2_6:<12.4f} {avg_ari_v1:<12.4f} {delta_avg_full:+.4f} ({pct_avg_full:+.1f}%)   {delta_avg_phys:+.4f} ({pct_avg_phys:+.1f}%)")
print()

# Summary
print("="*80)
print("Summary")
print("="*80)
print()

print("Feature Comparison:")
print(f"  v2.6.3 (RGB-only):          3D (R, G, B)")
print(f"  v2.6 (full features):       6D (GRA, MS, NGR, R, G, B)")
print(f"  v1 (physical-only):         3D (GRA, MS, NGR)")
print()

print(f"v2.6.3 Performance (RGB-only):")
print(f"  Average ARI: {avg_ari:.4f}")
print(f"  Best ARI: {max([r['ARI'] for r in results_v2_6_3]):.4f} (k={[r['k'] for r in results_v2_6_3][np.argmax([r['ARI'] for r in results_v2_6_3])]:.0f})")
print(f"  Average Silhouette: {avg_silhouette:.4f}")
print()

print(f"v2.6 Performance (full features):")
print(f"  Average ARI: {avg_ari_v2_6:.4f}")
print(f"  Best ARI: {max([r['ARI'] for r in results_v2_6]):.4f} (k=12)")
print()

print(f"v1 Performance (physical-only):")
print(f"  Average ARI: {avg_ari_v1:.4f}")
print(f"  Best ARI: {max([r['ARI'] for r in results_v1]):.4f} (k=20)")
print()

# Key findings
print("Key Findings:")
print()

if avg_ari > avg_ari_v2_6 * 0.95:
    print(f"✓ RGB-only achieves {pct_avg_full:+.1f}% vs full features (v2.6)")
    print(f"  → Physical properties (GRA+MS+NGR) add minimal value!")
    print(f"  → RGB color is the dominant signal for lithology clustering")
elif avg_ari > avg_ari_v1 * 1.5:
    print(f"○ RGB-only achieves {pct_avg_full:+.1f}% vs full features (v2.6)")
    print(f"  → Physical properties help, but RGB is primary driver")
    print(f"  → RGB-only is {pct_avg_phys:+.1f}% better than physical-only")
else:
    print(f"✗ RGB-only achieves {pct_avg_full:+.1f}% vs full features (v2.6)")
    print(f"  → Multi-modal features (physical + visual) are necessary")
    print(f"  → RGB and physical properties provide complementary information")

print()

# Save results
results_df = pd.DataFrame(results_v2_6_3)
results_df.to_csv('vae_v2_6_3_clustering_results.csv', index=False)
print("Results saved to: vae_v2_6_3_clustering_results.csv")
