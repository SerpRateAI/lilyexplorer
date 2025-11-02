"""
Compare VAE GRA v2.1 (no depth) vs v2.3 (with relative depth)

This script loads both models and compares their performance on clustering.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from collections import Counter

# Import model classes
import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_1_model import VAE as VAE_v2_1, DistributionAwareScaler as Scaler_v2_1
from vae_lithology_gra_v2_3_model import VAE as VAE_v2_3, DistributionAwareScaler as Scaler_v2_3

def load_model_and_data(model_path, data_path, input_dim, feature_cols):
    """Load model checkpoint and prepare data."""
    # Load checkpoint (model weights only)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    # Create model
    if input_dim == 6:
        model = VAE_v2_1(input_dim=6, latent_dim=checkpoint['latent_dim'], hidden_dims=[32, 16])
    else:
        model = VAE_v2_3(input_dim=7, latent_dim=checkpoint['latent_dim'], hidden_dims=[32, 16])

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    df = pd.read_csv(data_path)

    # Sample for faster comparison
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    X = df[feature_cols].values
    lithology = df['Principal'].values

    # Create and fit scaler (same as in training)
    if input_dim == 6:
        scaler = Scaler_v2_1()
    else:
        scaler = Scaler_v2_3()

    X_scaled = scaler.fit_transform(X)

    return model, X_scaled, lithology

def get_latent_representations(model, X):
    """Extract latent representations."""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        mu, _ = model.encode(X_tensor)
        return mu.numpy()

def evaluate_clustering(latent_vectors, lithology, k_values=[5, 10, 15, 20]):
    """Evaluate clustering performance."""
    results = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)

        silhouette = silhouette_score(latent_vectors, cluster_labels)
        ari = adjusted_rand_score(lithology, cluster_labels)

        # Find high-purity clusters
        high_purity_clusters = []
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_lithologies = lithology[cluster_mask]
            if len(cluster_lithologies) > 0:
                most_common = Counter(cluster_lithologies).most_common(1)
                top_lith = most_common[0][0]
                top_pct = most_common[0][1] / len(cluster_lithologies) * 100
                if top_pct > 80:
                    high_purity_clusters.append((top_lith, top_pct, len(cluster_lithologies)))

        results[k] = {
            'silhouette': silhouette,
            'ari': ari,
            'high_purity': high_purity_clusters
        }

    return results

def main():
    print("="*80)
    print("VAE GRA v2.1 vs v2.3 COMPARISON")
    print("Testing impact of relative depth on clustering performance")
    print("="*80)

    # Paths
    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')

    # Load v2.1 (no depth) - 8D model
    print("\nLoading v2.1 (6D features, no depth)...")
    model_v2_1_path = checkpoint_dir / 'vae_gra_v2_1_latent8.pth'
    data_v2_1_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    feature_cols_v2_1 = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    model_v2_1, X_v2_1, lithology_v2_1 = load_model_and_data(
        model_v2_1_path, data_v2_1_path, 6, feature_cols_v2_1
    )
    print(f"  Loaded {len(X_v2_1):,} samples")

    # Load v2.3 (with depth) - 8D model
    print("\nLoading v2.3 (7D features, with relative depth)...")
    model_v2_3_path = checkpoint_dir / 'vae_gra_v2_3_latent8.pth'
    data_v2_3_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_3_20cm.csv')
    feature_cols_v2_3 = feature_cols_v2_1 + ['Relative_Depth']

    model_v2_3, X_v2_3, lithology_v2_3 = load_model_and_data(
        model_v2_3_path, data_v2_3_path, 7, feature_cols_v2_3
    )
    print(f"  Loaded {len(X_v2_3):,} samples")

    # Get latent representations
    print("\nExtracting latent representations...")
    latent_v2_1 = get_latent_representations(model_v2_1, X_v2_1)
    latent_v2_3 = get_latent_representations(model_v2_3, X_v2_3)

    # Evaluate clustering
    print("\nEvaluating v2.1 (no depth)...")
    results_v2_1 = evaluate_clustering(latent_v2_1, lithology_v2_1)

    print("\nEvaluating v2.3 (with relative depth)...")
    results_v2_3 = evaluate_clustering(latent_v2_3, lithology_v2_3)

    # Print comparison
    print("\n" + "="*80)
    print("RESULTS COMPARISON (8D Latent Space)")
    print("="*80)
    print("\nPerformance by k:")
    print(f"{'k':>3} | {'Metric':>12} | {'v2.1 (no depth)':>15} | {'v2.3 (w/ depth)':>15} | {'Change':>10}")
    print("-" * 80)

    for k in [5, 10, 15, 20]:
        v2_1 = results_v2_1[k]
        v2_3 = results_v2_3[k]

        ari_change = (v2_3['ari'] - v2_1['ari']) / v2_1['ari'] * 100
        sil_change = (v2_3['silhouette'] - v2_1['silhouette']) / v2_1['silhouette'] * 100

        print(f"{k:3d} | {'ARI':>12} | {v2_1['ari']:15.3f} | {v2_3['ari']:15.3f} | {ari_change:+9.1f}%")
        print(f"{k:3d} | {'Silhouette':>12} | {v2_1['silhouette']:15.3f} | {v2_3['silhouette']:15.3f} | {sil_change:+9.1f}%")
        print("-" * 80)

    # Summary at k=10
    print("\n" + "="*80)
    print("SUMMARY AT k=10")
    print("="*80)

    v2_1_k10 = results_v2_1[10]
    v2_3_k10 = results_v2_3[10]

    print(f"\nv2.1 (no depth):")
    print(f"  ARI:        {v2_1_k10['ari']:.3f}")
    print(f"  Silhouette: {v2_1_k10['silhouette']:.3f}")
    print(f"  High-purity clusters (>80%): {len(v2_1_k10['high_purity'])}")
    for lith, pct, size in v2_1_k10['high_purity'][:3]:
        print(f"    - {lith}: {pct:.1f}% (n={size})")

    print(f"\nv2.3 (with relative depth):")
    print(f"  ARI:        {v2_3_k10['ari']:.3f}")
    print(f"  Silhouette: {v2_3_k10['silhouette']:.3f}")
    print(f"  High-purity clusters (>80%): {len(v2_3_k10['high_purity'])}")
    for lith, pct, size in v2_3_k10['high_purity'][:3]:
        print(f"    - {lith}: {pct:.1f}% (n={size})")

    ari_change = (v2_3_k10['ari'] - v2_1_k10['ari']) / v2_1_k10['ari'] * 100
    sil_change = (v2_3_k10['silhouette'] - v2_1_k10['silhouette']) / v2_1_k10['silhouette'] * 100

    print(f"\nChange with relative depth:")
    print(f"  ARI:        {ari_change:+.1f}%")
    print(f"  Silhouette: {sil_change:+.1f}%")

    if ari_change < 0:
        print(f"\n⚠ CONCLUSION: Adding relative depth HURTS performance")
        print(f"  - ARI decreased by {abs(ari_change):.1f}%")
        print(f"  - Depth adds confounding information that degrades lithology clustering")
        print(f"  - Theoretical concerns were correct: depth creates spurious correlations")
    else:
        print(f"\n✓ CONCLUSION: Adding relative depth IMPROVES performance")
        print(f"  - ARI increased by {ari_change:.1f}%")
        print(f"  - Depth provides useful compaction/diagenesis signal")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
