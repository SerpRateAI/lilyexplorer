"""
Re-evaluate VAE v2.1 using simplified lithology labels.

Compare ARI with:
- Original 139 fine-grained labels
- Simplified 12 major lithology types
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from collections import Counter
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_1_model import VAE, DistributionAwareScaler

def load_and_evaluate(use_simplified=True):
    """Load v2.1 model and evaluate with simplified labels."""

    # Load model
    model_path = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    if use_simplified:
        data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm_simplified.csv')
        label_col = 'Lithology_Simplified'
    else:
        data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
        label_col = 'Principal'

    df = pd.read_csv(data_path)

    # Sample for faster evaluation
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)

    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    X = df[feature_cols].values
    lithology = df[label_col].values

    # Scale
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Get latent
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        mu, _ = model.encode(X_tensor)
        latent_vectors = mu.numpy()

    return latent_vectors, lithology

def evaluate_clustering(latent_vectors, lithology, k_values):
    """Evaluate at multiple k values."""
    results = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)

        silhouette = silhouette_score(latent_vectors, cluster_labels)
        ari = adjusted_rand_score(lithology, cluster_labels)

        # Find high-purity clusters
        high_purity = []
        for cluster_id in range(k):
            cluster_mask = cluster_labels == cluster_id
            cluster_lithologies = lithology[cluster_mask]
            if len(cluster_lithologies) > 0:
                most_common = Counter(cluster_lithologies).most_common(1)
                purity = most_common[0][1] / len(cluster_lithologies)
                if purity > 0.8:
                    high_purity.append((most_common[0][0], purity, len(cluster_lithologies)))

        results.append({
            'k': k,
            'silhouette': silhouette,
            'ari': ari,
            'high_purity_count': len(high_purity),
            'high_purity': high_purity
        })

    return results

def main():
    print("="*80)
    print("VAE v2.1 - Simplified vs Original Labels")
    print("="*80)

    k_values = [5, 10, 12, 15, 20]

    # Evaluate with simplified labels
    print("\nEvaluating with SIMPLIFIED labels (12 major types)...")
    latent_simp, lithology_simp = load_and_evaluate(use_simplified=True)
    print(f"Loaded {len(latent_simp):,} samples")
    print(f"Unique lithologies: {len(np.unique(lithology_simp))}")
    results_simp = evaluate_clustering(latent_simp, lithology_simp, k_values)

    # Evaluate with original labels
    print("\nEvaluating with ORIGINAL labels (139 fine-grained)...")
    latent_orig, lithology_orig = load_and_evaluate(use_simplified=False)
    print(f"Loaded {len(latent_orig):,} samples")
    print(f"Unique lithologies: {len(np.unique(lithology_orig))}")
    results_orig = evaluate_clustering(latent_orig, lithology_orig, k_values)

    # Compare
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"\n{'k':>3} | {'Metric':>12} | {'Original (139)':>15} | {'Simplified (12)':>17} | {'Change':>10}")
    print("-"*85)

    for i, k in enumerate(k_values):
        orig = results_orig[i]
        simp = results_simp[i]

        ari_change = (simp['ari'] - orig['ari']) / orig['ari'] * 100 if orig['ari'] > 0 else 0
        sil_change = (simp['silhouette'] - orig['silhouette']) / orig['silhouette'] * 100

        print(f"{k:3d} | {'ARI':>12} | {orig['ari']:15.3f} | {simp['ari']:17.3f} | {ari_change:+9.1f}%")
        print(f"{'':3s} | {'Silhouette':>12} | {orig['silhouette']:15.3f} | {simp['silhouette']:17.3f} | {sil_change:+9.1f}%")
        print(f"{'':3s} | {'Pure (>80%)':>12} | {orig['high_purity_count']:15d} | {simp['high_purity_count']:17d} |")
        print("-"*85)

    # Highlight k=12 (matches number of simplified labels)
    print("\n" + "="*80)
    print("DETAILED ANALYSIS AT k=12 (matches 12 simplified labels)")
    print("="*80)

    k12_simp = results_simp[k_values.index(12)]
    k12_orig = results_orig[k_values.index(12)]

    print(f"\nSimplified labels (12 types):")
    print(f"  ARI:        {k12_simp['ari']:.3f}")
    print(f"  Silhouette: {k12_simp['silhouette']:.3f}")
    print(f"  High-purity clusters (>80%): {k12_simp['high_purity_count']}")
    if k12_simp['high_purity']:
        for lith, purity, size in sorted(k12_simp['high_purity'], key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {lith:30s}: {purity*100:.1f}% (n={size})")

    print(f"\nOriginal labels (139 types):")
    print(f"  ARI:        {k12_orig['ari']:.3f}")
    print(f"  Silhouette: {k12_orig['silhouette']:.3f}")
    print(f"  High-purity clusters (>80%): {k12_orig['high_purity_count']}")

    ari_improvement = (k12_simp['ari'] - k12_orig['ari']) / k12_orig['ari'] * 100
    print(f"\nImprovement: {ari_improvement:+.1f}% ARI")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nSimplified labels (12 major types) provide:")
    print("  - More interpretable groupings")
    print("  - Reduced noise from rare lithologies")
    best_simp_k = max(results_simp, key=lambda x: x['ari'])
    print(f"  - Best ARI: {best_simp_k['ari']:.3f} at k={best_simp_k['k']}")
    print(f"\nThis is {(best_simp_k['ari']/max(r['ari'] for r in results_orig))*100:.1f}% of original best performance")
    print("but with much more meaningful lithology categories.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
