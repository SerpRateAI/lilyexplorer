"""
Compare VAE GRA v2.1 vs v2.2 Performance

v2.1: 6D input (current position only) → 8D latent
v2.2: 18D input (with spatial context) → 8D latent

Evaluates clustering performance on k=5, 10, 15, 20 clusters.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from collections import Counter
import sys

# Import model classes
sys.path.append('/home/utig5/johna/bhai')
from ml_models.vae_lithology_gra_v2_1_model import VAE as VAE_v2_1, DistributionAwareScaler as Scaler_v2_1
from ml_models.vae_lithology_gra_v2_2_model import VAE as VAE_v2_2, DistributionAwareScaler as Scaler_v2_2

# Make DistributionAwareScaler available in __main__ namespace for unpickling
DistributionAwareScaler = Scaler_v2_1

def load_v2_1_model(checkpoint_path, device='cpu'):
    """Load v2.1 model (6D input)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VAE_v2_1(
        input_dim=6,
        latent_dim=checkpoint['latent_dim'],
        hidden_dims=[32, 16]
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['scaler'], checkpoint['label_encoder']

def load_v2_2_model(checkpoint_path, device='cpu'):
    """Load v2.2 model (18D input)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = VAE_v2_2(
        input_dim=18,
        output_dim=6,
        latent_dim=checkpoint['latent_dim'],
        hidden_dims=[32, 16]
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['scaler'], checkpoint['label_encoder']

def prepare_v2_1_data(data_path_v2_1, sample_size=50000):
    """Prepare v2.1 data (6D features)."""
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B',
        'Principal'
    ]

    # Read only necessary columns for speed
    df = pd.read_csv(data_path_v2_1, usecols=feature_cols)

    # Sample for faster evaluation
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Sampled {sample_size:,} rows")

    X = df[feature_cols[:-1]].values  # All except Principal
    lithology = df['Principal'].values

    return X, lithology

def prepare_v2_2_data(data_path_v2_2, sample_size=50000):
    """Prepare v2.2 data (18D features with spatial context)."""
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    # Build list of columns to read
    cols_to_read = feature_cols.copy()
    cols_to_read += [f'{col}_above' for col in feature_cols]
    cols_to_read += [f'{col}_below' for col in feature_cols]
    cols_to_read.append('Principal')

    # Read only necessary columns for speed
    df = pd.read_csv(data_path_v2_2, usecols=cols_to_read)

    # Sample for faster evaluation
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Sampled {sample_size:,} rows")

    # Current position
    X_current = df[feature_cols].values

    # Above position
    X_above = df[[f'{col}_above' for col in feature_cols]].values

    # Below position
    X_below = df[[f'{col}_below' for col in feature_cols]].values

    # Combine: [current, above, below]
    X_18d = np.concatenate([X_current, X_above, X_below], axis=1)

    lithology = df['Principal'].values

    return X_18d, X_current, lithology

def encode_to_latent(model, X_scaled, device='cpu'):
    """Encode data to latent space."""
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        mu, _ = model.encode(X_tensor)
        return mu.cpu().numpy()

def evaluate_clustering(latent_repr, lithology_labels, k_values=[5, 10, 15, 20]):
    """Evaluate clustering performance."""
    results = {}

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_repr)

        # Silhouette score
        sil_score = silhouette_score(latent_repr, cluster_labels)

        # Adjusted Rand Index
        ari_score = adjusted_rand_score(lithology_labels, cluster_labels)

        results[k] = {
            'silhouette': sil_score,
            'ari': ari_score,
            'cluster_labels': cluster_labels
        }

        print(f"  k={k:2d}: Silhouette={sil_score:.3f}, ARI={ari_score:.3f}")

    return results

def find_high_purity_clusters(cluster_labels, lithology_labels, k, top_n=3):
    """Find clusters with highest lithology purity."""
    cluster_info = []

    for cluster_id in range(k):
        cluster_mask = cluster_labels == cluster_id
        cluster_liths = lithology_labels[cluster_mask]

        if len(cluster_liths) == 0:
            continue

        # Most common lithology in this cluster
        lith_counts = Counter(cluster_liths)
        most_common_lith, count = lith_counts.most_common(1)[0]
        purity = count / len(cluster_liths) * 100

        cluster_info.append({
            'cluster_id': cluster_id,
            'size': len(cluster_liths),
            'lithology': most_common_lith,
            'purity': purity,
            'count': count
        })

    # Sort by purity
    cluster_info.sort(key=lambda x: x['purity'], reverse=True)

    return cluster_info[:top_n]

def main():
    print("="*80)
    print("VAE GRA v2.1 vs v2.2 Performance Comparison")
    print("="*80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Paths
    checkpoint_v2_1 = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth')
    checkpoint_v2_2 = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_2_latent8.pth')

    data_v2_1 = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    data_v2_2 = Path('/home/utig5/johna/bhai/vae_training_data_v2_2_20cm.csv')

    # =========================================================================
    # Evaluate v2.1
    # =========================================================================
    print("\n" + "="*80)
    print("VAE GRA v2.1 - 6D Input (Current Position Only)")
    print("="*80)

    print("\nLoading v2.1 model and data...")
    model_v2_1, scaler_v2_1, label_encoder = load_v2_1_model(checkpoint_v2_1, device)
    X_v2_1, lithology_v2_1 = prepare_v2_1_data(data_v2_1)

    print(f"  Data shape: {X_v2_1.shape}")
    print(f"  Unique lithologies: {len(np.unique(lithology_v2_1))}")

    print("\nScaling and encoding to latent space...")
    X_v2_1_scaled = scaler_v2_1.transform(X_v2_1)
    latent_v2_1 = encode_to_latent(model_v2_1, X_v2_1_scaled, device)

    print(f"  Latent shape: {latent_v2_1.shape}")

    print("\nClustering performance (v2.1):")
    results_v2_1 = evaluate_clustering(latent_v2_1, lithology_v2_1)

    print("\nHigh-purity clusters (k=20):")
    top_clusters_v2_1 = find_high_purity_clusters(
        results_v2_1[20]['cluster_labels'],
        lithology_v2_1,
        k=20,
        top_n=3
    )
    for i, cluster in enumerate(top_clusters_v2_1, 1):
        print(f"  {i}. Cluster {cluster['cluster_id']}: "
              f"{cluster['purity']:.1f}% {cluster['lithology']} "
              f"(n={cluster['size']})")

    # =========================================================================
    # Evaluate v2.2
    # =========================================================================
    print("\n" + "="*80)
    print("VAE GRA v2.2 - 18D Input (With Spatial Context)")
    print("="*80)

    print("\nLoading v2.2 model and data...")
    model_v2_2, scaler_v2_2, _ = load_v2_2_model(checkpoint_v2_2, device)
    X_v2_2, X_current_v2_2, lithology_v2_2 = prepare_v2_2_data(data_v2_2)

    print(f"  Data shape: {X_v2_2.shape}")
    print(f"  Unique lithologies: {len(np.unique(lithology_v2_2))}")

    print("\nScaling and encoding to latent space...")
    X_v2_2_scaled = scaler_v2_2.transform(X_v2_2)
    latent_v2_2 = encode_to_latent(model_v2_2, X_v2_2_scaled, device)

    print(f"  Latent shape: {latent_v2_2.shape}")

    print("\nClustering performance (v2.2):")
    results_v2_2 = evaluate_clustering(latent_v2_2, lithology_v2_2)

    print("\nHigh-purity clusters (k=20):")
    top_clusters_v2_2 = find_high_purity_clusters(
        results_v2_2[20]['cluster_labels'],
        lithology_v2_2,
        k=20,
        top_n=3
    )
    for i, cluster in enumerate(top_clusters_v2_2, 1):
        print(f"  {i}. Cluster {cluster['cluster_id']}: "
              f"{cluster['purity']:.1f}% {cluster['lithology']} "
              f"(n={cluster['size']})")

    # =========================================================================
    # Comparison Summary
    # =========================================================================
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*80)

    print("\n| k-clusters | Metric     | v2.1   | v2.2   | Change   |")
    print("|------------|------------|--------|--------|----------|")

    for k in [5, 10, 15, 20]:
        # ARI comparison
        ari_v2_1 = results_v2_1[k]['ari']
        ari_v2_2 = results_v2_2[k]['ari']
        ari_change = (ari_v2_2 - ari_v2_1) / ari_v2_1 * 100 if ari_v2_1 > 0 else 0

        # Silhouette comparison
        sil_v2_1 = results_v2_1[k]['silhouette']
        sil_v2_2 = results_v2_2[k]['silhouette']
        sil_change = (sil_v2_2 - sil_v2_1) / sil_v2_1 * 100 if sil_v2_1 > 0 else 0

        print(f"| k={k:2d}       | ARI        | {ari_v2_1:.3f}  | {ari_v2_2:.3f}  | {ari_change:+6.1f}% |")
        print(f"|            | Silhouette | {sil_v2_1:.3f}  | {sil_v2_2:.3f}  | {sil_change:+6.1f}% |")

    print("\n" + "="*80)
    print("Key Findings:")
    print("="*80)

    # Best improvements
    best_k = 10
    ari_improvement = (results_v2_2[best_k]['ari'] - results_v2_1[best_k]['ari']) / results_v2_1[best_k]['ari'] * 100

    print(f"\n1. Spatial context effect at k={best_k}:")
    print(f"   - v2.1 ARI: {results_v2_1[best_k]['ari']:.3f}")
    print(f"   - v2.2 ARI: {results_v2_2[best_k]['ari']:.3f}")
    print(f"   - Change: {ari_improvement:+.1f}%")

    print(f"\n2. Cluster purity comparison (k=20):")
    print(f"   v2.1 best: {top_clusters_v2_1[0]['purity']:.1f}% {top_clusters_v2_1[0]['lithology']}")
    print(f"   v2.2 best: {top_clusters_v2_2[0]['purity']:.1f}% {top_clusters_v2_2[0]['lithology']}")

    if ari_improvement > 0:
        print(f"\n3. ✓ Spatial context IMPROVES lithology discrimination")
    elif ari_improvement < -5:
        print(f"\n3. ✗ Spatial context HURTS lithology discrimination")
    else:
        print(f"\n3. ~ Spatial context has MINIMAL effect on lithology discrimination")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
