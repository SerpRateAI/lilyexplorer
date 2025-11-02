"""
VAE GRA v2.1 with Unsupervised Model Selection

Addresses the "cheating" problem: We can't use k=139 (number of lithology labels)
because that uses label information.

Solution: Evaluate k=5,10,15,20,30,40,50,60 and select best using ONLY unsupervised metrics:
- Silhouette score (cluster separation quality)
- BIC (Bayesian Information Criterion)
- Calinski-Harabasz score (variance ratio)

THEN compare to labels via ARI to see how well unsupervised selection did.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from collections import Counter
import sys

# Import model
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_1_model import VAE, DistributionAwareScaler

def compute_bic(X, labels, n_clusters):
    """
    Compute BIC (Bayesian Information Criterion) for K-Means clustering.

    BIC = log-likelihood - (k * d/2) * log(n)
    where k=n_clusters, d=latent_dim, n=n_samples

    Lower BIC is better.
    """
    n_samples, latent_dim = X.shape

    # Compute within-cluster sum of squares
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # Variance per cluster
    variance = np.sum((X - kmeans.cluster_centers_[labels])**2) / (n_samples - n_clusters)

    # Log-likelihood (assuming Gaussian)
    log_likelihood = -n_samples * latent_dim / 2 * np.log(2 * np.pi * variance) \
                     - (n_samples - n_clusters) / 2

    # BIC penalty
    n_params = n_clusters * latent_dim  # Cluster centers
    bic = -2 * log_likelihood + n_params * np.log(n_samples)

    return bic

def load_model_and_data():
    """Load v2.1 model and extract latent representations."""
    print("Loading VAE v2.1 model...")

    model_path = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth')
    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Create model
    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    df = pd.read_csv(data_path)

    # Sample for faster evaluation
    if len(df) > 50000:
        print(f"Sampling 50K from {len(df):,} samples for faster evaluation...")
        df = df.sample(n=50000, random_state=42)

    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    X = df[feature_cols].values
    lithology = df['Principal'].values

    # Scale data
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Get latent representations
    print("Extracting latent representations...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_scaled)
        mu, _ = model.encode(X_tensor)
        latent_vectors = mu.numpy()

    print(f"Loaded {len(latent_vectors):,} latent vectors (8D)\n")

    return latent_vectors, lithology

def evaluate_k_values(latent_vectors, lithology, k_values):
    """
    Evaluate clustering for different k values using ONLY unsupervised metrics.
    """
    print("="*80)
    print("UNSUPERVISED MODEL SELECTION")
    print("="*80)
    print("\nEvaluating k values using unsupervised metrics ONLY...")
    print("(ARI shown for comparison, but NOT used for selection)\n")

    results = []

    for k in k_values:
        print(f"k={k:3d}:", end=" ")

        # Run K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)

        # UNSUPERVISED metrics (can use for selection)
        silhouette = silhouette_score(latent_vectors, cluster_labels)
        ch_score = calinski_harabasz_score(latent_vectors, cluster_labels)
        bic = compute_bic(latent_vectors, cluster_labels, k)

        # SUPERVISED metric (for comparison only, NOT for selection)
        ari = adjusted_rand_score(lithology, cluster_labels)

        print(f"Silhouette={silhouette:.3f}, CH={ch_score:7.1f}, BIC={bic:8.1f} | ARI={ari:.3f}")

        results.append({
            'k': k,
            'silhouette': silhouette,
            'ch_score': ch_score,
            'bic': bic,
            'ari': ari,
            'labels': cluster_labels
        })

    return results

def select_best_k(results):
    """Select best k using ONLY unsupervised metrics."""
    print("\n" + "="*80)
    print("MODEL SELECTION (Unsupervised)")
    print("="*80)

    # Best by each unsupervised metric
    best_silhouette = max(results, key=lambda x: x['silhouette'])
    best_ch = max(results, key=lambda x: x['ch_score'])
    best_bic = min(results, key=lambda x: x['bic'])  # Lower BIC is better

    print(f"\nBest k by Silhouette score: k={best_silhouette['k']} (Silhouette={best_silhouette['silhouette']:.3f})")
    print(f"Best k by CH score:          k={best_ch['k']} (CH={best_ch['ch_score']:.1f})")
    print(f"Best k by BIC:               k={best_bic['k']} (BIC={best_bic['bic']:.1f})")

    # Consensus: Silhouette is most reliable for cluster quality
    selected_k = best_silhouette['k']
    selected_result = best_silhouette

    print(f"\n✓ SELECTED: k={selected_k} (based on Silhouette score - fully unsupervised)")
    print(f"  Silhouette: {selected_result['silhouette']:.3f}")
    print(f"  CH score:   {selected_result['ch_score']:.1f}")
    print(f"  BIC:        {selected_result['bic']:.1f}")
    print(f"  ARI:        {selected_result['ari']:.3f} (for comparison with labels)")

    return selected_result

def analyze_clusters(selected_result, lithology):
    """Analyze selected clusters."""
    print("\n" + "="*80)
    print(f"CLUSTER ANALYSIS (k={selected_result['k']})")
    print("="*80)

    labels = selected_result['labels']

    print("\nHigh-purity clusters (>80%):")
    for cluster_id in range(selected_result['k']):
        cluster_mask = labels == cluster_id
        cluster_lithologies = lithology[cluster_mask]
        cluster_size = len(cluster_lithologies)

        if cluster_size > 0:
            most_common = Counter(cluster_lithologies).most_common(1)
            top_lith = most_common[0][0]
            top_pct = most_common[0][1] / cluster_size * 100

            if top_pct > 80:
                print(f"  Cluster {cluster_id:2d} (n={cluster_size:5d}): {top_lith:30s} ({top_pct:.1f}%)")

def main():
    print("="*80)
    print("VAE v2.1 - Unsupervised Model Selection")
    print("="*80)
    print("\nPrinciple: Select k WITHOUT using label information")
    print("Approach:  Use Silhouette score (cluster separation quality)\n")

    # Load model and data
    latent_vectors, lithology = load_model_and_data()

    # Evaluate many k values
    k_values = [5, 10, 15, 20, 30, 40, 50, 60]
    results = evaluate_k_values(latent_vectors, lithology, k_values)

    # Select best k using unsupervised metrics
    selected_result = select_best_k(results)

    # Analyze selected clusters
    analyze_clusters(selected_result, lithology)

    # Final summary
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print(f"\nUsing ONLY unsupervised metrics (no label information):")
    print(f"  Selected k = {selected_result['k']}")
    print(f"  Silhouette = {selected_result['silhouette']:.3f}")
    print(f"\nPerformance when compared to labels:")
    print(f"  ARI = {selected_result['ari']:.3f}")
    print(f"\nThis is a fully principled unsupervised approach.")
    print(f"We didn't 'cheat' by using k=139 (number of lithology labels).")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
