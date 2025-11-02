"""
Analyze latent space distributions from VAE v2.6

Question: VAE assumes z ~ N(0,I), but do the learned latent dimensions
actually follow Gaussian distributions? If not, do the deviations
(e.g., bimodal peaks, heavy tails) correspond to meaningful geological clusters?
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler

print("="*80)
print("VAE Latent Space Distribution Analysis")
print("="*80)
print()

# Load v2.6 model
print("Loading VAE v2.6 model...")
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_5_annealing_Anneal_0.001to0.5_(50_epochs).pth',
                        map_location='cpu')
model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded")
print()

# Load data
print("Loading data...")
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
print()

# Scale data
print("Scaling data...")
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X)
print()

# Extract latent representations
print("Extracting latent representations...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    mu, logvar = model.encode(X_tensor)
    latent = mu.numpy()

print(f"Latent shape: {latent.shape}")
print()

# Analyze each latent dimension
print("="*80)
print("Latent Dimension Statistics")
print("="*80)
print()

for i in range(8):
    z_i = latent[:, i]

    # Basic statistics
    mean_z = np.mean(z_i)
    std_z = np.std(z_i)
    skew = stats.skew(z_i)
    kurt = stats.kurtosis(z_i)

    # Test for normality
    _, p_value = stats.normaltest(z_i)

    print(f"Latent Dimension {i}:")
    print(f"  Mean: {mean_z:.4f}, Std: {std_z:.4f}")
    print(f"  Skewness: {skew:.4f}, Kurtosis: {kurt:.4f}")
    print(f"  Normality test p-value: {p_value:.6f} {'(REJECT Gaussian!)' if p_value < 0.001 else '(accept Gaussian)'}")
    print()

# Create histogram plots
print("Creating histogram plots...")
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

for i in range(8):
    ax = axes[i]
    z_i = latent[:, i]

    # Plot histogram
    n, bins, patches = ax.hist(z_i, bins=100, alpha=0.7, color='steelblue', edgecolor='none', density=True)

    # Overlay Gaussian fit
    mu_fit, std_fit = z_i.mean(), z_i.std()
    x = np.linspace(z_i.min(), z_i.max(), 100)
    gaussian = stats.norm.pdf(x, mu_fit, std_fit)
    ax.plot(x, gaussian, 'r--', linewidth=2, label=f'N({mu_fit:.2f}, {std_fit:.2f}²)')

    # Mark mean and ±2σ
    ax.axvline(mu_fit, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(mu_fit - 2*std_fit, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax.axvline(mu_fit + 2*std_fit, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel(f'z_{i}')
    ax.set_ylabel('Density')
    ax.set_title(f'Latent Dimension {i}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('latent_distribution_histograms.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: latent_distribution_histograms.png")
print()

# Identify outliers and their lithology
print("="*80)
print("Analyzing Outliers in Each Latent Dimension")
print("="*80)
print()

outlier_analysis = []

for i in range(8):
    z_i = latent[:, i]
    mean_z = np.mean(z_i)
    std_z = np.std(z_i)

    # Define outliers as >2.5σ from mean
    outliers_low = z_i < (mean_z - 2.5*std_z)
    outliers_high = z_i > (mean_z + 2.5*std_z)

    n_low = outliers_low.sum()
    n_high = outliers_high.sum()

    print(f"Latent Dimension {i}:")
    print(f"  Low outliers (<-2.5σ): {n_low} samples ({100*n_low/len(z_i):.2f}%)")

    if n_low > 10:
        top_lithos = pd.Series(lithology[outliers_low]).value_counts().head(5)
        for lith, count in top_lithos.items():
            pct = 100*count/n_low
            print(f"    {lith:30s}: {count:5d} ({pct:5.1f}%)")

    print(f"  High outliers (>+2.5σ): {n_high} samples ({100*n_high/len(z_i):.2f}%)")

    if n_high > 10:
        top_lithos = pd.Series(lithology[outliers_high]).value_counts().head(5)
        for lith, count in top_lithos.items():
            pct = 100*count/n_high
            print(f"    {lith:30s}: {count:5d} ({pct:5.1f}%)")

    print()

    outlier_analysis.append({
        'dimension': i,
        'n_low': n_low,
        'n_high': n_high,
        'pct_outliers': 100*(n_low + n_high)/len(z_i)
    })

# Look for bimodal distributions
print("="*80)
print("Detecting Bimodal Distributions")
print("="*80)
print()

for i in range(8):
    z_i = latent[:, i]

    # Use Hartigan's dip test for unimodality
    from scipy.stats import mode as scipy_mode

    # Simple bimodality check: look for valley in histogram
    hist, bin_edges = np.histogram(z_i, bins=50)

    # Find local minima in histogram
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(hist, prominence=hist.max()*0.1)
    valleys, _ = find_peaks(-hist)

    if len(peaks) >= 2 and len(valleys) >= 1:
        print(f"Latent Dimension {i}: BIMODAL CANDIDATE ({len(peaks)} peaks)")

        # Find samples in each mode
        valley_idx = valleys[0]  # First valley
        valley_value = bin_edges[valley_idx]

        mode1 = z_i < valley_value
        mode2 = z_i >= valley_value

        print(f"  Mode 1 (z < {valley_value:.2f}): {mode1.sum()} samples")
        top_lithos = pd.Series(lithology[mode1]).value_counts().head(3)
        for lith, count in top_lithos.items():
            pct = 100*count/mode1.sum()
            print(f"    {lith:30s}: {count:5d} ({pct:5.1f}%)")

        print(f"  Mode 2 (z >= {valley_value:.2f}): {mode2.sum()} samples")
        top_lithos = pd.Series(lithology[mode2]).value_counts().head(3)
        for lith, count in top_lithos.items():
            pct = 100*count/mode2.sum()
            print(f"    {lith:30s}: {count:5d} ({pct:5.1f}%)")

        print()

# Compare raw features for outliers vs inliers
print("="*80)
print("Raw Feature Analysis for Outliers")
print("="*80)
print()

for i in [0, 1, 2]:  # Analyze first 3 dimensions
    z_i = latent[:, i]
    mean_z = np.mean(z_i)
    std_z = np.std(z_i)

    # Outliers vs inliers
    outliers = (z_i < mean_z - 2.5*std_z) | (z_i > mean_z + 2.5*std_z)
    inliers = ~outliers

    print(f"Latent Dimension {i}:")
    print(f"  Outliers: {outliers.sum()} samples")
    print(f"  Inliers: {inliers.sum()} samples")
    print()

    # Compare raw features
    print("  Raw feature means:")
    print(f"    {'Feature':<40s} {'Outliers':>12s} {'Inliers':>12s} {'Diff':>10s}")
    print("    " + "-"*76)

    for j, feat in enumerate(feature_cols):
        mean_out = X[outliers, j].mean()
        mean_in = X[inliers, j].mean()
        diff_pct = 100*(mean_out - mean_in)/mean_in if mean_in != 0 else 0

        print(f"    {feat:<40s} {mean_out:>12.3f} {mean_in:>12.3f} {diff_pct:>9.1f}%")

    print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("If latent dimensions show:")
print("  1. Non-Gaussian distributions (p < 0.001 in normality test)")
print("  2. Bimodal or multimodal structure")
print("  3. Outliers that cluster by lithology")
print()
print("Then the VAE is capturing real geological structure that doesn't")
print("fit the N(0,I) prior assumption. This is actually GOOD - it means")
print("the model is finding meaningful patterns in the data despite the")
print("Gaussian prior constraint.")
