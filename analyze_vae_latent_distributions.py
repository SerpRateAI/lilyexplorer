"""
VAE Latent Space Distribution Analysis

Analyzes the distributional properties of the 8D latent space from VAE GRA v2.6:
1. Q-Q plots for all 8 latent dimensions
2. Formal normality tests
3. 2D density visualizations
4. Cluster shape analysis
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, kstest, anderson, normaltest
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

print("="*100)
print("VAE LATENT SPACE DISTRIBUTION ANALYSIS")
print("="*100)
print()

# ============================================================================
# 1. Load VAE Model and Extract Latent Representations
# ============================================================================

print("Loading data...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

# Split by borehole
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

test_mask = df['Borehole_ID'].isin(test_boreholes)
train_mask = df['Borehole_ID'].isin(train_boreholes)

df_test = df[test_mask].copy()
df_train = df[train_mask].copy()

print(f"Train: {len(df_train):,} samples")
print(f"Test:  {len(df_test):,} samples")
print()

# Prepare features
X_train = df_train[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

# Scale
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load trained VAE model
print("Loading VAE model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
checkpoint = torch.load('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_hdbscan_comparison.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded: {model.latent_dim}D latent space")
print()

# Extract latent representations
print("Extracting latent representations...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test_scaled).to(device)
    mu, logvar = model.encode(X_tensor)
    latent = mu.cpu().numpy()

print(f"Latent shape: {latent.shape}")
print(f"Latent range: [{latent.min():.2f}, {latent.max():.2f}]")
print()

# ============================================================================
# 2. Q-Q Plots for All 8 Latent Dimensions
# ============================================================================

print("Generating Q-Q plots...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i in range(8):
    ax = axes[i]

    # Q-Q plot
    stats.probplot(latent[:, i], dist="norm", plot=ax)

    # Shapiro-Wilk test (sample for speed if needed)
    sample_size = min(5000, len(latent))
    sample_idx = np.random.choice(len(latent), sample_size, replace=False)
    stat, p_value = shapiro(latent[sample_idx, i])

    # Mean and std
    mean_val = latent[:, i].mean()
    std_val = latent[:, i].std()

    ax.set_title(f'Dimension {i+1}\nμ={mean_val:.3f}, σ={std_val:.3f}\nShapiro p={p_value:.2e}',
                 fontsize=11)
    ax.grid(True, alpha=0.3)

    # Color code based on normality
    if p_value > 0.05:
        ax.get_lines()[0].set_color('green')
        ax.get_lines()[0].set_alpha(0.6)
    else:
        ax.get_lines()[0].set_color('red')
        ax.get_lines()[0].set_alpha(0.6)

plt.suptitle('Q-Q Plots: Latent Dimensions vs Normal Distribution\n(Green=Gaussian p>0.05, Red=Non-Gaussian p<0.05)',
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('vae_latent_qq_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("Q-Q plots saved to: vae_latent_qq_plots.png")
print()

# ============================================================================
# 3. Formal Normality Tests
# ============================================================================

print("Running formal normality tests...")

normality_results = []

for i in range(8):
    dim_data = latent[:, i]

    # Sample for tests (some have sample size limits)
    sample_size = min(5000, len(dim_data))
    sample_idx = np.random.choice(len(dim_data), sample_size, replace=False)
    sample_data = dim_data[sample_idx]

    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = shapiro(sample_data)

    # Kolmogorov-Smirnov test (compare to normal with same mean/std)
    ks_stat, ks_p = kstest(sample_data, 'norm', args=(dim_data.mean(), dim_data.std()))

    # Anderson-Darling test
    anderson_result = anderson(sample_data, dist='norm')
    anderson_stat = anderson_result.statistic
    # Check at 5% significance level (index 2)
    anderson_reject = anderson_stat > anderson_result.critical_values[2]

    # D'Agostino-Pearson test
    dagostino_stat, dagostino_p = normaltest(sample_data)

    normality_results.append({
        'Dimension': i+1,
        'Mean': dim_data.mean(),
        'Std': dim_data.std(),
        'Skewness': stats.skew(dim_data),
        'Kurtosis': stats.kurtosis(dim_data),
        'Shapiro_p': shapiro_p,
        'KS_p': ks_p,
        'Anderson_stat': anderson_stat,
        'Anderson_reject': anderson_reject,
        'DAgostino_p': dagostino_p,
        'Is_Gaussian_05': shapiro_p > 0.05 and ks_p > 0.05 and not anderson_reject
    })

df_normality = pd.DataFrame(normality_results)

print("="*100)
print("NORMALITY TEST RESULTS (α=0.05)")
print("="*100)
print(df_normality.to_string(index=False))
print()
print(f"Gaussian dimensions (all tests pass): {df_normality['Is_Gaussian_05'].sum()}/8")
print(f"Non-Gaussian dimensions: {(~df_normality['Is_Gaussian_05']).sum()}/8")
print()

# Save
df_normality.to_csv('vae_latent_normality_tests.csv', index=False)
print("Results saved to: vae_latent_normality_tests.csv")
print()

# ============================================================================
# 4. Distribution Histograms with Normal Overlay
# ============================================================================

print("Generating distribution histograms...")

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i in range(8):
    ax = axes[i]
    dim_data = latent[:, i]

    # Histogram
    n, bins, patches = ax.hist(dim_data, bins=50, density=True, alpha=0.6,
                                 color='blue', edgecolor='black', linewidth=0.5)

    # Fit normal distribution
    mu, sigma = dim_data.mean(), dim_data.std()
    x = np.linspace(dim_data.min(), dim_data.max(), 100)
    normal_pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, normal_pdf, 'r-', linewidth=2, label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')

    # Kernel Density Estimate
    kde = stats.gaussian_kde(dim_data)
    kde_pdf = kde(x)
    ax.plot(x, kde_pdf, 'g--', linewidth=2, label='KDE (actual)')

    ax.set_xlabel(f'Latent Dimension {i+1}')
    ax.set_ylabel('Density')
    ax.set_title(f'Dim {i+1}: Skew={stats.skew(dim_data):.2f}, Kurt={stats.kurtosis(dim_data):.2f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Latent Dimension Distributions\n(Red=Fitted Normal, Green=Actual KDE, Blue=Histogram)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('vae_latent_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("Distribution plots saved to: vae_latent_distributions.png")
print()

# ============================================================================
# 5. 2D Projections: Density Visualization
# ============================================================================

print("Computing PCA projection...")
pca = PCA(n_components=2)
latent_2d = pca.fit_transform(latent)

print(f"PCA explained variance: {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%} = {pca.explained_variance_ratio_.sum():.1%}")
print()

print("Generating 2D density plots...")

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Scatter with density contours
ax = axes[0]

# Hexbin for density
hb = ax.hexbin(latent_2d[:, 0], latent_2d[:, 1], gridsize=50, cmap='viridis',
               mincnt=1, alpha=0.8)
plt.colorbar(hb, ax=ax, label='Point Density')

# KDE contours
xx, yy = np.meshgrid(
    np.linspace(latent_2d[:, 0].min(), latent_2d[:, 0].max(), 100),
    np.linspace(latent_2d[:, 1].min(), latent_2d[:, 1].max(), 100)
)
positions = np.vstack([xx.ravel(), yy.ravel()]).T

# Fit KDE
kde = stats.gaussian_kde(latent_2d.T)
density = kde(positions.T).reshape(xx.shape)

# Plot contours
ax.contour(xx, yy, density, levels=10, colors='white', alpha=0.4, linewidths=1)

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title('Latent Space Density (PCA projection)\nHexbin + KDE Contours', fontsize=13)
ax.grid(True, alpha=0.3)

# Plot 2: Top lithologies colored
ax = axes[1]

# Get top N lithologies
top_n = 10
top_lithologies = pd.Series(y_test).value_counts().head(top_n).index.tolist()

# Create color map
colors = plt.cm.tab10(np.linspace(0, 1, top_n))
color_map = {lith: colors[i] for i, lith in enumerate(top_lithologies)}

# Plot background (other lithologies)
mask_other = ~pd.Series(y_test).isin(top_lithologies)
ax.scatter(latent_2d[mask_other, 0], latent_2d[mask_other, 1],
           c='lightgray', alpha=0.3, s=5, label='Other')

# Plot top lithologies
for lith in top_lithologies:
    mask = y_test == lith
    ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
               c=[color_map[lith]], alpha=0.6, s=10, label=lith)

ax.set_xlabel('PC1', fontsize=12)
ax.set_ylabel('PC2', fontsize=12)
ax.set_title(f'Latent Space by Lithology\nTop {top_n} lithologies colored', fontsize=13)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('vae_latent_2d_density.png', dpi=300, bbox_inches='tight')
plt.close()

print("2D density plots saved to: vae_latent_2d_density.png")
print()

# ============================================================================
# 6. Correlation Analysis
# ============================================================================

print("Analyzing correlations between latent dimensions...")

# Compute correlation matrix
corr_matrix = np.corrcoef(latent.T)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(im, ax=ax, label='Correlation')

# Labels
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xticklabels([f'Dim {i+1}' for i in range(8)])
ax.set_yticklabels([f'Dim {i+1}' for i in range(8)])

# Annotate values
for i in range(8):
    for j in range(8):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

ax.set_title('Latent Dimension Correlations\n(VAE assumes independence, but...)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('vae_latent_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# Print max correlation
corr_matrix_offdiag = corr_matrix.copy()
np.fill_diagonal(corr_matrix_offdiag, 0)  # Ignore self-correlation
max_corr = np.abs(corr_matrix_offdiag).max()
max_idx = np.unravel_index(np.abs(corr_matrix_offdiag).argmax(), corr_matrix_offdiag.shape)
print(f"Max correlation: {corr_matrix_offdiag[max_idx]:.3f} between Dim {max_idx[0]+1} and Dim {max_idx[1]+1}")
print(f"Mean absolute correlation: {np.abs(corr_matrix_offdiag).mean():.3f}")
print()
print("Correlation matrix saved to: vae_latent_correlations.png")
print()

# ============================================================================
# 7. Summary Statistics
# ============================================================================

print("="*100)
print("LATENT SPACE SUMMARY")
print("="*100)
print(f"\nShape: {latent.shape}")
print(f"Device used for encoding: {device}")
print()

summary_stats = []
for i in range(8):
    dim_data = latent[:, i]
    summary_stats.append({
        'Dimension': i+1,
        'Mean': dim_data.mean(),
        'Std': dim_data.std(),
        'Min': dim_data.min(),
        'Max': dim_data.max(),
        'Range': dim_data.max() - dim_data.min(),
        'Skewness': stats.skew(dim_data),
        'Kurtosis': stats.kurtosis(dim_data)
    })

df_summary = pd.DataFrame(summary_stats)
print(df_summary.to_string(index=False))
print()

# Check for posterior collapse
collapsed_dims = df_summary[df_summary['Std'] < 0.1]['Dimension'].tolist()
if collapsed_dims:
    print(f"⚠️  POSTERIOR COLLAPSE detected in dimensions: {collapsed_dims}")
    print(f"   (std < 0.1, should be ~1 for N(0,I) prior)")
else:
    print("✓ No posterior collapse (all dimensions have std >= 0.1)")

print()
print("="*100)
print("KEY FINDINGS")
print("="*100)
print(f"1. Gaussian dimensions: {df_normality['Is_Gaussian_05'].sum()}/8")
print(f"2. Max dimension correlation: {max_corr:.3f}")
print(f"3. Posterior collapse: {len(collapsed_dims)} dimensions")
print(f"4. PCA variance explained (PC1+PC2): {pca.explained_variance_ratio_.sum():.1%}")
print()
print("INTERPRETATION:")
if df_normality['Is_Gaussian_05'].sum() == 0:
    print("  - ALL dimensions are NON-GAUSSIAN")
    print("  - VAE N(0,I) prior is violated")
    print("  - But clustering still works if structure is compact")
if max_corr > 0.5:
    print(f"  - Strong correlations (max {max_corr:.2f}) indicate dimensions are NOT independent")
    print("  - Disentanglement failed, but this may help clustering (preserves feature correlations)")
if len(collapsed_dims) > 0:
    print(f"  - {len(collapsed_dims)} dimensions collapsed → effective dimensionality < 8")
print("="*100)
print()

# ============================================================================
# 8. Cluster Shape Analysis
# ============================================================================

print("="*100)
print("CLUSTER SHAPE ANALYSIS (Top 5 Lithologies)")
print("="*100)
print()

# Analyze top lithologies
top_lithologies = pd.Series(y_test).value_counts().head(5).index.tolist()

for lith in top_lithologies:
    mask = y_test == lith
    lith_latent = latent[mask]

    print(f"\n{lith} (n={mask.sum()}):")
    print("-" * 80)

    # Compute covariance matrix
    cov = np.cov(lith_latent.T)

    # Eigenvalues indicate cluster shape
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[::-1]  # Sort descending

    # Sphericity: ratio of largest to smallest eigenvalue
    sphericity = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 0 else 0

    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Sphericity ratio (small/large): {sphericity:.3f}")
    if sphericity > 0.5:
        print(f"  Shape: SPHERICAL (eigenvalues similar)")
    elif sphericity > 0.1:
        print(f"  Shape: ELLIPTICAL (moderate elongation)")
    else:
        print(f"  Shape: HIGHLY ELONGATED (cigar-shaped)")

    # Volume (determinant)
    det_cov = np.linalg.det(cov)
    print(f"  Volume (det(Σ)): {det_cov:.2e}")

    # Trace (total variance)
    trace_cov = np.trace(cov)
    print(f"  Total variance (tr(Σ)): {trace_cov:.2f}")

print()
print("="*100)
print()

print("ANALYSIS COMPLETE!")
print()
print("Generated files:")
print("  - vae_latent_qq_plots.png")
print("  - vae_latent_normality_tests.csv")
print("  - vae_latent_distributions.png")
print("  - vae_latent_2d_density.png")
print("  - vae_latent_correlations.png")
