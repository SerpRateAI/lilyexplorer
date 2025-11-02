"""
Analyze the actual distribution of VAE v2.6 latent embeddings

This checks whether the Gaussian prior N(0, I) assumption is appropriate,
or if we should consider alternative distributions.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.stats import normaltest, shapiro, kstest

# Define model architecture
class VAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=8, hidden_dims=[32, 16]):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class DistributionAwareScaler:
    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]
        self.log_indices = [3, 4, 5]

    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit_transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        return self.scaler.fit_transform(X_transformed)

    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        return self.scaler.transform(X_transformed)

print("="*80)
print("LATENT DISTRIBUTION ANALYSIS - VAE v2.6")
print("="*80)

# Load data
print("\n1. Loading data and model...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X = df[feature_cols].values
lithology = df['Principal'].values
borehole_ids = df['Borehole_ID'].values

valid_mask = ~np.isnan(X).any(axis=1)
X = X[valid_mask]
lithology = lithology[valid_mask]
borehole_ids = borehole_ids[valid_mask]

# Split data
unique_boreholes = np.unique(borehole_ids)
train_boreholes, temp_boreholes = train_test_split(
    unique_boreholes, train_size=0.70, random_state=42
)
val_boreholes, test_boreholes = train_test_split(
    temp_boreholes, train_size=0.5, random_state=42
)

train_mask = np.isin(borehole_ids, train_boreholes)
test_mask = np.isin(borehole_ids, test_boreholes)

X_train = X[train_mask]
X_test = X[test_mask]
lithology_test = lithology[test_mask]

# Scale data
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Load model
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_5_annealing_Anneal_0.001to0.5_(50_epochs).pth',
                       weights_only=False)
model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"   Data: {len(X_test):,} test samples")

# Extract latent codes
print("\n2. Extracting latent representations...")
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_test_scaled)
    mu, logvar = model.encode(X_tensor)
    z = mu.numpy()  # Use mean (deterministic encoding)

print(f"   Latent codes shape: {z.shape}")

# Analyze distribution
print("\n" + "="*80)
print("DISTRIBUTION ANALYSIS")
print("="*80)

# Overall statistics
print("\nOverall statistics (each dimension):")
print(f"Mean: {z.mean(axis=0)}")
print(f"Std:  {z.std(axis=0)}")
print(f"\nExpected for N(0,I): Mean ≈ 0, Std ≈ 1")

# Test normality for each dimension
print("\n" + "-"*80)
print("Normality tests (per dimension):")
print("-"*80)

normality_results = []
for dim in range(8):
    # Shapiro-Wilk test (sample size sensitive, use subset)
    sample_idx = np.random.choice(len(z), min(5000, len(z)), replace=False)
    stat_sw, p_sw = shapiro(z[sample_idx, dim])

    # D'Agostino-Pearson test
    stat_dap, p_dap = normaltest(z[:, dim])

    # Kolmogorov-Smirnov test vs N(0,1)
    stat_ks, p_ks = kstest(z[:, dim], 'norm', args=(0, 1))

    is_normal = p_sw > 0.05 and p_dap > 0.05 and p_ks > 0.05

    print(f"\nDimension {dim}:")
    print(f"  Shapiro-Wilk:   stat={stat_sw:.4f}, p={p_sw:.4f} {'✓' if p_sw > 0.05 else '✗'}")
    print(f"  D'Agostino:     stat={stat_dap:.4f}, p={p_dap:.4f} {'✓' if p_dap > 0.05 else '✗'}")
    print(f"  KS vs N(0,1):   stat={stat_ks:.4f}, p={p_ks:.4f} {'✓' if p_ks > 0.05 else '✗'}")
    print(f"  Normal? {is_normal}")

    normality_results.append({
        'dim': dim,
        'mean': z[:, dim].mean(),
        'std': z[:, dim].std(),
        'p_shapiro': p_sw,
        'p_dagostino': p_dap,
        'p_ks': p_ks,
        'is_normal': is_normal
    })

# Check for multimodality
print("\n" + "-"*80)
print("Multimodality analysis:")
print("-"*80)

from scipy.stats import kurtosis, skew

for dim in range(8):
    kurt = kurtosis(z[:, dim])
    skewness = skew(z[:, dim])
    print(f"Dim {dim}: Kurtosis={kurt:6.3f}, Skewness={skewness:6.3f}", end="")
    if abs(kurt) > 1:
        print(" (heavy-tailed)" if kurt > 0 else " (light-tailed)", end="")
    if abs(skewness) > 0.5:
        print(f" (skewed)", end="")
    print()

# Check correlations
print("\n" + "-"*80)
print("Correlation analysis:")
print("-"*80)

corr_matrix = np.corrcoef(z.T)
off_diagonal = corr_matrix[np.triu_indices(8, k=1)]
max_corr = np.abs(off_diagonal).max()
mean_corr = np.abs(off_diagonal).mean()

print(f"Max correlation: {max_corr:.3f}")
print(f"Mean |correlation|: {mean_corr:.3f}")
print(f"Expected for independent N(0,I): correlations ≈ 0")

# Visualize
print("\n3. Creating visualizations...")

fig = plt.figure(figsize=(18, 12))

# Histograms + Q-Q plots for each dimension
for dim in range(8):
    # Histogram
    ax = plt.subplot(4, 8, dim + 1)
    ax.hist(z[:, dim], bins=50, density=True, alpha=0.7, color='steelblue')

    # Overlay N(0,1)
    x_range = np.linspace(z[:, dim].min(), z[:, dim].max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r--', linewidth=2, label='N(0,1)')

    ax.set_title(f'Dim {dim}', fontsize=10)
    ax.set_xlabel('Value', fontsize=8)
    if dim == 0:
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)

    # Q-Q plot
    ax = plt.subplot(4, 8, dim + 9)
    stats.probplot(z[:, dim], dist="norm", plot=ax)
    ax.set_title(f'Q-Q Dim {dim}', fontsize=10)
    ax.tick_params(labelsize=7)

# Correlation heatmap
ax = plt.subplot(4, 2, 5)
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(8))
ax.set_yticks(range(8))
ax.set_xlabel('Latent Dimension', fontsize=10)
ax.set_ylabel('Latent Dimension', fontsize=10)
ax.set_title('Latent Dimension Correlations', fontsize=12)
plt.colorbar(im, ax=ax)

# 2D projections colored by lithology
ax = plt.subplot(4, 2, 6)
top_lithologies = pd.Series(lithology_test).value_counts().head(5).index
colors_lith = []
for lith in lithology_test:
    if lith in top_lithologies:
        colors_lith.append(lith)
    else:
        colors_lith.append('Other')

for lith in list(top_lithologies) + ['Other']:
    mask = np.array(colors_lith) == lith
    ax.scatter(z[mask, 0], z[mask, 3], alpha=0.3, s=1, label=lith)
ax.set_xlabel('Latent Dim 0', fontsize=10)
ax.set_ylabel('Latent Dim 3', fontsize=10)
ax.set_title('Latent Space Clustering (Dims 0 vs 3)', fontsize=12)
ax.legend(markerscale=5, fontsize=8)
ax.grid(True, alpha=0.3)

# Summary statistics table
ax = plt.subplot(4, 2, 7)
ax.axis('off')
summary_text = "Distribution Summary:\n\n"
summary_text += f"Gaussian dimensions: {sum([r['is_normal'] for r in normality_results])}/8\n"
summary_text += f"Max correlation: {max_corr:.3f}\n"
summary_text += f"Mean |correlation|: {mean_corr:.3f}\n\n"
summary_text += "Non-Gaussian dimensions:\n"
for r in normality_results:
    if not r['is_normal']:
        summary_text += f"  Dim {r['dim']}: μ={r['mean']:.3f}, σ={r['std']:.3f}\n"

ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='top', family='monospace',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('VAE v2.6 Latent Distribution Analysis\nChecking N(0,I) Prior Assumption',
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('latent_distribution_analysis.png', dpi=300, bbox_inches='tight')
print("   Saved: latent_distribution_analysis.png")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

gaussian_dims = sum([r['is_normal'] for r in normality_results])
print(f"\n✓ Gaussian dimensions: {gaussian_dims}/8")

if gaussian_dims >= 6:
    print("\n✓ Latent distribution is approximately Gaussian")
    print("  N(0,I) prior is appropriate")
elif gaussian_dims >= 4:
    print("\n~ Latent distribution is partially Gaussian")
    print("  Consider:")
    print("  - Current model works well (ARI=0.258)")
    print("  - Could try normalizing flows for more flexible posterior")
elif max_corr > 0.5:
    print("\n✗ Strong correlations detected!")
    print("  Consider:")
    print("  - Factor VAE (encourage independence)")
    print("  - β-TCVAE (total correlation penalty)")
else:
    print("\n✗ Latent distribution is NOT Gaussian")
    print("  Alternative approaches:")
    print("  - VampPrior (mixture of posteriors)")
    print("  - Normalizing flows")
    print("  - Alternative distribution families")

print("\n" + "="*80)
