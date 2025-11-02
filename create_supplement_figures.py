"""
Create combined figures for supplement (max 5 figures)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load data
df = pd.read_csv('vae_training_data_v2_20cm.csv')
feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'R', 'G', 'B'
]
X = df[feature_cols].values

# Feature names for labels
feature_names = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']

print("Creating 5 combined supplement figures...")

# ============================================================================
# FIGURE S1: Raw Feature Distributions (6 histograms in 2x3 grid)
# ============================================================================
print("\nFigure S1: Raw Feature Distributions...")

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

colors = ['steelblue', 'steelblue', 'steelblue', 'red', 'green', 'blue']
alphas = [1.0, 1.0, 1.0, 0.7, 0.7, 0.7]

for i in range(6):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    ax.hist(X[:, i], bins=100, color=colors[i], edgecolor='none', alpha=alphas[i])
    ax.set_xlabel(f'{feature_names[i]}', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.text(0.02, 0.98, f'({chr(97+i)})', transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top')

plt.savefig('paper_draft/figS1_feature_distributions.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: figS1_feature_distributions.png")

# ============================================================================
# FIGURE S2: Feature Correlation Matrix (heatmap + key scatter plots)
# ============================================================================
print("\nFigure S2: Feature Correlations...")

# Calculate correlation matrix
corr = np.corrcoef(X.T)

fig = plt.figure(figsize=(14, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# Correlation matrix heatmap (spans 2x2 in top-left)
ax_heatmap = fig.add_subplot(gs[0:2, 0:2])
im = ax_heatmap.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax_heatmap.set_xticks(range(6))
ax_heatmap.set_yticks(range(6))
ax_heatmap.set_xticklabels(feature_names, rotation=45)
ax_heatmap.set_yticklabels(feature_names)
ax_heatmap.set_title('(a) Correlation Matrix', fontsize=12, fontweight='bold', loc='left')

# Add correlation values as text
for i in range(6):
    for j in range(6):
        text = ax_heatmap.text(j, i, f'{corr[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)

cbar = plt.colorbar(im, ax=ax_heatmap)
cbar.set_label('Correlation', fontsize=10)

# Sample for scatter plots
np.random.seed(42)
idx = np.random.choice(len(X), size=5000, replace=False)
X_sample = X[idx]

# Key scatter plots
scatter_configs = [
    (0, 1, 'GRA vs MS', gs[0, 2]),  # GRA vs MS
    (0, 3, 'GRA vs R', gs[1, 2]),   # GRA vs R
    (3, 4, 'R vs G', gs[2, 0]),     # R vs G
    (3, 5, 'R vs B', gs[2, 1]),     # R vs B
    (4, 5, 'G vs B', gs[2, 2]),     # G vs B
]

panel_labels = ['b', 'c', 'd', 'e', 'f']

for idx_plot, (i, j, title, grid_pos) in enumerate(scatter_configs):
    ax = fig.add_subplot(grid_pos)
    ax.scatter(X_sample[:, i], X_sample[:, j], alpha=0.3, s=0.5, color='steelblue')
    ax.set_xlabel(feature_names[i], fontsize=9)
    ax.set_ylabel(feature_names[j], fontsize=9)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.98, f'({panel_labels[idx_plot]}) r={corr[i,j]:.2f}',
            transform=ax.transAxes, fontsize=10, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.savefig('paper_draft/figS2_feature_correlations.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: figS2_feature_correlations.png")

# ============================================================================
# FIGURE S3: Validation Loss vs Clustering (use existing)
# ============================================================================
print("\nFigure S3: Using existing vae_v2_6_vs_v2_10_analysis.png")
# Already exists, just note it

# ============================================================================
# FIGURE S4: Latent Space Analysis (use existing)
# ============================================================================
print("\nFigure S4: Using existing latent_distribution_analysis.png")
# Already exists, just note it

# ============================================================================
# FIGURE S5: Model Architecture and Performance (combine 2 existing)
# ============================================================================
print("\nFigure S5: Model Architecture and Performance...")

from PIL import Image

# Load existing images
arch_img = Image.open('paper_draft/vae_v2_6_architecture_diagram.png')
perf_img = Image.open('paper_draft/vae_v2_6_performance_vs_k.png')

# Create combined figure
fig = plt.figure(figsize=(14, 6))
gs = GridSpec(1, 2, figure=fig, wspace=0.2)

# Architecture diagram
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(arch_img)
ax1.axis('off')
ax1.text(0.02, 0.98, '(a) VAE Architecture', transform=ax1.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Performance plot
ax2 = fig.add_subplot(gs[0, 1])
ax2.imshow(perf_img)
ax2.axis('off')
ax2.text(0.02, 0.98, '(b) Clustering Performance vs k', transform=ax2.transAxes,
         fontsize=12, fontweight='bold', va='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.savefig('paper_draft/figS5_architecture_performance.png', dpi=200, bbox_inches='tight')
plt.close()
print("  Saved: figS5_architecture_performance.png")

print("\nDone! Created 5 supplement figures:")
print("  - figS1_feature_distributions.png (6 histograms)")
print("  - figS2_feature_correlations.png (correlation matrix + 5 key scatters)")
print("  - vae_v2_6_vs_v2_10_analysis.png (existing, validation loss vs ARI)")
print("  - latent_distribution_analysis.png (existing, latent space analysis)")
print("  - figS5_architecture_performance.png (architecture + performance)")
