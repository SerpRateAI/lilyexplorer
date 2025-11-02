"""
PCA Analysis on VAE GRA v1 Dataset
Calculate and visualize PCA on the same data used for vae_gra_v1 model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Set style
sns.set_style('whitegrid')

# Load data
print("Loading data...")
data_path = '/home/utig5/johna/bhai/vae_training_data_20cm.csv'
df = pd.read_csv(data_path)

print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

# Extract features
feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)'
]

X = df[feature_cols].values
lithology = df['Principal'].values
borehole_ids = df['Borehole_ID'].values

# Remove NaN values
valid_mask = ~np.isnan(X).any(axis=1)
X = X[valid_mask]
lithology = lithology[valid_mask]
borehole_ids = borehole_ids[valid_mask]

print(f"Valid samples after removing NaN: {len(X):,}")

# Standardize features (same as VAE preprocessing)
print("Standardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
print("Performing PCA...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA Results:")
print(f"  PC1 variance explained: {pca.explained_variance_ratio_[0]*100:.2f}%")
print(f"  PC2 variance explained: {pca.explained_variance_ratio_[1]*100:.2f}%")
print(f"  Total variance explained: {pca.explained_variance_ratio_.sum()*100:.2f}%")

# Print component loadings
print(f"\nPrincipal Component Loadings:")
print(f"{'Feature':45s} {'PC1':>10s} {'PC2':>10s}")
print("="*67)
for i, feature in enumerate(feature_cols):
    print(f"{feature:45s} {pca.components_[0, i]:10.4f} {pca.components_[1, i]:10.4f}")

# Sample for visualization (50k points)
n_vis = min(50000, len(X_pca))
vis_idx = np.random.choice(len(X_pca), n_vis, replace=False)

X_pca_vis = X_pca[vis_idx]
lithology_vis = lithology[vis_idx]

# Get top lithologies for coloring
top_n = 10
top_lithologies = pd.Series(lithology_vis).value_counts().head(top_n).index.tolist()

# Prepare colors
palette = sns.color_palette('tab10', top_n)
palette.append((0.7, 0.7, 0.7))  # Gray for 'Other'
lithology_labels = top_lithologies + ['Other']

def get_lithology_colors(lithology_array, top_lithologies):
    """Assign colors to lithologies."""
    colors = []
    for lith in lithology_array:
        if lith in top_lithologies:
            colors.append(top_lithologies.index(lith))
        else:
            colors.append(top_n)  # 'Other' category
    return np.array(colors)

colors = get_lithology_colors(lithology_vis, top_lithologies)

# Create visualization
print("\nCreating visualization...")
fig, ax = plt.subplots(figsize=(12, 10))

# Plot each lithology
for i, (lith, color) in enumerate(zip(lithology_labels, palette)):
    mask = colors == i
    if mask.sum() > 0:
        ax.scatter(
            X_pca_vis[mask, 0],
            X_pca_vis[mask, 1],
            c=[color],
            label=lith,
            alpha=0.6,
            s=3,
            edgecolors='none'
        )

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('PCA on VAE GRA v1 Dataset (GRA + MS + NGR)', fontsize=14, fontweight='bold')
ax.legend(markerscale=3, fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = '/home/utig5/johna/bhai/pca_gra_v1.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_path}")

plt.close()

print("\nDone!")
