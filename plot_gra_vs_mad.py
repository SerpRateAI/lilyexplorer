"""
Plot GRA bulk density vs MAD bulk density to assess correlation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

print("Loading datasets...")
print("  Loading MAD (reference bulk density - discrete samples)...")
mad = pd.read_csv('datasets/MAD_DataLITH.csv', low_memory=False)
print(f"    MAD shape: {mad.shape}")

print("  Loading GRA (continuous bulk density - 4M measurements)...")
gra = pd.read_csv('datasets/GRA_DataLITH.csv', low_memory=False)
print(f"    GRA shape: {gra.shape}")

# Merge on core identification and depth
print("\nMerging datasets on core ID and depth...")
key_cols = ['Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W']

# Convert key columns to string for consistent merging
for col in key_cols:
    if col in mad.columns:
        mad[col] = mad[col].astype(str)
    if col in gra.columns:
        gra[col] = gra[col].astype(str)

# Select only needed columns
mad_subset = mad[key_cols + ['Depth CSF-A (m)', 'Bulk density (g/cm^3)']].copy()
gra_subset = gra[key_cols + ['Depth CSF-A (m)', 'Bulk density (GRA)']].copy()

# Rename for clarity
mad_subset = mad_subset.rename(columns={'Bulk density (g/cm^3)': 'MAD_density'})
gra_subset = gra_subset.rename(columns={'Bulk density (GRA)': 'GRA_density'})

# Merge on core ID and similar depth (within 0.05m)
print("  Merging on core section...")
merged = pd.merge(mad_subset, gra_subset, on=key_cols, suffixes=('_mad', '_gra'))

# Filter to measurements within 5cm depth
print("  Filtering to depth matches within 5cm...")
merged['depth_diff'] = np.abs(merged['Depth CSF-A (m)_mad'] - merged['Depth CSF-A (m)_gra'])
merged = merged[merged['depth_diff'] <= 0.05]

# Remove NaN
merged = merged.dropna(subset=['MAD_density', 'GRA_density'])

print(f"\nMatched measurements: {len(merged):,}")

# Calculate statistics
mad_vals = merged['MAD_density'].values
gra_vals = merged['GRA_density'].values

r2 = r2_score(mad_vals, gra_vals)
pearson_r, p_value = pearsonr(mad_vals, gra_vals)
rmse = np.sqrt(np.mean((mad_vals - gra_vals)**2))
mae = np.mean(np.abs(mad_vals - gra_vals))

print(f"\nStatistics:")
print(f"  R² = {r2:.4f}")
print(f"  Pearson r = {pearson_r:.4f} (p < {p_value:.2e})")
print(f"  RMSE = {rmse:.4f} g/cm³")
print(f"  MAE = {mae:.4f} g/cm³")
print(f"  MAD range: {mad_vals.min():.3f} - {mad_vals.max():.3f} g/cm³")
print(f"  GRA range: {gra_vals.min():.3f} - {gra_vals.max():.3f} g/cm³")

# Create plot
print("\nCreating plot...")
fig, ax = plt.subplots(figsize=(10, 10))

# Hexbin for density (too many points for scatter)
hb = ax.hexbin(mad_vals, gra_vals, gridsize=50, cmap='viridis',
               mincnt=1, bins='log', alpha=0.8)

# 1:1 line
min_val = min(mad_vals.min(), gra_vals.min())
max_val = max(mad_vals.max(), gra_vals.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
        label='1:1 line', alpha=0.7)

# Labels and title
ax.set_xlabel('MAD Bulk Density (g/cm³)', fontsize=14, fontweight='bold')
ax.set_ylabel('GRA Bulk Density (g/cm³)', fontsize=14, fontweight='bold')
ax.set_title('GRA vs MAD Bulk Density Comparison', fontsize=16, fontweight='bold', pad=20)

# Add statistics text
stats_text = f'n = {len(merged):,}\nR² = {r2:.4f}\nPearson r = {pearson_r:.4f}\nRMSE = {rmse:.4f} g/cm³'
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Colorbar
cbar = plt.colorbar(hb, ax=ax)
cbar.set_label('Count (log scale)', fontsize=12)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Equal aspect ratio
ax.set_aspect('equal', adjustable='box')

# Legend
ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('gra_vs_mad_density.png', dpi=300, bbox_inches='tight')
print("Plot saved to: gra_vs_mad_density.png")

plt.close()
print("Done!")
