"""
Scatter plot of GRA bulk density vs MAD bulk density.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

print("Loading datasets...")
mad = pd.read_csv('datasets/MAD_DataLITH.csv', low_memory=False)
gra = pd.read_csv('datasets/GRA_DataLITH.csv', low_memory=False)

key_cols = ['Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W']

for col in key_cols:
    if col in mad.columns:
        mad[col] = mad[col].astype(str)
    if col in gra.columns:
        gra[col] = gra[col].astype(str)

mad_subset = mad[key_cols + ['Depth CSF-A (m)', 'Bulk density (g/cm^3)']].copy()
gra_subset = gra[key_cols + ['Depth CSF-A (m)', 'Bulk density (GRA)']].copy()

mad_subset = mad_subset.rename(columns={'Bulk density (g/cm^3)': 'MAD_density'})
gra_subset = gra_subset.rename(columns={'Bulk density (GRA)': 'GRA_density'})

merged = pd.merge(mad_subset, gra_subset, on=key_cols, suffixes=('_mad', '_gra'))
merged['depth_diff'] = np.abs(merged['Depth CSF-A (m)_mad'] - merged['Depth CSF-A (m)_gra'])
merged = merged[merged['depth_diff'] <= 0.05]
merged = merged.dropna(subset=['MAD_density', 'GRA_density'])

mad_vals = merged['MAD_density'].values
gra_vals = merged['GRA_density'].values

r2 = r2_score(mad_vals, gra_vals)
pearson_r, _ = pearsonr(mad_vals, gra_vals)
rmse = np.sqrt(np.mean((mad_vals - gra_vals)**2))

fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(mad_vals, gra_vals, alpha=0.3, s=10, c='blue')

min_val = min(mad_vals.min(), gra_vals.min())
max_val = max(mad_vals.max(), gra_vals.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 line')

ax.set_xlabel('MAD Bulk Density (g/cm³)', fontsize=14, fontweight='bold')
ax.set_ylabel('GRA Bulk Density (g/cm³)', fontsize=14, fontweight='bold')
ax.set_title('GRA vs MAD Bulk Density', fontsize=16, fontweight='bold')

stats_text = f'n = {len(merged):,}\nR² = {r2:.4f}\nr = {pearson_r:.4f}\nRMSE = {rmse:.4f} g/cm³'
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=12)

plt.tight_layout()
plt.savefig('gra_vs_mad_scatter.png', dpi=300, bbox_inches='tight')
print("Saved: gra_vs_mad_scatter.png")
