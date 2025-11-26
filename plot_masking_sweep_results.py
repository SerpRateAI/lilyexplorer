#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot Masking Sweep Results
2x3 subplot showing R² vs masking percentage for each feature
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read results
df = pd.read_csv('masking_sweep_results.csv')
df = df.sort_values('mask_prob')

# Feature names
features = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']
r2_cols = ['r2_gra', 'r2_ms', 'r2_ngr', 'r2_r', 'r2_g', 'r2_b']

# Create 2x3 subplot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (feature, r2_col) in enumerate(zip(features, r2_cols)):
    ax = axes[i]

    # Plot R² vs masking percentage
    ax.plot(df['mask_prob'] * 100, df[r2_col], 'b-', linewidth=2)

    # Add horizontal line at R²=0
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Styling
    ax.set_xlabel('Masking Percentage (%)', fontsize=12)
    ax.set_ylabel('Reconstruction R²', fontsize=12)
    ax.set_title(feature, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)

    # Find optimal masking percentage (max R²)
    max_idx = df[r2_col].idxmax()
    max_r2 = df.loc[max_idx, r2_col]
    max_mask = df.loc[max_idx, 'mask_prob'] * 100

    # Mark optimal point
    ax.plot(max_mask, max_r2, 'ro', markersize=8, zorder=5)
    ax.text(max_mask, max_r2, f'  {max_mask:.0f}%\n  R²={max_r2:.3f}',
            fontsize=10, ha='left', va='bottom')

# Adjust layout
plt.tight_layout()
plt.savefig('masking_sweep_results.png', dpi=300, bbox_inches='tight')
print("Plot saved to masking_sweep_results.png")

# Print summary statistics
print("\nOptimal Masking Percentages:")
print("=" * 60)
print(f"{'Feature':<10} {'Optimal %':<12} {'Max R²':<12} {'R² at 0%':<12} {'Improvement':<12}")
print("=" * 60)

for feature, r2_col in zip(features, r2_cols):
    max_idx = df[r2_col].idxmax()
    max_r2 = df.loc[max_idx, r2_col]
    max_mask = df.loc[max_idx, 'mask_prob'] * 100

    # R² at 0% masking
    r2_at_0 = df.loc[df['mask_prob'] == 0.0, r2_col].values[0]
    improvement = ((max_r2 - r2_at_0) / abs(r2_at_0) * 100) if r2_at_0 != 0 else float('inf')

    print(f"{feature:<10} {max_mask:>8.0f}%    {max_r2:>8.4f}    {r2_at_0:>8.4f}    {improvement:>8.1f}%")

print("=" * 60)
