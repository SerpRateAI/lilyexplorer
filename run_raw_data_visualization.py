"""
Raw Data Visualization

Visualizing the 6 input features from VAE training dataset:
- GRA (Bulk density)
- MS (Magnetic susceptibility)
- NGR (Natural gamma radiation)
- R (Red channel)
- G (Green channel)
- B (Blue channel)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'R',
    'G',
    'B'
]

X = df[feature_cols].values

print(f"Dataset: {len(X):,} samples")
print(f"Features: {len(feature_cols)}")
print()

# ============================================================================
# HISTOGRAMS
# ============================================================================
print("Creating histograms...")

# GRA histogram
plt.figure(figsize=(8, 6))
plt.hist(X[:, 0], bins=100, color='steelblue', edgecolor='none')
plt.xlabel('Bulk density (GRA) [g/cm³]')
plt.ylabel('Frequency')
plt.savefig('hist_gra.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - hist_gra.png")

# MS histogram
plt.figure(figsize=(8, 6))
plt.hist(X[:, 1], bins=100, color='steelblue', edgecolor='none')
plt.xlabel('Magnetic susceptibility [instr. units]')
plt.ylabel('Frequency')
plt.savefig('hist_ms.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - hist_ms.png")

# NGR histogram
plt.figure(figsize=(8, 6))
plt.hist(X[:, 2], bins=100, color='steelblue', edgecolor='none')
plt.xlabel('NGR total counts [cps]')
plt.ylabel('Frequency')
plt.savefig('hist_ngr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - hist_ngr.png")

# R histogram
plt.figure(figsize=(8, 6))
plt.hist(X[:, 3], bins=100, color='red', edgecolor='none', alpha=0.7)
plt.xlabel('R [0-255]')
plt.ylabel('Frequency')
plt.savefig('hist_r.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - hist_r.png")

# G histogram
plt.figure(figsize=(8, 6))
plt.hist(X[:, 4], bins=100, color='green', edgecolor='none', alpha=0.7)
plt.xlabel('G [0-255]')
plt.ylabel('Frequency')
plt.savefig('hist_g.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - hist_g.png")

# B histogram
plt.figure(figsize=(8, 6))
plt.hist(X[:, 5], bins=100, color='blue', edgecolor='none', alpha=0.7)
plt.xlabel('B [0-255]')
plt.ylabel('Frequency')
plt.savefig('hist_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - hist_b.png")

print()

# ============================================================================
# SCATTER PLOTS
# ============================================================================
print("Creating scatter plots (using 10k samples)...")

# Sample data for scatter plots
np.random.seed(42)
idx = np.random.choice(len(X), size=10000, replace=False)
X_sample = X[idx]

# GRA vs MS
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 0], X_sample[:, 1], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Bulk density (GRA) [g/cm³]')
plt.ylabel('Magnetic susceptibility [instr. units]')
plt.savefig('scatter_gra_ms.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_gra_ms.png")

# GRA vs NGR
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 0], X_sample[:, 2], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Bulk density (GRA) [g/cm³]')
plt.ylabel('NGR total counts [cps]')
plt.savefig('scatter_gra_ngr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_gra_ngr.png")

# GRA vs R
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 0], X_sample[:, 3], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Bulk density (GRA) [g/cm³]')
plt.ylabel('R [0-255]')
plt.savefig('scatter_gra_r.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_gra_r.png")

# GRA vs G
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 0], X_sample[:, 4], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Bulk density (GRA) [g/cm³]')
plt.ylabel('G [0-255]')
plt.savefig('scatter_gra_g.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_gra_g.png")

# GRA vs B
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 0], X_sample[:, 5], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Bulk density (GRA) [g/cm³]')
plt.ylabel('B [0-255]')
plt.savefig('scatter_gra_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_gra_b.png")

# MS vs NGR
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 1], X_sample[:, 2], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Magnetic susceptibility [instr. units]')
plt.ylabel('NGR total counts [cps]')
plt.savefig('scatter_ms_ngr.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ms_ngr.png")

# MS vs R
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 1], X_sample[:, 3], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Magnetic susceptibility [instr. units]')
plt.ylabel('R [0-255]')
plt.savefig('scatter_ms_r.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ms_r.png")

# MS vs G
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 1], X_sample[:, 4], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Magnetic susceptibility [instr. units]')
plt.ylabel('G [0-255]')
plt.savefig('scatter_ms_g.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ms_g.png")

# MS vs B
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 1], X_sample[:, 5], alpha=0.3, s=1, color='steelblue')
plt.xlabel('Magnetic susceptibility [instr. units]')
plt.ylabel('B [0-255]')
plt.savefig('scatter_ms_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ms_b.png")

# NGR vs R
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 2], X_sample[:, 3], alpha=0.3, s=1, color='steelblue')
plt.xlabel('NGR total counts [cps]')
plt.ylabel('R [0-255]')
plt.savefig('scatter_ngr_r.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ngr_r.png")

# NGR vs G
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 2], X_sample[:, 4], alpha=0.3, s=1, color='steelblue')
plt.xlabel('NGR total counts [cps]')
plt.ylabel('G [0-255]')
plt.savefig('scatter_ngr_g.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ngr_g.png")

# NGR vs B
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 2], X_sample[:, 5], alpha=0.3, s=1, color='steelblue')
plt.xlabel('NGR total counts [cps]')
plt.ylabel('B [0-255]')
plt.savefig('scatter_ngr_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_ngr_b.png")

# R vs G
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 3], X_sample[:, 4], alpha=0.3, s=1, color='steelblue')
plt.xlabel('R [0-255]')
plt.ylabel('G [0-255]')
plt.savefig('scatter_r_g.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_r_g.png")

# R vs B
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 3], X_sample[:, 5], alpha=0.3, s=1, color='steelblue')
plt.xlabel('R [0-255]')
plt.ylabel('B [0-255]')
plt.savefig('scatter_r_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_r_b.png")

# G vs B
plt.figure(figsize=(8, 6))
plt.scatter(X_sample[:, 4], X_sample[:, 5], alpha=0.3, s=1, color='steelblue')
plt.xlabel('G [0-255]')
plt.ylabel('B [0-255]')
plt.savefig('scatter_g_b.png', dpi=150, bbox_inches='tight')
plt.close()
print("  - scatter_g_b.png")

print()
print("Done! Generated 21 plots:")
print("  - 6 histograms")
print("  - 15 scatter plots")
