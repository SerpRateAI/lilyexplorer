"""
Analyze overlap of continuous MSCL measurements with 50cm depth tolerance
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*80)
print("MSCL Feature Overlap Analysis (50cm tolerance)")
print("="*80)
print()

datasets_dir = Path('/home/utig5/johna/bhai/datasets')

# Define datasets to analyze
datasets = {
    'GRA': 'GRA_DataLITH.csv',
    'MS': 'MS_DataLITH.csv',
    'MSP': 'MSP_DataLITH.csv',
    'NGR': 'NGR_DataLITH.csv',
    'RGB': 'RGB_DataLITH.csv',
    'RSC': 'RSC_DataLITH.csv',
    'PWL': 'PWL_DataLITH.csv'
}

# Load data with only necessary columns
print("Loading datasets (this may take a few minutes)...")
print()

data = {}
for name, filename in datasets.items():
    print(f"Loading {name}...", end=' ', flush=True)

    df = pd.read_csv(
        datasets_dir / filename,
        usecols=['Exp', 'Site', 'Hole', 'Depth CSF-A (m)']
    )

    # Create Borehole_ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

    # Remove NaN depths
    df = df.dropna(subset=['Depth CSF-A (m)'])

    # Create 50cm depth bins (0.5m)
    df['depth_bin'] = (df['Depth CSF-A (m)'] / 0.5).round() * 0.5

    # Get unique borehole-depth combinations
    unique_bins = df[['Borehole_ID', 'depth_bin']].drop_duplicates()

    data[name] = unique_bins

    print(f"{len(df):,} measurements → {len(unique_bins):,} unique 50cm bins")

print()
print("="*80)
print("Overlap Analysis")
print("="*80)
print()

# Merge all datasets on borehole + depth bin
print("Computing overlaps...")
print()

# Start with GRA (most common)
merged = data['GRA'].copy()
merged['GRA'] = True

for name in ['MS', 'MSP', 'NGR', 'RGB', 'RSC', 'PWL']:
    df = data[name].copy()
    df[name] = True
    merged = merged.merge(df, on=['Borehole_ID', 'depth_bin'], how='outer')

# Fill NaN with False
for name in datasets.keys():
    merged[name] = merged[name].fillna(False)

print(f"Total unique 50cm bins across all boreholes: {len(merged):,}")
print()

# Compute overlap statistics
print("="*80)
print("Coverage Statistics (% of bins with each measurement)")
print("="*80)
print()

for name in datasets.keys():
    count = merged[name].sum()
    pct = 100 * count / len(merged)
    print(f"{name:6s}: {count:>10,} bins ({pct:5.1f}%)")

print()
print("="*80)
print("Pairwise Overlaps")
print("="*80)
print()

feature_list = list(datasets.keys())
for i, name1 in enumerate(feature_list):
    for name2 in feature_list[i+1:]:
        overlap = (merged[name1] & merged[name2]).sum()
        total1 = merged[name1].sum()
        total2 = merged[name2].sum()

        pct1 = 100 * overlap / total1 if total1 > 0 else 0
        pct2 = 100 * overlap / total2 if total2 > 0 else 0

        print(f"{name1:6s} ∩ {name2:6s}: {overlap:>10,} bins "
              f"({pct1:5.1f}% of {name1}, {pct2:5.1f}% of {name2})")

print()
print("="*80)
print("Multi-Feature Combinations")
print("="*80)
print()

# Check common combinations
combos = [
    (['GRA', 'MS', 'NGR'], 'Physical properties (v1 VAE)'),
    (['GRA', 'MS', 'NGR', 'RGB'], 'Physical + RGB (v2 VAE)'),
    (['GRA', 'MS', 'NGR', 'RGB', 'RSC'], 'Physical + RGB + RSC'),
    (['GRA', 'MS', 'NGR', 'RGB', 'PWL'], 'Physical + RGB + P-wave'),
    (['GRA', 'MS', 'NGR', 'RGB', 'RSC', 'PWL'], 'All 6 features'),
    (['GRA', 'MS', 'MSP', 'NGR', 'RGB', 'RSC', 'PWL'], 'All 7 features'),
]

for features, description in combos:
    mask = merged[features[0]].copy()
    for feat in features[1:]:
        mask = mask & merged[feat]

    count = mask.sum()
    pct = 100 * count / len(merged)

    # Count boreholes
    n_boreholes = merged[mask]['Borehole_ID'].nunique()

    print(f"{description}")
    print(f"  Features: {', '.join(features)}")
    print(f"  Bins: {count:,} ({pct:.1f}% of all bins)")
    print(f"  Boreholes: {n_boreholes}")
    print()

print("="*80)
print("Borehole Coverage")
print("="*80)
print()

# Count boreholes with each feature
for name in datasets.keys():
    n_boreholes = merged[merged[name]]['Borehole_ID'].nunique()
    print(f"{name:6s}: {n_boreholes:>4} boreholes")

print()

# Boreholes with all 7 features
mask_all = merged['GRA'] & merged['MS'] & merged['MSP'] & merged['NGR'] & merged['RGB'] & merged['RSC'] & merged['PWL']
boreholes_all = merged[mask_all]['Borehole_ID'].unique()
print(f"Boreholes with all 7 features: {len(boreholes_all)}")

# Boreholes with GRA+MS+NGR+RGB (v2 VAE combo)
mask_vae = merged['GRA'] & merged['MS'] & merged['NGR'] & merged['RGB']
boreholes_vae = merged[mask_vae]['Borehole_ID'].unique()
print(f"Boreholes with GRA+MS+NGR+RGB: {len(boreholes_vae)}")

print()
print("="*80)
print("Comparison to 20cm VAE Datasets")
print("="*80)
print()
print("Current VAE v2 (20cm bins, GRA+MS+NGR+RGB):")
print("  - 238,506 samples")
print("  - 296 boreholes")
print()
print(f"With 50cm bins (GRA+MS+NGR+RGB): {mask_vae.sum():,} samples from {len(boreholes_vae)} boreholes")
print(f"Difference: {mask_vae.sum() - 238506:,} samples ({100*(mask_vae.sum() - 238506)/238506:+.1f}%)")
print()

# Save overlap data for further analysis
output_file = 'mscl_overlap_50cm.csv'
merged.to_csv(output_file, index=False)
print(f"Saved detailed overlap data to: {output_file}")
