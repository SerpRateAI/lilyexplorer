"""
Check what percentage of LILY data has all four measurement types co-located
"""

import pandas as pd
from pathlib import Path

print("="*80)
print("LILY DATA COVERAGE ANALYSIS")
print("="*80)

data_dir = Path('/home/utig5/johna/bhai/datasets')

# Count total rows in each dataset
print("\n1. Total raw measurements in each dataset:")
print("-"*80)

datasets = {
    'GRA': 'GRA_DataLITH.csv',
    'MS': 'MS_DataLITH.csv',
    'NGR': 'NGR_DataLITH.csv',
    'RGB': 'RGB_DataLITH.csv'
}

row_counts = {}
for name, filename in datasets.items():
    filepath = data_dir / filename
    # Use pandas to count rows efficiently
    df = pd.read_csv(filepath, usecols=[0], nrows=None)  # Read just first column
    row_counts[name] = len(df)
    print(f"  {name:3s}: {len(df):>12,} measurements")

# Now check the VAE training dataset
print("\n2. Co-located measurements (20cm bins, all 4 types):")
print("-"*80)

vae_data = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"  VAE v2 dataset: {len(vae_data):>12,} samples")
print(f"  From {vae_data['Borehole_ID'].nunique()} boreholes")

# Calculate percentages
print("\n3. Coverage percentages:")
print("-"*80)
for name in ['GRA', 'MS', 'NGR', 'RGB']:
    if name in row_counts:
        pct = (len(vae_data) / row_counts[name]) * 100
        print(f"  {name:3s}: {pct:5.2f}% of raw measurements have all 4 types co-located")

# Check GRA-only dataset (v1, no RGB)
print("\n4. Without RGB requirement (GRA + MS + NGR only):")
print("-"*80)
try:
    vae_v1_data = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_20cm.csv')
    print(f"  VAE v1 dataset: {len(vae_v1_data):>12,} samples")
    print(f"  From {vae_v1_data['Borehole_ID'].nunique()} boreholes")

    pct_gra = (len(vae_v1_data) / row_counts['GRA']) * 100
    print(f"\n  {pct_gra:5.2f}% of GRA measurements have MS + NGR co-located")
except FileNotFoundError:
    print("  VAE v1 dataset not found")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total GRA measurements: {row_counts['GRA']:,}")
print(f"With MS + NGR + RGB (v2): {len(vae_data):,} ({len(vae_data)/row_counts['GRA']*100:.1f}%)")
if 'vae_v1_data' in locals():
    print(f"With MS + NGR only (v1): {len(vae_v1_data):,} ({len(vae_v1_data)/row_counts['GRA']*100:.1f}%)")

print("\nNote: Percentages are after 20cm depth binning (averaging)")
print("Each bin may contain multiple raw measurements")
print("="*80)
