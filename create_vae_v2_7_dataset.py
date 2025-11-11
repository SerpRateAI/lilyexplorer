"""
Create VAE v2.7 training dataset with MAD features added to v2.6.

Strategy: Conservative feature enrichment
- Start with v2.6 boreholes (GRA+MS+NGR+RGB): 296 boreholes
- Add MAD measurements (porosity, grain density, water content): 197 overlap
- 8 features total: GRA, MS, NGR, RGB, Porosity, Grain Density, Water Content
- Expected: 197 boreholes (-33% coverage, but +3 discriminative features)

Goal: Test if porosity/grain density improve classification beyond 42% ceiling.
"""

import pandas as pd
import numpy as np

print("="*100)
print("VAE v2.7 DATASET CREATION: v2.6 + MAD FEATURES")
print("="*100)
print()

# Load v2.6 dataset (GRA+MS+NGR+RGB)
print("Loading v2.6 dataset (GRA+MS+NGR+RGB)...")
v2_6_df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"v2.6 dataset: {len(v2_6_df):,} samples, {v2_6_df['Borehole_ID'].nunique()} boreholes")
print()

# Load MAD dataset
print("Loading MAD dataset...")
mad_df = pd.read_csv('/home/utig5/johna/bhai/datasets/MAD_DataLITH.csv')
print(f"MAD dataset: {len(mad_df):,} measurements")
print()

# Create Borehole_ID for MAD (use dash separator to match v2.6 format)
mad_df['Borehole_ID'] = mad_df['Exp'].astype(str) + '-' + mad_df['Site'].astype(str) + '-' + mad_df['Hole'].astype(str)

# Convert depth to float
mad_df['Depth CSF-A (m)'] = pd.to_numeric(mad_df['Depth CSF-A (m)'], errors='coerce')

# Create 20cm depth bins for MAD (same as v2.6)
mad_df['depth_bin'] = (mad_df['Depth CSF-A (m)'] / 0.2).round() * 0.2

print("MAD features available:")
mad_features = ['Porosity (vol%)', 'Grain density (g/cm^3)', 'Bulk density (g/cm^3)',
                'Dry density (g/cm^3)', 'Moisture wet (wt%)']
for feat in mad_features:
    if feat in mad_df.columns:
        non_null = mad_df[feat].notna().sum()
        print(f"  {feat:30s}: {non_null:>8,} measurements")
print()

# Aggregate MAD by depth bin (mean)
print("Aggregating MAD measurements by 20cm depth bins...")
mad_agg = mad_df.groupby(['Borehole_ID', 'depth_bin']).agg({
    'Porosity (vol%)': 'mean',
    'Grain density (g/cm^3)': 'mean',
    'Moisture wet (wt%)': 'mean'
}).reset_index()

print(f"MAD aggregated: {len(mad_agg):,} depth bins")
print()

# Rename Depth_Bin to depth_bin for merge
v2_6_df['depth_bin'] = v2_6_df['Depth_Bin']

# Merge with v2.6 dataset
print("Merging MAD features with v2.6 dataset...")
v2_7_df = v2_6_df.merge(
    mad_agg,
    on=['Borehole_ID', 'depth_bin'],
    how='inner'  # Only keep samples with BOTH v2.6 AND MAD
)

print(f"v2.7 dataset after merge: {len(v2_7_df):,} samples")
print(f"Boreholes: {v2_7_df['Borehole_ID'].nunique()}")
print()

# Check for missing values in new features
print("Missing values in MAD features:")
for feat in ['Porosity (vol%)', 'Grain density (g/cm^3)', 'Moisture wet (wt%)']:
    missing = v2_7_df[feat].isna().sum()
    pct = (missing / len(v2_7_df)) * 100
    print(f"  {feat:30s}: {missing:>6,} ({pct:5.2f}%)")
print()

# Drop rows with any missing MAD values
v2_7_df = v2_7_df.dropna(subset=['Porosity (vol%)', 'Grain density (g/cm^3)', 'Moisture wet (wt%)'])
print(f"After removing missing MAD values: {len(v2_7_df):,} samples")
print()

# Rename columns for consistency
v2_7_df = v2_7_df.rename(columns={
    'Bulk density (GRA)': 'GRA_bulk_density',
    'Magnetic susceptibility (instr. units)': 'MS',
    'NGR total counts (cps)': 'NGR',
    'R': 'RGB_R',
    'G': 'RGB_G',
    'B': 'RGB_B',
    'Porosity (vol%)': 'Porosity',
    'Grain density (g/cm^3)': 'Grain_Density',
    'Moisture wet (wt%)': 'Water_Content'
})

# Final feature list
features_v2_7 = ['GRA_bulk_density', 'MS', 'NGR',
                 'RGB_R', 'RGB_G', 'RGB_B',
                 'Porosity', 'Grain_Density', 'Water_Content']

print("="*100)
print("FINAL v2.7 DATASET STATISTICS")
print("="*100)
print()

print(f"Total samples: {len(v2_7_df):,}")
print(f"Total boreholes: {v2_7_df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {v2_7_df['Principal'].nunique()}")
print()

print("Feature statistics:")
for feat in features_v2_7:
    mean_val = v2_7_df[feat].mean()
    std_val = v2_7_df[feat].std()
    min_val = v2_7_df[feat].min()
    max_val = v2_7_df[feat].max()
    print(f"  {feat:20s}: mean={mean_val:8.3f}, std={std_val:8.3f}, range=[{min_val:8.3f}, {max_val:8.3f}]")
print()

# Distribution analysis
print("Distribution shapes:")
print(f"  GRA bulk density: Gaussian")
print(f"  MS, NGR: Poisson/Bimodal (will use signed log)")
print(f"  RGB: Log-normal (will use log transform)")
print(f"  Porosity: Log-normal (will use log transform)")
print(f"  Grain Density: Gaussian")
print(f"  Water Content: Log-normal (will use log transform)")
print()

# Lithology distribution
print("Top 10 lithologies:")
lith_counts = v2_7_df['Principal'].value_counts().head(10)
for lith, count in lith_counts.items():
    pct = (count / len(v2_7_df)) * 100
    print(f"  {lith:40s}: {count:>6,} ({pct:5.2f}%)")
print()

# Comparison to v2.6
v2_6_samples = len(v2_6_df)
v2_6_boreholes = v2_6_df['Borehole_ID'].nunique()
v2_7_samples = len(v2_7_df)
v2_7_boreholes = v2_7_df['Borehole_ID'].nunique()

sample_change = ((v2_7_samples - v2_6_samples) / v2_6_samples) * 100
borehole_change = ((v2_7_boreholes - v2_6_boreholes) / v2_6_boreholes) * 100

print("="*100)
print("COMPARISON TO v2.6")
print("="*100)
print()
print(f"v2.6: {v2_6_samples:>7,} samples, {v2_6_boreholes:>3d} boreholes, 6 features")
print(f"v2.7: {v2_7_samples:>7,} samples, {v2_7_boreholes:>3d} boreholes, 9 features")
print()
print(f"Change: {sample_change:+.1f}% samples, {borehole_change:+.1f}% boreholes, +50% features")
print()

if borehole_change < -20:
    print("⚠ WARNING: Significant borehole coverage loss")
    print(f"  Trade-off: {abs(borehole_change):.0f}% fewer boreholes for +3 discriminative features")
    print(f"  Hypothesis: Porosity/grain density improve lithology classification beyond 42% ceiling")
else:
    print("✓ Good coverage retained while adding features")

print()

# Save dataset
output_path = '/home/utig5/johna/bhai/vae_training_data_v2_7_20cm.csv'
v2_7_df.to_csv(output_path, index=False)
print(f"✓ Dataset saved to: {output_path}")
print(f"  Size: {len(v2_7_df):,} samples")
print(f"  Columns: {len(v2_7_df.columns)}")
print()

print("="*100)
print("v2.7 DATASET CREATION COMPLETE")
print("="*100)
