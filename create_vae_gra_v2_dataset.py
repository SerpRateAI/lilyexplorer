"""
Create VAE GRA v2 Dataset - Adding RGB color features to GRA + MS + NGR

This script creates a training dataset with 6 features:
- GRA bulk density (g/cm³)
- Magnetic susceptibility (instrument units)
- NGR total counts (cps)
- R (red channel, 0-255)
- G (green channel, 0-255)
- B (blue channel, 0-255)

Uses 20cm depth binning to align measurements from different instruments.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def load_and_bin_data(filepath, measurement_cols, bin_size=0.2):
    """Load data and bin by depth."""
    print(f"Loading {filepath.name}...")

    # Only load necessary columns for efficiency
    required_cols = ['Exp', 'Site', 'Hole', 'Depth CSF-A (m)', 'Principal'] + measurement_cols

    df = pd.read_csv(filepath, usecols=required_cols, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + '-' + df['Hole'].astype(str)

    # Remove rows with missing depth or measurement values
    required_cols = ['Depth CSF-A (m)', 'Borehole_ID', 'Principal'] + measurement_cols
    df = df.dropna(subset=required_cols)

    print(f"  After removing NaN: {len(df):,} rows")

    # Create depth bins (round to nearest bin_size)
    df['Depth_Bin'] = (df['Depth CSF-A (m)'] / bin_size).round() * bin_size

    # Group by borehole and depth bin
    agg_dict = {col: 'mean' for col in measurement_cols}
    agg_dict['Principal'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]  # Most common lithology

    binned = df.groupby(['Borehole_ID', 'Depth_Bin']).agg(agg_dict).reset_index()

    print(f"  After binning: {len(binned):,} bins from {binned['Borehole_ID'].nunique()} boreholes")

    return binned

def main():
    print("="*80)
    print("VAE GRA v2 DATASET CREATION - Adding RGB Features")
    print("="*80)

    start_time = time.time()

    # Paths
    data_dir = Path('/home/utig5/johna/bhai/datasets')
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

    bin_size = 0.2  # 20cm bins

    # Load and bin GRA data
    print("\n1. Processing GRA data...")
    gra_binned = load_and_bin_data(
        data_dir / 'GRA_DataLITH.csv',
        ['Bulk density (GRA)'],
        bin_size
    )

    # Load and bin MS data
    print("\n2. Processing MS data...")
    ms_binned = load_and_bin_data(
        data_dir / 'MS_DataLITH.csv',
        ['Magnetic susceptibility (instr. units)'],
        bin_size
    )

    # Load and bin NGR data
    print("\n3. Processing NGR data...")
    ngr_binned = load_and_bin_data(
        data_dir / 'NGR_DataLITH.csv',
        ['NGR total counts (cps)'],
        bin_size
    )

    # Load and bin RGB data
    print("\n4. Processing RGB data...")
    rgb_binned = load_and_bin_data(
        data_dir / 'RGB_DataLITH.csv',
        ['R', 'G', 'B'],
        bin_size
    )

    # Merge datasets
    print("\n5. Merging datasets...")
    merged = gra_binned.copy()
    print(f"  Starting with GRA: {len(merged):,} bins")

    merged = merged.merge(
        ms_binned[['Borehole_ID', 'Depth_Bin', 'Magnetic susceptibility (instr. units)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"  After merging MS: {len(merged):,} bins")

    merged = merged.merge(
        ngr_binned[['Borehole_ID', 'Depth_Bin', 'NGR total counts (cps)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"  After merging NGR: {len(merged):,} bins")

    merged = merged.merge(
        rgb_binned[['Borehole_ID', 'Depth_Bin', 'R', 'G', 'B']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"  After merging RGB: {len(merged):,} bins")

    # Final cleanup
    print("\n6. Final cleanup...")

    # Remove any remaining NaN values
    initial_count = len(merged)
    merged = merged.dropna()
    print(f"  Removed {initial_count - len(merged):,} rows with NaN values")

    # Remove outliers (keep values within reasonable ranges)
    print("\n7. Removing outliers...")

    # GRA: typically 0.5-3.0 g/cm³
    outliers = (merged['Bulk density (GRA)'] < 0.3) | (merged['Bulk density (GRA)'] > 4.0)
    print(f"  GRA outliers: {outliers.sum():,}")
    merged = merged[~outliers]

    # RGB: 0-255
    for color in ['R', 'G', 'B']:
        outliers = (merged[color] < 0) | (merged[color] > 255)
        print(f"  {color} outliers: {outliers.sum():,}")
        merged = merged[~outliers]

    # Statistics
    print("\n" + "="*80)
    print("FINAL DATASET STATISTICS")
    print("="*80)

    print(f"\nTotal samples: {len(merged):,}")
    print(f"Number of boreholes: {merged['Borehole_ID'].nunique()}")
    print(f"Unique lithologies: {merged['Principal'].nunique()}")

    print(f"\nTop 10 lithologies:")
    top_lithologies = merged['Principal'].value_counts().head(10)
    for lith, count in top_lithologies.items():
        pct = count / len(merged) * 100
        print(f"  {lith:30s}: {count:6d} ({pct:5.2f}%)")

    print(f"\nFeature statistics:")
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]
    print(merged[feature_cols].describe())

    # Save dataset
    print(f"\n8. Saving dataset to {output_path}...")
    merged.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n✓ Dataset creation complete in {elapsed:.1f} seconds")
    print(f"✓ Saved {len(merged):,} samples to {output_path}")
    print(f"✓ Dataset size: {output_path.stat().st_size / 1024**2:.1f} MB")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
