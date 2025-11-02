"""
Create VAE GRA v2.2 Dataset - Adding Spatial Context

This script creates a training dataset with spatial context:
- Input: 18D (6 features × 3 positions: above, current, below)
- Output: 6D (only current position reconstructed)

Features:
- GRA bulk density (g/cm³)
- Magnetic susceptibility (instrument units)
- NGR total counts (cps)
- R (red channel, 0-255)
- G (green channel, 0-255)
- B (blue channel, 0-255)

Uses 20cm depth binning to align measurements from different instruments.
Adds above/below context for stratigraphic information.
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

def add_spatial_context(df, feature_cols):
    """
    Add above/below features for spatial context using vectorized operations.

    For each row, adds features from:
    - Above: bin at depth - 0.2m
    - Below: bin at depth + 0.2m

    Edge handling: Pads with current values where above/below missing.
    """
    print("\nAdding spatial context (above/below features)...")

    # Sort by borehole and depth for proper context window
    df = df.sort_values(['Borehole_ID', 'Depth_Bin']).reset_index(drop=True)

    # Use groupby and shift for efficient vectorized operation
    grouped = df.groupby('Borehole_ID')

    # For each feature, shift by 1 position (above) and -1 position (below)
    for col in feature_cols:
        # Shift up (previous row = above in depth)
        df[f'{col}_above'] = grouped[col].shift(1)
        # Shift down (next row = below in depth)
        df[f'{col}_below'] = grouped[col].shift(-1)

        # Fill NaN with current values (edge padding)
        df[f'{col}_above'] = df[f'{col}_above'].fillna(df[col])
        df[f'{col}_below'] = df[f'{col}_below'].fillna(df[col])

    # Verify no NaN in context features
    context_cols = [f'{col}_{pos}' for col in feature_cols for pos in ['above', 'below']]
    nan_count = df[context_cols].isna().sum().sum()
    print(f"  Added spatial context: {nan_count} NaN values (should be 0)")

    num_boreholes = df['Borehole_ID'].nunique()
    print(f"  Processed {len(df):,} samples from {num_boreholes} boreholes")

    return df

def main():
    print("="*80)
    print("VAE GRA v2.2 DATASET CREATION - Adding Spatial Context")
    print("="*80)

    start_time = time.time()

    # Paths
    data_dir = Path('/home/utig5/johna/bhai/datasets')
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_2_20cm.csv')

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

    # Remove any remaining NaN values
    print("\n6. Initial cleanup...")
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

    # Add spatial context (above/below features)
    print("\n8. Adding spatial context...")
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]

    merged = add_spatial_context(merged, feature_cols)

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

    print(f"\nFeature statistics (current bin):")
    print(merged[feature_cols].describe())

    print(f"\nDataset dimensions:")
    print(f"  Input features: 18D (6 features × 3 positions)")
    print(f"  Output features: 6D (current position only)")
    print(f"  Columns in CSV: {len(merged.columns)}")

    # Save dataset
    print(f"\n9. Saving dataset to {output_path}...")
    merged.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n✓ Dataset creation complete in {elapsed:.1f} seconds")
    print(f"✓ Saved {len(merged):,} samples to {output_path}")
    print(f"✓ Dataset size: {output_path.stat().st_size / 1024**2:.1f} MB")

    print("\n" + "="*80)

if __name__ == "__main__":
    main()
