"""
Create VAE GRA v2.6.1 Dataset - RSC color + MSP for maximum coverage

This script creates a training dataset with 5 features:
- GRA bulk density (g/cm³)
- Magnetic susceptibility MS (instrument units)
- NGR total counts (cps)
- RSC: Reflectance L* (lightness, 0-100)
- RSC: Reflectance a* (red-green axis)
- RSC: Reflectance b* (blue-yellow axis)
- MSP: Point magnetic susceptibility (instrument units)

Uses 20cm depth binning to align measurements.
Expected: ~341,000 samples from 484 boreholes (+43% vs v2.6)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def load_and_bin_data(filepath, measurement_cols, bin_size=0.2):
    """Load data and bin by depth."""
    print(f"Loading {filepath.name}...")

    required_cols = ['Exp', 'Site', 'Hole', 'Depth CSF-A (m)', 'Principal'] + measurement_cols

    df = pd.read_csv(filepath, usecols=required_cols, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + '-' + df['Hole'].astype(str)

    # Remove rows with missing depth or measurement values
    required_cols = ['Depth CSF-A (m)', 'Borehole_ID', 'Principal'] + measurement_cols
    df = df.dropna(subset=required_cols)

    print(f"  After removing NaN: {len(df):,} rows")

    # Create depth bins
    df['Depth_Bin'] = (df['Depth CSF-A (m)'] / bin_size).round() * bin_size

    # Group by borehole and depth bin
    agg_dict = {col: 'mean' for col in measurement_cols}
    agg_dict['Principal'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]

    binned = df.groupby(['Borehole_ID', 'Depth_Bin']).agg(agg_dict).reset_index()

    print(f"  After binning: {len(binned):,} bins from {binned['Borehole_ID'].nunique()} boreholes")

    return binned

def main():
    print("="*80)
    print("VAE GRA v2.6.1 DATASET CREATION - RSC Color + MSP")
    print("="*80)
    print()
    print("Target: ~341,000 samples from 484 boreholes")
    print("Features: GRA, MS, NGR, RSC (L*a*b*), MSP")
    print()

    start_time = time.time()

    data_dir = Path('/home/utig5/johna/bhai/datasets')
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_6_1_20cm.csv')

    bin_size = 0.2  # 20cm bins

    # Load and bin each dataset
    print("1. Processing GRA data...")
    gra_binned = load_and_bin_data(
        data_dir / 'GRA_DataLITH.csv',
        ['Bulk density (GRA)'],
        bin_size
    )

    print("\n2. Processing MS data...")
    ms_binned = load_and_bin_data(
        data_dir / 'MS_DataLITH.csv',
        ['Magnetic susceptibility (instr. units)'],
        bin_size
    )

    print("\n3. Processing NGR data...")
    ngr_binned = load_and_bin_data(
        data_dir / 'NGR_DataLITH.csv',
        ['NGR total counts (cps)'],
        bin_size
    )

    print("\n4. Processing RSC data...")
    rsc_binned = load_and_bin_data(
        data_dir / 'RSC_DataLITH.csv',
        ['Reflectance L*', 'Reflectance a*', 'Reflectance b*'],
        bin_size
    )

    print("\n5. Processing MSP data...")
    msp_binned = load_and_bin_data(
        data_dir / 'MSP_DataLITH.csv',
        ['Magnetic susceptibility (instr. units)'],
        bin_size
    )
    # Rename MSP column to distinguish from MS
    msp_binned = msp_binned.rename(columns={'Magnetic susceptibility (instr. units)': 'MSP (instr. units)'})

    # Merge datasets
    print("\n6. Merging datasets...")
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
        rsc_binned[['Borehole_ID', 'Depth_Bin', 'Reflectance L*', 'Reflectance a*', 'Reflectance b*']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"  After merging RSC: {len(merged):,} bins")

    merged = merged.merge(
        msp_binned[['Borehole_ID', 'Depth_Bin', 'MSP (instr. units)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"  After merging MSP: {len(merged):,} bins")

    # Final cleanup
    print("\n7. Final cleanup...")
    initial_count = len(merged)
    merged = merged.dropna()
    print(f"  Removed {initial_count - len(merged):,} rows with NaN values")

    # Remove outliers
    print("\n8. Removing outliers...")

    # GRA: typically 0.5-3.0 g/cm³
    outliers = (merged['Bulk density (GRA)'] < 0.3) | (merged['Bulk density (GRA)'] > 4.0)
    print(f"  GRA outliers: {outliers.sum():,}")
    merged = merged[~outliers]

    # RSC L*: 0-100
    outliers = (merged['Reflectance L*'] < 0) | (merged['Reflectance L*'] > 100)
    print(f"  RSC L* outliers: {outliers.sum():,}")
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
        'Reflectance L*',
        'Reflectance a*',
        'Reflectance b*',
        'MSP (instr. units)'
    ]
    print(merged[feature_cols].describe())

    # Save dataset
    print(f"\n9. Saving dataset to {output_path}...")
    merged.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\n✓ Dataset creation complete in {elapsed:.1f} seconds")
    print(f"✓ Saved {len(merged):,} samples to {output_path}")
    print(f"✓ Dataset size: {output_path.stat().st_size / 1024**2:.1f} MB")

    print("\n" + "="*80)
    print("COMPARISON TO v2.6")
    print("="*80)
    print(f"\nv2.6 (GRA+MS+NGR+RGB): 238,506 samples, 296 boreholes")
    print(f"v2.6.1 (GRA+MS+NGR+RSC+MSP): {len(merged):,} samples, {merged['Borehole_ID'].nunique()} boreholes")

    improvement = (len(merged) - 238506) / 238506 * 100
    bh_improvement = (merged['Borehole_ID'].nunique() - 296) / 296 * 100

    print(f"\nImprovement: {improvement:+.1f}% more samples, {bh_improvement:+.1f}% more boreholes")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
