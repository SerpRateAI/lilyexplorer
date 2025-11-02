"""
Create VAE GRA v2.9 training dataset with porosity feature.

Merges:
- GRA bulk density (continuous)
- MAD grain density (discrete)
- MS, NGR (continuous)
- RGB (continuous)

Then computes derived porosity feature:
  porosity = (ρ_grain - ρ_bulk) / (ρ_grain - ρ_fluid)
  where ρ_fluid = 1.024 g/cm³ (seawater density)

This physically meaningful feature captures lithology-dependent compaction state.

Expected impact: Reduced sample count (MAD is sparse) but potentially better discrimination.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def load_gra_data():
    """Load GRA bulk density measurements."""
    print("Loading GRA data...")
    gra_path = Path('/home/utig5/johna/bhai/datasets/GRA_DataLITH.csv')

    # Load only needed columns for efficiency
    cols = [
        'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
        'Depth CSF-A (m)', 'Bulk density (GRA)',
        'Prefix', 'Principal', 'Suffix'
    ]

    df = pd.read_csv(gra_path, usecols=cols)

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

    print(f"  Loaded {len(df):,} GRA measurements")
    return df

def load_mad_data():
    """Load MAD grain density measurements."""
    print("Loading MAD data...")
    mad_path = Path('/home/utig5/johna/bhai/datasets/MAD_DataLITH.csv')

    cols = [
        'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
        'Depth CSF-A (m)', 'Grain density (g/cm^3)'
    ]

    df = pd.read_csv(mad_path, usecols=cols)

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

    print(f"  Loaded {len(df):,} MAD measurements")
    return df

def load_ms_data():
    """Load magnetic susceptibility measurements."""
    print("Loading MS data...")
    ms_path = Path('/home/utig5/johna/bhai/datasets/MS_DataLITH.csv')

    cols = [
        'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
        'Depth CSF-A (m)', 'Magnetic susceptibility (instr. units)'
    ]

    df = pd.read_csv(ms_path, usecols=cols)

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

    print(f"  Loaded {len(df):,} MS measurements")
    return df

def load_ngr_data():
    """Load natural gamma radiation measurements."""
    print("Loading NGR data...")
    ngr_path = Path('/home/utig5/johna/bhai/datasets/NGR_DataLITH.csv')

    cols = [
        'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
        'Depth CSF-A (m)', 'NGR total counts (cps)'
    ]

    df = pd.read_csv(ngr_path, usecols=cols)

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

    print(f"  Loaded {len(df):,} NGR measurements")
    return df

def load_rgb_data():
    """Load RGB color measurements."""
    print("Loading RGB data...")
    rgb_path = Path('/home/utig5/johna/bhai/datasets/RGB_DataLITH.csv')

    cols = [
        'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
        'Depth CSF-A (m)', 'R', 'G', 'B'
    ]

    # RGB is huge, sample if needed
    df = pd.read_csv(rgb_path, usecols=cols)

    # Create borehole ID
    df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

    print(f"  Loaded {len(df):,} RGB measurements")
    return df

def bin_depth(depth, bin_size=0.2):
    """Bin depth to nearest bin_size meters."""
    return np.round(depth / bin_size) * bin_size

def merge_measurements_by_depth(gra_df, mad_df, ms_df, ngr_df, rgb_df, bin_size=0.2):
    """
    Merge all measurements using depth binning strategy.

    Strategy:
    1. Bin all depths to 20cm intervals
    2. Average measurements within each bin for each borehole
    3. Merge bins across measurement types
    4. Compute porosity from GRA + MAD
    """
    print(f"\nBinning measurements to {bin_size}m intervals...")

    # Bin all datasets
    for df in [gra_df, mad_df, ms_df, ngr_df, rgb_df]:
        df['Depth_Bin'] = bin_depth(df['Depth CSF-A (m)'], bin_size)

    # Aggregate GRA by borehole + depth bin
    print("Aggregating GRA measurements...")
    gra_agg = gra_df.groupby(['Borehole_ID', 'Depth_Bin']).agg({
        'Bulk density (GRA)': 'mean',
        'Prefix': 'first',
        'Principal': 'first',
        'Suffix': 'first',
        'Exp': 'first',
        'Site': 'first',
        'Hole': 'first'
    }).reset_index()

    # Aggregate MAD by borehole + depth bin
    print("Aggregating MAD measurements...")
    mad_agg = mad_df.groupby(['Borehole_ID', 'Depth_Bin']).agg({
        'Grain density (g/cm^3)': 'mean'
    }).reset_index()

    # Aggregate MS by borehole + depth bin
    print("Aggregating MS measurements...")
    ms_agg = ms_df.groupby(['Borehole_ID', 'Depth_Bin']).agg({
        'Magnetic susceptibility (instr. units)': 'mean'
    }).reset_index()

    # Aggregate NGR by borehole + depth bin
    print("Aggregating NGR measurements...")
    ngr_agg = ngr_df.groupby(['Borehole_ID', 'Depth_Bin']).agg({
        'NGR total counts (cps)': 'mean'
    }).reset_index()

    # Aggregate RGB by borehole + depth bin
    print("Aggregating RGB measurements...")
    rgb_agg = rgb_df.groupby(['Borehole_ID', 'Depth_Bin']).agg({
        'R': 'mean',
        'G': 'mean',
        'B': 'mean'
    }).reset_index()

    # Merge all measurements
    print("\nMerging measurements...")
    merged = gra_agg.copy()

    print(f"  Starting with GRA: {len(merged):,} bins")

    # Merge MAD (this will be the limiting factor)
    merged = merged.merge(mad_agg, on=['Borehole_ID', 'Depth_Bin'], how='inner')
    print(f"  After MAD merge: {len(merged):,} bins")

    # Merge MS
    merged = merged.merge(ms_agg, on=['Borehole_ID', 'Depth_Bin'], how='inner')
    print(f"  After MS merge: {len(merged):,} bins")

    # Merge NGR
    merged = merged.merge(ngr_agg, on=['Borehole_ID', 'Depth_Bin'], how='inner')
    print(f"  After NGR merge: {len(merged):,} bins")

    # Merge RGB
    merged = merged.merge(rgb_agg, on=['Borehole_ID', 'Depth_Bin'], how='inner')
    print(f"  After RGB merge: {len(merged):,} bins")

    return merged

def compute_porosity(df, fluid_density=1.024):
    """
    Compute porosity from bulk density and grain density.

    Formula: φ = (ρ_grain - ρ_bulk) / (ρ_grain - ρ_fluid)

    where:
    - φ = porosity (fraction)
    - ρ_grain = grain density (g/cm³) from MAD
    - ρ_bulk = bulk density (g/cm³) from GRA
    - ρ_fluid = fluid density (g/cm³) = 1.024 for seawater
    """
    print(f"\nComputing porosity (ρ_fluid = {fluid_density} g/cm³)...")

    rho_grain = df['Grain density (g/cm^3)']
    rho_bulk = df['Bulk density (GRA)']
    rho_fluid = fluid_density

    # Compute porosity
    porosity = (rho_grain - rho_bulk) / (rho_grain - rho_fluid)

    # Clip to valid range [0, 1]
    porosity = porosity.clip(0, 1)

    df['Porosity'] = porosity

    print(f"  Porosity range: {porosity.min():.3f} - {porosity.max():.3f}")
    print(f"  Porosity mean: {porosity.mean():.3f}")
    print(f"  Porosity std: {porosity.std():.3f}")

    return df

def remove_outliers(df):
    """Remove outlier measurements."""
    print("\nRemoving outliers...")
    initial_count = len(df)

    # Remove samples with any feature outside reasonable range
    df = df[
        (df['Bulk density (GRA)'] > 0.5) & (df['Bulk density (GRA)'] < 5.0) &
        (df['Grain density (g/cm^3)'] > 1.5) & (df['Grain density (g/cm^3)'] < 4.0) &
        (df['Porosity'] >= 0) & (df['Porosity'] <= 1) &
        (df['R'] >= 0) & (df['R'] <= 255) &
        (df['G'] >= 0) & (df['G'] <= 255) &
        (df['B'] >= 0) & (df['B'] <= 255)
    ]

    removed = initial_count - len(df)
    print(f"  Removed {removed:,} outliers ({removed/initial_count*100:.1f}%)")
    print(f"  Remaining: {len(df):,} samples")

    return df

def main():
    """Create VAE v2.9 dataset with porosity feature."""
    print("="*80)
    print("CREATE VAE GRA v2.9 DATASET WITH POROSITY")
    print("="*80)

    start_time = time.time()

    # Load all measurement types
    gra_df = load_gra_data()
    mad_df = load_mad_data()
    ms_df = load_ms_data()
    ngr_df = load_ngr_data()
    rgb_df = load_rgb_data()

    # Merge using depth binning
    merged_df = merge_measurements_by_depth(gra_df, mad_df, ms_df, ngr_df, rgb_df)

    # Compute porosity
    merged_df = compute_porosity(merged_df)

    # Remove outliers
    merged_df = remove_outliers(merged_df)

    # Create full lithology description
    merged_df['Full Lithology'] = (
        merged_df['Prefix'].fillna('') + ' ' +
        merged_df['Principal'].fillna('') + ' ' +
        merged_df['Suffix'].fillna('')
    ).str.strip()

    # Summary statistics
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(merged_df):,}")
    print(f"Unique boreholes: {merged_df['Borehole_ID'].nunique()}")
    print(f"Unique principal lithologies: {merged_df['Principal'].nunique()}")

    print("\nTop 10 lithologies:")
    print(merged_df['Principal'].value_counts().head(10))

    print("\nFeature statistics:")
    feature_cols = [
        'Bulk density (GRA)',
        'Grain density (g/cm^3)',
        'Porosity',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R', 'G', 'B'
    ]
    print(merged_df[feature_cols].describe())

    # Save dataset
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_9_20cm_porosity.csv')
    merged_df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Time elapsed: {elapsed:.1f}s")

    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
