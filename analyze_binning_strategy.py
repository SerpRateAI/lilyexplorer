"""
Analyze optimal depth binning strategy for VAE lithology model.
Compare sample counts and data quality at different bin sizes.
Optimized for large datasets and realistic co-location.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_borehole_id(df):
    """Create consistent borehole identifier."""
    return df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

def bin_depth(depth, bin_size_cm):
    """Round depth to nearest bin."""
    bin_size_m = bin_size_cm / 100.0
    return np.round(depth / bin_size_m) * bin_size_m

def load_and_process_dataset(file_path, value_col, bin_size_cm, chunk_size=100000):
    """Load dataset and bin by depth - optimized for large files."""
    print(f"Loading {file_path.name}...")

    # Define columns to read
    required_cols = ['Exp', 'Site', 'Hole', 'Depth CSF-A (m)', value_col, 'Principal']

    # Read in chunks and process
    chunks = []
    total_rows = 0

    for chunk in pd.read_csv(file_path, usecols=required_cols, chunksize=chunk_size, low_memory=False):
        # Remove NaN values in the measurement
        chunk = chunk.dropna(subset=[value_col, 'Depth CSF-A (m)'])

        if len(chunk) == 0:
            continue

        # Create borehole ID
        chunk['Borehole_ID'] = create_borehole_id(chunk)

        # Bin depth
        chunk['Depth_Bin'] = bin_depth(chunk['Depth CSF-A (m)'], bin_size_cm)

        # Keep only needed columns
        chunk = chunk[['Borehole_ID', 'Depth_Bin', value_col, 'Principal']].copy()

        chunks.append(chunk)
        total_rows += len(chunk)

        # Print progress for large files
        if len(chunks) % 20 == 0:
            print(f"  Processed {total_rows:,} rows...")

    if len(chunks) == 0:
        print(f"  Warning: No valid data in {file_path.name}")
        return None

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} valid measurements from {df['Borehole_ID'].nunique()} boreholes")

    # Group by borehole and depth bin, take mean
    print(f"  Binning data...")
    group_cols = ['Borehole_ID', 'Depth_Bin']
    agg_dict = {
        value_col: 'mean',
        'Principal': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    }

    df_binned = df.groupby(group_cols, as_index=False).agg(agg_dict)

    print(f"  After binning: {len(df_binned):,} bins from {df_binned['Borehole_ID'].nunique()} boreholes")

    return df_binned

def analyze_bin_size(bin_size_cm):
    """Analyze dataset for a specific bin size."""
    print(f"\n{'='*80}")
    print(f"ANALYZING BIN SIZE: {bin_size_cm} cm")
    print(f"{'='*80}")

    datasets_dir = Path('/home/utig5/johna/bhai/datasets')

    # Strategy: Use measurements that are commonly collected together
    # GRA, MS, and NGR are all whole-core measurements done on MSCL
    # They should have excellent co-location

    # Load each dataset with binning
    print("\nLoading GRA (Gamma Ray Attenuation bulk density)...")
    gra = load_and_process_dataset(
        datasets_dir / 'GRA_DataLITH.csv',
        'Bulk density (GRA)',
        bin_size_cm,
        chunk_size=200000
    )

    print("\nLoading MS (Magnetic Susceptibility)...")
    ms = load_and_process_dataset(
        datasets_dir / 'MS_DataLITH.csv',
        'Magnetic susceptibility (instr. units)',
        bin_size_cm,
        chunk_size=200000
    )

    print("\nLoading NGR (Natural Gamma Radiation)...")
    ngr = load_and_process_dataset(
        datasets_dir / 'NGR_DataLITH.csv',
        'NGR total counts (cps)',
        bin_size_cm,
        chunk_size=200000
    )

    print(f"\n{'-'*80}")
    print("MERGING DATASETS (GRA + MS + NGR)")
    print(f"{'-'*80}")

    # Merge datasets - all three are MSCL measurements
    merged = gra.copy()
    print(f"Starting with GRA: {len(merged):,} bins, {merged['Borehole_ID'].nunique()} boreholes")

    # Merge MS
    merged = merged.merge(
        ms[['Borehole_ID', 'Depth_Bin', 'Magnetic susceptibility (instr. units)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"After adding MS: {len(merged):,} bins, {merged['Borehole_ID'].nunique()} boreholes")

    # Merge NGR
    merged = merged.merge(
        ngr[['Borehole_ID', 'Depth_Bin', 'NGR total counts (cps)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"After adding NGR: {len(merged):,} bins, {merged['Borehole_ID'].nunique()} boreholes")

    # Analyze lithology distribution
    lithology_counts = merged['Principal'].value_counts()
    print(f"\nTop 10 lithologies:")
    for lith, count in lithology_counts.head(10).items():
        print(f"  {lith}: {count} ({count/len(merged)*100:.1f}%)")
    print(f"Total unique lithologies: {len(lithology_counts)}")

    # Analyze depth coverage
    depth_stats = merged.groupby('Borehole_ID')['Depth_Bin'].agg(['min', 'max', 'count'])
    depth_stats['depth_range'] = depth_stats['max'] - depth_stats['min']

    print(f"\nDepth coverage statistics:")
    print(f"  Bins per borehole: {depth_stats['count'].mean():.1f} ± {depth_stats['count'].std():.1f}")
    print(f"  Depth range per borehole: {depth_stats['depth_range'].mean():.1f} ± {depth_stats['depth_range'].std():.1f} m")
    print(f"  Min bins in borehole: {depth_stats['count'].min()}")
    print(f"  Max bins in borehole: {depth_stats['count'].max()}")

    # Summary statistics
    summary = {
        'bin_size_cm': bin_size_cm,
        'total_samples': len(merged),
        'total_boreholes': merged['Borehole_ID'].nunique(),
        'unique_lithologies': len(lithology_counts),
        'avg_bins_per_borehole': depth_stats['count'].mean(),
        'avg_depth_range': depth_stats['depth_range'].mean(),
        'min_bins_per_borehole': depth_stats['count'].min(),
        'max_bins_per_borehole': depth_stats['count'].max()
    }

    return summary, merged

def main():
    """Main analysis function."""
    print("DEPTH BINNING STRATEGY ANALYSIS FOR VAE LITHOLOGY MODEL")
    print("="*80)
    print("Using MSCL measurements: GRA + MS + NGR")
    print("These are collected together on the same instrument")
    print("="*80)

    # Test different bin sizes - start with larger bins for speed
    bin_sizes = [50, 20, 10]

    results = []
    datasets = {}

    for bin_size in bin_sizes:
        summary, merged = analyze_bin_size(bin_size)
        results.append(summary)
        datasets[bin_size] = merged

    # Compare results
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save results
    output_path = Path('/home/utig5/johna/bhai/binning_analysis_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    # Find optimal bin size (maximize samples while maintaining good coverage)
    optimal_idx = results_df['total_samples'].idxmax()
    optimal = results_df.iloc[optimal_idx]

    print(f"\nOptimal bin size: {optimal['bin_size_cm']:.0f} cm")
    print(f"  Total samples: {optimal['total_samples']:.0f}")
    print(f"  Boreholes: {optimal['total_boreholes']:.0f}")
    print(f"  Unique lithologies: {optimal['unique_lithologies']:.0f}")
    print(f"  Avg bins per borehole: {optimal['avg_bins_per_borehole']:.1f}")
    print(f"  Depth range per borehole: {optimal['avg_depth_range']:.1f} m")

    # Save optimal dataset
    optimal_bin_size = int(optimal['bin_size_cm'])
    optimal_dataset = datasets[optimal_bin_size]

    output_data_path = Path(f'/home/utig5/johna/bhai/vae_training_data_{optimal_bin_size}cm.csv')
    optimal_dataset.to_csv(output_data_path, index=False)
    print(f"\nOptimal dataset saved to: {output_data_path}")

    print(f"\nDataset preview:")
    print(optimal_dataset.head(10).to_string())

    print(f"\nBasic statistics:")
    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)', 'NGR total counts (cps)']
    print(optimal_dataset[feature_cols].describe())

    print(f"\n{'='*80}")
    print(f"COMPARISON TO PREVIOUS MODEL")
    print(f"{'='*80}")
    print(f"Previous model (MAD-based): 151 samples")
    print(f"New model (GRA+MS+NGR): {len(optimal_dataset):,} samples")
    print(f"Improvement: {len(optimal_dataset)/151:.1f}x more samples")
    print(f"\nPrevious boreholes: ~5")
    print(f"New boreholes: {optimal_dataset['Borehole_ID'].nunique()}")

if __name__ == "__main__":
    main()
