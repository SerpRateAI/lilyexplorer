"""
Create aligned VAE training dataset using depth binning.
Uses GRA + MS + NGR (all MSCL measurements with good co-location).
Optimized for large files - uses 20cm bins directly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import time
warnings.filterwarnings('ignore')

def create_borehole_id(df):
    """Create consistent borehole identifier."""
    return df['Exp'].astype(str) + '-' + df['Site'].astype(str) + df['Hole'].astype(str)

def bin_depth(depth, bin_size_cm=20):
    """Round depth to nearest bin (default 20cm)."""
    bin_size_m = bin_size_cm / 100.0
    return np.round(depth / bin_size_m) * bin_size_m

def load_and_bin_dataset(file_path, value_col, bin_size_cm=20, chunk_size=200000):
    """Load dataset and bin by depth - optimized for large files."""
    print(f"\n{'='*60}")
    print(f"Loading {file_path.name}")
    print(f"{'='*60}")
    start_time = time.time()

    # Define columns to read
    required_cols = ['Exp', 'Site', 'Hole', 'Depth CSF-A (m)', value_col, 'Principal']

    # Read in chunks and process
    all_bins = {}  # Dict to accumulate measurements per bin
    total_rows = 0
    chunk_num = 0

    for chunk in pd.read_csv(file_path, usecols=required_cols, chunksize=chunk_size, low_memory=False):
        chunk_num += 1
        # Remove NaN values
        chunk = chunk.dropna(subset=[value_col, 'Depth CSF-A (m)'])

        if len(chunk) == 0:
            continue

        # Create borehole ID
        chunk['Borehole_ID'] = create_borehole_id(chunk)

        # Bin depth
        chunk['Depth_Bin'] = bin_depth(chunk['Depth CSF-A (m)'], bin_size_cm)

        # Accumulate measurements by bin
        for _, row in chunk.iterrows():
            key = (row['Borehole_ID'], row['Depth_Bin'])
            if key not in all_bins:
                all_bins[key] = {
                    'Borehole_ID': row['Borehole_ID'],
                    'Depth_Bin': row['Depth_Bin'],
                    'Principal': row['Principal'],
                    f'{value_col}_values': []
                }
            all_bins[key][f'{value_col}_values'].append(row[value_col])

        total_rows += len(chunk)

        if chunk_num % 5 == 0:
            elapsed = time.time() - start_time
            print(f"  Chunk {chunk_num}: {total_rows:,} rows processed ({elapsed:.1f}s)")

    # Convert to dataframe with averaged values
    print(f"  Converting {len(all_bins):,} bins to dataframe...")
    records = []
    for bin_data in all_bins.values():
        record = {
            'Borehole_ID': bin_data['Borehole_ID'],
            'Depth_Bin': bin_data['Depth_Bin'],
            'Principal': bin_data['Principal'],
            value_col: np.mean(bin_data[f'{value_col}_values'])
        }
        records.append(record)

    df_binned = pd.DataFrame(records)

    elapsed = time.time() - start_time
    print(f"  DONE: {len(df_binned):,} bins from {df_binned['Borehole_ID'].nunique()} boreholes ({elapsed:.1f}s)")

    return df_binned

def main():
    """Create VAE training dataset."""
    print("="*80)
    print("VAE LITHOLOGY MODEL - DATASET CREATION")
    print("="*80)
    print("Features: GRA bulk density + MS + NGR (MSCL measurements)")
    print("Bin size: 20 cm")
    print("="*80)

    datasets_dir = Path('/home/utig5/johna/bhai/datasets')

    # Load and bin each dataset
    gra = load_and_bin_dataset(
        datasets_dir / 'GRA_DataLITH.csv',
        'Bulk density (GRA)',
        bin_size_cm=20,
        chunk_size=200000
    )

    ms = load_and_bin_dataset(
        datasets_dir / 'MS_DataLITH.csv',
        'Magnetic susceptibility (instr. units)',
        bin_size_cm=20,
        chunk_size=200000
    )

    ngr = load_and_bin_dataset(
        datasets_dir / 'NGR_DataLITH.csv',
        'NGR total counts (cps)',
        bin_size_cm=20,
        chunk_size=100000  # Smaller chunks for NGR
    )

    # Merge datasets
    print(f"\n{'='*80}")
    print("MERGING DATASETS")
    print(f"{'='*80}")

    merged = gra.copy()
    print(f"Starting with GRA: {len(merged):,} bins, {merged['Borehole_ID'].nunique()} boreholes")

    # Merge MS
    merged = merged.merge(
        ms[['Borehole_ID', 'Depth_Bin', 'Magnetic susceptibility (instr. units)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"After adding MS:  {len(merged):,} bins, {merged['Borehole_ID'].nunique()} boreholes")

    # Merge NGR
    merged = merged.merge(
        ngr[['Borehole_ID', 'Depth_Bin', 'NGR total counts (cps)']],
        on=['Borehole_ID', 'Depth_Bin'],
        how='inner'
    )
    print(f"After adding NGR: {len(merged):,} bins, {merged['Borehole_ID'].nunique()} boreholes")

    # Analyze results
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")

    # Lithology distribution
    lithology_counts = merged['Principal'].value_counts()
    print(f"\nTop 10 lithologies:")
    for i, (lith, count) in enumerate(lithology_counts.head(10).items(), 1):
        print(f"  {i:2d}. {lith:40s}: {count:5d} ({count/len(merged)*100:5.1f}%)")
    print(f"\nTotal unique lithologies: {len(lithology_counts)}")

    # Depth coverage per borehole
    depth_stats = merged.groupby('Borehole_ID')['Depth_Bin'].agg(['min', 'max', 'count'])
    depth_stats['depth_range'] = depth_stats['max'] - depth_stats['min']

    print(f"\nDepth coverage per borehole:")
    print(f"  Mean bins/borehole: {depth_stats['count'].mean():.1f} ± {depth_stats['count'].std():.1f}")
    print(f"  Mean depth range:   {depth_stats['depth_range'].mean():.1f} ± {depth_stats['depth_range'].std():.1f} m")
    print(f"  Min bins:           {depth_stats['count'].min()}")
    print(f"  Max bins:           {depth_stats['count'].max()}")

    # Feature statistics
    print(f"\nFeature statistics:")
    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)', 'NGR total counts (cps)']
    print(merged[feature_cols].describe().to_string())

    # Save dataset
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_20cm.csv')
    merged.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*80}")

    # Comparison
    print(f"\nCOMPARISON TO PREVIOUS MODEL:")
    print(f"  Previous (MAD-based):  151 samples from ~5 boreholes")
    print(f"  New (GRA+MS+NGR):      {len(merged):,} samples from {merged['Borehole_ID'].nunique()} boreholes")
    print(f"  Improvement:           {len(merged)/151:.0f}x more samples")

    # Show sample of data
    print(f"\nSample data:")
    print(merged.head(15).to_string())

if __name__ == "__main__":
    main()
