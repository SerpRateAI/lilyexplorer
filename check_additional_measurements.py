"""
Check what additional measurements could be added without reducing the 238K sample count
"""

import pandas as pd
from pathlib import Path
import numpy as np

print("="*80)
print("CHECKING ADDITIONAL MEASUREMENTS FOR VAE v2 DATASET")
print("="*80)

data_dir = Path('/home/utig5/johna/bhai/datasets')

# Load existing VAE v2 dataset to get borehole/depth combinations
print("\nLoading existing VAE v2 dataset...")
vae_v2 = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"Current dataset: {len(vae_v2):,} samples from {vae_v2['Borehole_ID'].nunique()} boreholes")

# Create keys for matching
vae_v2['match_key'] = vae_v2['Borehole_ID'] + '_' + vae_v2['Depth_Bin'].astype(str)
existing_keys = set(vae_v2['match_key'])

print("\n" + "="*80)
print("Checking coverage of other measurement types...")
print("="*80)

# List of other datasets to check
other_datasets = [
    ('PWC', 'PWC_DataLITH.csv', ['P-wave velocity x (m/s)', 'P-wave velocity y (m/s)', 'P-wave velocity z (m/s)']),
    ('RSC', 'RSC_DataLITH.csv', ['Reflectance L*', 'Reflectance a*', 'Reflectance b*']),  # Color reflectance (alternative to RGB)
    ('CARB', 'CARB_DataLITH.csv', ['Calcium carbonate (wt%)']),
    ('MAD', 'MAD_DataLITH.csv', ['Grain density (g/cm^3)', 'Bulk density (g/cm^3)', 'Porosity (vol%)']),
    ('SRM', 'SRM_DataLITH.csv', ['Intensity background & drift corrected (A/m)']),
]

bin_size = 0.2

results = []

for name, filename, measurement_cols in other_datasets:
    filepath = data_dir / filename

    if not filepath.exists():
        print(f"\n{name}: File not found - {filename}")
        continue

    print(f"\n{name}: {filename}")
    try:
        # Handle special depth column names
        depth_col = 'Top depth CSF-A (m)' if name == 'CARB' else 'Depth CSF-A (m)'

        # Load minimal columns
        cols_to_load = ['Exp', 'Site', 'Hole', depth_col] + measurement_cols
        df = pd.read_csv(filepath, usecols=cols_to_load, low_memory=False)

        # Standardize depth column name
        if depth_col != 'Depth CSF-A (m)':
            df['Depth CSF-A (m)'] = df[depth_col]

        print(f"  Total measurements: {len(df):,}")

        # Create borehole ID
        df['Borehole_ID'] = df['Exp'].astype(str) + '-' + df['Site'].astype(str) + '-' + df['Hole'].astype(str)

        # Remove NaN
        df = df.dropna(subset=['Depth CSF-A (m)', 'Borehole_ID'] + measurement_cols)
        print(f"  After removing NaN: {len(df):,}")

        # Create depth bins
        df['Depth_Bin'] = (df['Depth CSF-A (m)'] / bin_size).round() * bin_size

        # Create match key
        df['match_key'] = df['Borehole_ID'] + '_' + df['Depth_Bin'].astype(str)

        # Check overlap with existing VAE v2 dataset
        matching_keys = set(df['match_key']) & existing_keys
        match_count = len(matching_keys)
        match_pct = (match_count / len(vae_v2)) * 100

        print(f"  Matches with VAE v2: {match_count:,} / {len(vae_v2):,} ({match_pct:.1f}%)")

        # If coverage is high, this could be added
        if match_pct > 90:
            print(f"  ✓ HIGH COVERAGE - Could add with minimal data loss!")
        elif match_pct > 50:
            print(f"  ~ MODERATE COVERAGE - Would lose {100-match_pct:.1f}% of samples")
        else:
            print(f"  ✗ LOW COVERAGE - Would lose {100-match_pct:.1f}% of samples")

        results.append({
            'Measurement': name,
            'Total_Measurements': len(df),
            'Matches': match_count,
            'Coverage_%': match_pct,
            'Columns': ', '.join(measurement_cols)
        })

    except Exception as e:
        print(f"  Error: {e}")

# Print summary
print("\n" + "="*80)
print("SUMMARY - Potential additions ranked by coverage")
print("="*80)

if results:
    df_results = pd.DataFrame(results).sort_values('Coverage_%', ascending=False)
    print(df_results.to_string(index=False))

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    high_coverage = df_results[df_results['Coverage_%'] > 90]
    if len(high_coverage) > 0:
        print("\n✓ These measurements could be added with <10% data loss:")
        for _, row in high_coverage.iterrows():
            print(f"  - {row['Measurement']}: {row['Columns']} ({row['Coverage_%']:.1f}% coverage)")
    else:
        print("\n✗ No measurements have >90% coverage of current dataset")
        print("  Adding any additional measurement would reduce sample count")

    moderate = df_results[(df_results['Coverage_%'] > 50) & (df_results['Coverage_%'] <= 90)]
    if len(moderate) > 0:
        print("\n~ These would require moderate data loss (10-50%):")
        for _, row in moderate.iterrows():
            print(f"  - {row['Measurement']}: {row['Columns']} ({row['Coverage_%']:.1f}% coverage)")
else:
    print("\nNo measurements could be checked")

print("\n" + "="*80)
