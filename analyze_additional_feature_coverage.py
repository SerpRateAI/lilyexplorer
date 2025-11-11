"""
Analyze what additional features could supplement VAE v2.6.7 training data.

Current v2.6.7:
- 238,506 samples from 296 boreholes
- 6D features: GRA, MS, NGR, RGB
- 20cm depth binning

This script checks coverage of additional measurements that could be added.
"""

import pandas as pd
import numpy as np

print("="*100)
print("ADDITIONAL FEATURE COVERAGE ANALYSIS FOR VAE v2.6.7")
print("="*100)
print()

# Load current VAE v2.6.7 training data
print("Loading current VAE v2.6.7 dataset...")
vae_df = pd.read_csv('vae_training_data_v2_20cm.csv')
print(f"Current dataset: {len(vae_df):,} samples from {vae_df['Borehole_ID'].nunique()} boreholes")
print()

# Get unique borehole IDs and depth bins from current dataset
current_boreholes = set(vae_df['Borehole_ID'].unique())
print(f"Current boreholes: {len(current_boreholes)}")
print()

# Use existing depth bins (already computed in dataset)
current_samples = set(zip(vae_df['Borehole_ID'], vae_df['Depth_Bin']))
print(f"Current depth bins: {len(current_samples):,}")
print()

# List of additional features to check
additional_features = {
    'PWC': ('datasets/PWC_DataLITH.csv', 'P-wave velocity x (m/s)', 'P-wave velocity'),
    'MAD_grain': ('datasets/MAD_DataLITH.csv', 'Grain density (g/cm³)', 'Grain density (discrete)'),
    'MAD_porosity': ('datasets/MAD_DataLITH.csv', 'Porosity (%)', 'Porosity (discrete)'),
    'SRM': ('datasets/SRM_DataLITH.csv', 'Inclination (deg)', 'Natural remanent magnetization'),
}

print("="*100)
print("CHECKING COVERAGE OF ADDITIONAL FEATURES")
print("="*100)
print()

coverage_results = []

for feature_name, (filepath, column_name, description) in additional_features.items():
    print(f"Checking {feature_name} ({description})...")

    try:
        # Load dataset
        if feature_name.startswith('MAD'):
            # Only load once for MAD
            if 'mad_df' not in locals():
                mad_df = pd.read_csv(filepath, usecols=[
                    'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
                    'Depth CSF-A (m)', 'Grain density (g/cm³)', 'Porosity (%)'
                ])
                mad_df['Borehole_ID'] = (
                    mad_df['Exp'].astype(str) + '_' +
                    mad_df['Site'].astype(str) + '_' +
                    mad_df['Hole'].astype(str)
                )
                mad_df['Depth_bin'] = (mad_df['Depth CSF-A (m)'] / 0.2).round().astype(int)
            df = mad_df
        else:
            df = pd.read_csv(filepath, usecols=[
                'Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
                'Depth CSF-A (m)', column_name
            ])
            df['Borehole_ID'] = (
                df['Exp'].astype(str) + '_' +
                df['Site'].astype(str) + '_' +
                df['Hole'].astype(str)
            )
            df['Depth_bin'] = (df['Depth CSF-A (m)'] / 0.2).round().astype(int)

        # Check coverage
        total_measurements = len(df)
        total_boreholes = df['Borehole_ID'].nunique()

        # Boreholes in common with current dataset
        feature_boreholes = set(df['Borehole_ID'].unique())
        common_boreholes = current_boreholes & feature_boreholes

        # Depth bins in common
        feature_samples = set(zip(df['Borehole_ID'], df['Depth_bin']))
        common_samples = current_samples & feature_samples

        # Coverage percentages
        borehole_coverage = len(common_boreholes) / len(current_boreholes) * 100
        sample_coverage = len(common_samples) / len(current_samples) * 100

        coverage_results.append({
            'feature': feature_name,
            'description': description,
            'total_measurements': total_measurements,
            'total_boreholes': total_boreholes,
            'common_boreholes': len(common_boreholes),
            'borehole_coverage_pct': borehole_coverage,
            'common_samples': len(common_samples),
            'sample_coverage_pct': sample_coverage
        })

        print(f"  Total measurements: {total_measurements:,}")
        print(f"  Total boreholes: {total_boreholes}")
        print(f"  Common boreholes: {len(common_boreholes)}/{len(current_boreholes)} ({borehole_coverage:.1f}%)")
        print(f"  Common depth bins: {len(common_samples):,}/{len(current_samples):,} ({sample_coverage:.1f}%)")
        print()

    except Exception as e:
        print(f"  Error: {e}")
        print()

print("="*100)
print("COVERAGE SUMMARY")
print("="*100)
print()

# Sort by sample coverage
coverage_results.sort(key=lambda x: x['sample_coverage_pct'], reverse=True)

print(f"{'Feature':<20s} {'Description':<35s} {'Borehole %':>12s} {'Sample %':>12s} {'N_Samples':>12s}")
print("-"*100)

for result in coverage_results:
    print(f"{result['feature']:<20s} "
          f"{result['description']:<35s} "
          f"{result['borehole_coverage_pct']:>11.1f}% "
          f"{result['sample_coverage_pct']:>11.1f}% "
          f"{result['common_samples']:>12,d}")

print()
print("="*100)
print("RECOMMENDATIONS")
print("="*100)
print()

# High coverage features
high_coverage = [r for r in coverage_results if r['sample_coverage_pct'] > 50]
medium_coverage = [r for r in coverage_results if 20 < r['sample_coverage_pct'] <= 50]
low_coverage = [r for r in coverage_results if r['sample_coverage_pct'] <= 20]

if high_coverage:
    print("HIGH COVERAGE (>50% of current samples):")
    for r in high_coverage:
        print(f"  ✓ {r['feature']} ({r['description']}): {r['sample_coverage_pct']:.1f}% coverage")
        print(f"    - Could add to {r['common_samples']:,} samples with minimal data loss")
    print()

if medium_coverage:
    print("MEDIUM COVERAGE (20-50% of current samples):")
    for r in medium_coverage:
        print(f"  ○ {r['feature']} ({r['description']}): {r['sample_coverage_pct']:.1f}% coverage")
        print(f"    - Would reduce dataset to {r['common_samples']:,} samples ({r['sample_coverage_pct']:.1f}%)")
    print()

if low_coverage:
    print("LOW COVERAGE (<20% of current samples):")
    for r in low_coverage:
        print(f"  ✗ {r['feature']} ({r['description']}): {r['sample_coverage_pct']:.1f}% coverage")
        print(f"    - Not recommended (only {r['common_samples']:,} samples)")
    print()

print("NOTES:")
print()
print("1. High coverage features can be added with minimal dataset size reduction")
print("2. Medium coverage features create trade-off: more features vs fewer samples")
print("3. Low coverage features likely not worth the large dataset reduction")
print()
print("4. Consider previously tested alternatives:")
print("   - RSC (reflectance spectroscopy): Tested in v2.6.1, performed -54% worse than RGB")
print("   - MSP (point magnetic susceptibility): Tested in v2.6.1, redundant with MS loop")
print("   - Spatial context (above/below bins): Tested in v2.2, only +3.9% improvement")
print()
print("5. Remember: Feature quality > dataset size")
print("   - v2.6.1 with +44% more data performed -54% worse (RSC < RGB)")
print("   - Only add features if they provide discriminative information")
print()
print("="*100)
