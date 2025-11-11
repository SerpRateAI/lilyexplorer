"""
Test fuzzy matching with multiple tolerance levels for VAE dataset creation.

Tolerance range: 5cm to 5m
Goal: Maximize samples while maintaining clustering performance
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time

print("="*100)
print("FUZZY MATCHING TOLERANCE EXPERIMENT")
print("="*100)
print()

# Test these tolerance values
tolerances = [0.05, 0.10, 0.20, 0.30, 0.50, 1.0, 2.0, 3.0, 5.0]  # meters

print("Testing tolerances:")
for tol in tolerances:
    print(f"  ± {tol*100:>5.0f} cm ({tol:>4.2f} m)")
print()

# Load raw datasets
print("Loading raw IODP datasets...")
print("  GRA...")
gra_df = pd.read_csv('datasets/GRA_DataLITH.csv', low_memory=False)
gra_df['Borehole_ID'] = gra_df['Exp'].astype(str) + '_' + gra_df['Site'].astype(str) + '_' + gra_df['Hole'].astype(str)

print("  MS...")
ms_df = pd.read_csv('datasets/MS_DataLITH.csv', low_memory=False)
ms_df['Borehole_ID'] = ms_df['Exp'].astype(str) + '_' + ms_df['Site'].astype(str) + '_' + ms_df['Hole'].astype(str)

print("  NGR...")
ngr_df = pd.read_csv('datasets/NGR_DataLITH.csv', low_memory=False)
ngr_df['Borehole_ID'] = ngr_df['Exp'].astype(str) + '_' + ngr_df['Site'].astype(str) + '_' + ngr_df['Hole'].astype(str)

print("  RGB...")
rgb_df = pd.read_csv('datasets/RGB_DataLITH.csv', low_memory=False)
rgb_df['Borehole_ID'] = rgb_df['Exp'].astype(str) + '_' + rgb_df['Site'].astype(str) + '_' + rgb_df['Hole'].astype(str)

print()

# Get boreholes with all 4 measurements
gra_boreholes = set(gra_df['Borehole_ID'].unique())
ms_boreholes = set(ms_df['Borehole_ID'].unique())
ngr_boreholes = set(ngr_df['Borehole_ID'].unique())
rgb_boreholes = set(rgb_df['Borehole_ID'].unique())
target_boreholes = sorted(list(gra_boreholes & ms_boreholes & ngr_boreholes & rgb_boreholes))

print(f"Boreholes with all 4 measurements: {len(target_boreholes)}")
print()

def create_fuzzy_dataset(tolerance_m, sample_boreholes=None):
    """
    Create dataset with fuzzy depth matching.

    For each 20cm depth bin center, find measurements within ±tolerance.
    Take the closest measurement within tolerance.
    """
    if sample_boreholes is None:
        sample_boreholes = target_boreholes

    bin_size = 0.2  # 20cm bins
    results = []

    for bh_id in sample_boreholes:
        # Get data for this borehole
        bh_gra = gra_df[gra_df['Borehole_ID'] == bh_id].copy()
        bh_ms = ms_df[ms_df['Borehole_ID'] == bh_id].copy()
        bh_ngr = ngr_df[ngr_df['Borehole_ID'] == bh_id].copy()
        bh_rgb = rgb_df[rgb_df['Borehole_ID'] == bh_id].copy()

        if len(bh_gra) == 0 or len(bh_ms) == 0 or len(bh_ngr) == 0 or len(bh_rgb) == 0:
            continue

        # Get depth range
        min_depth = max(bh_gra['Depth CSF-A (m)'].min(), bh_ms['Depth CSF-A (m)'].min(),
                       bh_ngr['Depth CSF-A (m)'].min(), bh_rgb['Depth CSF-A (m)'].min())
        max_depth = min(bh_gra['Depth CSF-A (m)'].max(), bh_ms['Depth CSF-A (m)'].max(),
                       bh_ngr['Depth CSF-A (m)'].max(), bh_rgb['Depth CSF-A (m)'].max())

        if max_depth <= min_depth:
            continue

        # Create bin centers every 20cm
        bin_centers = np.arange(np.floor(min_depth / bin_size) * bin_size,
                               np.ceil(max_depth / bin_size) * bin_size,
                               bin_size)

        for bin_center in bin_centers:
            # Find measurements within tolerance of bin center
            gra_match = bh_gra[np.abs(bh_gra['Depth CSF-A (m)'] - bin_center) <= tolerance_m]
            ms_match = bh_ms[np.abs(bh_ms['Depth CSF-A (m)'] - bin_center) <= tolerance_m]
            ngr_match = bh_ngr[np.abs(bh_ngr['Depth CSF-A (m)'] - bin_center) <= tolerance_m]
            rgb_match = bh_rgb[np.abs(bh_rgb['Depth CSF-A (m)'] - bin_center) <= tolerance_m]

            # All four measurements must be present
            if len(gra_match) == 0 or len(ms_match) == 0 or len(ngr_match) == 0 or len(rgb_match) == 0:
                continue

            # Take closest measurement (or average if multiple)
            gra_val = gra_match.iloc[(gra_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['Bulk density (GRA)']
            ms_val = ms_match.iloc[(ms_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['Magnetic susceptibility (instr. units)']
            ngr_val = ngr_match.iloc[(ngr_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['NGR total counts (cps)']

            # RGB: average R, G, B from closest measurement
            rgb_closest = rgb_match.iloc[(rgb_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]
            r_val = rgb_closest['R']
            g_val = rgb_closest['G']
            b_val = rgb_closest['B']

            # Get lithology from closest measurement with lithology
            # Prefer GRA for lithology (most continuous)
            principal = gra_match.iloc[(gra_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['Principal']

            results.append({
                'Borehole_ID': bh_id,
                'Depth_Bin': int(bin_center / bin_size),
                'Bulk density (GRA)': gra_val,
                'Magnetic susceptibility (instr. units)': ms_val,
                'NGR total counts (cps)': ngr_val,
                'R': r_val,
                'G': g_val,
                'B': b_val,
                'Principal': principal
            })

    return pd.DataFrame(results)

# Test each tolerance on a sample of boreholes first (for speed)
print("="*100)
print("QUICK TEST ON 20 SAMPLE BOREHOLES")
print("="*100)
print()

sample_bh = target_boreholes[:20]
quick_results = []

for tolerance in tolerances:
    print(f"Testing tolerance ± {tolerance*100:.0f} cm...")
    start_time = time.time()

    df = create_fuzzy_dataset(tolerance, sample_boreholes=sample_bh)

    elapsed = time.time() - start_time

    quick_results.append({
        'tolerance_m': tolerance,
        'tolerance_cm': tolerance * 100,
        'n_samples': len(df),
        'n_boreholes': df['Borehole_ID'].nunique(),
        'n_lithologies': df['Principal'].nunique(),
        'time_seconds': elapsed
    })

    print(f"  Samples: {len(df):>6,d}  Boreholes: {df['Borehole_ID'].nunique():>3d}  "
          f"Lithologies: {df['Principal'].nunique():>3d}  Time: {elapsed:>5.1f}s")

print()

# Display results
quick_df = pd.DataFrame(quick_results)
print("Quick test results (20 boreholes):")
print()
print(f"{'Tolerance':>12s} {'Samples':>10s} {'vs ±20cm':>10s} {'Lithologies':>12s}")
print("-"*100)

baseline_samples = quick_df[quick_df['tolerance_cm'] == 20.0]['n_samples'].values[0]

for _, row in quick_df.iterrows():
    pct_change = (row['n_samples'] - baseline_samples) / baseline_samples * 100
    print(f"± {row['tolerance_cm']:>6.0f} cm   {int(row['n_samples']):>8,d}   {pct_change:>+8.1f}%   "
          f"{int(row['n_lithologies']):>10d}")

print()

# Estimate full dataset sizes
print("="*100)
print("ESTIMATED FULL DATASET SIZES (ALL 296 BOREHOLES)")
print("="*100)
print()

scale_factor = len(target_boreholes) / len(sample_bh)

print(f"{'Tolerance':>12s} {'Estimated Samples':>18s} {'vs Baseline':>12s}")
print("-"*100)

baseline_20cm = 238506  # Current v2.6.7 dataset size

for _, row in quick_df.iterrows():
    estimated_full = int(row['n_samples'] * scale_factor)
    pct_vs_baseline = (estimated_full - baseline_20cm) / baseline_20cm * 100

    print(f"± {row['tolerance_cm']:>6.0f} cm   {estimated_full:>16,d}   {pct_vs_baseline:>+10.1f}%")

print()
print(f"Current v2.6.7 baseline: {baseline_20cm:,} samples")
print()

# Select promising tolerances for full dataset creation
promising_tolerances = [0.10, 0.20, 0.50, 1.0, 2.0]  # Based on quick test

print("="*100)
print("CREATING FULL DATASETS FOR PROMISING TOLERANCES")
print("="*100)
print()

full_results = []

for tolerance in promising_tolerances:
    print(f"\nCreating full dataset with tolerance ± {tolerance*100:.0f} cm...")
    start_time = time.time()

    df = create_fuzzy_dataset(tolerance, sample_boreholes=target_boreholes)

    elapsed = time.time() - start_time

    # Save dataset
    filename = f'vae_training_data_fuzzy_{int(tolerance*100)}cm.csv'
    df.to_csv(filename, index=False)

    full_results.append({
        'tolerance_m': tolerance,
        'tolerance_cm': tolerance * 100,
        'n_samples': len(df),
        'n_boreholes': df['Borehole_ID'].nunique(),
        'n_lithologies': df['Principal'].nunique(),
        'time_seconds': elapsed,
        'filename': filename
    })

    print(f"  ✓ Created {len(df):,} samples")
    print(f"  ✓ {df['Borehole_ID'].nunique()} boreholes")
    print(f"  ✓ {df['Principal'].nunique()} unique lithologies")
    print(f"  ✓ Saved to {filename}")
    print(f"  ✓ Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

print()
print("="*100)
print("FULL DATASET RESULTS")
print("="*100)
print()

full_df = pd.DataFrame(full_results)

print(f"{'Tolerance':>12s} {'Samples':>12s} {'vs Baseline':>12s} {'Boreholes':>10s} {'File':>40s}")
print("-"*100)

for _, row in full_df.iterrows():
    pct_change = (row['n_samples'] - baseline_20cm) / baseline_20cm * 100
    print(f"± {row['tolerance_cm']:>6.0f} cm   {row['n_samples']:>10,d}   {pct_change:>+10.1f}%   "
          f"{row['n_boreholes']:>8d}   {row['filename']:>40s}")

print()
print(f"Baseline (v2.6.7):     {baseline_20cm:>10,d}        0.0%        296")
print()

print("="*100)
print("NEXT STEPS")
print("="*100)
print()
print("1. Train VAE models on each fuzzy-matched dataset")
print("2. Compare clustering performance (ARI) vs sample count")
print("3. Find optimal tolerance that maximizes ARI")
print()
print("Expected trade-off:")
print("  - Larger tolerance → More samples → But measurements farther apart")
print("  - Smaller tolerance → Fewer samples → But tighter depth alignment")
print()
print("Geologically reasonable tolerances:")
print("  ± 10-20 cm: Measurement precision + core disturbance")
print("  ± 50-100 cm: Within same lithological unit typically")
print("  ± 1-2 m: May span lithological boundaries")
print("  ± 5 m: Definitely spans multiple units")
print()
print("Recommendation: Test ARI on ±10cm, ±20cm, ±50cm, ±1m, ±2m datasets")
print()
print("="*100)
