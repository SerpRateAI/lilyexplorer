"""
Create fuzzy matched dataset with ±20cm tolerance for VAE v2.6.8

Simplified version - creates only the ±20cm fuzzy dataset.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import time

print("="*100)
print("FUZZY MATCHING ±20cm DATASET CREATION")
print("="*100)
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

def create_fuzzy_dataset(tolerance_m, sample_boreholes):
    """
    Create dataset with fuzzy depth matching.

    For each 20cm depth bin center, find measurements within ±tolerance.
    Take the closest measurement within tolerance.
    """
    bin_size = 0.2  # 20cm bins
    results = []

    print(f"Creating fuzzy ±{tolerance_m*100:.0f}cm dataset for {len(sample_boreholes)} boreholes...")

    for bh_id in tqdm(sample_boreholes, desc="Processing boreholes"):
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

            # Take closest measurement
            gra_val = gra_match.iloc[(gra_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['Bulk density (GRA)']
            ms_val = ms_match.iloc[(ms_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['Magnetic susceptibility (instr. units)']
            ngr_val = ngr_match.iloc[(ngr_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]['NGR total counts (cps)']

            # RGB: closest measurement
            rgb_closest = rgb_match.iloc[(rgb_match['Depth CSF-A (m)'] - bin_center).abs().argmin()]
            r_val = rgb_closest['R']
            g_val = rgb_closest['G']
            b_val = rgb_closest['B']

            # Get lithology from closest GRA measurement
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

# Create fuzzy ±20cm dataset
print("="*100)
print("CREATING FUZZY ±20cm DATASET (ALL 296 BOREHOLES)")
print("="*100)
print()

start_time = time.time()

df = create_fuzzy_dataset(0.2, target_boreholes)  # 0.2m = 20cm

elapsed = time.time() - start_time

# Save dataset
filename = 'vae_training_data_fuzzy_20cm.csv'
df.to_csv(filename, index=False)

print()
print(f"✓ Created {len(df):,} samples")
print(f"✓ {df['Borehole_ID'].nunique()} boreholes")
print(f"✓ {df['Principal'].nunique()} unique lithologies")
print(f"✓ Saved to {filename}")
print(f"✓ Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print()

# Compare to baseline
baseline_20cm = 238506  # Current v2.6.7 exact matching
pct_change = (len(df) - baseline_20cm) / baseline_20cm * 100

print("="*100)
print("COMPARISON TO v2.6.7 BASELINE")
print("="*100)
print()
print(f"v2.6.7 (exact 20cm bins):  {baseline_20cm:>10,d} samples")
print(f"v2.6.8 (fuzzy ±20cm):       {len(df):>10,d} samples  ({pct_change:+.1f}%)")
print()

if pct_change > 0:
    print(f"✓ Fuzzy matching gained {pct_change:+.1f}% more samples!")
else:
    print(f"⚠ Fuzzy matching found {pct_change:.1f}% samples (similar to exact matching)")

print()
print("="*100)
print("NEXT: Train VAE v2.6.8 with fuzzy ±20cm dataset")
print("="*100)
print()
print(f"Command: python3 train_vae_v2_6_8_fuzzy_comparison.py --tolerance_cm 20 --gpu_id 0")
print()
