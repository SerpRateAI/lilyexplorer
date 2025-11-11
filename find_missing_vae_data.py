"""
Find if there's more IODP data available that's not in VAE v2.6.7.

Checks:
1. How many boreholes have GRA+MS+NGR+RGB measurements
2. Why only 296/534 boreholes are included
3. If we can extract more data with different strategies
"""

import pandas as pd
import numpy as np

print("="*100)
print("FINDING MISSING VAE DATA")
print("="*100)
print()

# Load current VAE dataset
print("Loading current VAE v2.6.7 dataset...")
vae_df = pd.read_csv('vae_training_data_v2_20cm.csv')
current_boreholes = set(vae_df['Borehole_ID'].unique())
print(f"Current dataset: {len(vae_df):,} samples from {len(current_boreholes)} boreholes")
print()

# Load raw datasets
print("Loading raw IODP datasets...")
print("  Loading GRA...")
gra_df = pd.read_csv('datasets/GRA_DataLITH.csv', low_memory=False)
gra_df['Borehole_ID'] = gra_df['Exp'].astype(str) + '_' + gra_df['Site'].astype(str) + '_' + gra_df['Hole'].astype(str)

print("  Loading MS...")
ms_df = pd.read_csv('datasets/MS_DataLITH.csv', low_memory=False)
ms_df['Borehole_ID'] = ms_df['Exp'].astype(str) + '_' + ms_df['Site'].astype(str) + '_' + ms_df['Hole'].astype(str)

print("  Loading NGR...")
ngr_df = pd.read_csv('datasets/NGR_DataLITH.csv', low_memory=False)
ngr_df['Borehole_ID'] = ngr_df['Exp'].astype(str) + '_' + ngr_df['Site'].astype(str) + '_' + ngr_df['Hole'].astype(str)

print("  Loading RGB...")
rgb_df = pd.read_csv('datasets/RGB_DataLITH.csv', low_memory=False)
rgb_df['Borehole_ID'] = rgb_df['Exp'].astype(str) + '_' + rgb_df['Site'].astype(str) + '_' + rgb_df['Hole'].astype(str)

print()

# Get unique boreholes per dataset
gra_boreholes = set(gra_df['Borehole_ID'].unique())
ms_boreholes = set(ms_df['Borehole_ID'].unique())
ngr_boreholes = set(ngr_df['Borehole_ID'].unique())
rgb_boreholes = set(rgb_df['Borehole_ID'].unique())

print("Raw dataset borehole counts:")
print(f"  GRA: {len(gra_boreholes)} boreholes")
print(f"  MS:  {len(ms_boreholes)} boreholes")
print(f"  NGR: {len(ngr_boreholes)} boreholes")
print(f"  RGB: {len(rgb_boreholes)} boreholes")
print()

# Find intersections
all_four = gra_boreholes & ms_boreholes & ngr_boreholes & rgb_boreholes
print(f"Boreholes with ALL four measurements: {len(all_four)}")
print(f"Current VAE dataset uses: {len(current_boreholes)}")
print()

if len(all_four) > len(current_boreholes):
    missing_boreholes = all_four - current_boreholes
    print(f"⚠ MISSING BOREHOLES: {len(missing_boreholes)} boreholes have all 4 measurements but aren't in VAE dataset!")
    print()
    print("Missing boreholes:")
    for i, bh in enumerate(sorted(missing_boreholes)[:20], 1):
        print(f"  {i:3d}. {bh}")
    if len(missing_boreholes) > 20:
        print(f"  ... and {len(missing_boreholes)-20} more")
    print()

    # Estimate potential additional samples
    print("Estimating potential additional samples...")

    # For each missing borehole, count measurements
    potential_samples = 0
    for bh in list(missing_boreholes)[:10]:  # Check first 10
        bh_gra = gra_df[gra_df['Borehole_ID'] == bh]
        bh_ms = ms_df[ms_df['Borehole_ID'] == bh]
        bh_ngr = ngr_df[ngr_df['Borehole_ID'] == bh]
        bh_rgb = rgb_df[rgb_df['Borehole_ID'] == bh]

        # Check depth overlap (simplified)
        min_depth = max(bh_gra['Depth CSF-A (m)'].min(), bh_ms['Depth CSF-A (m)'].min(),
                       bh_ngr['Depth CSF-A (m)'].min(), bh_rgb['Depth CSF-A (m)'].min())
        max_depth = min(bh_gra['Depth CSF-A (m)'].max(), bh_ms['Depth CSF-A (m)'].max(),
                       bh_ngr['Depth CSF-A (m)'].max(), bh_rgb['Depth CSF-A (m)'].max())

        if max_depth > min_depth:
            depth_range = max_depth - min_depth
            estimated_bins = int(depth_range / 0.2)
            potential_samples += estimated_bins
            print(f"  {bh}: {depth_range:.1f}m range → ~{estimated_bins} potential bins")

    print(f"\n  Estimated from 10 sample boreholes: ~{potential_samples} bins")
    print(f"  Extrapolated to all {len(missing_boreholes)} missing boreholes: ~{potential_samples * len(missing_boreholes) / 10:,.0f} potential samples")
    print()

else:
    print("✓ All boreholes with all 4 measurements are already included")
    print()

# Check what's limiting the dataset
print("="*100)
print("WHAT'S LIMITING THE DATASET?")
print("="*100)
print()

# For each current borehole, check why some might have low sample counts
bh_coverage = []
for bh in list(current_boreholes)[:50]:  # Sample 50 boreholes
    vae_bh = vae_df[vae_df['Borehole_ID'] == bh]
    n_vae_samples = len(vae_bh)

    # Get raw measurement counts
    n_gra = len(gra_df[gra_df['Borehole_ID'] == bh])
    n_ms = len(ms_df[ms_df['Borehole_ID'] == bh])
    n_ngr = len(ngr_df[ngr_df['Borehole_ID'] == bh])
    n_rgb = len(rgb_df[rgb_df['Borehole_ID'] == bh])

    bh_coverage.append({
        'borehole': bh,
        'vae_samples': n_vae_samples,
        'gra_measurements': n_gra,
        'ms_measurements': n_ms,
        'ngr_measurements': n_ngr,
        'rgb_measurements': n_rgb,
        'min_measurements': min(n_gra, n_ms, n_ngr, n_rgb)
    })

coverage_df = pd.DataFrame(bh_coverage)

# Find bottleneck measurement
coverage_df['bottleneck'] = coverage_df[['gra_measurements', 'ms_measurements',
                                         'ngr_measurements', 'rgb_measurements']].idxmin(axis=1)

print("Bottleneck analysis (which measurement limits each borehole):")
bottleneck_counts = coverage_df['bottleneck'].value_counts()
for bottleneck, count in bottleneck_counts.items():
    measurement = bottleneck.replace('_measurements', '').upper()
    print(f"  {measurement}: {count}/50 boreholes ({count/50*100:.0f}%)")
print()

# Show examples where RGB is the bottleneck
rgb_bottleneck = coverage_df[coverage_df['bottleneck'] == 'rgb_measurements'].head(5)
if len(rgb_bottleneck) > 0:
    print("Example boreholes where RGB is the limiting factor:")
    for _, row in rgb_bottleneck.iterrows():
        print(f"  {row['borehole']}: GRA={row['gra_measurements']:,}, MS={row['ms_measurements']:,}, "
              f"NGR={row['ngr_measurements']:,}, RGB={row['rgb_measurements']:,} ← bottleneck")
    print()

print("="*100)
print("DEPTH BINNING STRATEGY")
print("="*100)
print()

print("Current strategy: 20cm bins with exact alignment")
print()
print("Could we get more data by:")
print("1. Using larger bins (e.g., 50cm, 1m)?")
print("2. Allowing fuzzy matching (±10cm tolerance)?")
print("3. Interpolating missing measurements?")
print()
print("Let me check depth spacing in raw data...")
print()

# Sample GRA and RGB depth spacing
sample_bh = list(current_boreholes)[0]
sample_gra = gra_df[gra_df['Borehole_ID'] == sample_bh]['Depth CSF-A (m)'].values
sample_rgb = rgb_df[rgb_df['Borehole_ID'] == sample_bh]['Depth CSF-A (m)'].values

if len(sample_gra) > 1:
    gra_spacing = np.diff(sorted(sample_gra[:100]))  # First 100 measurements
    print(f"GRA typical spacing: {np.median(gra_spacing):.4f}m (median), {np.mean(gra_spacing):.4f}m (mean)")

if len(sample_rgb) > 1:
    rgb_spacing = np.diff(sorted(sample_rgb[:100]))
    print(f"RGB typical spacing: {np.median(rgb_spacing):.4f}m (median), {np.mean(rgb_spacing):.4f}m (mean)")

print()
print("="*100)
print("RECOMMENDATIONS")
print("="*100)
print()

if len(all_four) > len(current_boreholes):
    print(f"✓ IMMEDIATE ACTION: Add {len(missing_boreholes)} missing boreholes with all 4 measurements")
    print(f"  - Potential increase: ~{potential_samples * len(missing_boreholes) / 10:,.0f} samples")
    print(f"  - This is 'free' data using the same approach")
    print()

print("Additional strategies to explore:")
print("1. Re-run dataset creation script to capture all boreholes")
print("2. Investigate why some boreholes were excluded (QC filters? missing lithology?)")
print("3. Check if depth binning tolerance can be relaxed slightly")
print("4. Consider 50cm bins as alternative (trade resolution for coverage)")
print()
print("="*100)
