"""Debug why v2.7 merge produces 0 samples."""

import pandas as pd
import numpy as np

# Load datasets
print("Loading v2.6 dataset...")
v2_6_df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"v2.6: {len(v2_6_df):,} samples, {v2_6_df['Borehole_ID'].nunique()} boreholes")

print("\nLoading MAD dataset...")
mad_df = pd.read_csv('/home/utig5/johna/bhai/datasets/MAD_DataLITH.csv')
mad_df['Borehole_ID'] = mad_df['Exp'].astype(str) + '_' + mad_df['Site'].astype(str) + mad_df['Hole'].astype(str)
mad_df['Depth CSF-A (m)'] = pd.to_numeric(mad_df['Depth CSF-A (m)'], errors='coerce')
mad_df['depth_bin'] = (mad_df['Depth CSF-A (m)'] / 0.2).round() * 0.2
print(f"MAD: {len(mad_df):,} measurements, {mad_df['Borehole_ID'].nunique()} boreholes")

# Check borehole overlap
v2_6_boreholes = set(v2_6_df['Borehole_ID'].unique())
mad_boreholes = set(mad_df['Borehole_ID'].unique())
overlap_boreholes = v2_6_boreholes & mad_boreholes

print(f"\nBorehole overlap:")
print(f"  v2.6 boreholes: {len(v2_6_boreholes)}")
print(f"  MAD boreholes: {len(mad_boreholes)}")
print(f"  Overlap: {len(overlap_boreholes)}")

if len(overlap_boreholes) == 0:
    print("\n❌ ERROR: NO BOREHOLE OVERLAP!")
    print("\nSample v2.6 borehole IDs:")
    for bh in list(v2_6_boreholes)[:5]:
        print(f"  {bh}")
    print("\nSample MAD borehole IDs:")
    for bh in list(mad_boreholes)[:5]:
        print(f"  {bh}")
else:
    print(f"\n✓ {len(overlap_boreholes)} boreholes overlap")

    # Check depth bin overlap for a sample borehole
    sample_bh = list(overlap_boreholes)[0]
    print(f"\nChecking depth bins for sample borehole: {sample_bh}")

    v2_6_depths = sorted(v2_6_df[v2_6_df['Borehole_ID'] == sample_bh]['Depth_Bin'].unique())
    mad_depths = sorted(mad_df[mad_df['Borehole_ID'] == sample_bh]['depth_bin'].unique())

    print(f"  v2.6 depth bins: {len(v2_6_depths)} bins")
    print(f"    First 5: {v2_6_depths[:5]}")
    print(f"  MAD depth bins: {len(mad_depths)} bins")
    print(f"    First 5: {mad_depths[:5]}")

    # Check if any depths overlap
    v2_6_depth_set = set(v2_6_depths)
    mad_depth_set = set(mad_depths)
    depth_overlap = v2_6_depth_set & mad_depth_set

    print(f"\n  Depth bin overlap: {len(depth_overlap)} bins")
    if len(depth_overlap) > 0:
        print(f"    Sample overlapping depths: {sorted(depth_overlap)[:5]}")

    # Check data types
    print(f"\nData types:")
    print(f"  v2.6 Depth_Bin type: {v2_6_df['Depth_Bin'].dtype}")
    print(f"  MAD depth_bin type: {mad_df['depth_bin'].dtype}")

    # Try a fuzzy merge with tolerance
    print(f"\nTrying fuzzy merge with ±10cm tolerance...")

    # For one borehole, try to match depths
    v2_6_sample = v2_6_df[v2_6_df['Borehole_ID'] == sample_bh].head(10)
    mad_sample = mad_df[mad_df['Borehole_ID'] == sample_bh]

    matches = 0
    for idx, row in v2_6_sample.iterrows():
        depth = row['Depth_Bin']
        # Find MAD measurements within 10cm
        close_mad = mad_sample[abs(mad_sample['depth_bin'] - depth) <= 0.1]
        if len(close_mad) > 0:
            matches += 1
            print(f"  v2.6 depth {depth:.2f}m matches {len(close_mad)} MAD measurements")

    print(f"\nMatches: {matches}/10 v2.6 samples have MAD within 10cm")
