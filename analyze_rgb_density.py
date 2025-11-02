"""
Analyze why RGB has 10.8M measurements but limited coverage
"""
import pandas as pd
import numpy as np

print("="*80)
print("RGB Measurement Density Analysis")
print("="*80)
print()

# Load RGB data
print("Loading RGB data...")
rgb = pd.read_csv(
    '/home/utig5/johna/bhai/datasets/RGB_DataLITH.csv',
    usecols=['Exp', 'Site', 'Hole', 'Depth CSF-A (m)', 'R']
)

rgb['Borehole_ID'] = rgb['Exp'].astype(str) + '-' + rgb['Site'].astype(str) + rgb['Hole'].astype(str)
rgb = rgb.dropna(subset=['Depth CSF-A (m)'])

print(f"Total RGB measurements: {len(rgb):,}")
print(f"Boreholes with RGB: {rgb['Borehole_ID'].nunique()}")
print()

# Analyze measurement density
print("RGB Measurement Density:")
print()

# Measurements per borehole
rgb_per_borehole = rgb.groupby('Borehole_ID').size()
print(f"Measurements per borehole:")
print(f"  Mean: {rgb_per_borehole.mean():,.0f}")
print(f"  Median: {rgb_per_borehole.median():,.0f}")
print(f"  Min: {rgb_per_borehole.min():,}")
print(f"  Max: {rgb_per_borehole.max():,}")
print()

# Depth spacing between consecutive measurements
print("Analyzing depth spacing...")
depth_diffs = []
for bh in rgb['Borehole_ID'].unique()[:20]:  # Sample 20 boreholes
    bh_data = rgb[rgb['Borehole_ID'] == bh]['Depth CSF-A (m)'].sort_values()
    if len(bh_data) > 1:
        diffs = bh_data.diff().dropna()
        depth_diffs.extend(diffs.values)

depth_diffs = np.array(depth_diffs)
depth_diffs = depth_diffs[depth_diffs > 0]  # Remove zeros

print(f"Depth spacing between consecutive RGB measurements (sample of 20 boreholes):")
print(f"  Mean: {depth_diffs.mean()*100:.2f} cm")
print(f"  Median: {np.median(depth_diffs)*100:.2f} cm")
print(f"  5th percentile: {np.percentile(depth_diffs, 5)*100:.2f} cm")
print()

print("="*80)
print("Comparison: Load GRA data (same analysis)")
print("="*80)
print()

print("Loading GRA data...")
gra = pd.read_csv(
    '/home/utig5/johna/bhai/datasets/GRA_DataLITH.csv',
    usecols=['Exp', 'Site', 'Hole', 'Depth CSF-A (m)', 'Bulk density (GRA)']
)

gra['Borehole_ID'] = gra['Exp'].astype(str) + '-' + gra['Site'].astype(str) + gra['Hole'].astype(str)
gra = gra.dropna(subset=['Depth CSF-A (m)'])

print(f"Total GRA measurements: {len(gra):,}")
print(f"Boreholes with GRA: {gra['Borehole_ID'].nunique()}")
print()

# Measurements per borehole
gra_per_borehole = gra.groupby('Borehole_ID').size()
print(f"Measurements per borehole:")
print(f"  Mean: {gra_per_borehole.mean():,.0f}")
print(f"  Median: {gra_per_borehole.median():,.0f}")
print(f"  Min: {gra_per_borehole.min():,}")
print(f"  Max: {gra_per_borehole.max():,}")
print()

print("="*80)
print("Key Insight")
print("="*80)
print()
print("RGB has 2.6× MORE measurements than GRA (10.8M vs 4.1M)")
print(f"  → RGB measurements are more dense (avg {rgb_per_borehole.mean():,.0f} per borehole)")
print(f"  → GRA measurements are sparser (avg {gra_per_borehole.mean():,.0f} per borehole)")
print()
print("BUT RGB is only collected on 304/534 boreholes (57%)")
print(f"  → Where RGB exists: {rgb_per_borehole.mean():,.0f} measurements/borehole")
print(f"  → Where GRA exists: {gra_per_borehole.mean():,.0f} measurements/borehole")
print()
print("This explains why:")
print("  1. RGB has 10.8M raw measurements (very high density)")
print("  2. But only 253,916 bins at 20cm (many measurements per bin)")
print("  3. And limited overlap with GRA/MS/NGR (only 57% of boreholes)")
print()

# Check overlap
print("="*80)
print("Checking which boreholes have both RGB and GRA")
print("="*80)
print()

rgb_boreholes = set(rgb['Borehole_ID'].unique())
gra_boreholes = set(gra['Borehole_ID'].unique())

both = rgb_boreholes & gra_boreholes
only_rgb = rgb_boreholes - gra_boreholes
only_gra = gra_boreholes - rgb_boreholes

print(f"Boreholes with RGB: {len(rgb_boreholes)}")
print(f"Boreholes with GRA: {len(gra_boreholes)}")
print(f"Boreholes with BOTH: {len(both)}")
print(f"Boreholes with only RGB: {len(only_rgb)}")
print(f"Boreholes with only GRA: {len(only_gra)}")
print()
print(f"RGB coverage of GRA boreholes: {100*len(both)/len(gra_boreholes):.1f}%")
