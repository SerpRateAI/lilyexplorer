"""
Analyze measurement coverage overlap to determine if proposed v2.7 features
exist on the same 296 boreholes as v2.6 (GRA+MS+NGR+RGB) or expand to new boreholes.

This answers: Are we adding feature richness (more measurements per sample)
or dataset expansion (more boreholes)?
"""

import pandas as pd
import numpy as np

print("="*100)
print("FEATURE COVERAGE OVERLAP ANALYSIS")
print("="*100)
print()

# Helper function to extract unique boreholes
def get_unique_boreholes(filepath, description):
    """Load dataset and return unique borehole IDs"""
    try:
        df = pd.read_csv(filepath, usecols=['Exp', 'Site', 'Hole'])
        df['Borehole_ID'] = df['Exp'].astype(str) + '_' + df['Site'].astype(str) + df['Hole'].astype(str)
        boreholes = set(df['Borehole_ID'].unique())
        print(f"{description:30s}: {len(boreholes):4d} boreholes")
        return boreholes
    except Exception as e:
        print(f"{description:30s}: ERROR - {e}")
        return set()

print("Loading measurement datasets...")
print()

# Existing v2.6 features
gra_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/GRA_DataLITH.csv', 'GRA (bulk density)')
ms_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/MS_DataLITH.csv', 'MS (magnetic susceptibility)')
ngr_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/NGR_DataLITH.csv', 'NGR (natural gamma)')
rgb_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/RGB_DataLITH.csv', 'RGB (color imaging)')

# Proposed new features
pwc_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/PWC_DataLITH.csv', 'PWC (P-wave velocity)')
thcn_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/THCN_DataLITH.csv', 'THCN (thermal conductivity)')
carb_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/CARB_DataLITH.csv', 'CARB (carbonate content)')
mad_boreholes = get_unique_boreholes('/home/utig5/johna/bhai/datasets/MAD_DataLITH.csv', 'MAD (porosity/grain density)')

print()
print("="*100)
print("V2.6 BASELINE COVERAGE")
print("="*100)
print()

# v2.6 requires GRA + MS + NGR + RGB
v2_6_boreholes = gra_boreholes & ms_boreholes & ngr_boreholes & rgb_boreholes
print(f"Boreholes with GRA+MS+NGR+RGB (v2.6): {len(v2_6_boreholes)}")
print()

print("="*100)
print("PROPOSED V2.7 FEATURE OVERLAPS")
print("="*100)
print()

# Check overlap of each new feature with v2.6 baseline
features = [
    ('P-wave velocity (PWC)', pwc_boreholes),
    ('Thermal conductivity (THCN)', thcn_boreholes),
    ('Carbonate content (CARB)', carb_boreholes),
    ('MAD (porosity/grain density)', mad_boreholes)
]

overlap_summary = []

for feature_name, feature_boreholes in features:
    overlap = v2_6_boreholes & feature_boreholes
    only_in_feature = feature_boreholes - v2_6_boreholes
    only_in_v2_6 = v2_6_boreholes - feature_boreholes

    overlap_pct = (len(overlap) / len(v2_6_boreholes)) * 100 if len(v2_6_boreholes) > 0 else 0

    print(f"{feature_name}:")
    print(f"  Total boreholes with this feature: {len(feature_boreholes)}")
    print(f"  Overlap with v2.6 baseline: {len(overlap)} ({overlap_pct:.1f}% of v2.6)")
    print(f"  New boreholes (has feature but not v2.6): {len(only_in_feature)}")
    print(f"  v2.6 boreholes missing this feature: {len(only_in_v2_6)}")
    print()

    overlap_summary.append({
        'Feature': feature_name,
        'Total_Boreholes': len(feature_boreholes),
        'Overlap_v2.6': len(overlap),
        'Overlap_Pct': overlap_pct,
        'New_Boreholes': len(only_in_feature),
        'v2.6_Missing': len(only_in_v2_6)
    })

print("="*100)
print("COMBINED FEATURE SET ANALYSIS")
print("="*100)
print()

# What if we require all proposed features?
all_features = v2_6_boreholes & pwc_boreholes & thcn_boreholes & carb_boreholes & mad_boreholes
print(f"Boreholes with ALL features (GRA+MS+NGR+RGB+PWC+THCN+CARB+MAD): {len(all_features)}")
print()

# What about different combinations?
print("Partial feature combinations:")
print()

# v2.6 + PWC
v2_6_pwc = v2_6_boreholes & pwc_boreholes
print(f"  v2.6 + P-wave velocity: {len(v2_6_pwc)} boreholes")

# v2.6 + MAD (for porosity/grain density)
v2_6_mad = v2_6_boreholes & mad_boreholes
print(f"  v2.6 + MAD (porosity): {len(v2_6_mad)} boreholes")

# v2.6 + CARB
v2_6_carb = v2_6_boreholes & carb_boreholes
print(f"  v2.6 + Carbonate content: {len(v2_6_carb)} boreholes")

# v2.6 + THCN
v2_6_thcn = v2_6_boreholes & thcn_boreholes
print(f"  v2.6 + Thermal conductivity: {len(v2_6_thcn)} boreholes")

print()

# Most permissive: v2.6 + at least one new feature
v2_6_with_any_new = v2_6_boreholes & (pwc_boreholes | thcn_boreholes | carb_boreholes | mad_boreholes)
print(f"  v2.6 + at least ONE new feature: {len(v2_6_with_any_new)} boreholes")

print()
print("="*100)
print("SUMMARY: FEATURE RICHNESS vs DATASET EXPANSION")
print("="*100)
print()

print(f"v2.6 baseline: {len(v2_6_boreholes)} boreholes with GRA+MS+NGR+RGB")
print()

# Create summary table
summary_df = pd.DataFrame(overlap_summary)
print("Overlap percentages:")
print(summary_df[['Feature', 'Overlap_v2.6', 'Overlap_Pct']].to_string(index=False))
print()

# Determine if we're adding richness or expansion
avg_overlap = summary_df['Overlap_Pct'].mean()
print(f"Average overlap with v2.6 baseline: {avg_overlap:.1f}%")
print()

if avg_overlap > 80:
    print("✓ CONCLUSION: Adding these features provides FEATURE RICHNESS")
    print("  Most new measurements exist on the same 296 boreholes.")
    print(f"  Expected v2.7 dataset: ~{len(v2_6_boreholes)} boreholes with 10-12 features")
elif avg_overlap < 30:
    print("✓ CONCLUSION: Adding these features provides DATASET EXPANSION")
    print("  New measurements exist on DIFFERENT boreholes.")
    print("  Would need to decide: keep v2.6 features only, or accept missing data")
else:
    print("⚠ CONCLUSION: MIXED - some features add richness, others expand dataset")
    print("  Need to carefully select which features to include")
    print()
    print("Recommendation: Create multiple v2.7 variants:")
    print(f"  - v2.7a (maximal richness): {len(all_features)} boreholes, all features")
    print(f"  - v2.7b (balanced): {len(v2_6_with_any_new)} boreholes, v2.6 + selective features")
    print(f"  - v2.7c (expansion): Use features individually to maximize coverage")

print()
print("="*100)
