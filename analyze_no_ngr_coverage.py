"""
Analyze coverage if we remove NGR (the bottleneck).
"""

import pandas as pd

print("Loading raw datasets...")

# GRA
print("  GRA...")
gra_df = pd.read_csv('datasets/GRA_DataLITH.csv',
                     usecols=['Exp', 'Site', 'Hole'],
                     low_memory=False)
gra_df['Borehole_ID'] = gra_df['Exp'].astype(str) + '-' + gra_df['Site'].astype(str) + '-' + gra_df['Hole'].astype(str)
gra_boreholes = set(gra_df['Borehole_ID'].unique())

# MS
print("  MS...")
ms_df = pd.read_csv('datasets/MS_DataLITH.csv',
                    usecols=['Exp', 'Site', 'Hole'],
                    low_memory=False)
ms_df['Borehole_ID'] = ms_df['Exp'].astype(str) + '-' + ms_df['Site'].astype(str) + '-' + ms_df['Hole'].astype(str)
ms_boreholes = set(ms_df['Borehole_ID'].unique())

# RGB
print("  RGB...")
rgb_df = pd.read_csv('datasets/RGB_DataLITH.csv',
                     usecols=['Exp', 'Site', 'Hole'],
                     low_memory=False)
rgb_df['Borehole_ID'] = rgb_df['Exp'].astype(str) + '-' + rgb_df['Site'].astype(str) + '-' + rgb_df['Hole'].astype(str)
rgb_boreholes = set(rgb_df['Borehole_ID'].unique())

# NGR
print("  NGR...")
ngr_df = pd.read_csv('datasets/NGR_DataLITH.csv',
                     usecols=['Exp', 'Site', 'Hole'],
                     low_memory=False)
ngr_df['Borehole_ID'] = ngr_df['Exp'].astype(str) + '-' + ngr_df['Site'].astype(str) + '-' + ngr_df['Hole'].astype(str)
ngr_boreholes = set(ngr_df['Borehole_ID'].unique())

print("\n" + "="*80)
print("BOREHOLE COVERAGE")
print("="*80)

print(f"\nTotal boreholes with each measurement:")
print(f"  GRA: {len(gra_boreholes):3d}")
print(f"  MS:  {len(ms_boreholes):3d}")
print(f"  RGB: {len(rgb_boreholes):3d}")
print(f"  NGR: {len(ngr_boreholes):3d}")

# GRA+MS+RGB (no NGR)
gra_ms_rgb = gra_boreholes & ms_boreholes & rgb_boreholes
print(f"\nGRA+MS+RGB (no NGR): {len(gra_ms_rgb)} boreholes")

# GRA+MS+NGR+RGB (current v2.6.7)
all_four = gra_boreholes & ms_boreholes & ngr_boreholes & rgb_boreholes
print(f"GRA+MS+NGR+RGB (v2.6.7): {len(all_four)} boreholes")

print(f"\n→ Removing NGR gains {len(gra_ms_rgb) - len(all_four)} boreholes (+{(len(gra_ms_rgb) - len(all_four)) / len(all_four) * 100:.0f}%)")

# Load VAE dataset to estimate sample gain
vae_df = pd.read_csv('vae_training_data_v2_20cm.csv')
avg_samples_per_bh = len(vae_df) / vae_df['Borehole_ID'].nunique()

estimated_samples_no_ngr = len(gra_ms_rgb) * avg_samples_per_bh

print(f"\n" + "="*80)
print("ESTIMATED SAMPLE COUNTS")
print("="*80)
print(f"\nv2.6.7 (GRA+MS+NGR+RGB):")
print(f"  Boreholes: {len(all_four)}")
print(f"  Samples: {len(vae_df):,}")
print(f"  Avg samples/borehole: {avg_samples_per_bh:.0f}")

print(f"\nv2.6.9 Stage 1 (GRA+MS+RGB, no NGR):")
print(f"  Boreholes: {len(gra_ms_rgb)}")
print(f"  Estimated samples: {estimated_samples_no_ngr:,.0f}")
print(f"  → +{(estimated_samples_no_ngr - len(vae_df)) / len(vae_df) * 100:.0f}% more pre-training data")

print(f"\nv2.6.9 Stage 2 (add NGR):")
print(f"  Boreholes: {len(all_four)} (same as v2.6.7)")
print(f"  Samples: {len(vae_df):,} (same as v2.6.7)")

print(f"\n" + "="*80)
print("v2.6.9 STRATEGY")
print("="*80)
print("""
Stage 1: Pre-train on GRA+MS+RGB (no NGR)
  • ~300 boreholes (+4 vs v2.6.7)
  • ~240K samples (similar to v2.6.7, slight gain)
  • Learns GRA-MS-RGB correlations

Stage 2: Expand input to include NGR, fine-tune
  • Same 296 boreholes as v2.6.7
  • Same 239K samples as v2.6.7
  • NGR pathway initialized randomly, others frozen or low LR

Expected outcome:
  • Pre-training on cross-modal GRA+MS+RGB (not just physical)
  • NGR added as supplementary feature (not critical like RGB)
  • Different from v2.6.2 (which added RGB, the key feature)

Risk:
  • All transfer learning failed (-50% to -79%)
  • Pre-training optimizes for 3-feature reconstruction
  • Adding 4th feature may break learned representations
  • Lesson: "joint training from scratch is optimal"

Worth testing because:
  • Only +4 boreholes (not much gain actually!)
  • But conceptually different: adding NGR not RGB
  • NGR less critical than RGB for visual discrimination
""")
