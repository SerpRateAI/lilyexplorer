"""
Create VAE v2.6.10 dataset with predicted RGB.

Dataset composition:
- 296 boreholes with real RGB camera data
- 229 boreholes with predicted RGB (from GRA+MS+NGR)
- Total: 525 boreholes (+77%)
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tqdm import tqdm

print("="*80)
print("VAE v2.6.10 DATASET CREATION")
print("="*80)
print()
print("Strategy: Combine real RGB (296 BH) + predicted RGB (229 BH)")
print()

# Load RGB predictors
print("Loading RGB prediction models...")
model_r = CatBoostRegressor()
model_r.load_model('ml_models/rgb_predictor_r.cbm')
model_g = CatBoostRegressor()
model_g.load_model('ml_models/rgb_predictor_g.cbm')
model_b = CatBoostRegressor()
model_b.load_model('ml_models/rgb_predictor_b.cbm')
print("✓ Models loaded")

# Load existing v2.6.7 dataset (296 boreholes with real RGB)
print("\nLoading v2.6.7 dataset (real RGB)...")
real_rgb_df = pd.read_csv('vae_training_data_v2_20cm.csv')
real_rgb_boreholes = set(real_rgb_df['Borehole_ID'].unique())
print(f"✓ {len(real_rgb_df):,} samples from {len(real_rgb_boreholes)} boreholes")

# Load raw datasets to find boreholes with GRA+MS+NGR but no RGB
print("\nLoading raw datasets...")

print("  GRA...")
gra_df = pd.read_csv('datasets/GRA_DataLITH.csv', low_memory=False)
gra_df['Borehole_ID'] = gra_df['Exp'].astype(str) + '-' + gra_df['Site'].astype(str) + '-' + gra_df['Hole'].astype(str)

print("  MS...")
ms_df = pd.read_csv('datasets/MS_DataLITH.csv', low_memory=False)
ms_df['Borehole_ID'] = ms_df['Exp'].astype(str) + '-' + ms_df['Site'].astype(str) + '-' + ms_df['Hole'].astype(str)

print("  NGR...")
ngr_df = pd.read_csv('datasets/NGR_DataLITH.csv', low_memory=False)
ngr_df['Borehole_ID'] = ngr_df['Exp'].astype(str) + '-' + ngr_df['Site'].astype(str) + '-' + ngr_df['Hole'].astype(str)

# Find boreholes with GRA+MS+NGR but NOT RGB
gra_boreholes = set(gra_df['Borehole_ID'].unique())
ms_boreholes = set(ms_df['Borehole_ID'].unique())
ngr_boreholes = set(ngr_df['Borehole_ID'].unique())

# Boreholes with all three physical measurements but no RGB camera
predict_rgb_boreholes = (gra_boreholes & ms_boreholes & ngr_boreholes) - real_rgb_boreholes
predict_rgb_boreholes = sorted(list(predict_rgb_boreholes))

print(f"\nBoreholes to add (GRA+MS+NGR, no RGB camera): {len(predict_rgb_boreholes)}")

# Create dataset for boreholes needing RGB prediction
print("\nCreating dataset with predicted RGB...")

def create_dataset_with_predicted_rgb(boreholes):
    """
    Create 20cm binned dataset with predicted RGB for given boreholes.
    """
    bin_size = 0.2  # 20cm bins
    results = []

    for bh_id in tqdm(boreholes, desc="Processing boreholes"):
        # Get data for this borehole
        bh_gra = gra_df[gra_df['Borehole_ID'] == bh_id].copy()
        bh_ms = ms_df[ms_df['Borehole_ID'] == bh_id].copy()
        bh_ngr = ngr_df[ngr_df['Borehole_ID'] == bh_id].copy()

        if len(bh_gra) == 0 or len(bh_ms) == 0 or len(bh_ngr) == 0:
            continue

        # Get depth range
        min_depth = max(bh_gra['Depth CSF-A (m)'].min(),
                       bh_ms['Depth CSF-A (m)'].min(),
                       bh_ngr['Depth CSF-A (m)'].min())
        max_depth = min(bh_gra['Depth CSF-A (m)'].max(),
                       bh_ms['Depth CSF-A (m)'].max(),
                       bh_ngr['Depth CSF-A (m)'].max())

        if max_depth <= min_depth:
            continue

        # Create 20cm bins
        bins = np.arange(np.floor(min_depth / bin_size) * bin_size,
                        np.ceil(max_depth / bin_size) * bin_size + bin_size,
                        bin_size)

        for i in range(len(bins) - 1):
            bin_start = bins[i]
            bin_end = bins[i + 1]
            bin_center = (bin_start + bin_end) / 2

            # Find measurements in this bin
            gra_in_bin = bh_gra[(bh_gra['Depth CSF-A (m)'] >= bin_start) &
                               (bh_gra['Depth CSF-A (m)'] < bin_end)]
            ms_in_bin = bh_ms[(bh_ms['Depth CSF-A (m)'] >= bin_start) &
                             (bh_ms['Depth CSF-A (m)'] < bin_end)]
            ngr_in_bin = bh_ngr[(bh_ngr['Depth CSF-A (m)'] >= bin_start) &
                               (bh_ngr['Depth CSF-A (m)'] < bin_end)]

            # All three measurements must be present
            if len(gra_in_bin) == 0 or len(ms_in_bin) == 0 or len(ngr_in_bin) == 0:
                continue

            # Average measurements in bin
            gra_val = gra_in_bin['Bulk density (GRA)'].mean()
            ms_val = ms_in_bin['Magnetic susceptibility (instr. units)'].mean()
            ngr_val = ngr_in_bin['NGR total counts (cps)'].mean()

            # Get lithology (from GRA)
            principal = gra_in_bin['Principal'].mode()[0] if len(gra_in_bin['Principal'].mode()) > 0 else gra_in_bin['Principal'].iloc[0]

            results.append({
                'Borehole_ID': bh_id,
                'Depth_Bin': int(bin_center / bin_size),
                'Bulk density (GRA)': gra_val,
                'Magnetic susceptibility (instr. units)': ms_val,
                'NGR total counts (cps)': ngr_val,
                'Principal': principal
            })

    return pd.DataFrame(results)

predicted_rgb_df = create_dataset_with_predicted_rgb(predict_rgb_boreholes)

print(f"\n✓ Created {len(predicted_rgb_df):,} samples from {predicted_rgb_df['Borehole_ID'].nunique()} boreholes")

# Predict RGB for new samples
print("\nPredicting RGB values...")
X_predict = predicted_rgb_df[['Bulk density (GRA)',
                              'Magnetic susceptibility (instr. units)',
                              'NGR total counts (cps)']].values

predicted_rgb_df['R'] = model_r.predict(X_predict)
predicted_rgb_df['G'] = model_g.predict(X_predict)
predicted_rgb_df['B'] = model_b.predict(X_predict)

# Clip RGB to valid range [0, 255]
predicted_rgb_df['R'] = predicted_rgb_df['R'].clip(0, 255)
predicted_rgb_df['G'] = predicted_rgb_df['G'].clip(0, 255)
predicted_rgb_df['B'] = predicted_rgb_df['B'].clip(0, 255)

print("✓ RGB predicted and clipped to [0, 255]")

# Add flag to distinguish real vs predicted RGB
real_rgb_df['RGB_Source'] = 'real'
predicted_rgb_df['RGB_Source'] = 'predicted'

# Combine datasets
print("\nCombining real + predicted RGB datasets...")
combined_df = pd.concat([real_rgb_df, predicted_rgb_df], ignore_index=True)

# Load lithology hierarchy for grouping
hierarchy_df = pd.read_csv('lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'],
                             hierarchy_df['Lithology_Group']))
combined_df['Lithology_Group'] = combined_df['Principal'].map(principal_to_group)

# Remove samples with NaN lithology
combined_df = combined_df.dropna(subset=['Lithology_Group'])

print(f"✓ Combined dataset created")

# Statistics
print("\n" + "="*80)
print("DATASET STATISTICS")
print("="*80)

print(f"\nTotal samples: {len(combined_df):,}")
print(f"Total boreholes: {combined_df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {combined_df['Lithology_Group'].nunique()}")

print(f"\nBreakdown by RGB source:")
print(f"  Real RGB:      {len(combined_df[combined_df['RGB_Source']=='real']):>8,} samples ({len(combined_df[combined_df['RGB_Source']=='real'])/len(combined_df)*100:5.1f}%)")
print(f"  Predicted RGB: {len(combined_df[combined_df['RGB_Source']=='predicted']):>8,} samples ({len(combined_df[combined_df['RGB_Source']=='predicted'])/len(combined_df)*100:5.1f}%)")

print(f"\nComparison to v2.6.7:")
print(f"  v2.6.7 samples:     {len(real_rgb_df):>8,}")
print(f"  v2.6.10 samples:    {len(combined_df):>8,} (+{(len(combined_df)-len(real_rgb_df))/len(real_rgb_df)*100:.0f}%)")
print(f"  v2.6.7 boreholes:   {len(real_rgb_boreholes):>8}")
print(f"  v2.6.10 boreholes:  {combined_df['Borehole_ID'].nunique():>8} (+{(combined_df['Borehole_ID'].nunique()-len(real_rgb_boreholes))/len(real_rgb_boreholes)*100:.0f}%)")

# Save combined dataset
output_file = 'vae_training_data_v2_6_10.csv'
combined_df.to_csv(output_file, index=False)
print(f"\n✓ Saved to {output_file}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Train VAE v2.6.10 with same architecture as v2.6.7:
   - 10D latent space
   - β annealing: 1e-10 → 0.75 over 50 epochs
   - Distribution-aware scaling

2. Compare performance:
   - v2.6.7: 239K samples, 296 BH, all real RGB
   - v2.6.10: ???K samples, 525 BH, mixed real/predicted RGB

3. Expected outcomes:
   ✓ Best case: +77% boreholes improves clustering despite RGB noise
   ⚠ Moderate: Similar performance (noise cancels data gain)
   ✗ Worst case: Predicted RGB degrades performance

This is a novel approach - different from all previous failures!
""")

print(f"\nCommand: python3 train_vae_v2_6_10.py")
