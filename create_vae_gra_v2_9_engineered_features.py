"""
Create VAE GRA v2.9 dataset with engineered features.

Takes v2 dataset (GRA + MS + NGR + RGB) and adds 6 derived features:
1. red_green_ratio: R/(G+1) - oxidation state (iron oxides are red)
2. brightness: (R+G+B)/3 - carbonate (bright) vs siliciclastic (dark)
3. chroma: color saturation - iron staining, weathering
4. density_contrast: local density heterogeneity vs depth trend
5. mag_gamma_interaction: MS * NGR - cross-property correlation
6. depth_normalized_gra: GRA with compaction trend removed

Total: 12D features (6 original + 6 derived)
Expected: Same 238K samples as v2, better discrimination from feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def compute_rolling_mean(series, window=5):
    """Compute rolling mean with edge handling."""
    return series.rolling(window=window, center=True, min_periods=1).mean()

def compute_engineered_features(df):
    """
    Compute 6 engineered features from raw measurements.

    Features are computed per borehole to avoid mixing different holes.
    """
    print("\nComputing engineered features...")

    # Extract raw features
    R = df['R'].values
    G = df['G'].values
    B = df['B'].values
    gra = df['Bulk density (GRA)'].values
    ms = df['Magnetic susceptibility (instr. units)'].values
    ngr = df['NGR total counts (cps)'].values
    depth = df['Depth CSF-A (m)'].values if 'Depth CSF-A (m)' in df.columns else df['Depth_Bin'].values

    # 1. Red-green ratio (oxidation state)
    # Red = oxidized iron (hematite), green = reduced/chlorite
    red_green_ratio = R / (G + 1)  # +1 to avoid division by zero

    # 2. Brightness (overall reflectance)
    # Carbonates are bright, organic-rich clays are dark
    brightness = (R + G + B) / 3

    # 3. Chroma (color saturation)
    # Euclidean distance in RGB space from neutral gray
    # High chroma = vivid colors (iron staining, oxidation)
    chroma = np.sqrt((R - G)**2 + (G - B)**2 + (B - R)**2)

    # Initialize arrays for per-borehole features
    density_contrast = np.zeros(len(df))
    mag_gamma_interaction = ms * ngr  # Can compute globally
    depth_normalized_gra = np.zeros(len(df))

    # Compute per-borehole features (rolling mean, depth trends)
    print("  Computing per-borehole density contrast and depth normalization...")
    for borehole_id in df['Borehole_ID'].unique():
        mask = df['Borehole_ID'] == borehole_id
        indices = np.where(mask)[0]

        if len(indices) < 5:
            # Too few samples for rolling mean
            density_contrast[mask] = 0
            depth_normalized_gra[mask] = 0
            continue

        # Get data for this borehole
        bh_gra = gra[mask]
        bh_depth = depth[mask]

        # Sort by depth for rolling operations
        sort_idx = np.argsort(bh_depth)
        bh_gra_sorted = bh_gra[sort_idx]
        bh_depth_sorted = bh_depth[sort_idx]

        # 4. Density contrast (local heterogeneity)
        # Deviation from local rolling mean
        rolling_gra = pd.Series(bh_gra_sorted).rolling(window=5, center=True, min_periods=1).mean().values
        contrast_sorted = bh_gra_sorted - rolling_gra

        # Unsort to match original order
        unsort_idx = np.argsort(sort_idx)
        density_contrast[indices] = contrast_sorted[unsort_idx]

        # 6. Depth-normalized GRA (remove compaction trend)
        # Fit linear trend: GRA ~ depth
        if len(bh_depth_sorted) > 2:
            # Simple linear detrending
            coeffs = np.polyfit(bh_depth_sorted, bh_gra_sorted, deg=1)
            trend = np.polyval(coeffs, bh_depth_sorted)
            residuals = bh_gra_sorted - trend

            # Normalize by std if non-zero
            std = residuals.std()
            if std > 1e-6:
                normalized = residuals / std
            else:
                normalized = residuals

            depth_normalized_gra[indices] = normalized[unsort_idx]
        else:
            depth_normalized_gra[indices] = 0

    # Add to dataframe
    df['red_green_ratio'] = red_green_ratio
    df['brightness'] = brightness
    df['chroma'] = chroma
    df['density_contrast'] = density_contrast
    df['mag_gamma_interaction'] = mag_gamma_interaction
    df['depth_normalized_gra'] = depth_normalized_gra

    print("\nEngineered feature statistics:")
    eng_features = ['red_green_ratio', 'brightness', 'chroma', 'density_contrast',
                    'mag_gamma_interaction', 'depth_normalized_gra']
    print(df[eng_features].describe())

    return df

def main():
    """Create v2.9 dataset with engineered features."""
    print("="*80)
    print("CREATE VAE GRA v2.9 DATASET WITH ENGINEERED FEATURES")
    print("="*80)

    start_time = time.time()

    # Load v2 dataset
    print("\nLoading v2 dataset...")
    v2_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    df = pd.read_csv(v2_path)

    print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

    # Compute engineered features
    df = compute_engineered_features(df)

    # Summary
    print("\n" + "="*80)
    print("DATASET SUMMARY")
    print("="*80)
    print(f"Total samples: {len(df):,}")
    print(f"Unique boreholes: {df['Borehole_ID'].nunique()}")
    print(f"Unique principal lithologies: {df['Principal'].nunique()}")

    print("\nOriginal features (6D):")
    orig_features = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                     'NGR total counts (cps)', 'R', 'G', 'B']
    print(df[orig_features].describe())

    print("\nEngineered features (6D):")
    eng_features = ['red_green_ratio', 'brightness', 'chroma', 'density_contrast',
                    'mag_gamma_interaction', 'depth_normalized_gra']
    print(df[eng_features].describe())

    print("\nTotal: 12D feature space")

    # Save dataset
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_9_engineered.csv')
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    print(f"\nDataset saved to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"Time elapsed: {elapsed:.1f}s")

    print("\n" + "="*80)
    print("DATASET CREATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
