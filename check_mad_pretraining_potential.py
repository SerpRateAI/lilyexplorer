"""
Check if MAD bulk density could be used for pre-training VAE.

Analyzes:
1. MAD measurement coverage vs current VAE v2.6.7 boreholes
2. Potential sample increase from MAD
3. Why pre-training failed in v2.6.2 and v2.6.4
"""

import pandas as pd
import numpy as np

print("="*100)
print("MAD PRE-TRAINING POTENTIAL ANALYSIS")
print("="*100)
print()

# Load current VAE v2.6.7 training data
print("Loading current VAE v2.6.7 dataset...")
vae_df = pd.read_csv('vae_training_data_v2_20cm.csv')
current_boreholes = set(vae_df['Borehole_ID'].unique())
current_samples = set(zip(vae_df['Borehole_ID'], vae_df['Depth_Bin']))

print(f"Current VAE v2.6.7:")
print(f"  Samples: {len(vae_df):,}")
print(f"  Boreholes: {len(current_boreholes)}")
print(f"  Features: 6D (GRA, MS, NGR, RGB)")
print()

# Load MAD dataset
print("Loading MAD dataset...")
mad_df = pd.read_csv('datasets/MAD_DataLITH.csv', low_memory=False)

# Create borehole IDs
mad_df['Borehole_ID'] = (
    mad_df['Exp'].astype(str) + '_' +
    mad_df['Site'].astype(str) + '_' +
    mad_df['Hole'].astype(str)
)

print(f"MAD dataset:")
print(f"  Total measurements: {len(mad_df):,}")
print(f"  Total boreholes: {mad_df['Borehole_ID'].nunique()}")
print()

# Check available MAD columns
print("Available MAD measurements:")
mad_cols = ['Bulk density (g/cm³)', 'Grain density (g/cm³)', 'Porosity (%)']
for col in mad_cols:
    if col in mad_df.columns:
        non_null = mad_df[col].notna().sum()
        print(f"  {col:30s}: {non_null:8,d} measurements")
print()

# Check overlap with current boreholes
mad_boreholes = set(mad_df['Borehole_ID'].unique())
common_boreholes = current_boreholes & mad_boreholes
mad_only_boreholes = mad_boreholes - current_boreholes

print("="*100)
print("BOREHOLE OVERLAP")
print("="*100)
print()
print(f"MAD boreholes in current VAE dataset: {len(common_boreholes)}/{len(current_boreholes)} ({len(common_boreholes)/len(current_boreholes)*100:.1f}%)")
print(f"MAD-only boreholes (not in VAE):     {len(mad_only_boreholes)}")
print()

# If we did pre-training on MAD-only boreholes
if mad_only_boreholes:
    mad_only_df = mad_df[mad_df['Borehole_ID'].isin(mad_only_boreholes)]
    print(f"MAD-only dataset potential:")
    print(f"  Boreholes: {len(mad_only_boreholes)}")
    print(f"  Measurements: {len(mad_only_df):,}")
    print()

print("="*100)
print("WHY PRE-TRAINING WOULD FAIL")
print("="*100)
print()

print("Previous pre-training experiments (from VAE classifier investigation):")
print()
print("1. v2.6.2 (Sequential transfer learning):")
print("   - Pre-train on GRA+MS+NGR (403K samples, 524 boreholes)")
print("   - Fine-tune with RGB added (239K samples, 296 boreholes)")
print("   - Result: ARI = 0.125 (-51% vs v2.6)")
print("   - Why: Pre-trained encoder optimized for physical-only patterns")
print()

print("2. v2.6.4 (Dual pre-training):")
print("   - Pre-train physical encoder (GRA+MS+NGR) separately")
print("   - Pre-train RGB encoder separately")
print("   - Combine with fusion layer")
print("   - Result: ARI = 0.122 (-53% vs v2.6)")
print("   - Why: Both encoders pre-optimized for wrong objective")
print()

print("3. Key lesson:")
print("   'Multi-modal learning is NOT compositional'")
print("   optimal(A) + optimal(B) ≠ optimal(A+B)")
print()
print("   Cross-modal correlations ('dark + dense = basalt') must be")
print("   learned jointly during training, not composed from separate")
print("   pre-trained representations.")
print()

print("="*100)
print("WHY MAD PRE-TRAINING WOULD FAIL")
print("="*100)
print()

print("Fundamental issues:")
print()
print("1. Different objectives:")
print("   - MAD pre-training would optimize: MAD bulk density reconstruction")
print("   - VAE v2.6.7 needs: Cross-modal GRA+MS+NGR+RGB clustering")
print("   - These are incompatible objectives")
print()

print("2. Measurement mismatch:")
print("   - MAD: Discrete samples (~every 1-2 meters)")
print("   - GRA: Continuous high-resolution (~every 2-5 cm)")
print("   - Encoder would learn discrete sampling patterns, not continuous")
print()

print("3. Representational commitment:")
print("   - MAD pre-training creates encoder optimized for discrete MAD patterns")
print("   - Cannot adapt to continuous GRA + cross-modal RGB correlations")
print("   - Lower learning rate in fine-tuning prevents sufficient adaptation")
print()

print("4. Different boreholes:")
print(f"   - MAD-only boreholes: {len(mad_only_boreholes)}")
print(f"   - These don't overlap with VAE v2.6.7's 296 boreholes")
print(f"   - Pre-training on different boreholes doesn't transfer")
print()

print("5. Empirical evidence:")
print("   - v2.6.2 had 228 extra physical-only boreholes → FAILED (-51%)")
print("   - v2.6.4 dual pre-training → FAILED (-53%)")
print("   - All transfer learning approaches: -50% to -79% performance")
print()

print("="*100)
print("RECOMMENDATION")
print("="*100)
print()

print("❌ DO NOT use MAD pre-training")
print()
print("Instead:")
print("✓ Continue with joint training from scratch (current v2.6.7 approach)")
print("✓ v2.6.7 achieves ARI = 0.196 ± 0.037 (5-fold CV)")
print("✓ This is optimal for multi-modal clustering")
print()

print("If you want more data:")
print()
print("Option 1: Find additional boreholes with ALL measurements (GRA+MS+NGR+RGB)")
print("  - Check if any boreholes beyond current 296 have complete coverage")
print("  - Add to existing dataset (expand, don't pre-train)")
print()

print("Option 2: Accept larger but lower-performing dataset")
print("  - Use VAE v1 (GRA+MS+NGR only): 403K samples, ARI = 0.084")
print("  - 69% more boreholes, but -57% performance (no RGB)")
print()

print("Option 3: Wait for more expeditions")
print("  - Future IODP expeditions with RGB camera imaging")
print("  - Add new data to existing v2.6.7 approach")
print()

print("Remember: Feature quality > dataset size")
print("  - v2.6.1 with +44% data performed -54% worse (RSC < RGB)")
print("  - Joint training with optimal features >> pre-training with more data")
print()
print("="*100)
