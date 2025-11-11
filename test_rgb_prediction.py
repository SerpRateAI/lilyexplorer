"""
Test if we can predict RGB from GRA+MS+NGR using CatBoost.

If R² > 0.5, we could use this to expand the dataset to 525 boreholes.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

print("Loading VAE v2.6.7 dataset (296 boreholes with all 4 measurements)...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

# Features and targets
X = df[['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)']].values
y_r = df['R'].values
y_g = df['G'].values
y_b = df['B'].values

print(f"\nFeature statistics:")
print(f"  GRA: {X[:,0].mean():.2f} ± {X[:,0].std():.2f}")
print(f"  MS:  {X[:,1].mean():.2f} ± {X[:,1].std():.2f}")
print(f"  NGR: {X[:,2].mean():.2f} ± {X[:,2].std():.2f}")

print(f"\nTarget statistics:")
print(f"  R: {y_r.mean():.1f} ± {y_r.std():.1f}")
print(f"  G: {y_g.mean():.1f} ± {y_g.std():.1f}")
print(f"  B: {y_b.mean():.1f} ± {y_b.std():.1f}")

# Check correlations
print(f"\nCorrelations with RGB:")
for i, name in enumerate(['GRA', 'MS', 'NGR']):
    corr_r = np.corrcoef(X[:,i], y_r)[0,1]
    corr_g = np.corrcoef(X[:,i], y_g)[0,1]
    corr_b = np.corrcoef(X[:,i], y_b)[0,1]
    print(f"  {name:3s} vs R: {corr_r:6.3f}")
    print(f"  {name:3s} vs G: {corr_g:6.3f}")
    print(f"  {name:3s} vs B: {corr_b:6.3f}")

print("\n" + "="*80)
print("TRAINING CATBOOST REGRESSORS")
print("="*80)

# Split data
X_train, X_test, y_r_train, y_r_test = train_test_split(X, y_r, test_size=0.2, random_state=42)
_, _, y_g_train, y_g_test = train_test_split(X, y_g, test_size=0.2, random_state=42)
_, _, y_b_train, y_b_test = train_test_split(X, y_b, test_size=0.2, random_state=42)

print(f"\nTrain: {len(X_train):,} samples")
print(f"Test:  {len(X_test):,} samples")

# Train R predictor
print("\nTraining R predictor...")
model_r = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, verbose=False)
model_r.fit(X_train, y_r_train)
y_r_pred = model_r.predict(X_test)
r2_r = r2_score(y_r_test, y_r_pred)
rmse_r = np.sqrt(mean_squared_error(y_r_test, y_r_pred))

# Train G predictor
print("Training G predictor...")
model_g = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, verbose=False)
model_g.fit(X_train, y_g_train)
y_g_pred = model_g.predict(X_test)
r2_g = r2_score(y_g_test, y_g_pred)
rmse_g = np.sqrt(mean_squared_error(y_g_test, y_g_pred))

# Train B predictor
print("Training B predictor...")
model_b = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.1, verbose=False)
model_b.fit(X_train, y_b_train)
y_b_pred = model_b.predict(X_test)
r2_b = r2_score(y_b_test, y_b_pred)
rmse_b = np.sqrt(mean_squared_error(y_b_test, y_b_pred))

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\nR channel:")
print(f"  R² = {r2_r:.4f}")
print(f"  RMSE = {rmse_r:.2f} (mean={y_r.mean():.1f}, std={y_r.std():.1f})")
print(f"  RMSE/std = {rmse_r/y_r.std():.2f}")

print(f"\nG channel:")
print(f"  R² = {r2_g:.4f}")
print(f"  RMSE = {rmse_g:.2f} (mean={y_g.mean():.1f}, std={y_g.std():.1f})")
print(f"  RMSE/std = {rmse_g/y_g.std():.2f}")

print(f"\nB channel:")
print(f"  R² = {r2_b:.4f}")
print(f"  RMSE = {rmse_b:.2f} (mean={y_b.mean():.1f}, std={y_b.std():.1f})")
print(f"  RMSE/std = {rmse_b/y_b.std():.2f}")

print(f"\nAverage R² = {(r2_r + r2_g + r2_b)/3:.4f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

avg_r2 = (r2_r + r2_g + r2_b)/3

if avg_r2 > 0.5:
    print(f"""
✓ RGB prediction is FEASIBLE (avg R² = {avg_r2:.3f})
  → Could expand to 525 boreholes (+77%)
  → Train v2.6.10 with 296 real + 229 predicted RGB
""")
elif avg_r2 > 0.3:
    print(f"""
⚠ RGB prediction is MARGINAL (avg R² = {avg_r2:.3f})
  → Predictions have substantial noise
  → May degrade clustering performance
  → Worth testing, but risky
""")
else:
    print(f"""
✗ RGB prediction is TOO WEAK (avg R² = {avg_r2:.3f})
  → Predictions are barely better than mean
  → Would add noisy/uninformative features
  → Not worth pursuing

Lesson from v2.11: RGB imputation R² = -16.8 to -79.3
This confirms that GRA/MS/NGR → RGB mappings are too weak.
Physical properties don't reliably predict visual color.
""")

print(f"\nConclusion:")
if avg_r2 < 0.5:
    print("v2.6.7 (296 boreholes with real RGB) remains optimal.")
    print("Cannot reliably expand dataset with predicted RGB.")
