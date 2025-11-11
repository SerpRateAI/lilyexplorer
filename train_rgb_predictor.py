"""
Train CatBoost models to predict RGB from GRA+MS+NGR.

Use these models to expand the VAE dataset to boreholes without RGB camera data.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

print("="*80)
print("RGB PREDICTOR TRAINING")
print("="*80)

print("\nLoading VAE v2.6.7 dataset (296 boreholes with all 4 measurements)...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

# Features and targets
X = df[['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)']].values
y_r = df['R'].values
y_g = df['G'].values
y_b = df['B'].values

print(f"\nInput features:")
print(f"  GRA:  mean={X[:,0].mean():.2f}, std={X[:,0].std():.2f}")
print(f"  MS:   mean={X[:,1].mean():.2f}, std={X[:,1].std():.2f}")
print(f"  NGR:  mean={X[:,2].mean():.2f}, std={X[:,2].std():.2f}")

print(f"\nTarget RGB channels:")
print(f"  R:    mean={y_r.mean():.1f}, std={y_r.std():.1f}")
print(f"  G:    mean={y_g.mean():.1f}, std={y_g.std():.1f}")
print(f"  B:    mean={y_b.mean():.1f}, std={y_b.std():.1f}")

# Check correlations
print(f"\nFeature-target correlations:")
for i, fname in enumerate(['GRA', 'MS', 'NGR']):
    corr_r = np.corrcoef(X[:,i], y_r)[0,1]
    corr_g = np.corrcoef(X[:,i], y_g)[0,1]
    corr_b = np.corrcoef(X[:,i], y_b)[0,1]
    print(f"  {fname:3s} → R: {corr_r:+.3f}  |  G: {corr_g:+.3f}  |  B: {corr_b:+.3f}")

# Train/test split
X_train, X_test, y_r_train, y_r_test = train_test_split(
    X, y_r, test_size=0.2, random_state=42
)
_, _, y_g_train, y_g_test = train_test_split(
    X, y_g, test_size=0.2, random_state=42
)
_, _, y_b_train, y_b_test = train_test_split(
    X, y_b, test_size=0.2, random_state=42
)

print(f"\nTrain/test split:")
print(f"  Train: {len(X_train):,} samples")
print(f"  Test:  {len(X_test):,} samples")

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

# CatBoost hyperparameters
params = {
    'iterations': 500,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'RMSE',
    'random_seed': 42,
    'verbose': False
}

# Train R predictor
print("\n[1/3] Training R channel predictor...")
model_r = CatBoostRegressor(**params)
model_r.fit(X_train, y_r_train, eval_set=(X_test, y_r_test), early_stopping_rounds=50)
y_r_pred = model_r.predict(X_test)
r2_r = r2_score(y_r_test, y_r_pred)
rmse_r = np.sqrt(mean_squared_error(y_r_test, y_r_pred))
print(f"  R² = {r2_r:.4f}, RMSE = {rmse_r:.2f}")

# Train G predictor
print("\n[2/3] Training G channel predictor...")
model_g = CatBoostRegressor(**params)
model_g.fit(X_train, y_g_train, eval_set=(X_test, y_g_test), early_stopping_rounds=50)
y_g_pred = model_g.predict(X_test)
r2_g = r2_score(y_g_test, y_g_pred)
rmse_g = np.sqrt(mean_squared_error(y_g_test, y_g_pred))
print(f"  R² = {r2_g:.4f}, RMSE = {rmse_g:.2f}")

# Train B predictor
print("\n[3/3] Training B channel predictor...")
model_b = CatBoostRegressor(**params)
model_b.fit(X_train, y_b_train, eval_set=(X_test, y_b_test), early_stopping_rounds=50)
y_b_pred = model_b.predict(X_test)
r2_b = r2_score(y_b_test, y_b_pred)
rmse_b = np.sqrt(mean_squared_error(y_b_test, y_b_pred))
print(f"  R² = {r2_b:.4f}, RMSE = {rmse_b:.2f}")

print("\n" + "="*80)
print("RESULTS")
print("="*80)

avg_r2 = (r2_r + r2_g + r2_b) / 3
avg_rmse = (rmse_r + rmse_g + rmse_b) / 3

print(f"\nPer-channel performance:")
print(f"  R: R²={r2_r:.4f}, RMSE={rmse_r:.2f} ({rmse_r/y_r.std()*100:.1f}% of std)")
print(f"  G: R²={r2_g:.4f}, RMSE={rmse_g:.2f} ({rmse_g/y_g.std()*100:.1f}% of std)")
print(f"  B: R²={r2_b:.4f}, RMSE={rmse_b:.2f} ({rmse_b/y_b.std()*100:.1f}% of std)")

print(f"\nOverall performance:")
print(f"  Average R² = {avg_r2:.4f}")
print(f"  Average RMSE = {avg_rmse:.2f}")

# Feature importance
print("\n" + "="*80)
print("FEATURE IMPORTANCE")
print("="*80)

feature_names = ['GRA', 'MS', 'NGR']

print("\nR channel:")
for i, name in enumerate(feature_names):
    print(f"  {name:3s}: {model_r.feature_importances_[i]:5.1f}")

print("\nG channel:")
for i, name in enumerate(feature_names):
    print(f"  {name:3s}: {model_g.feature_importances_[i]:5.1f}")

print("\nB channel:")
for i, name in enumerate(feature_names):
    print(f"  {name:3s}: {model_b.feature_importances_[i]:5.1f}")

# Save models
print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

model_r.save_model('ml_models/rgb_predictor_r.cbm')
print("✓ Saved ml_models/rgb_predictor_r.cbm")

model_g.save_model('ml_models/rgb_predictor_g.cbm')
print("✓ Saved ml_models/rgb_predictor_g.cbm")

model_b.save_model('ml_models/rgb_predictor_b.cbm')
print("✓ Saved ml_models/rgb_predictor_b.cbm")

# Also save a summary
summary = {
    'r2_r': r2_r,
    'r2_g': r2_g,
    'r2_b': r2_b,
    'avg_r2': avg_r2,
    'rmse_r': rmse_r,
    'rmse_g': rmse_g,
    'rmse_b': rmse_b,
    'avg_rmse': avg_rmse,
    'n_train': len(X_train),
    'n_test': len(X_test)
}

import json
with open('ml_models/rgb_predictor_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved ml_models/rgb_predictor_summary.json")

print("\n" + "="*80)
print("USAGE")
print("="*80)
print("""
To predict RGB for new data:

```python
from catboost import CatBoostRegressor
import pandas as pd

# Load models
model_r = CatBoostRegressor()
model_r.load_model('ml_models/rgb_predictor_r.cbm')
model_g = CatBoostRegressor()
model_g.load_model('ml_models/rgb_predictor_g.cbm')
model_b = CatBoostRegressor()
model_b.load_model('ml_models/rgb_predictor_b.cbm')

# Prepare features
X = df[['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)']].values

# Predict
df['R_predicted'] = model_r.predict(X)
df['G_predicted'] = model_g.predict(X)
df['B_predicted'] = model_b.predict(X)
```

Next step: Create v2.6.10 dataset with predicted RGB for 229 boreholes
""")

print("\n✓ Training complete!")
