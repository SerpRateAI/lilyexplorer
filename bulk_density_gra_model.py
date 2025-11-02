"""
Gradient Boosted Trees model for predicting GRA bulk density from LILY database.

Uses CatBoost to predict GRA bulk density (4M measurements - high frequency).
IMPORTANT: Splits data by BOREHOLE to avoid data leakage and test generalization
to unseen boreholes.
"""

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Loading datasets...")
print("Using GRA bulk density as target (4M samples)")
print("CRITICAL: Will split by BOREHOLE to test generalization\n")

# Load GRA data (contains target: Bulk density (GRA))
print("  Loading GRA (contains target bulk density - 4M samples!)...")
gra = pd.read_csv('datasets/GRA_DataLITH.csv', low_memory=False)
print(f"    GRA shape: {gra.shape}")

# Rename target for clarity
gra = gra.rename(columns={'Bulk density (GRA)': 'target_bulk_density'})

# Key columns for merging
key_cols = ['Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W']

# Convert key columns to string for consistent merging
for col in key_cols:
    if col in gra.columns:
        gra[col] = gra[col].astype(str)

# Create borehole ID for proper splitting
gra['borehole_id'] = gra['Exp'].astype(str) + '_' + gra['Site'].astype(str) + '_' + gra['Hole'].astype(str)
print(f"    Unique boreholes: {gra['borehole_id'].nunique()}")

# Load other datasets (aggregated by section to keep memory manageable)
datasets = {}

print("\n  Loading and aggregating other datasets by section...")

print("  Loading MAD (moisture and density - will aggregate)...")
mad = pd.read_csv('datasets/MAD_DataLITH.csv')
for col in key_cols:
    if col in mad.columns:
        mad[col] = mad[col].astype(str)
# Aggregate MAD per section (EXCLUDE ALL DENSITY/POROSITY to avoid circular prediction)
mad_num_cols = [c for c in mad.select_dtypes(include=[np.number]).columns.tolist()
                if not any(x in c.lower() for x in ['density', 'porosity', 'void'])]
agg_dict = {col: ['mean', 'std'] for col in mad_num_cols[:10] if col}
if agg_dict:
    mad_agg = mad.groupby(key_cols).agg(agg_dict).reset_index()
    mad_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in mad_agg.columns.values]
    datasets['MAD'] = mad_agg
    print(f"    MAD aggregated shape: {mad_agg.shape}")
del mad

print("  Loading MS (magnetic susceptibility)...")
ms = pd.read_csv('datasets/MS_DataLITH.csv')
for col in key_cols:
    if col in ms.columns:
        ms[col] = ms[col].astype(str)
ms_col = [c for c in ms.columns if 'susceptibility' in c.lower() and c not in key_cols][0]
ms_agg = ms.groupby(key_cols).agg({
    ms_col: ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
ms_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in ms_agg.columns.values]
datasets['MS'] = ms_agg
print(f"    MS aggregated shape: {ms_agg.shape}")
del ms

print("  Loading NGR (natural gamma radiation)...")
ngr = pd.read_csv('datasets/NGR_DataLITH.csv')
for col in key_cols:
    if col in ngr.columns:
        ngr[col] = ngr[col].astype(str)
ngr_num_cols = ngr.select_dtypes(include=[np.number]).columns.tolist()
agg_dict = {col: ['mean', 'std', 'min', 'max'] for col in ngr_num_cols[:5] if col}
ngr_agg = ngr.groupby(key_cols).agg(agg_dict).reset_index()
ngr_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in ngr_agg.columns.values]
datasets['NGR'] = ngr_agg
print(f"    NGR aggregated shape: {ngr_agg.shape}")
del ngr

print("  Loading RGB (color data)...")
rgb = pd.read_csv('datasets/RGB_DataLITH.csv')
for col in key_cols:
    if col in rgb.columns:
        rgb[col] = rgb[col].astype(str)
rgb_num_cols = rgb.select_dtypes(include=[np.number]).columns.tolist()
agg_dict = {col: ['mean', 'std'] for col in rgb_num_cols[:10] if col}
rgb_agg = rgb.groupby(key_cols).agg(agg_dict).reset_index()
rgb_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in rgb_agg.columns.values]
datasets['RGB'] = rgb_agg
print(f"    RGB aggregated shape: {rgb_agg.shape}")
del rgb

print("  Loading SRM (remanent magnetization)...")
srm = pd.read_csv('datasets/SRM_DataLITH.csv')
for col in key_cols:
    if col in srm.columns:
        srm[col] = srm[col].astype(str)
srm_num_cols = srm.select_dtypes(include=[np.number]).columns.tolist()
agg_dict = {col: ['mean', 'std'] for col in srm_num_cols[:8] if col}
srm_agg = srm.groupby(key_cols).agg(agg_dict).reset_index()
srm_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in srm_agg.columns.values]
datasets['SRM'] = srm_agg
print(f"    SRM aggregated shape: {srm_agg.shape}")
del srm

print("  Loading PWL (P-wave logger)...")
pwl = pd.read_csv('datasets/PWL_DataLITH.csv')
for col in key_cols:
    if col in pwl.columns:
        pwl[col] = pwl[col].astype(str)
pwl_num_cols = pwl.select_dtypes(include=[np.number]).columns.tolist()
agg_dict = {col: ['mean', 'std'] for col in pwl_num_cols[:5] if col}
pwl_agg = pwl.groupby(key_cols).agg(agg_dict).reset_index()
pwl_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in pwl_agg.columns.values]
datasets['PWL'] = pwl_agg
print(f"    PWL aggregated shape: {pwl_agg.shape}")
del pwl

print("  Loading smaller datasets (no aggregation)...")

print("  Loading CARB (carbonate content)...")
carb = pd.read_csv('datasets/CARB_DataLITH.csv')
for col in key_cols:
    if col in carb.columns:
        carb[col] = carb[col].astype(str)
datasets['CARB'] = carb
print(f"    CARB shape: {carb.shape}")

print("  Loading ICP (inductively coupled plasma)...")
icp = pd.read_csv('datasets/ICP_DataLITH.csv')
for col in key_cols:
    if col in icp.columns:
        icp[col] = icp[col].astype(str)
datasets['ICP'] = icp
print(f"    ICP shape: {icp.shape}")

print("  Loading PWC (P-wave velocity)...")
pwc = pd.read_csv('datasets/PWC_DataLITH.csv')
for col in key_cols:
    if col in pwc.columns:
        pwc[col] = pwc[col].astype(str)
datasets['PWC'] = pwc
print(f"    PWC shape: {pwc.shape}")

# Start with GRA as base
print("\nMerging datasets on core identification fields...")
merged = gra.copy()

# Merge each dataset
for name, df in datasets.items():
    print(f"  Merging {name}...")

    # Remove duplicate lithology/location columns (keep from GRA)
    lithology_cols = ['Prefix', 'Principal', 'Suffix', 'Full Lithology', 'Simplified Lithology',
                     'Lithology Type', 'Degree of Consolidation', 'Lithology Subtype']
    location_cols = ['Latitude (DD)', 'Longitude (DD)', 'Water Depth (mbsl)', 'Expanded Core Type']
    cols_to_drop = [col for col in lithology_cols + location_cols if col in df.columns]

    df_to_merge = df.drop(columns=cols_to_drop, errors='ignore')

    # Add suffix to numeric columns
    rename_dict = {col: f"{name}_{col}" for col in df_to_merge.columns
                   if col not in key_cols}
    df_to_merge = df_to_merge.rename(columns=rename_dict)

    # Merge
    merged = pd.merge(merged, df_to_merge, on=key_cols, how='left')

print(f"\nMerged dataset shape: {merged.shape}")

# Remove rows where target is missing
print(f"Rows before removing missing targets: {len(merged)}")
merged = merged.dropna(subset=['target_bulk_density'])
print(f"Rows after removing missing targets: {len(merged)}")

# Identify feature columns
exclude_cols = ['target_bulk_density', 'borehole_id', 'Timestamp (UTC)', 'Text ID', 'Test No.',
                'Comments', 'Sample comments', 'Test comments', 'Result comments',
                'Instrument', 'Instrument group']

# Exclude timestamp/metadata columns and any density/porosity columns
exclude_cols.extend([col for col in merged.columns if any(x in col for x in ['Timestamp', 'Text ID', 'Test No'])])

# Get feature columns
all_cols = merged.columns.tolist()
feature_cols = [col for col in all_cols if col not in exclude_cols]

# Double-check no density/porosity leaked through
feature_cols = [col for col in feature_cols if not any(x in col.lower() for x in ['density', 'porosity', 'void'])]

# Identify categorical features
categorical_features = []
for col in feature_cols:
    if merged[col].dtype == 'object' or merged[col].dtype.name == 'string':
        categorical_features.append(col)
    elif col in ['Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W']:
        categorical_features.append(col)

print(f"\nTotal features: {len(feature_cols)}")
print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features: {len([f for f in feature_cols if f not in categorical_features])}")

# Prepare data
X = merged[feature_cols].copy()
y = merged['target_bulk_density'].values
groups = merged['borehole_id'].values

# Handle NaN in categorical features
for col in categorical_features:
    if col in X.columns:
        X[col] = X[col].fillna('missing').astype(str)

cat_features_idx = [i for i, col in enumerate(feature_cols) if col in categorical_features]

print(f"\nDataset info:")
print(f"  Total samples: {len(X):,}")
print(f"  Total boreholes: {len(np.unique(groups))}")
print(f"  Features: {len(feature_cols)}")
print(f"  Categorical features: {len(cat_features_idx)}")

# Sample for faster training (use 20% for better representation)
sample_frac = 0.2
print(f"\nSampling {sample_frac*100}% of data for tractable training...")
sample_idx = np.random.RandomState(42).choice(len(X), size=int(len(X)*sample_frac), replace=False)
X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_sample = y[sample_idx]
groups_sample = groups[sample_idx]
print(f"  Sampled size: {len(X_sample):,}")
print(f"  Sampled boreholes: {len(np.unique(groups_sample))}")

# Split by BOREHOLE using GroupShuffleSplit
print(f"\nSplitting by borehole (no borehole appears in multiple splits)...")
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, temp_idx = next(gss.split(X_sample, y_sample, groups_sample))

X_train = X_sample.iloc[train_idx]
y_train = y_sample[train_idx]
groups_train = groups_sample[train_idx]

X_temp = X_sample.iloc[temp_idx]
y_temp = y_sample[temp_idx]
groups_temp = groups_sample[temp_idx]

# Split temp into val and test
gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss_val.split(X_temp, y_temp, groups_temp))

X_val = X_temp.iloc[val_idx]
y_val = y_temp[val_idx]
groups_val = groups_temp[val_idx]

X_test = X_temp.iloc[test_idx]
y_test = y_temp[test_idx]
groups_test = groups_temp[test_idx]

print(f"\nData splits (BY BOREHOLE):")
print(f"  Train: {len(X_train):,} samples from {len(np.unique(groups_train))} boreholes ({len(X_train)/len(X_sample)*100:.1f}%)")
print(f"  Validation: {len(X_val):,} samples from {len(np.unique(groups_val))} boreholes ({len(X_val)/len(X_sample)*100:.1f}%)")
print(f"  Test: {len(X_test):,} samples from {len(np.unique(groups_test))} boreholes ({len(X_test)/len(X_sample)*100:.1f}%)")

# Verify no overlap
train_boreholes = set(np.unique(groups_train))
val_boreholes = set(np.unique(groups_val))
test_boreholes = set(np.unique(groups_test))
print(f"\nBorehole overlap check:")
print(f"  Train ∩ Val: {len(train_boreholes & val_boreholes)} (should be 0)")
print(f"  Train ∩ Test: {len(train_boreholes & test_boreholes)} (should be 0)")
print(f"  Val ∩ Test: {len(val_boreholes & test_boreholes)} (should be 0)")

# Create pools
train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
val_pool = Pool(X_val, y_val, cat_features=cat_features_idx)
test_pool = Pool(X_test, y_test, cat_features=cat_features_idx)

# Train model
print("\nTraining CatBoost model with early stopping...")
model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.1,
    depth=8,
    loss_function='RMSE',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=100,
    cat_features=cat_features_idx
)

model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Evaluate
print("\n" + "="*70)
print("MODEL EVALUATION - GRA BULK DENSITY PREDICTION (BOREHOLE-SPLIT)")
print("="*70)

for name, pool, X_set, y_set in [
    ("Train", train_pool, X_train, y_train),
    ("Validation", val_pool, X_val, y_val),
    ("Test", test_pool, X_test, y_test)
]:
    y_pred = model.predict(pool)
    rmse = np.sqrt(mean_squared_error(y_set, y_pred))
    mae = mean_absolute_error(y_set, y_pred)
    r2 = r2_score(y_set, y_pred)

    print(f"\n{name} Set:")
    print(f"  RMSE: {rmse:.4f} g/cm³")
    print(f"  MAE:  {mae:.4f} g/cm³")
    print(f"  R²:   {r2:.4f}")

# Feature importance
print("\n" + "="*70)
print("TOP 30 MOST IMPORTANT FEATURES")
print("="*70)

feature_importance = model.get_feature_importance(train_pool)
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(30).to_string(index=False))

# Save
print("\nSaving model...")
model.save_model('bulk_density_gra_model_borehole_split.cbm')
print("Model saved to: bulk_density_gra_model_borehole_split.cbm")

importance_df.to_csv('feature_importance_gra_borehole_split.csv', index=False)
print("Feature importance saved to: feature_importance_gra_borehole_split.csv")

print("\n" + "="*70)
print("DONE!")
print("="*70)
print(f"\nSummary:")
print(f"  - Predicting GRA bulk density (4M total measurements)")
print(f"  - Used {len(feature_cols)} features from {len(datasets)} datasets")
print(f"  - Trained on {len(X_train):,} samples from {len(np.unique(groups_train))} boreholes (20% sample)")
print(f"  - Test R²: {r2_score(y_test, model.predict(test_pool)):.4f}")
print(f"  - Test set has {len(np.unique(groups_test))} UNSEEN boreholes")
print(f"  - Early stopping at iteration {model.tree_count_}")
print(f"\nNote: Proper borehole-based split ensures model generalizes to new boreholes!")
