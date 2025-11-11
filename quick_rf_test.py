"""Quick test: RF/CatBoost with scaled vs unscaled features."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
hierarchy_df = pd.read_csv('/home/utig5/johna/bhai/lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'], hierarchy_df['Lithology_Group']))
df['Lithology_Group'] = df['Principal'].map(principal_to_group).dropna()

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
y = df['Lithology_Group'].values
lithology_to_idx = {lith: i for i, lith in enumerate(sorted(df['Lithology_Group'].unique()))}
y_encoded = np.array([lithology_to_idx[lith] for lith in y])

# Same split
unique_boreholes = df['Borehole_ID'].unique()
np.random.seed(42)
np.random.shuffle(unique_boreholes)
n_train = int(0.70 * len(unique_boreholes))
train_boreholes = unique_boreholes[:n_train]
test_boreholes = unique_boreholes[n_train:]

train_mask = df['Borehole_ID'].isin(train_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

X_train, y_train = X[train_mask], y_encoded[train_mask]
X_test, y_test = X[test_mask], y_encoded[test_mask]

# Test 1: Unscaled
rf1 = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
rf1.fit(X_train, y_train)
y_pred1 = rf1.predict(X_test)
acc1 = balanced_accuracy_score(y_test, y_pred1)

# Test 2: Standard scaled
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf2 = RandomForestClassifier(n_estimators=200, max_depth=20, class_weight='balanced', n_jobs=-1, random_state=42)
rf2.fit(X_train_scaled, y_train)
y_pred2 = rf2.predict(X_test_scaled)
acc2 = balanced_accuracy_score(y_test, y_pred2)

print(f"RF unscaled:  {acc1:.4f} ({acc1*100:.2f}%)")
print(f"RF scaled:    {acc2:.4f} ({acc2*100:.2f}%)")
print(f"Difference: {(acc2-acc1)*100:+.2f}%")
