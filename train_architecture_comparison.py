"""
Architecture Comparison: Test if 42% ceiling is model limitation, not feature limitation.

Models:
1. Random Forest
2. CatBoost
3. Deep Neural Network (deeper than 42% baseline)
4. Ensemble (voting)

Dataset: v2.6 (239K samples, 6 features: GRA, MS, NGR, RGB)
Target: 14 lithology classes
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("ARCHITECTURE COMPARISON: Breaking the 42% Ceiling")
print("="*100)
print()

# Load v2.6 dataset
print("Loading v2.6 dataset...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"Total samples: {len(df):,}")
print(f"Boreholes: {df['Borehole_ID'].nunique()}")
print()

# Load lithology hierarchy
print("Loading lithology hierarchy...")
hierarchy_df = pd.read_csv('/home/utig5/johna/bhai/lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'], hierarchy_df['Lithology_Group']))

df['Lithology_Group'] = df['Principal'].map(principal_to_group)
df = df.dropna(subset=['Lithology_Group'])
print(f"Samples after hierarchy: {len(df):,}")
print(f"Classes: {df['Lithology_Group'].nunique()}")
print()

# Features and labels
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
y = df['Lithology_Group'].values

# Encode labels
lithology_groups = sorted(df['Lithology_Group'].unique())
lithology_to_idx = {lith: i for i, lith in enumerate(lithology_groups)}
y_encoded = np.array([lithology_to_idx[lith] for lith in y])

print("Class distribution:")
for i, lith in enumerate(lithology_groups):
    count = (y_encoded == i).sum()
    print(f"  {i:2d}. {lith:30s}: {count:>8,} ({count/len(y)*100:5.2f}%)")
print()

# Borehole-level split (same as baseline)
print("Creating borehole-level split (70/15/15)...")
unique_boreholes = df['Borehole_ID'].unique()
np.random.seed(42)
np.random.shuffle(unique_boreholes)

n_train = int(0.70 * len(unique_boreholes))
n_val = int(0.15 * len(unique_boreholes))

train_boreholes = unique_boreholes[:n_train]
val_boreholes = unique_boreholes[n_train:n_train+n_val]
test_boreholes = unique_boreholes[n_train+n_val:]

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

X_train, y_train = X[train_mask], y_encoded[train_mask]
X_val, y_val = X[val_mask], y_encoded[val_mask]
X_test, y_test = X[test_mask], y_encoded[test_mask]

print(f"Train: {len(X_train):>6,} samples from {len(train_boreholes):>3d} boreholes")
print(f"Val:   {len(X_val):>6,} samples from {len(val_boreholes):>3d} boreholes")
print(f"Test:  {len(X_test):>6,} samples from {len(test_boreholes):>3d} boreholes")
print()

# Standard scaling (for neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Class weights (for imbalanced classes)
class_counts = np.bincount(y_train)
class_weights_array = len(y_train) / (len(class_counts) * class_counts)

results = {}

# ============================================================================
# MODEL 1: RANDOM FOREST
# ============================================================================
print("="*100)
print("MODEL 1: RANDOM FOREST")
print("="*100)
print()

start = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=50,
    min_samples_leaf=20,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
bal_acc_rf = balanced_accuracy_score(y_test, y_pred_rf)
elapsed = time.time() - start

print(f"Training time: {elapsed:.1f}s")
print(f"Balanced accuracy: {bal_acc_rf:.4f} ({bal_acc_rf*100:.2f}%)")
print()

results['Random Forest'] = bal_acc_rf

# ============================================================================
# MODEL 2: CATBOOST
# ============================================================================
print("="*100)
print("MODEL 2: CATBOOST")
print("="*100)
print()

start = time.time()
cb_model = CatBoostClassifier(
    iterations=200,
    depth=8,
    learning_rate=0.1,
    class_weights=class_weights_array,
    random_seed=42,
    verbose=False
)
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
y_pred_cb = cb_model.predict(X_test)
bal_acc_cb = balanced_accuracy_score(y_test, y_pred_cb)
elapsed = time.time() - start

print(f"Training time: {elapsed:.1f}s")
print(f"Balanced accuracy: {bal_acc_cb:.4f} ({bal_acc_cb*100:.2f}%)")
print()

results['CatBoost'] = bal_acc_cb

# ============================================================================
# MODEL 3: DEEP NEURAL NETWORK
# ============================================================================
print("="*100)
print("MODEL 3: DEEP NEURAL NETWORK")
print("="*100)
print()

class DeepClassifier(nn.Module):
    """Deeper network than 42% baseline."""
    def __init__(self, input_dim=6, num_classes=14):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deep_model = DeepClassifier().to(device)

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights_array).to(device))
optimizer = optim.Adam(deep_model.parameters(), lr=1e-3)

start = time.time()
best_val_acc = 0
patience = 10
patience_counter = 0

for epoch in range(100):
    deep_model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = deep_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    # Validation
    deep_model.eval()
    y_pred_val = []
    y_true_val = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits = deep_model(batch_x)
            y_pred_val.extend(logits.argmax(dim=1).cpu().numpy())
            y_true_val.extend(batch_y.numpy())

    val_acc = balanced_accuracy_score(y_true_val, y_pred_val)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = deep_model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break

# Load best model and evaluate on test
deep_model.load_state_dict(best_model_state)
deep_model.eval()
y_pred_deep = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        logits = deep_model(batch_x)
        y_pred_deep.extend(logits.argmax(dim=1).cpu().numpy())

bal_acc_deep = balanced_accuracy_score(y_test, y_pred_deep)
elapsed = time.time() - start

print(f"Training time: {elapsed:.1f}s")
print(f"Balanced accuracy: {bal_acc_deep:.4f} ({bal_acc_deep*100:.2f}%)")
print()

results['Deep NN'] = bal_acc_deep

# ============================================================================
# MODEL 4: ENSEMBLE (VOTING)
# ============================================================================
print("="*100)
print("MODEL 4: ENSEMBLE (VOTING)")
print("="*100)
print()

# Use predictions from all 3 models
all_preds = np.vstack([y_pred_rf, y_pred_cb, y_pred_deep])
y_pred_ensemble = []

for i in range(len(y_test)):
    # Majority vote
    votes = all_preds[:, i]
    y_pred_ensemble.append(np.bincount(votes).argmax())

y_pred_ensemble = np.array(y_pred_ensemble)
bal_acc_ensemble = balanced_accuracy_score(y_test, y_pred_ensemble)

print(f"Balanced accuracy: {bal_acc_ensemble:.4f} ({bal_acc_ensemble*100:.2f}%)")
print()

results['Ensemble'] = bal_acc_ensemble

# ============================================================================
# COMPARISON
# ============================================================================
print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()

print("| Model | Balanced Accuracy | vs 42% Baseline |")
print("|-------|-------------------|-----------------|")

baseline = 0.4232
for model_name, bal_acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    improvement = ((bal_acc - baseline) / baseline) * 100
    print(f"| {model_name:20s} | {bal_acc:.4f} ({bal_acc*100:5.2f}%) | {improvement:+6.1f}% |")

print()
print(f"Baseline (simple NN): {baseline:.4f} ({baseline*100:.2f}%)")
print()

best_model = max(results.items(), key=lambda x: x[1])
print(f"Best model: {best_model[0]} with {best_model[1]:.4f} ({best_model[1]*100:.2f}%)")
print()

if best_model[1] > baseline:
    improvement = ((best_model[1] - baseline) / baseline) * 100
    print(f"✓ SUCCESS: {best_model[0]} breaks the 42% ceiling by {improvement:.1f}%!")
    print("  The bottleneck was model architecture, not feature quality.")
else:
    print("✗ RESULT: No model breaks the 42% ceiling.")
    print("  The bottleneck is feature insufficiency, not model architecture.")
    print("  Need porosity/grain density (v2.7) or additional measurements.")

print()

# Per-class breakdown for best model
print("="*100)
print(f"PER-CLASS PERFORMANCE: {best_model[0]}")
print("="*100)
print()

if best_model[0] == 'Random Forest':
    y_pred_best = y_pred_rf
elif best_model[0] == 'CatBoost':
    y_pred_best = y_pred_cb
elif best_model[0] == 'Deep NN':
    y_pred_best = y_pred_deep
else:
    y_pred_best = y_pred_ensemble

for i, lith in enumerate(lithology_groups):
    mask = y_test == i
    if mask.sum() > 0:
        acc = (y_pred_best[mask] == i).mean()
        print(f"  {lith:30s}: {acc:6.2%} ({mask.sum():>6,} samples)")

print()
print("="*100)
print("ARCHITECTURE COMPARISON COMPLETE")
print("="*100)
