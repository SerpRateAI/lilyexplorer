"""
Analyze train/test distribution mismatch caused by random borehole splitting.
Compare class distributions across splits.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('vae_training_data_v2_20cm.csv')
y = df['Principal'].values

# Same random split as classifier
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.8235, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

y_train = y[train_mask]
y_val = y[val_mask]
y_test = y[test_mask]

print("="*100)
print("TRAIN/TEST DISTRIBUTION MISMATCH ANALYSIS")
print("="*100)
print()

print(f"Total samples: {len(df):,}")
print(f"Train: {len(y_train):,} samples ({100*len(y_train)/len(df):.1f}%)")
print(f"Val:   {len(y_val):,} samples ({100*len(y_val)/len(df):.1f}%)")
print(f"Test:  {len(y_test):,} samples ({100*len(y_test)/len(df):.1f}%)")
print()

# Get class distributions
unique_labels = np.unique(y)
train_dist = pd.Series(y_train).value_counts()
val_dist = pd.Series(y_val).value_counts()
test_dist = pd.Series(y_test).value_counts()

# Calculate percentages
train_pct = train_dist / len(y_train) * 100
val_pct = val_dist / len(y_val) * 100
test_pct = test_dist / len(y_test) * 100

# Find worst mismatches
results = []
for label in unique_labels:
    train_samples = train_dist.get(label, 0)
    val_samples = val_dist.get(label, 0)
    test_samples = test_dist.get(label, 0)

    train_p = train_pct.get(label, 0)
    val_p = val_pct.get(label, 0)
    test_p = test_pct.get(label, 0)

    # Calculate mismatch (test % vs train %)
    if train_p > 0:
        mismatch_ratio = test_p / train_p
    else:
        mismatch_ratio = float('inf') if test_p > 0 else 0

    results.append({
        'lithology': label,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'test_samples': test_samples,
        'train_pct': train_p,
        'val_pct': val_p,
        'test_pct': test_p,
        'mismatch_ratio': mismatch_ratio
    })

results_df = pd.DataFrame(results)

print("="*100)
print("TOP 20 WORST TRAIN/TEST DISTRIBUTION MISMATCHES")
print("="*100)
print("(Sorted by test/train percentage ratio)")
print()

# Filter to classes with at least 100 test samples to focus on significant mismatches
significant = results_df[results_df['test_samples'] >= 100].copy()
significant = significant.sort_values('mismatch_ratio', ascending=False)

print(f"{'Lithology':<30s} {'Train':>8s} {'Test':>8s} {'Train%':>8s} {'Test%':>8s} {'Ratio':>8s}")
print("-"*100)

for _, row in significant.head(20).iterrows():
    print(f"{row['lithology']:<30s} "
          f"{int(row['train_samples']):>8d} "
          f"{int(row['test_samples']):>8d} "
          f"{row['train_pct']:>7.2f}% "
          f"{row['test_pct']:>7.2f}% "
          f"{row['mismatch_ratio']:>7.1f}x")

print()
print("="*100)
print("CLASSES MISSING FROM TRAINING SET (but present in test)")
print("="*100)
print()

missing_from_train = results_df[(results_df['train_samples'] == 0) & (results_df['test_samples'] > 0)]
print(f"Found {len(missing_from_train)} classes with 0 training samples but >0 test samples")
print()

if len(missing_from_train) > 0:
    print(f"{'Lithology':<40s} {'Test Samples':>15s}")
    print("-"*100)
    for _, row in missing_from_train.sort_values('test_samples', ascending=False).iterrows():
        print(f"{row['lithology']:<40s} {int(row['test_samples']):>15d}")

print()
print("="*100)
print("ENTROPY ANALYSIS")
print("="*100)
print()

# Calculate entropy for each split
def entropy(counts):
    """Calculate Shannon entropy"""
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log2(probs))

train_entropy = entropy(train_dist.values)
val_entropy = entropy(val_dist.values)
test_entropy = entropy(test_dist.values)

print(f"Training set entropy:   {train_entropy:.4f}")
print(f"Validation set entropy: {val_entropy:.4f}")
print(f"Test set entropy:       {test_entropy:.4f}")
print()

print(f"Entropy difference (test vs train): {abs(test_entropy - train_entropy):.4f}")
print()

print("="*100)
