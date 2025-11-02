"""
Analyze lithology class distribution in the dataset.
Show percentage of samples per class.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('vae_training_data_v2_20cm.csv')

print("="*100)
print("LITHOLOGY CLASS DISTRIBUTION")
print("="*100)
print()

# Overall distribution
lithology_counts = df['Principal'].value_counts()
total_samples = len(df)

print(f"Total samples: {total_samples:,}")
print(f"Unique lithologies: {len(lithology_counts)}")
print()

# Calculate percentages
percentages = (lithology_counts / total_samples * 100)

print("="*100)
print("TOP 20 MOST COMMON LITHOLOGIES")
print("="*100)
print(f"{'Lithology':<40s} {'Count':>10s} {'Percentage':>12s} {'Cumulative':>12s}")
print("-"*100)

cumulative = 0
for i, (lithology, count) in enumerate(lithology_counts.head(20).items(), 1):
    pct = percentages[lithology]
    cumulative += pct
    print(f"{i:2d}. {lithology:<37s} {count:10,d} {pct:11.2f}% {cumulative:11.2f}%")

print()
print(f"Top 20 lithologies account for {cumulative:.2f}% of all samples")
print()

# Bottom 20 rarest
print("="*100)
print("BOTTOM 20 RAREST LITHOLOGIES")
print("="*100)
print(f"{'Lithology':<40s} {'Count':>10s} {'Percentage':>12s}")
print("-"*100)

for i, (lithology, count) in enumerate(lithology_counts.tail(20).items(), 1):
    pct = percentages[lithology]
    print(f"{lithology:<40s} {count:10,d} {pct:11.2f}%")

print()

# Distribution statistics
print("="*100)
print("CLASS IMBALANCE STATISTICS")
print("="*100)
print()

print(f"Most common class:  {lithology_counts.iloc[0]:,} samples ({percentages.iloc[0]:.2f}%)")
print(f"Least common class: {lithology_counts.iloc[-1]:,} samples ({percentages.iloc[-1]:.4f}%)")
print(f"Median samples per class: {lithology_counts.median():.0f}")
print(f"Mean samples per class: {lithology_counts.mean():.0f}")
print()

# Classes with <1% of data
rare_classes = lithology_counts[percentages < 1.0]
print(f"Classes with <1% of data: {len(rare_classes)}/{len(lithology_counts)} ({len(rare_classes)/len(lithology_counts)*100:.1f}%)")
print(f"Total samples in rare classes: {rare_classes.sum():,} ({rare_classes.sum()/total_samples*100:.2f}%)")
print()

# Classes with <0.1% of data
very_rare_classes = lithology_counts[percentages < 0.1]
print(f"Classes with <0.1% of data: {len(very_rare_classes)}/{len(lithology_counts)} ({len(very_rare_classes)/len(lithology_counts)*100:.1f}%)")
print(f"Total samples in very rare classes: {very_rare_classes.sum():,} ({very_rare_classes.sum()/total_samples*100:.2f}%)")
print()

# Imbalance ratio
imbalance_ratio = lithology_counts.iloc[0] / lithology_counts.iloc[-1]
print(f"Imbalance ratio (max/min): {imbalance_ratio:.1f}:1")
print()

# Train/val/test split analysis
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

print("="*100)
print("LITHOLOGY DISTRIBUTION ACROSS SPLITS")
print("="*100)
print()

print(f"Train: {train_mask.sum():,} samples, {df[train_mask]['Principal'].nunique()} unique lithologies")
print(f"Val:   {val_mask.sum():,} samples, {df[val_mask]['Principal'].nunique()} unique lithologies")
print(f"Test:  {test_mask.sum():,} samples, {df[test_mask]['Principal'].nunique()} unique lithologies")
print()

# Save full distribution
print("Saving full distribution to file...")
distribution_df = pd.DataFrame({
    'Lithology': lithology_counts.index,
    'Count': lithology_counts.values,
    'Percentage': percentages.values,
    'Rank': range(1, len(lithology_counts) + 1)
})

distribution_df.to_csv('lithology_class_distribution.csv', index=False)
print(f"✓ Saved to: lithology_class_distribution.csv")
print()

print("="*100)
