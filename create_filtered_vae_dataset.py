"""
Create filtered VAE training dataset (≥100 samples per class).

This ensures VAE pre-training uses exact same data as downstream classification.
"""

import pandas as pd

print("="*100)
print("CREATING FILTERED VAE DATASET")
print("="*100)
print()

# Load original dataset
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"Original dataset: {len(df):,} samples")

# Load hierarchy mapping
hierarchy_df = pd.read_csv('/home/utig5/johna/bhai/lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'], hierarchy_df['Lithology_Group']))

# Map to lithology groups
df['Lithology_Group'] = df['Principal'].map(principal_to_group)
df = df.dropna(subset=['Lithology_Group'])

# Count samples per group
group_counts = df['Lithology_Group'].value_counts()
print("\nOriginal class distribution:")
for group, count in group_counts.items():
    print(f"  {group:40s}: {count:>8,} samples")

# Filter to ≥100 samples
MIN_SAMPLES = 100
valid_groups = group_counts[group_counts >= MIN_SAMPLES].index.tolist()
df_filtered = df[df['Lithology_Group'].isin(valid_groups)].copy()

print(f"\nFiltering to groups with ≥{MIN_SAMPLES} samples...")
removed_groups = group_counts[group_counts < MIN_SAMPLES].index.tolist()
print(f"Groups removed ({len(removed_groups)}): {', '.join(removed_groups)}")
print(f"Samples removed: {len(df) - len(df_filtered):,}")

print(f"\nFiltered dataset: {len(df_filtered):,} samples ({len(df_filtered)/len(df)*100:.2f}%)")
print(f"Classes remaining: {len(valid_groups)}")

print("\nFinal class distribution:")
for group in sorted(valid_groups):
    count = (df_filtered['Lithology_Group'] == group).sum()
    print(f"  {group:40s}: {count:>8,} samples")

# Drop the temporary Lithology_Group column (keep original Principal)
df_filtered = df_filtered.drop('Lithology_Group', axis=1)

# Save filtered dataset
output_path = '/home/utig5/johna/bhai/vae_training_data_v2_20cm_filtered_100.csv'
df_filtered.to_csv(output_path, index=False)
print(f"\n✓ Saved to: {output_path}")
print(f"  Columns: {list(df_filtered.columns)}")
print(f"  Size: {len(df_filtered):,} rows × {len(df_filtered.columns)} columns")

print("\n" + "="*100)
print("FILTERED VAE DATASET CREATED")
print("="*100)
