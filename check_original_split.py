"""Check what's special about the original v2.6.6 test split"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

# Reproduce exact original split
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)

print(f"Original v2.6.6 test set: {len(test_boreholes)} boreholes")
print()

# Analyze test set
test_mask = df['Borehole_ID'].isin(test_boreholes)
df_test = df[test_mask].copy()

print("Test set lithology distribution:")
lith_counts = df_test['Principal'].value_counts()
print(lith_counts.head(15))
print()

print(f"Test set unique lithologies: {df_test['Principal'].nunique()}")
print(f"Full dataset unique lithologies: {df['Principal'].nunique()}")
print()

# Check geographic distribution
gra = pd.read_csv('/home/utig5/johna/bhai/datasets/GRA_DataLITH.csv')
gra['Borehole_ID'] = gra['Exp'].astype(str) + '-' + gra['Site'] + '-' + gra['Hole']

test_locations = gra[gra['Borehole_ID'].isin(test_boreholes)].groupby('Borehole_ID').agg({
    'Latitude (DD)': 'first',
    'Longitude (DD)': 'first'
}).reset_index()

print("Test set geographic distribution:")
print(f"  Latitude range: [{test_locations['Latitude (DD)'].min():.1f}, {test_locations['Latitude (DD)'].max():.1f}]")
print(f"  Longitude range: [{test_locations['Longitude (DD)'].min():.1f}, {test_locations['Longitude (DD)'].max():.1f}]")
print()

# Compare to full dataset
all_locations = gra.groupby('Borehole_ID').agg({
    'Latitude (DD)': 'first',
    'Longitude (DD)': 'first'
}).reset_index()

print("Full dataset geographic distribution:")
print(f"  Latitude range: [{all_locations['Latitude (DD)'].min():.1f}, {all_locations['Latitude (DD)'].max():.1f}]")
print(f"  Longitude range: [{all_locations['Longitude (DD)'].min():.1f}, {all_locations['Longitude (DD)'].max():.1f}]")
print()

# Check lithology entropy (how diverse is test set)
from scipy.stats import entropy

test_entropy = entropy(lith_counts.values)
full_entropy = entropy(df['Principal'].value_counts().values)

print(f"Lithology diversity:")
print(f"  Test set entropy: {test_entropy:.3f}")
print(f"  Full dataset entropy: {full_entropy:.3f}")
print()

if test_entropy < full_entropy:
    print("⚠️ Test set is LESS diverse than full dataset (easier to cluster)")
else:
    print("✓ Test set is equally/more diverse than full dataset")

# Check dominant lithologies
print()
print("Dominant lithologies in test set:")
for lith, count in lith_counts.head(5).items():
    pct = 100 * count / len(df_test)
    full_pct = 100 * (df['Principal'] == lith).sum() / len(df)
    print(f"  {lith}: {pct:.1f}% (vs {full_pct:.1f}% in full dataset)")
