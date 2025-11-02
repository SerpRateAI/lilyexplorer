"""
Create hierarchical lithology groupings for classification.

Groups 139 fine-grained lithologies into ~15-20 geological families.
"""

import pandas as pd
import numpy as np

# Load lithology distribution
dist_df = pd.read_csv('lithology_class_distribution.csv')

print("="*100)
print("CREATING LITHOLOGY HIERARCHY")
print("="*100)
print()
print(f"Total fine-grained lithologies: {len(dist_df)}")
print()

# Define lithology hierarchy (Principal lithology → Lithology Group)
hierarchy = {}

# CARBONATES (calcareous oozes, chalks, limestones)
carbonate_keywords = ['nannofossil', 'foraminifera', 'calcareous', 'chalk', 'limestone',
                      'wackestone', 'packstone', 'grainstone', 'rudstone', 'floatstone',
                      'boundstone', 'calcilutite', 'calcarenite', 'marl', 'marlstone',
                      'carbonate ooze', 'carbonate']
for lith in dist_df['Lithology']:
    if any(kw in lith.lower() for kw in carbonate_keywords):
        hierarchy[lith] = 'Carbonate'

# SILICICLASTIC - CLAYS/MUDS
clay_mud_keywords = ['clay', 'mud', 'mudstone']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in clay_mud_keywords):
        hierarchy[lith] = 'Clay/Mud'

# SILICICLASTIC - SILTS
silt_keywords = ['silt', 'siltstone']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in silt_keywords):
        hierarchy[lith] = 'Silt'

# SILICICLASTIC - SANDS
sand_keywords = ['sand', 'sandstone']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in sand_keywords):
        hierarchy[lith] = 'Sand'

# SILICICLASTIC - MIXED/INTERBEDDED
interbedded_keywords = ['interbedded']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in interbedded_keywords):
        hierarchy[lith] = 'Interbedded'

# SILICICLASTIC - DIAMICT
diamict_keywords = ['diamict', 'pebbly mud', 'debris flow']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in diamict_keywords):
        hierarchy[lith] = 'Diamict'

# SILICICLASTIC - CONGLOMERATE/BRECCIA
conglomerate_keywords = ['conglomerate', 'breccia', 'gravel']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in conglomerate_keywords):
        hierarchy[lith] = 'Conglomerate/Breccia'

# BIOGENIC SILICA (diatom, radiolarian, biosiliceous)
biosilica_keywords = ['diatom', 'radiolarian', 'biosiliceous', 'diatomite', 'porcellanite', 'chert']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in biosilica_keywords):
        hierarchy[lith] = 'Biogenic Silica'

# VOLCANICS - MAFIC (basalt, gabbro, boninite, dolerite, diabase)
mafic_keywords = ['basalt', 'gabbro', 'boninite', 'dolerite', 'diabase', 'gabbronorite',
                  'metabasite', 'hyaloclastite']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in mafic_keywords):
        hierarchy[lith] = 'Mafic Igneous'

# VOLCANICS - INTERMEDIATE/FELSIC
felsic_keywords = ['andesite', 'dacite', 'rhyodacite', 'felsic', 'diorite']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in felsic_keywords):
        hierarchy[lith] = 'Intermediate/Felsic Igneous'

# VOLCANICS - ULTRAMAFIC (harzburgite, dunite, serpentinite)
ultramafic_keywords = ['harzburgite', 'dunite', 'ultramafic', 'serpentinite']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in ultramafic_keywords):
        hierarchy[lith] = 'Ultramafic'

# VOLCANICLASTICS (ash, tuff, tephra, lapilli, agglomerate)
volcaniclastic_keywords = ['ash', 'tuff', 'tephra', 'lapilli', 'agglomerate', 'volcaniclastic']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in volcaniclastic_keywords):
        hierarchy[lith] = 'Volcaniclastic'

# METAMORPHIC
metamorphic_keywords = ['schist', 'granofels', 'marble', 'cataclasite', 'metavolcanic', 'metaclaystone']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in metamorphic_keywords):
        hierarchy[lith] = 'Metamorphic'

# EVAPORITES
evaporite_keywords = ['gypsum', 'anhydrite', 'dolostone']
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if any(kw in lith.lower() for kw in evaporite_keywords):
        hierarchy[lith] = 'Evaporite'

# CATCH-ALL (volcanic rock, lava, ooze without specific type, coal, etc.)
for lith in dist_df['Lithology']:
    if lith in hierarchy:
        continue
    if lith.lower() in ['volcanic rock', 'lava', 'ooze', 'coal']:
        hierarchy[lith] = 'Other'

# Any remaining unmapped lithologies
for lith in dist_df['Lithology']:
    if lith not in hierarchy:
        print(f"WARNING: Unmapped lithology: {lith}")
        hierarchy[lith] = 'Unclassified'

# Create mapping dataframe
mapping_df = pd.DataFrame([
    {'Principal_Lithology': lith, 'Lithology_Group': group}
    for lith, group in hierarchy.items()
])

# Merge with distribution data
mapping_df = mapping_df.merge(dist_df, left_on='Principal_Lithology', right_on='Lithology')
mapping_df = mapping_df[['Principal_Lithology', 'Lithology_Group', 'Count', 'Percentage']]
mapping_df = mapping_df.sort_values('Count', ascending=False)

# Save mapping
mapping_df.to_csv('lithology_hierarchy_mapping.csv', index=False)

print("✓ Hierarchy mapping created")
print()

# Summary statistics
print("="*100)
print("LITHOLOGY GROUP SUMMARY")
print("="*100)
print()

group_stats = mapping_df.groupby('Lithology_Group').agg({
    'Count': 'sum',
    'Principal_Lithology': 'count'
}).rename(columns={'Principal_Lithology': 'N_Lithologies'})

group_stats['Percentage'] = 100 * group_stats['Count'] / group_stats['Count'].sum()
group_stats = group_stats.sort_values('Count', ascending=False)

print(f"{'Lithology Group':<30s} {'N_Lithologies':>15s} {'Total_Samples':>15s} {'Percentage':>12s}")
print("-"*100)

for group, row in group_stats.iterrows():
    print(f"{group:<30s} {int(row['N_Lithologies']):>15d} {int(row['Count']):>15,d} {row['Percentage']:>11.2f}%")

print()
print(f"Total groups: {len(group_stats)}")
print(f"Total lithologies: {len(mapping_df)}")
print(f"Total samples: {mapping_df['Count'].sum():,}")
print()

# Check coverage
total_samples = dist_df['Count'].sum()
mapped_samples = mapping_df['Count'].sum()
print(f"Coverage: {100*mapped_samples/total_samples:.2f}% of samples mapped")
print()

print("="*100)
print(f"✓ Lithology hierarchy saved to: lithology_hierarchy_mapping.csv")
print("="*100)
