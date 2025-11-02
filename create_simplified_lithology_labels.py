"""
Create simplified lithology grouping for more meaningful clustering evaluation.

Problem: 139 fine-grained lithologies, 66 with <100 samples
Solution: Group into ~15-20 major lithology types based on sediment classification

Grouping strategy:
1. Biogenic sediments (oozes, chalks)
2. Siliciclastic sediments (clay, silt, sand, mud)
3. Carbonate rocks (limestone, wackestone, packstone, grainstone)
4. Igneous rocks (basalt, gabbro)
5. Mixed/other
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_lithology_groups():
    """
    Define simplified lithology groupings based on sediment type.

    Returns hierarchical mapping: original → simplified
    """

    # Biogenic oozes and chalks
    biogenic_ooze = [
        'nannofossil ooze', 'diatom ooze', 'calcareous ooze',
        'foraminifera ooze', 'radiolarian ooze', 'siliceous ooze',
        'foraminifer ooze', 'nannofossil-rich clay', 'diatom-rich clay'
    ]

    biogenic_chalk = [
        'nannofossil chalk', 'chalk', 'foraminifer chalk',
        'calcareous chalk', 'nannofossil-rich chalk'
    ]

    # Clays and clayey sediments
    clay_sediments = [
        'clay', 'nannofossil clay', 'silty clay', 'claystone',
        'silty claystone', 'sandy clay', 'diatom clay',
        'calcareous clay', 'clayey silt', 'clay with ash'
    ]

    # Muds and mudstones
    mud_sediments = [
        'mud', 'mudstone', 'pebbly mud', 'sandy mud',
        'silty mud', 'clayey mud', 'diatom mud'
    ]

    # Silts and siltstones
    silt_sediments = [
        'silt', 'siltstone', 'sandy silt', 'clayey silt',
        'interbedded silt and clay', 'interbedded sand and silt'
    ]

    # Sands and sandstones
    sand_sediments = [
        'sand', 'sandstone', 'silty sand', 'medium to coarse sandstone',
        'fine sandstone', 'coarse sandstone', 'pebbly sand',
        'clayey sand', 'volcanic sand'
    ]

    # Diamicts (glacial sediments)
    diamict = [
        'clast-poor sandy diamict', 'clast-rich sandy diamict',
        'diamict', 'sandy diamict', 'muddy diamict'
    ]

    # Carbonate rocks
    limestone = [
        'wackestone', 'packstone', 'grainstone', 'boundstone',
        'floatstone', 'rudstone', 'framestone', 'bafflestone'
    ]

    # Igneous - basaltic
    basalt = [
        'basalt', 'aphyric basalt', 'plagioclase-phyric basalt',
        'olivine-phyric basalt', 'phyric basalt', 'pillow basalt',
        'basaltic breccia', 'altered basalt'
    ]

    # Igneous - gabbroic
    gabbro = [
        'gabbro', 'olivine gabbro', 'oxide gabbro',
        'gabbronorite', 'troctolite', 'diabase', 'dolerite'
    ]

    # Volcaniclastic
    volcanic = [
        'lapilli tuff', 'tuff', 'ash', 'volcanic ash',
        'vitric tuff', 'crystal tuff', 'lapillistone',
        'tuff breccia', 'volcanic breccia'
    ]

    # Create mapping dictionary
    mapping = {}

    for lith in biogenic_ooze:
        mapping[lith] = 'Biogenic ooze'
    for lith in biogenic_chalk:
        mapping[lith] = 'Biogenic chalk'
    for lith in clay_sediments:
        mapping[lith] = 'Clay/Claystone'
    for lith in mud_sediments:
        mapping[lith] = 'Mud/Mudstone'
    for lith in silt_sediments:
        mapping[lith] = 'Silt/Siltstone'
    for lith in sand_sediments:
        mapping[lith] = 'Sand/Sandstone'
    for lith in diamict:
        mapping[lith] = 'Diamict'
    for lith in limestone:
        mapping[lith] = 'Limestone'
    for lith in basalt:
        mapping[lith] = 'Basalt'
    for lith in gabbro:
        mapping[lith] = 'Gabbro'
    for lith in volcanic:
        mapping[lith] = 'Volcaniclastic'

    return mapping

def simplify_lithology_labels(data_path, output_path):
    """Apply simplified lithology grouping to dataset."""

    print("="*80)
    print("LITHOLOGY LABEL SIMPLIFICATION")
    print("="*80)

    # Load data
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df):,} samples")
    print(f"Original lithologies: {df['Principal'].nunique()}")

    # Get mapping
    mapping = create_lithology_groups()

    # Apply mapping
    df['Lithology_Simplified'] = df['Principal'].map(mapping)

    # Unmapped lithologies → "Other"
    unmapped_mask = df['Lithology_Simplified'].isna()
    df.loc[unmapped_mask, 'Lithology_Simplified'] = 'Other'

    # Statistics
    print(f"Simplified lithologies: {df['Lithology_Simplified'].nunique()}")
    print(f"Unmapped → 'Other': {unmapped_mask.sum():,} samples")

    print("\nSimplified lithology distribution:")
    print(df['Lithology_Simplified'].value_counts())

    # Save
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to {output_path}")
    print(f"  Dataset size: {output_path.stat().st_size / 1024**2:.1f} MB")

    # Show what got grouped
    print("\n" + "="*80)
    print("GROUPING EXAMPLES")
    print("="*80)
    for simplified in sorted(df['Lithology_Simplified'].unique()):
        originals = df[df['Lithology_Simplified'] == simplified]['Principal'].unique()
        count = len(df[df['Lithology_Simplified'] == simplified])
        print(f"\n{simplified} ({count:,} samples):")
        for orig in sorted(originals)[:5]:  # Show first 5
            orig_count = len(df[df['Principal'] == orig])
            print(f"  - {orig} ({orig_count:,})")
        if len(originals) > 5:
            print(f"  ... and {len(originals)-5} more")

    print("\n" + "="*80)

if __name__ == "__main__":
    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    output_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm_simplified.csv')

    simplify_lithology_labels(data_path, output_path)
