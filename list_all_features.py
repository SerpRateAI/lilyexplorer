"""
List all available features across all LILY datasets
"""
import pandas as pd
from pathlib import Path

datasets_dir = Path('/home/utig5/johna/bhai/datasets')

# Get all CSV files
csv_files = sorted(datasets_dir.glob('*_DataLITH.csv'))

print("="*80)
print("LILY DATABASE - AVAILABLE FEATURES BY DATASET")
print("="*80)
print()

for csv_file in csv_files:
    dataset_name = csv_file.stem.replace('_DataLITH', '')

    try:
        # Read just the header
        df = pd.read_csv(csv_file, nrows=0)

        # Get measurement columns (exclude common metadata)
        common_cols = ['Exp', 'Site', 'Hole', 'Core', 'Type', 'Sect', 'A/W',
                       'Depth CSF-A (m)', 'Depth CSF-B (m)',
                       'Prefix', 'Principal', 'Suffix', 'Full Lithology', 'Simplified Lithology',
                       'Latitude (DD)', 'Longitude (DD)', 'Water Depth (mbsl)',
                       'Expanded Core Type', 'Borehole_ID']

        measurement_cols = [col for col in df.columns if col not in common_cols]

        # Count rows
        row_count = sum(1 for _ in open(csv_file)) - 1  # -1 for header

        print(f"{dataset_name:10s} ({row_count:>10,} measurements)")
        print(f"  Measurement features:")
        for col in measurement_cols:
            print(f"    - {col}")
        print()

    except Exception as e:
        print(f"{dataset_name:10s} - Error: {e}")
        print()

print("="*80)
print("Common metadata fields (present in all datasets):")
print("="*80)
print("  - Exp, Site, Hole, Core, Type, Sect, A/W")
print("  - Depth CSF-A (m), Depth CSF-B (m)")
print("  - Lithology: Prefix, Principal, Suffix, Full Lithology, Simplified Lithology")
print("  - Location: Latitude (DD), Longitude (DD), Water Depth (mbsl)")
print("  - Expanded Core Type (coring system: APC, XCB, RCB, etc.)")
print("  - Borehole_ID")
