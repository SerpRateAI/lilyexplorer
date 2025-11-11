"""
Create a summary of total borehole depths from GRA data.
This avoids loading the entire GRA dataset repeatedly.
"""

import pandas as pd
import numpy as np

print("Loading GRA data in chunks to build borehole depth summary...")

borehole_depths = {}
chunk_size = 100000
chunk_num = 0

for chunk in pd.read_csv('datasets/GRA_DataLITH.csv',
                         usecols=['Exp', 'Site', 'Hole', 'Depth CSF-A (m)'],
                         chunksize=chunk_size,
                         low_memory=False):
    chunk_num += 1
    print(f"  Processing chunk {chunk_num} ({len(chunk):,} rows)...")

    # Create Borehole_ID
    chunk['Borehole_ID'] = chunk['Exp'].astype(str) + '_' + chunk['Site'].astype(str) + '_' + chunk['Hole'].astype(str)

    # Update min/max for each borehole
    for bh_id in chunk['Borehole_ID'].unique():
        bh_data = chunk[chunk['Borehole_ID'] == bh_id]
        min_depth = bh_data['Depth CSF-A (m)'].min()
        max_depth = bh_data['Depth CSF-A (m)'].max()

        if bh_id in borehole_depths:
            borehole_depths[bh_id]['min_depth'] = min(borehole_depths[bh_id]['min_depth'], min_depth)
            borehole_depths[bh_id]['max_depth'] = max(borehole_depths[bh_id]['max_depth'], max_depth)
        else:
            borehole_depths[bh_id] = {'min_depth': min_depth, 'max_depth': max_depth}

print(f"\n✓ Processed {chunk_num} chunks")
print(f"✓ Found depth ranges for {len(borehole_depths)} boreholes")

# Convert to DataFrame and save
summary_data = []
for bh_id, depths in borehole_depths.items():
    summary_data.append({
        'Borehole_ID': bh_id,
        'total_min_depth': depths['min_depth'],
        'total_max_depth': depths['max_depth'],
        'total_depth_range': depths['max_depth'] - depths['min_depth']
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Borehole_ID')
summary_df.to_csv('borehole_total_depths.csv', index=False)

print(f"\n✓ Saved summary to borehole_total_depths.csv")
print(f"  Depth range: {summary_df['total_min_depth'].min():.1f} - {summary_df['total_max_depth'].max():.1f} m")
print(f"  Mean total depth: {summary_df['total_depth_range'].mean():.1f} m")
print(f"  Longest borehole: {summary_df['total_depth_range'].max():.1f} m")
