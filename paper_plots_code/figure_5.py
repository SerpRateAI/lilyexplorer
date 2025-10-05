"""
Figure 5: Extent of physical, chemical, and magnetic datasets paired
with lithologic descriptions
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Count rows in each dataset file
print("Counting data in each dataset file...")
datasets_dir = 'datasets/'
dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith('_DataLITH.csv')]

# Data structure for plot
data_counts = {}

for file in dataset_files:
    data_type = file.replace('_DataLITH.csv', '')
    filepath = os.path.join(datasets_dir, file)

    # Count rows (subtract 1 for header)
    with open(filepath, 'r') as f:
        count = sum(1 for line in f) - 1

    data_counts[data_type] = count
    print(f"{data_type}: {count:,} rows")

# Sort by count (descending)
sorted_data = sorted(data_counts.items(), key=lambda x: x[1], reverse=True)
labels = [item[0] for item in sorted_data]
counts = [item[1] for item in sorted_data]

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Create bar chart with viridis colormap
colors_map = plt.cm.viridis(np.linspace(0.2, 0.95, len(labels)))
bars = ax.bar(range(len(labels)), counts, color=colors_map, edgecolor='black',
              linewidth=0.5)

# Labels and formatting
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Rows of matched data', fontsize=11)
ax.set_xlabel('Dataset Type', fontsize=11)
ax.set_title('Extent of physical, chemical, and magnetic datasets', fontsize=12)

# Add value labels on top of bars for largest datasets
for i, (bar, count) in enumerate(zip(bars[:5], counts[:5])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count:,.0f}',
            ha='center', va='bottom', fontsize=8)

# Use linear scale (as in original)
ax.grid(True, alpha=0.3, axis='y')

# Add inset showing smaller datasets in detail
# (simplified version - full implementation would show specific expeditions)
axins = ax.inset_axes([0.6, 0.5, 0.35, 0.4])
smaller_data = sorted_data[-10:]
axins.barh(range(len(smaller_data)),
           [item[1] for item in smaller_data],
           color='gray', alpha=0.6)
axins.set_yticks(range(len(smaller_data)))
axins.set_yticklabels([item[0] for item in smaller_data], fontsize=7)
axins.set_xlabel('Count', fontsize=8)
axins.set_title('Inset', fontsize=9)
axins.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('paper_plots/figure_5.png', dpi=300, bbox_inches='tight')
print("Figure 5 saved successfully")
plt.close()
