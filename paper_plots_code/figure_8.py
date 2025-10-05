"""
Figure 8: Mean and median densities table for principal lithologies
This is Table 3 in the paper, shown as a table image (not a plot)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading MAD dataset...")
df = pd.read_csv('datasets/MAD_DataLITH.csv')

# Clean data
df = df.dropna(subset=['Grain density (g/cm^3)', 'Dry density (g/cm^3)',
                       'Bulk density (g/cm^3)', 'Porosity (vol%)', 'Principal'])

# Group by principal lithology and calculate statistics
lith_stats = df.groupby('Principal').agg({
    'Grain density (g/cm^3)': ['count', 'mean', 'std'],
    'Bulk density (g/cm^3)': ['mean', 'std'],
    'Dry density (g/cm^3)': ['mean', 'std'],
    'Porosity (vol%)': ['mean', 'std']
}).reset_index()

# Flatten column names
lith_stats.columns = ['Principal lithology', 'N', 'Grain Mean', 'Grain σ',
                      'Bulk Mean', 'Bulk σ', 'Dry Mean', 'Dry σ',
                      'Porosity Mean', 'Porosity σ']

# Filter for lithologies with >50 observations
lith_stats = lith_stats[lith_stats['N'] > 50]

# Sort by grain density descending
lith_stats = lith_stats.sort_values('Grain Mean', ascending=False).reset_index(drop=True)

# Create figure for table
fig, ax = plt.subplots(figsize=(14, 12))
ax.axis('tight')
ax.axis('off')

# Format the data for display
table_data = []
# Header row
header = ['Principal lithology', 'N', 'Grain density', '', 'Bulk density', '',
          'Dry density', '', 'Porosity', '']
subheader = ['', '', 'Mean', 'deviation', 'Mean', 'deviation',
             'Mean', 'deviation', 'Mean', 'deviation']

table_data.append(header)
table_data.append(subheader)

# Data rows
for idx, row in lith_stats.head(60).iterrows():
    table_data.append([
        row['Principal lithology'],
        f"{int(row['N'])}",
        f"{row['Grain Mean']:.3f}",
        f"{row['Grain σ']:.3f}",
        f"{row['Bulk Mean']:.3f}",
        f"{row['Bulk σ']:.3f}",
        f"{row['Dry Mean']:.3f}",
        f"{row['Dry σ']:.3f}",
        f"{row['Porosity Mean']:.1f}",
        f"{row['Porosity σ']:.1f}"
    ])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.15, 0.05, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.08])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 1.5)

# Bold header rows
for i in range(10):
    table[(0, i)].set_facecolor('#E0E0E0')
    table[(0, i)].set_text_props(weight='bold', fontsize=8)
    table[(1, i)].set_facecolor('#F0F0F0')
    table[(1, i)].set_text_props(weight='bold', fontsize=7)

# Add borders
for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)
    cell.set_edgecolor('black')

# Add title
plt.title('Table 3. Mean (and standard deviation) of grain, bulk, dry densities and porosity\n' +
          'for principal lithologies with more than 50 observations',
          fontsize=11, pad=20, weight='bold')

# Add note at bottom
fig.text(0.5, 0.05,
         'Note: A more complete list of grain densities, bulk densities, dry densities, and porosities '
         'for the principal lithologies is given in the supporting information.\nStandard deviations '
         'less than the mean, median, and their standard deviations are recommended for use when '
         'representative mean values from this dataset as the most representative estimate of the true values.',
         ha='center', fontsize=7, style='italic', wrap=True)

plt.tight_layout()
plt.savefig('paper_plots/figure_8.png', dpi=300, bbox_inches='tight')
print("Figure 8 (Table 3) saved successfully")
plt.close()
