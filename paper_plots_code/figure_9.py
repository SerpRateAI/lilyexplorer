"""
Figure 9: Mean bulk densities and porosities and their standard deviations
for some of the principal lithologies with more than 50 observations each
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load MAD data
print("Loading MAD dataset...")
df = pd.read_csv('datasets/MAD_DataLITH.csv')

# Clean data
df = df.dropna(subset=['Bulk density (g/cm^3)', 'Porosity (vol%)', 'Principal'])

# Group by principal lithology and calculate statistics
lith_stats = df.groupby('Principal').agg({
    'Bulk density (g/cm^3)': ['mean', 'std', 'count'],
    'Porosity (vol%)': ['mean', 'std']
}).reset_index()

# Flatten column names
lith_stats.columns = ['Principal', 'Bulk_mean', 'Bulk_std', 'Count', 'Por_mean', 'Por_std']

# Filter for lithologies with >50 observations
lith_stats = lith_stats[lith_stats['Count'] > 50]

# Sort by bulk density (descending)
lith_stats = lith_stats.sort_values('Bulk_mean', ascending=False).reset_index(drop=True)

# Select top lithologies for display (to match paper)
n_display = min(60, len(lith_stats))
lith_stats = lith_stats.head(n_display)

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Set up positions
x = np.arange(len(lith_stats))
width = 0.8

# Plot bulk density as red squares with error bars
ax.errorbar(x, lith_stats['Bulk_mean'], yerr=lith_stats['Bulk_std'],
            fmt='rs', markersize=6, capsize=4, capthick=1.5, ecolor='red',
            elinewidth=1.5, label='Bulk Density Mean\nand Standard Deviation')

# Create twin axis for porosity
ax2 = ax.twinx()

# Plot porosity as blue diamonds with error bars
ax2.errorbar(x, lith_stats['Por_mean'], yerr=lith_stats['Por_std'],
             fmt='bo', markersize=6, capsize=4, capthick=1.5, ecolor='blue',
             elinewidth=1.5, label='Porosity Mean\nand Standard Deviation')

# Labels and formatting
ax.set_xlabel('', fontsize=11)
ax.set_ylabel('Bulk Density (g/cm³)', fontsize=12, color='red')
ax2.set_ylabel('Porosity (vol%)', fontsize=12, color='blue')

ax.set_xticks(x)
ax.set_xticklabels(lith_stats['Principal'], rotation=90, ha='right', fontsize=8)
ax.set_ylim(1.0, 3.0)
ax2.set_ylim(0, 100)

ax.tick_params(axis='y', labelcolor='red')
ax2.tick_params(axis='y', labelcolor='blue')

ax.grid(True, alpha=0.3, axis='y')

# Title
plt.title('MAD Bulk Density and Porosity by Lithology', fontsize=13, pad=20)

# Combined legend
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('paper_plots/figure_9.png', dpi=300, bbox_inches='tight')
print("Figure 9 saved successfully")
plt.close()
