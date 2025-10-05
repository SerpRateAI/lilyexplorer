"""
Figure 11: The new high-resolution GRA porosities (gray line) are plotted
along with MAD porosities (blue dots) for Hole U1387A
The GRA porosities for intervals with silty mud are plotted as red dots
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

print("Loading GRA and MAD datasets for Hole U1387A...")

# Load MAD data for specific hole
mad = pd.read_csv('datasets/MAD_DataLITH.csv')
mad_hole = mad[(mad['Site'] == 'U1387') & (mad['Hole'] == 'A')]

# Load GRA data for specific hole (sample to reduce memory)
gra_cols = ['Site', 'Hole', 'Depth CSF-A (m)', 'Bulk density (GRA)', 'Principal']
gra = pd.read_csv('datasets/GRA_DataLITH.csv', usecols=gra_cols, low_memory=False)
gra_hole = gra[(gra['Site'] == 'U1387') & (gra['Hole'] == 'A')]

# Calculate porosity from GRA bulk density
# Porosity = (grain_density - bulk_density) / (grain_density - fluid_density)
# For nannofossil mud: grain_density ≈ 2.766 g/cm³ (from paper Table 3)
# For silty mud: grain_density ≈ 2.761 g/cm³
# Fluid density = 1.024 g/cm³

gra_hole = gra_hole.copy()
gra_hole['Grain_density'] = 2.766  # Default for nannofossil mud
gra_hole.loc[gra_hole['Principal'] == 'silty mud', 'Grain_density'] = 2.761

gra_hole['Porosity_GRA'] = ((gra_hole['Grain_density'] - gra_hole['Bulk density (GRA)']) /
                             (gra_hole['Grain_density'] - 1.024)) * 100

# Create figure
fig, ax = plt.subplots(figsize=(8, 12))

# Focus on depth range shown in paper (approximately 100-200 m)
depth_min, depth_max = 100, 200

# Filter data for plot range
gra_plot = gra_hole[(gra_hole['Depth CSF-A (m)'] >= depth_min) &
                    (gra_hole['Depth CSF-A (m)'] <= depth_max)]
mad_plot = mad_hole[(mad_hole['Depth CSF-A (m)'] >= depth_min) &
                    (mad_hole['Depth CSF-A (m)'] <= depth_max)]

# Plot GRA porosity as gray line
gra_plot = gra_plot.sort_values('Depth CSF-A (m)')
ax.plot(gra_plot['Porosity_GRA'], gra_plot['Depth CSF-A (m)'],
        'gray', linewidth=1.0, alpha=0.7, label='GRA porosity', zorder=1)

# Highlight silty mud intervals in red
silty_mud = gra_plot[gra_plot['Principal'] == 'silty mud']
if len(silty_mud) > 0:
    ax.scatter(silty_mud['Porosity_GRA'], silty_mud['Depth CSF-A (m)'],
               c='red', s=2, alpha=0.6, label='GRA porosity\nfor silty mud',
               zorder=2)

# Plot MAD porosity as blue dots
ax.scatter(mad_plot['Porosity (vol%)'], mad_plot['Depth CSF-A (m)'],
           c='blue', s=30, alpha=0.7, edgecolors='darkblue',
           linewidths=0.5, label='MAD porosity', zorder=3)

# Labels and formatting
ax.set_xlabel('Porosity (vol%)', fontsize=12)
ax.set_ylabel('Depth CSF-A (m)', fontsize=12)
ax.set_xlim(40, 60)
ax.set_ylim(depth_max, depth_min)  # Inverted y-axis (depth increases down)
ax.set_title('Hole U1387A', fontsize=13)
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('paper_plots/figure_11.png', dpi=300, bbox_inches='tight')
print("Figure 11 saved successfully")
plt.close()
