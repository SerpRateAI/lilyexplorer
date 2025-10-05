"""
Figure 2: Geographic location of sites for the 42 expeditions
Symbol size proportional to meters of lithology described for each site
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not available, using basic plot")

# Read one dataset to get location and expedition data
print("Loading dataset...")
df = pd.read_csv('datasets/MAD_DataLITH.csv')

# Calculate meters of lithology described per site
# Group by Site, Latitude, Longitude and count unique depth measurements
site_data = df.groupby(['Site', 'Latitude (DD)', 'Longitude (DD)']).agg({
    'Depth CSF-A (m)': 'count'
}).reset_index()
site_data.columns = ['Site', 'Latitude', 'Longitude', 'Count']

# Create figure with proper map projection
if HAS_CARTOPY:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5, zorder=1)
    ax.set_global()

    # Plot sites with size proportional to meters described
    sizes = (site_data['Count'] / site_data['Count'].max() * 1000) + 20

    scatter = ax.scatter(site_data['Longitude'], site_data['Latitude'],
                         s=sizes, c='red', alpha=0.6, edgecolors='darkred',
                         linewidth=0.5, zorder=5, transform=ccrs.PlateCarree())

    # Add legend showing size scale
    legend_sizes = [100, 500, 1000]
    legend_elements = [plt.scatter([], [], s=(sz/site_data['Count'].max()*1000)+20,
                                   c='red', alpha=0.6, edgecolors='darkred',
                                   label=f'{sz}\nMeters of\nlithology\ndescribed')
                       for sz in legend_sizes]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True,
              title='', fontsize=8)
else:
    # Fallback if cartopy not available
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Plot sites only
    sizes = (site_data['Count'] / site_data['Count'].max() * 1000) + 20
    scatter = ax.scatter(site_data['Longitude'], site_data['Latitude'],
                         s=sizes, c='red', alpha=0.6, edgecolors='darkred',
                         linewidth=0.5, zorder=5)

    # Add legend
    legend_sizes = [100, 500, 1000]
    legend_elements = [plt.scatter([], [], s=(sz/site_data['Count'].max()*1000)+20,
                                   c='red', alpha=0.6, edgecolors='darkred',
                                   label=f'{sz}\nMeters of\nlithology\ndescribed')
                       for sz in legend_sizes]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True,
              title='', fontsize=8)

    # Set map boundaries and labels
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=10)
    ax.set_ylabel('Latitude', fontsize=10)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_aspect('equal')

plt.title('Geographic location of sites for the 42 expeditions', fontsize=11)
plt.tight_layout()
plt.savefig('paper_plots/figure_2.png', dpi=300, bbox_inches='tight')
print("Figure 2 saved successfully")
plt.close()
