"""
Figure 6: Histograms of MAD data showing distributions of combined data
and differences in distributions for two lithologies (nannofossil chalk and basalt)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load MAD data
print("Loading MAD dataset...")
df = pd.read_csv('datasets/MAD_DataLITH.csv')

# Clean data - remove missing values
df = df.dropna(subset=['Bulk density (g/cm^3)', 'Porosity (vol%)', 'Grain density (g/cm^3)'])

# Filter for specific lithologies
nannofossil_chalk = df[df['Principal'] == 'nannofossil chalk']
basalt = df[df['Principal'] == 'basalt']

# Create figure with 3x2 subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Top row: Combined MAD data
# Bulk Density - use many narrow bins like original
axes[0, 0].hist(df['Bulk density (g/cm^3)'], bins=100, color='#2d8659',
                edgecolor='black', linewidth=0.3, alpha=0.7)
axes[0, 0].set_xlabel('Density (g/cm³)', fontsize=10)
axes[0, 0].set_ylabel('Count', fontsize=10)
axes[0, 0].set_title('Bulk Density', fontsize=11)
axes[0, 0].set_xlim(1.0, 3.0)
axes[0, 0].set_ylim(0, 17000)
axes[0, 0].text(0.05, 0.95, 'Combined\nMAD data', transform=axes[0, 0].transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Porosity
axes[0, 1].hist(df['Porosity (vol%)'], bins=100, color='#2d8659',
                edgecolor='black', linewidth=0.3, alpha=0.7)
axes[0, 1].set_xlabel('Porosity (vol%)', fontsize=10)
axes[0, 1].set_ylabel('Count', fontsize=10)
axes[0, 1].set_title('Porosity', fontsize=11)
axes[0, 1].set_xlim(0, 100)
axes[0, 1].set_ylim(0, 5500)

# Grain Density
axes[0, 2].hist(df['Grain density (g/cm^3)'], bins=100, color='#2d8659',
                edgecolor='black', linewidth=0.3, alpha=0.7)
axes[0, 2].set_xlabel('Density (g/cm³)', fontsize=10)
axes[0, 2].set_ylabel('Count', fontsize=10)
axes[0, 2].set_title('Grain Density', fontsize=11)
axes[0, 2].set_xlim(2.2, 3.2)
axes[0, 2].set_ylim(0, 22000)

# Bottom row: Comparison of Nannofossil Chalk and Basalt
# Bulk Density comparison
axes[1, 0].hist(nannofossil_chalk['Bulk density (g/cm^3)'], bins=25, color='blue',
                edgecolor='black', alpha=0.6, label='Nannofossil\nChalk')
axes[1, 0].hist(basalt['Bulk density (g/cm^3)'], bins=25, color='red',
                edgecolor='black', alpha=0.6, label='Basalt')
axes[1, 0].set_xlabel('Density (g/cm³)', fontsize=10)
axes[1, 0].set_ylabel('Count', fontsize=10)
axes[1, 0].set_xlim(1.0, 3.0)
axes[1, 0].legend(fontsize=9, frameon=True, fancybox=True)

# Porosity comparison
axes[1, 1].hist(nannofossil_chalk['Porosity (vol%)'], bins=25, color='blue',
                edgecolor='black', alpha=0.6, label='Nannofossil\nChalk')
axes[1, 1].hist(basalt['Porosity (vol%)'], bins=25, color='red',
                edgecolor='black', alpha=0.6, label='Basalt')
axes[1, 1].set_xlabel('Porosity (vol%)', fontsize=10)
axes[1, 1].set_ylabel('Count', fontsize=10)
axes[1, 1].set_xlim(0, 100)
axes[1, 1].legend(fontsize=9, frameon=True, fancybox=True)

# Grain Density comparison
axes[1, 2].hist(nannofossil_chalk['Grain density (g/cm^3)'], bins=25, color='blue',
                edgecolor='black', alpha=0.6, label='Nannofossil\nChalk')
axes[1, 2].hist(basalt['Grain density (g/cm^3)'], bins=25, color='red',
                edgecolor='black', alpha=0.6, label='Basalt')
axes[1, 2].set_xlabel('Density (g/cm³)', fontsize=10)
axes[1, 2].set_ylabel('Count', fontsize=10)
axes[1, 2].set_xlim(2.2, 3.2)
axes[1, 2].legend(fontsize=9, frameon=True, fancybox=True)

plt.tight_layout()
plt.savefig('paper_plots/figure_6.png', dpi=300, bbox_inches='tight')
print("Figure 6 saved successfully")
plt.close()
