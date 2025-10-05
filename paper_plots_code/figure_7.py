"""
Figure 7: (a) MAD bulk densities versus porosity data and (b) MAD grain densities versus porosity
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load MAD data
print("Loading MAD dataset...")
df = pd.read_csv('datasets/MAD_DataLITH.csv')

# Clean data
df = df.dropna(subset=['Bulk density (g/cm^3)', 'Porosity (vol%)', 'Grain density (g/cm^3)'])

# Get specific lithologies for highlighting
nannofossil_ooze = df[df['Principal'] == 'nannofossil ooze']
diatom_ooze = df[df['Principal'] == 'diatom ooze']
basalt = df[df['Principal'] == 'basalt']
gabbro = df[df['Principal'] == 'gabbro']

# Create figure with marginal histograms using gridspec
from matplotlib import gridspec

fig = plt.figure(figsize=(16, 7))
gs = gridspec.GridSpec(2, 4, height_ratios=[1, 4], width_ratios=[4, 1, 4, 1],
                       hspace=0.02, wspace=0.02)

# Main scatter plots
ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 2])

# Top histograms
ax1_histx = fig.add_subplot(gs[0, 0], sharex=ax1)
ax2_histx = fig.add_subplot(gs[0, 2], sharex=ax2)

# Right histograms
ax1_histy = fig.add_subplot(gs[1, 1], sharey=ax1)
ax2_histy = fig.add_subplot(gs[1, 3], sharey=ax2)

# Panel (a): Bulk Density vs Porosity
ax1.scatter(df['Bulk density (g/cm^3)'], df['Porosity (vol%)'],
            edgecolors="none", s=5, c='gray', alpha=0.3, label='All MAD data')
ax1.scatter(nannofossil_ooze['Bulk density (g/cm^3)'], nannofossil_ooze['Porosity (vol%)'],
            edgecolors="none", s=5, c='green', alpha=0.3, label='Nannofossil ooze')
ax1.scatter(diatom_ooze['Bulk density (g/cm^3)'], diatom_ooze['Porosity (vol%)'],
            edgecolors="none", s=5, c='orange', alpha=0.3, label='Diatom ooze')
ax1.scatter(basalt['Bulk density (g/cm^3)'], basalt['Porosity (vol%)'],
            edgecolors="none", s=5, c='red', alpha=0.3, label='Basalt')
ax1.scatter(gabbro['Bulk density (g/cm^3)'], gabbro['Porosity (vol%)'],
            edgecolors="none", s=5, c='purple', alpha=0.3, label='Gabbro')

# Add theoretical lines for different grain densities
# Porosity = (ρ_fluid - ρ_grain)/(ρ_bulk - ρ_grain) * 100
# where ρ_fluid = 1.024 g/cm³
rho_fluid = 1.024
bulk_range = np.linspace(1.0, 3.0, 100)

# Nannofossil ooze line (grain density ~2.72)
porosity_nanno = (1 - (bulk_range - rho_fluid) / (2.72 - rho_fluid)) * 100
ax1.plot(bulk_range, porosity_nanno, 'g-', linewidth=1.5, label='Nannofossil ooze mean')

# Diatom ooze line (grain density ~2.46)
porosity_diatom = (1 - (bulk_range - rho_fluid) / (2.46 - rho_fluid)) * 100
ax1.plot(bulk_range, porosity_diatom, 'orange', linestyle='--', linewidth=1.5,
         label='Diatom ooze mean')

# Basalt line (grain density ~2.89)
porosity_basalt = (1 - (bulk_range - rho_fluid) / (2.89 - rho_fluid)) * 100
ax1.plot(bulk_range, porosity_basalt, 'r--', linewidth=1.5, label='Basalt mean')

# Add 1:1 line equivalent (MAD = GRA)
ax1.plot([1.0, 3.0], [100, 0], 'k--', linewidth=0.8, alpha=0.3)

ax1.set_xlabel('Bulk Density (g/cm³)', fontsize=11)
ax1.set_ylabel('Porosity (vol%)', fontsize=11)
ax1.set_xlim(1.0, 3.0)
ax1.set_ylim(0, 100)
ax1.legend(loc='upper right', fontsize=8, markerscale=3)
ax1.grid(True, alpha=0.3)
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes, fontsize=14,
         verticalalignment='top', fontweight='bold')

# Panel (b): Grain Density vs Porosity
ax2.scatter(df['Grain density (g/cm^3)'], df['Porosity (vol%)'],
            edgecolors="none", s=5, c='gray', alpha=0.3, label='All MAD data')
ax2.scatter(nannofossil_ooze['Grain density (g/cm^3)'], nannofossil_ooze['Porosity (vol%)'],
            edgecolors="none", s=5, c='green', alpha=0.3, label='Nannofossil ooze')
ax2.scatter(diatom_ooze['Grain density (g/cm^3)'], diatom_ooze['Porosity (vol%)'],
            edgecolors="none", s=5, c='orange', alpha=0.3, label='Diatom ooze')
ax2.scatter(basalt['Grain density (g/cm^3)'], basalt['Porosity (vol%)'],
            edgecolors="none", s=5, c='red', alpha=0.3, label='Basalt')
ax2.scatter(gabbro['Grain density (g/cm^3)'], gabbro['Porosity (vol%)'],
            edgecolors="none", s=5, c='purple', alpha=0.3, label='Gabbro')

# Vertical lines for expected grain densities
ax2.axvline(x=2.72, color='g', linestyle='-', linewidth=1.5, alpha=0.3)  # Nannofossil
ax2.axvline(x=2.46, color='orange', linestyle='--', linewidth=1.5, alpha=0.3)  # Diatom
ax2.axvline(x=2.89, color='r', linestyle='--', linewidth=1.5, alpha=0.3)  # Basalt

# Model line for incomplete drying effect
porosity_model = np.linspace(80, 95, 50)
grain_model = 2.65 - 0.015 * (porosity_model - 80) / 15
ax2.plot(grain_model, porosity_model, 'k^-', linewidth=1, markersize=3,
         alpha=0.3, label='Incomplete drying model')

ax2.set_xlabel('Grain Density (g/cm³)', fontsize=11)
ax2.set_ylabel('Porosity (vol%)', fontsize=11)
ax2.set_xlim(1.5, 3.5)
ax2.set_ylim(0, 100)
ax2.legend(loc='upper left', fontsize=8, markerscale=3)
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes, fontsize=14,
         verticalalignment='top', fontweight='bold')

# Add marginal histograms for panel (a)
ax1_histx.hist(df['Bulk density (g/cm^3)'].dropna(), bins=50, color='steelblue',
               edgecolor='black', linewidth=0.5)
ax1_histx.tick_params(axis='x', labelbottom=False)
ax1_histx.set_ylabel('Count', fontsize=8)

ax1_histy.hist(df['Porosity (vol%)'].dropna(), bins=50, orientation='horizontal',
               color='steelblue', edgecolor='black', linewidth=0.5)
ax1_histy.tick_params(axis='y', labelleft=False)
ax1_histy.set_xlabel('Count', fontsize=8)

# Add marginal histograms for panel (b)
ax2_histx.hist(df['Grain density (g/cm^3)'].dropna(), bins=50, color='steelblue',
               edgecolor='black', linewidth=0.5)
ax2_histx.tick_params(axis='x', labelbottom=False)
ax2_histx.set_ylabel('Count', fontsize=8)

ax2_histy.hist(df['Porosity (vol%)'].dropna(), bins=50, orientation='horizontal',
               color='steelblue', edgecolor='black', linewidth=0.5)
ax2_histy.tick_params(axis='y', labelleft=False)
ax2_histy.set_xlabel('Count', fontsize=8)

plt.savefig('paper_plots/figure_7.png', dpi=300, bbox_inches='tight')
print("Figure 7 saved successfully")
plt.close()
