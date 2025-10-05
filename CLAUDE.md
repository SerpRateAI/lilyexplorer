# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the LILY Database (LIMS with Lithology) - a comprehensive dataset linking lithological information to physical, chemical, and magnetic properties data from IODP (International Ocean Discovery Program) expeditions 2009-2019. The dataset contains over 34 million measurements from 89 km of core recovered across 42 expeditions.

## Key Reference Paper

The paper at `/papers/lilypaper.pdf` (Childress et al., 2024, Geochemistry, Geophysics, Geosystems) describes the complete dataset construction, methodology, and boundary conditions. **Read this paper first** to understand the data structure and scientific context.

## Data Structure

### Dataset Location
All datasets are located in `/datasets/` (note: CLAUDE.md mentions `/data_set` but actual directory is `/datasets/`)

### Available Datasets
The repository contains multiple CSV files with different measurement types, all following the naming pattern `{MEASUREMENT_TYPE}_DataLITH.csv`:

- **GRA_DataLITH.csv** - Gamma ray attenuation bulk density (largest file, >1GB, 3.7M+ measurements)
- **MAD_DataLITH.csv** - Moisture and density measurements (grain density, bulk density, porosity)
- **RGB_DataLITH.csv** - Digital image color data (largest dataset by count, >10M measurements)
- **MS_DataLITH.csv** - Magnetic susceptibility
- **NGR_DataLITH.csv** - Natural gamma radiation
- **RSC_DataLITH.csv** - Reflectance spectroscopy
- **SRM_DataLITH.csv** - Natural remanent magnetization
- **IW_DataLITH.csv** - Interstitial water chemistry
- **CARB_DataLITH.csv** - Carbonate content
- **ICP_DataLITH.csv** - Inductively coupled plasma measurements
- Plus 14 additional measurement types (see paper Table S1 for complete list)

### Common Data Fields
All datasets share core identification fields:
- Expedition, Site, Hole, Core, Type, Section (Sect), Archive/Working half (A/W)
- Depth measurements: `Depth CSF-A (m)` and `Depth CSF-B (m)`
- Lithology: `Prefix`, `Principal`, `Suffix`, `Full Lithology`, `Simplified Lithology`
- Location metadata: `Latitude (DD)`, `Longitude (DD)`, `Water Depth (mbsl)`
- `Expanded Core Type`: Coring system (APC, HLAPC, XCB, RCB, etc.)

### Lithology Classification
Lithologies follow modified Mazzullo et al. (1988) and Dean et al. (1985) schemes:
- **Principal lithology**: Dominant composition (>50-60%)
- **Prefix (major modifier)**: Components 25-50% of composition
- **Suffix (minor modifier)**: Components 10-25% of composition

Common principal lithologies include: nannofossil ooze, clay, silty clay, diatom ooze, basalt, mud, chalk, sand, etc.

## Primary Task: Figure Reproduction

The main objective is to recreate all plots from the paper in `/papers/lilypaper.pdf`.

### Workflow
1. **Output directory**: Generate Python scripts in `/paper_plots_code/`
2. **Script naming**: `figure_{number}.py` for each figure in the paper
3. **Image output**: Save plots as PNG in `/paper_plots/`
4. **Execution script**: Create `generate_paper_plots.sh` in repo root
5. **Validation**: Compare generated plots with originals, iterate to improve accuracy
6. **Documentation**: Create `data_comparison.pdf` with side-by-side comparisons

### Critical Rules
- **Never fabricate data** - all data must come from the datasets in `/datasets/`
- Use Python for all code unless explicitly specified otherwise
- Match original plots as closely as possible (colors, scales, labels, styles)
- Handle large files efficiently (GRA and RGB datasets are multi-GB)

## Data Analysis Insights from Paper

### Key Statistics
- 209 unique principal lithologies
- Most common: nannofossil ooze (>20% of descriptions)
- 431 unique prefix values
- 185 unique suffix values
- Unconsolidated sediments: ~71% of cores
- Biogenic oozes: 41% of unconsolidated sediments

### Coring Systems
- **APC** (Advanced Piston Corer): Soft sediments, high recovery (~103%)
- **HLAPC** (Half-Length APC): Intermediate sediments
- **XCB** (Extended Core Barrel): Semi-lithified materials
- **RCB** (Rotary Core Barrel): Hard rock, variable recovery

### Important Data Relationships
- Grain density depends on lithology (e.g., calcite-rich ~2.71 g/cm³, basalt ~2.89 g/cm³)
- Porosity = (ρ_fluid - ρ_grain)/(ρ_bulk - ρ_grain) for seawater (ρ_fluid = 1.024 g/cm³)
- GRA bulk densities have systematic biases for RCB cores (need correction factors from paper Table S8)
- MAD measurements are discrete samples; GRA provides continuous high-resolution data

## Development Environment

### Python Setup
- Python version: 3.11+ (see `.python-version`)
- Project configuration: `pyproject.toml`
- Main entry point: `main.py` (currently minimal)

### Common Python Libraries (Recommended)
- pandas: CSV data manipulation
- numpy: Numerical operations
- matplotlib/seaborn: Plotting
- scipy: Statistical analysis

## Working with Large Files

Several datasets exceed 1GB. Use efficient loading strategies:
```python
# Read in chunks for large files
import pandas as pd
df = pd.read_csv('datasets/GRA_DataLITH.csv', chunksize=100000)

# Or read specific columns only
df = pd.read_csv('datasets/GRA_DataLITH.csv', usecols=['Exp', 'Depth CSF-A (m)', 'Bulk density (GRA)', 'Principal'])
```

## Figure Types in Paper

Based on the paper structure, expect to reproduce:
- Histograms (density distributions, recovery percentages)
- Scatter plots (MAD vs GRA comparisons, density vs porosity)
- Geographic maps (expedition locations, Figure 2)
- Bar charts (lithology statistics)
- Box plots (coring system performance)
- Multi-panel comparison figures

## Additional Resources

- Paper supporting information contains extensive tables (S1-S8) with:
  - Complete data type lists
  - Data quantity by expedition
  - Grain density lookup tables by lithology
  - GRA correction factors
  - Lithology dictionaries
