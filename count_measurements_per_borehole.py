#!/usr/bin/env python3
"""
Count measurements per borehole for:
- Porosity (from MAD)
- Grain density (from MAD)
- P-wave velocity (from PWC)
- Thermal conductivity (from TCON)
"""

import pandas as pd
from pathlib import Path

# Dataset paths
DATASETS_DIR = Path("datasets")
MAD_FILE = DATASETS_DIR / "MAD_DataLITH.csv"
PWC_FILE = DATASETS_DIR / "PWC_DataLITH.csv"
TCON_FILE = DATASETS_DIR / "TCON_DataLITH.csv"

def load_and_count_measurements():
    """Load datasets and count measurements per borehole."""

    print("Loading MAD data (porosity and grain density)...")
    # Load MAD for porosity and grain density
    mad_df = pd.read_csv(MAD_FILE, usecols=['Exp', 'Site', 'Hole', 'Porosity (vol%)', 'Grain density (g/cm^3)'], low_memory=False)
    mad_df['Exp'] = mad_df['Exp'].astype(str)
    mad_df['Site'] = mad_df['Site'].astype(str)
    mad_df['Hole'] = mad_df['Hole'].astype(str)

    print("Loading PWC data (P-wave velocity)...")
    # Load PWC for P-wave velocity
    pwc_df = pd.read_csv(PWC_FILE, usecols=['Exp', 'Site', 'Hole', 'P-wave velocity x (m/s)',
                                             'P-wave velocity y (m/s)', 'P-wave velocity z (m/s)',
                                             'P-wave velocity unknown (m/s)'], low_memory=False)
    pwc_df['Exp'] = pwc_df['Exp'].astype(str)
    pwc_df['Site'] = pwc_df['Site'].astype(str)
    pwc_df['Hole'] = pwc_df['Hole'].astype(str)

    print("Loading TCON data (thermal conductivity)...")
    # Load TCON for thermal conductivity
    tcon_df = pd.read_csv(TCON_FILE, usecols=['Exp', 'Site', 'Hole', 'Thermal conductivity mean (W/(m*K))'], low_memory=False)
    tcon_df['Exp'] = tcon_df['Exp'].astype(str)
    tcon_df['Site'] = tcon_df['Site'].astype(str)
    tcon_df['Hole'] = tcon_df['Hole'].astype(str)

    # Count measurements per borehole
    print("\nCounting measurements per borehole...")

    # Porosity counts (non-null values)
    porosity_counts = mad_df[mad_df['Porosity (vol%)'].notna()].groupby(['Exp', 'Site', 'Hole']).size().reset_index(name='Porosity_count')

    # Grain density counts (non-null values)
    grain_density_counts = mad_df[mad_df['Grain density (g/cm^3)'].notna()].groupby(['Exp', 'Site', 'Hole']).size().reset_index(name='Grain_density_count')

    # P-wave velocity counts (count if ANY of the velocity columns have data)
    pwc_df['has_pwave'] = pwc_df[['P-wave velocity x (m/s)', 'P-wave velocity y (m/s)',
                                    'P-wave velocity z (m/s)', 'P-wave velocity unknown (m/s)']].notna().any(axis=1)
    pwave_counts = pwc_df[pwc_df['has_pwave']].groupby(['Exp', 'Site', 'Hole']).size().reset_index(name='Pwave_velocity_count')

    # Thermal conductivity counts (non-null values)
    tcon_counts = tcon_df[tcon_df['Thermal conductivity mean (W/(m*K))'].notna()].groupby(['Exp', 'Site', 'Hole']).size().reset_index(name='Thermal_conductivity_count')

    # Get all unique boreholes
    all_boreholes = set()
    for df in [porosity_counts, grain_density_counts, pwave_counts, tcon_counts]:
        for _, row in df.iterrows():
            all_boreholes.add((row['Exp'], row['Site'], row['Hole']))

    # Create comprehensive dataframe
    borehole_data = []
    for exp, site, hole in sorted(all_boreholes, key=lambda x: (x[0], x[1], x[2])):
        borehole_id = f"{exp}-{site}{hole}"

        # Get counts for each measurement type
        por_count = porosity_counts[(porosity_counts['Exp'] == exp) &
                                    (porosity_counts['Site'] == site) &
                                    (porosity_counts['Hole'] == hole)]['Porosity_count'].values
        por_count = por_count[0] if len(por_count) > 0 else 0

        grain_count = grain_density_counts[(grain_density_counts['Exp'] == exp) &
                                          (grain_density_counts['Site'] == site) &
                                          (grain_density_counts['Hole'] == hole)]['Grain_density_count'].values
        grain_count = grain_count[0] if len(grain_count) > 0 else 0

        pwave_count = pwave_counts[(pwave_counts['Exp'] == exp) &
                                   (pwave_counts['Site'] == site) &
                                   (pwave_counts['Hole'] == hole)]['Pwave_velocity_count'].values
        pwave_count = pwave_count[0] if len(pwave_count) > 0 else 0

        tcon_count = tcon_counts[(tcon_counts['Exp'] == exp) &
                                (tcon_counts['Site'] == site) &
                                (tcon_counts['Hole'] == hole)]['Thermal_conductivity_count'].values
        tcon_count = tcon_count[0] if len(tcon_count) > 0 else 0

        has_all = (por_count > 0 and grain_count > 0 and pwave_count > 0 and tcon_count > 0)

        borehole_data.append({
            'Borehole_ID': borehole_id,
            'Expedition': exp,
            'Site': site,
            'Hole': hole,
            'Porosity_count': por_count,
            'Grain_density_count': grain_count,
            'Pwave_velocity_count': pwave_count,
            'Thermal_conductivity_count': tcon_count,
            'Has_all_measurements': has_all
        })

    results_df = pd.DataFrame(borehole_data)
    return results_df

def generate_report(results_df):
    """Generate markdown report."""

    total_boreholes = len(results_df)
    boreholes_with_all = results_df['Has_all_measurements'].sum()

    # Summary statistics
    total_porosity = results_df['Porosity_count'].sum()
    total_grain_density = results_df['Grain_density_count'].sum()
    total_pwave = results_df['Pwave_velocity_count'].sum()
    total_tcon = results_df['Thermal_conductivity_count'].sum()

    boreholes_with_porosity = (results_df['Porosity_count'] > 0).sum()
    boreholes_with_grain = (results_df['Grain_density_count'] > 0).sum()
    boreholes_with_pwave = (results_df['Pwave_velocity_count'] > 0).sum()
    boreholes_with_tcon = (results_df['Thermal_conductivity_count'] > 0).sum()

    with open('report.md', 'w') as f:
        f.write("# Borehole Measurements Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total boreholes analyzed**: {total_boreholes}\n")
        f.write(f"- **Boreholes with ALL four measurement types**: {boreholes_with_all}\n")
        f.write(f"- **Percentage with all measurements**: {100*boreholes_with_all/total_boreholes:.2f}%\n\n")

        f.write("## Measurement Type Coverage\n\n")
        f.write("| Measurement Type | Total Measurements | Boreholes with Data | % Coverage |\n")
        f.write("|-----------------|-------------------|---------------------|------------|\n")
        f.write(f"| Porosity | {total_porosity:,} | {boreholes_with_porosity} | {100*boreholes_with_porosity/total_boreholes:.2f}% |\n")
        f.write(f"| Grain Density | {total_grain_density:,} | {boreholes_with_grain} | {100*boreholes_with_grain/total_boreholes:.2f}% |\n")
        f.write(f"| P-wave Velocity | {total_pwave:,} | {boreholes_with_pwave} | {100*boreholes_with_pwave/total_boreholes:.2f}% |\n")
        f.write(f"| Thermal Conductivity | {total_tcon:,} | {boreholes_with_tcon} | {100*boreholes_with_tcon/total_boreholes:.2f}% |\n\n")

        f.write("## Boreholes with All Four Measurements\n\n")
        if boreholes_with_all > 0:
            complete_boreholes = results_df[results_df['Has_all_measurements']]
            f.write("| Borehole ID | Expedition | Site | Hole | Porosity | Grain Density | P-wave Velocity | Thermal Conductivity |\n")
            f.write("|-------------|-----------|------|------|----------|---------------|-----------------|----------------------|\n")
            for _, row in complete_boreholes.iterrows():
                f.write(f"| {row['Borehole_ID']} | {row['Expedition']} | {row['Site']} | {row['Hole']} | "
                       f"{row['Porosity_count']} | {row['Grain_density_count']} | "
                       f"{row['Pwave_velocity_count']} | {row['Thermal_conductivity_count']} |\n")
        else:
            f.write("No boreholes have all four measurement types.\n")

        f.write("\n## Detailed Counts by Borehole\n\n")
        f.write("| Borehole ID | Expedition | Site | Hole | Porosity | Grain Density | P-wave Velocity | Thermal Conductivity | Has All |\n")
        f.write("|-------------|-----------|------|------|----------|---------------|-----------------|----------------------|---------|\n")
        for _, row in results_df.iterrows():
            has_all_marker = "✓" if row['Has_all_measurements'] else "✗"
            f.write(f"| {row['Borehole_ID']} | {row['Expedition']} | {row['Site']} | {row['Hole']} | "
                   f"{row['Porosity_count']} | {row['Grain_density_count']} | "
                   f"{row['Pwave_velocity_count']} | {row['Thermal_conductivity_count']} | {has_all_marker} |\n")

    print(f"\n✓ Report saved to report.md")
    print(f"  - {total_boreholes} total boreholes")
    print(f"  - {boreholes_with_all} boreholes with all four measurement types")

if __name__ == "__main__":
    print("=" * 70)
    print("BOREHOLE MEASUREMENTS ANALYSIS")
    print("=" * 70)

    results_df = load_and_count_measurements()
    generate_report(results_df)

    print("\nAnalysis complete!")
