"""
Visualize VAE v2.6.7 borehole coverage.

Creates a plot showing all boreholes side-by-side with:
- Gray bars showing total borehole depth range
- Green bars (on top) showing where VAE training data exists
Heights are not rescaled so long boreholes appear taller.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

print("Loading VAE v2.6.7 dataset...")
vae_df = pd.read_csv('vae_training_data_v2_20cm.csv')

# Convert depth bins back to depths (depth_bin * 0.2m)
vae_df['Depth_m'] = vae_df['Depth_Bin'] * 0.2

print(f"Loaded {len(vae_df):,} samples from {vae_df['Borehole_ID'].nunique()} boreholes")

print("Loading borehole total depths summary...")
borehole_total_depths = pd.read_csv('borehole_total_depths.csv')
borehole_depth_dict = dict(zip(borehole_total_depths['Borehole_ID'],
                                zip(borehole_total_depths['total_min_depth'],
                                    borehole_total_depths['total_max_depth'])))
print(f"Loaded total depths for {len(borehole_total_depths)} boreholes")

# Get borehole statistics (both VAE data coverage and total depth)
borehole_stats = []
for bh_id in vae_df['Borehole_ID'].unique():
    bh_data = vae_df[vae_df['Borehole_ID'] == bh_id]

    # Get total borehole depth from summary dictionary
    if bh_id in borehole_depth_dict:
        total_min_depth, total_max_depth = borehole_depth_dict[bh_id]
    else:
        # Fallback to VAE data if not in summary (shouldn't happen)
        total_min_depth = bh_data['Depth_m'].min()
        total_max_depth = bh_data['Depth_m'].max()

    borehole_stats.append({
        'borehole_id': bh_id,
        'n_samples': len(bh_data),
        'min_depth': bh_data['Depth_m'].min(),
        'max_depth': bh_data['Depth_m'].max(),
        'depth_range': bh_data['Depth_m'].max() - bh_data['Depth_m'].min(),
        'depth_bins': sorted(bh_data['Depth_Bin'].unique()),
        'total_min_depth': total_min_depth,
        'total_max_depth': total_max_depth,
        'total_depth_range': total_max_depth - total_min_depth
    })

# Sort boreholes by total depth range (longest first)
borehole_stats.sort(key=lambda x: x['total_depth_range'], reverse=True)

print(f"\nBorehole depth statistics:")
print(f"  Total depths:")
print(f"    Longest: {borehole_stats[0]['total_depth_range']:.1f}m ({borehole_stats[0]['borehole_id']})")
print(f"    Shortest: {borehole_stats[-1]['total_depth_range']:.1f}m ({borehole_stats[-1]['borehole_id']})")
print(f"    Mean: {np.mean([b['total_depth_range'] for b in borehole_stats]):.1f}m")
print(f"  VAE data coverage:")
print(f"    Longest: {max([b['depth_range'] for b in borehole_stats]):.1f}m")
print(f"    Mean: {np.mean([b['depth_range'] for b in borehole_stats]):.1f}m")

# Create visualization
fig, ax = plt.subplots(figsize=(24, 12))

bar_width = 0.8
x_positions = np.arange(len(borehole_stats))

# Find global max depth for consistent y-axis (use total depths)
global_max_depth = max([b['total_max_depth'] for b in borehole_stats])

print(f"\nCreating visualization...")
print(f"  Global max depth: {global_max_depth:.1f}m")

# For each borehole, draw bars
for i, bh_stat in enumerate(borehole_stats):
    # First, draw gray bar for total borehole depth (background)
    total_height = bh_stat['total_max_depth'] - bh_stat['total_min_depth']
    ax.bar(i, total_height, bar_width, bottom=bh_stat['total_min_depth'],
           color='lightgray', alpha=0.6, edgecolor='gray', linewidth=0.3)

    # Then, draw green bars for VAE data coverage (on top)
    # Get all depth bins for this borehole
    depth_bins = bh_stat['depth_bins']

    # Create continuous segments (find gaps)
    segments = []
    current_segment_start = depth_bins[0]
    current_segment_end = depth_bins[0]

    for j in range(1, len(depth_bins)):
        if depth_bins[j] == depth_bins[j-1] + 1:
            # Continuous
            current_segment_end = depth_bins[j]
        else:
            # Gap found, save current segment
            segments.append((current_segment_start * 0.2, current_segment_end * 0.2))
            current_segment_start = depth_bins[j]
            current_segment_end = depth_bins[j]

    # Save final segment
    segments.append((current_segment_start * 0.2, current_segment_end * 0.2))

    # Draw each continuous segment as a green bar on top of gray
    for seg_start, seg_end in segments:
        height = seg_end - seg_start + 0.2  # +0.2 to include the bin
        bottom = seg_start
        ax.bar(i, height, bar_width, bottom=bottom, color='green', alpha=0.8,
               edgecolor='darkgreen', linewidth=0.3)

# Formatting
ax.set_xlim(-0.5, len(borehole_stats) - 0.5)
ax.set_ylim(0, global_max_depth * 1.05)
ax.invert_yaxis()  # Depth increases downward

ax.set_xlabel('Borehole Index (sorted by depth range, longest first)', fontsize=16, fontweight='bold')
ax.set_ylabel('Depth (mbsf)', fontsize=16, fontweight='bold')
ax.set_title(f'VAE v2.6.7 Borehole Coverage\n{len(borehole_stats)} boreholes, {len(vae_df):,} samples (20cm bins)',
             fontsize=18, fontweight='bold', pad=20)

# Add grid for depth reference
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Remove x-tick labels (too many boreholes to label individually)
ax.set_xticks([])

# Increase tick label sizes
ax.tick_params(axis='both', which='major', labelsize=14)

# Add legend in LOWER right
gray_patch = mpatches.Patch(color='gray', alpha=0.3, label='Total borehole depth')
green_patch = mpatches.Patch(color='green', alpha=0.7, label='VAE data available (20cm bins)')
ax.legend(handles=[gray_patch, green_patch], loc='lower right', fontsize=14)

# Add statistics text box
stats_text = (
    f"Dataset Statistics:\n"
    f"  Total samples: {len(vae_df):,}\n"
    f"  Total boreholes: {len(borehole_stats)}\n"
    f"  Depth range: 0 - {global_max_depth:.1f} m\n"
    f"  Longest borehole: {borehole_stats[0]['total_depth_range']:.1f} m\n"
    f"  Shortest borehole: {borehole_stats[-1]['total_depth_range']:.1f} m\n"
    f"  Mean total depth: {np.mean([b['total_depth_range'] for b in borehole_stats]):.1f} m\n"
    f"  Mean VAE coverage: {np.mean([b['depth_range'] for b in borehole_stats]):.1f} m"
)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
        fontsize=13, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('vae_borehole_coverage.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Visualization saved: vae_borehole_coverage.png")

# Also create a summary CSV
summary_df = pd.DataFrame(borehole_stats)
summary_df = summary_df[['borehole_id', 'n_samples', 'min_depth', 'max_depth', 'depth_range',
                         'total_min_depth', 'total_max_depth', 'total_depth_range']]
summary_df = summary_df.sort_values('total_depth_range', ascending=False)
summary_df.to_csv('borehole_coverage_summary.csv', index=False)
print(f"✓ Summary saved: borehole_coverage_summary.csv")

print(f"\nTop 5 longest boreholes (by total depth):")
for i, bh in enumerate(borehole_stats[:5], 1):
    coverage_pct = (bh['depth_range'] / bh['total_depth_range'] * 100) if bh['total_depth_range'] > 0 else 0
    print(f"  {i}. {bh['borehole_id']}: {bh['total_depth_range']:.1f}m total, "
          f"{bh['depth_range']:.1f}m VAE ({coverage_pct:.1f}%)")

print("\n✓ Complete!")
