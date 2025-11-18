#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize Semi-Supervised VAE v2.14 Architecture Diagram

Shows the complete data flow including the classification head that branches
from the latent space for guided representation learning.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(figsize=(20, 14))
ax.set_xlim(0, 20)
ax.set_ylim(0, 14)
ax.axis('off')

# Colors
color_input = '#E8F4F8'
color_transform = '#FFE6CC'
color_encoder = '#D4E6F1'
color_latent = '#D5F4E6'
color_decoder = '#F9E79F'
color_classifier = '#F8BBD0'  # Pink for classification head
color_output = '#E8F4F8'
color_loss = '#FADBD8'

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, label, color, fontsize=10, bold=False):
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, weight=weight)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label='', color='black', style='->'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=2,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y + 0.2, label, ha='center', va='bottom',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                facecolor='white', edgecolor='none', alpha=0.8))

# Title
ax.text(10, 13.5, 'Semi-Supervised VAE v2.14 Architecture (α=0.1)',
        ha='center', va='top', fontsize=18, weight='bold')
ax.text(10, 13, 'Guided Representation Learning with Classification Head',
        ha='center', va='top', fontsize=12, style='italic', color='#666')

# ============================================================================
# INPUT LAYER
# ============================================================================
y_input = 11
draw_box(ax, 2, y_input, 2.5, 1, 'Raw Input\n6D Features', color_input, fontsize=10, bold=True)

# Feature list
features = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']
for i, feat in enumerate(features):
    ax.text(2, y_input - 0.6 - i*0.15, f'• {feat}', ha='center', va='top', fontsize=7)

# ============================================================================
# DISTRIBUTION-AWARE SCALING
# ============================================================================
draw_arrow(ax, 2, y_input - 0.5, 2, 9.5, 'Input')

y_scale = 9
draw_box(ax, 2, y_scale, 2.5, 1.2, 'Distribution-Aware\nScaling', color_transform, fontsize=9, bold=True)

# Transformations
transforms = [
    'GRA: Standard',
    'MS: Sign×log(|x|+1)',
    'NGR: Sign×log(|x|+1)',
    'RGB: log(x+1)'
]
for i, t in enumerate(transforms):
    ax.text(2, y_scale - 0.7 - i*0.12, t, ha='center', va='top', fontsize=6)

# ============================================================================
# ENCODER
# ============================================================================
draw_arrow(ax, 2, y_scale - 0.6, 2, 7.5, 'Scaled\n6D')

y_enc1 = 7
draw_box(ax, 2, y_enc1, 2, 0.8, 'Linear(6→32)\nReLU', color_encoder, fontsize=9)

draw_arrow(ax, 2, y_enc1 - 0.4, 2, 5.7, '32D')

y_enc2 = 5.2
draw_box(ax, 2, y_enc2, 2, 0.8, 'Linear(32→16)\nReLU', color_encoder, fontsize=9)

draw_arrow(ax, 2, y_enc2 - 0.4, 2, 4, '16D')

# ============================================================================
# LATENT SPACE (with reparameterization)
# ============================================================================
y_latent = 3
draw_box(ax, 2, y_latent + 0.5, 1.5, 0.6, 'fc_mu\n16→10', color_latent, fontsize=8)
draw_box(ax, 2, y_latent - 0.5, 1.5, 0.6, 'fc_logvar\n16→10', color_latent, fontsize=8)

draw_arrow(ax, 2.8, y_latent, 4.5, y_latent, 'μ (10D)')
draw_arrow(ax, 2.8, y_latent - 0.5, 4.5, y_latent - 0.3, 'σ (10D)')

# Reparameterization trick
y_reparam = y_latent - 0.15
draw_box(ax, 5.5, y_reparam, 2, 0.9, 'Reparameterization\nz = μ + σ⊙ε\nε ~ N(0,I)',
         color_latent, fontsize=8, bold=True)

# ============================================================================
# LATENT REPRESENTATION (central node)
# ============================================================================
draw_arrow(ax, 6.5, y_reparam, 8.5, y_reparam, 'z\n10D', color='#2E86AB')

y_z = y_reparam
draw_box(ax, 9.5, y_z, 2.2, 1, 'Latent Space\n10D', color_latent, fontsize=10, bold=True)
ax.text(9.5, y_z - 0.6, 'All dims active', ha='center', va='top', fontsize=7, style='italic')

# ============================================================================
# DECODER PATH (continues down)
# ============================================================================
draw_arrow(ax, 9.5, y_z - 0.5, 9.5, 1.7, '10D', color='#2E86AB')

y_dec1 = 1.2
draw_box(ax, 9.5, y_dec1, 2, 0.8, 'Linear(10→16)\nReLU', color_decoder, fontsize=9)

draw_arrow(ax, 9.5, y_dec1 - 0.4, 9.5, 0.4, '16D')

# ============================================================================
# DECODER CONTINUES
# ============================================================================
y_dec2 = 11
x_dec2 = 14
draw_arrow(ax, 9.5, y_dec1 - 0.4, 9.5, y_dec2 + 0.6, '', color='#2E86AB')
draw_arrow(ax, 9.5, y_dec2 + 0.6, x_dec2, y_dec2 + 0.6, '16D', color='#2E86AB')

draw_box(ax, x_dec2, y_dec2, 2, 0.8, 'Linear(16→32)\nReLU', color_decoder, fontsize=9)

draw_arrow(ax, x_dec2, y_dec2 - 0.4, x_dec2, 9.5, '32D')

y_dec3 = 9
draw_box(ax, x_dec2, y_dec3, 2, 0.8, 'fc_out\nLinear(32→6)', color_decoder, fontsize=9)

draw_arrow(ax, x_dec2, y_dec3 - 0.4, x_dec2, 7.5, '6D')

# ============================================================================
# OUTPUT
# ============================================================================
y_output = 7
draw_box(ax, x_dec2, y_output, 2.5, 1, 'Reconstructed\n6D Features', color_output, fontsize=10, bold=True)

# Feature list
for i, feat in enumerate(features):
    ax.text(x_dec2, y_output - 0.6 - i*0.15, f'• {feat}', ha='center', va='top', fontsize=7)

# ============================================================================
# CLASSIFICATION HEAD (branches from latent space)
# ============================================================================
# Arrow from latent to classifier (going right)
draw_arrow(ax, 10.7, y_z, 13.5, y_z, '10D', color='#C2185B')

x_class = 15
y_class = y_z
draw_box(ax, x_class, y_class, 2.2, 0.8, 'Linear(10→32)\nReLU',
         color_classifier, fontsize=9, bold=True)

draw_arrow(ax, x_class, y_class - 0.4, x_class, y_class - 1.2, '32D', color='#C2185B')

y_dropout = y_class - 1.6
draw_box(ax, x_class, y_dropout, 2.2, 0.6, 'Dropout(0.2)',
         color_classifier, fontsize=9)

draw_arrow(ax, x_class, y_dropout - 0.3, x_class, y_dropout - 0.9, '32D', color='#C2185B')

y_logits = y_dropout - 1.3
draw_box(ax, x_class, y_logits, 2.2, 0.8, 'Linear(32→139)\nLogits',
         color_classifier, fontsize=9, bold=True)

# Classification output
draw_arrow(ax, x_class, y_logits - 0.4, x_class, y_logits - 1.2, 'Logits\n139 classes', color='#C2185B')

y_class_out = y_logits - 1.6
draw_box(ax, x_class, y_class_out, 2.2, 0.7, 'Lithology\nPredictions',
         color_classifier, fontsize=9, bold=True)

# ============================================================================
# LOSS FUNCTIONS (bottom)
# ============================================================================
y_loss = 0.3
x_loss_start = 4

# Reconstruction Loss
draw_arrow(ax, 2, y_output - 0.5, x_loss_start, y_loss, '', color='#555', style='->')
draw_arrow(ax, x_dec2, y_output - 0.5, x_loss_start + 2, y_loss, '', color='#555', style='->')
draw_box(ax, x_loss_start + 1, y_loss, 2.5, 0.5, 'L_recon = MSE', color_loss, fontsize=8)

# KL Divergence Loss
draw_arrow(ax, 5.5, y_reparam - 0.45, x_loss_start + 4, y_loss, '', color='#555', style='->')
draw_box(ax, x_loss_start + 4, y_loss, 2.5, 0.5, 'L_KL = β×KL', color_loss, fontsize=8)
ax.text(x_loss_start + 4, y_loss - 0.35, 'β: 1e-10→0.75', ha='center', va='top', fontsize=6)

# Classification Loss
draw_arrow(ax, x_class, y_class_out - 0.35, x_loss_start + 7, y_loss, '', color='#C2185B', style='->')
draw_box(ax, x_loss_start + 7, y_loss, 2.5, 0.5, 'L_class = α×CE', color_loss, fontsize=8)
ax.text(x_loss_start + 7, y_loss - 0.35, 'α = 0.1', ha='center', va='top', fontsize=6, weight='bold')

# Total Loss
draw_box(ax, x_loss_start + 10, y_loss, 3, 0.6,
         'L_total = L_recon + β×L_KL + α×L_class',
         color_loss, fontsize=9, bold=True)

# ============================================================================
# ANNOTATIONS
# ============================================================================
# Parameter count
ax.text(18.5, 12.5, 'Parameters: 6,949', ha='right', va='top',
        fontsize=10, bbox=dict(boxstyle='round,pad=0.5',
        facecolor='lightgray', alpha=0.8))
ax.text(18.5, 12, 'vs v2.6.7: 2,010', ha='right', va='top', fontsize=9)
ax.text(18.5, 11.6, '+3.5× (classification head)', ha='right', va='top', fontsize=8)

# Performance
perf_box = FancyBboxPatch(
    (18.5 - 2.5, 10 - 1), 2.5, 1,
    boxstyle="round,pad=0.15",
    edgecolor='#2E7D32',
    facecolor='#E8F5E9',
    linewidth=2.5
)
ax.add_patch(perf_box)
ax.text(17.25, 9.7, 'Performance', ha='center', va='top', fontsize=10, weight='bold', color='#2E7D32')
ax.text(17.25, 9.4, 'GMM ARI: 0.285', ha='center', va='top', fontsize=9)
ax.text(17.25, 9.15, '+45.6% vs v2.6.7', ha='center', va='top', fontsize=9, weight='bold', color='#2E7D32')
ax.text(17.25, 8.85, 'Reconstruction R²:', ha='center', va='top', fontsize=8)
ax.text(17.25, 8.6, 'GRA: 0.949, RGB: 0.92', ha='center', va='top', fontsize=7)

# Architecture highlights
ax.text(0.5, 5, 'Encoder', ha='left', va='center', fontsize=12, weight='bold',
        rotation=90, color='#1565C0')
ax.text(18.5, y_z, 'Classification\nHead (NEW)', ha='right', va='center',
        fontsize=11, weight='bold', color='#C2185B',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FCE4EC',
                 edgecolor='#C2185B', linewidth=2))

# Key innovation
ax.text(10, 5.5, 'Key Innovation:', ha='center', va='top', fontsize=10, weight='bold')
ax.text(10, 5.2, 'Lithology labels guide latent space', ha='center', va='top', fontsize=9)
ax.text(10, 4.95, 'organization during training,', ha='center', va='top', fontsize=9)
ax.text(10, 4.7, 'improving clustering with GMM', ha='center', va='top', fontsize=9)

# Legend
legend_y = 12
legend_x = 0.5
ax.text(legend_x, legend_y, 'Data Flow:', ha='left', va='top', fontsize=9, weight='bold')
ax.plot([legend_x, legend_x + 0.8], [legend_y - 0.2, legend_y - 0.2],
        'k-', linewidth=2, marker='>')
ax.text(legend_x + 1, legend_y - 0.2, 'Standard VAE path', ha='left', va='center', fontsize=7)
ax.plot([legend_x, legend_x + 0.8], [legend_y - 0.45, legend_y - 0.45],
        color='#C2185B', linewidth=2, marker='>')
ax.text(legend_x + 1, legend_y - 0.45, 'Classification path', ha='left', va='center', fontsize=7)

plt.tight_layout()
plt.savefig('vae_v2_14_semisupervised_architecture_diagram.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✓ Saved: vae_v2_14_semisupervised_architecture_diagram.png")
plt.close()

print("\nArchitecture Diagram Generated Successfully!")
print("="*60)
print("Shows:")
print("  • Encoder: 6D → [32, 16] → 10D latent")
print("  • Decoder: 10D → [16, 32] → 6D (symmetric)")
print("  • Classification head: 10D → [32, ReLU, Dropout(0.2)] → 139")
print("  • Three-part loss: Reconstruction + β×KL + α×Classification")
print("  • Parameters: 6,949 (vs 2,010 for v2.6.7)")
print("  • Performance: ARI=0.285 (+45.6% improvement)")
