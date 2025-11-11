"""
Generate figures for CLEAN semi-supervised classifier paper.
Uses results from entropy-balanced borehole split with ≥100 sample filter.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300

# Figure 1: Performance comparison across methods (CLEAN DATA)
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['Direct Classifier\n(CatBoost, raw 6D)',
           'Semi-supervised\n(frozen encoder)',
           'Semi-supervised\n(fine-tuned)']
accuracies = [25.60, 22.05, 17.60]
colors = ['#2ecc71', '#f39c12', '#e74c3c']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.axhline(25.60, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5, label='Direct baseline')
ax.set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Lithology Classification: Entropy-Balanced Split (≥100 samples/class)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 35)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper right', framealpha=0.9)

# Add annotation about filtering
ax.text(1, 32, '12 classes, 238K samples\nEntropy-balanced borehole split',
        ha='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('paper_draft/fig_clean_semisupervised_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_clean_semisupervised_comparison.png")
plt.close()

# Figure 2: Per-class comparison (Direct vs Frozen)
fig, ax = plt.subplots(figsize=(12, 6))

lithologies = ['Carbonate', 'Clay/Mud', 'Biogenic\nSilica', 'Mafic\nIgneous',
               'Volcaniclastic', 'Sand', 'Silt', 'Conglomerate/\nBreccia',
               'Other', 'Intermediate/\nFelsic Igneous', 'Metamorphic', 'Evaporite']

# Per-class accuracy from training log
direct_acc = [89.78, 78.29, 24.67, 51.58, 33.54, 23.67, 4.07, 1.60, 0.00, 0.00, 0.00, 0.00]
frozen_acc = [63.32, 14.76, 62.10, 40.47, 34.95, 27.39, 14.26, 5.49, 0.00, 0.00, 0.00, 1.87]
sample_counts = [26276, 27186, 1512, 2377, 1774, 6561, 2237, 875, 297, 14, 18, 107]

x = np.arange(len(lithologies))
width = 0.35

bars1 = ax.bar(x - width/2, direct_acc, width, label='Direct (CatBoost)',
               color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, frozen_acc, width, label='Semi-supervised (frozen)',
               color='#f39c12', alpha=0.8, edgecolor='black', linewidth=1)

ax.set_xlabel('Lithology Group', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance: Direct vs Semi-Supervised (Frozen Encoder)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(lithologies, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper right', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# Add sample count annotations
for i, count in enumerate(sample_counts):
    ax.text(i, -8, f'n={count:,}', ha='center', fontsize=7, style='italic', rotation=45)

plt.tight_layout()
plt.savefig('paper_draft/fig_clean_semisupervised_per_class.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_clean_semisupervised_per_class.png")
plt.close()

# Figure 3: Training curves (clean data)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Frozen encoder
epochs_frozen = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 77]
loss_frozen = [2.3239, 1.8956, 1.7817, 1.7179, 1.6824, 1.6435, 1.6249, 1.6094, 1.5958, 1.5774, 1.5707, 1.5558, 1.5308, 1.5238, 1.5163, 1.5051, 1.5051]
bal_acc_frozen = [0.1816, 0.1756, 0.1846, 0.2032, 0.2005, 0.2086, 0.2116, 0.2126, 0.2133, 0.2076, 0.2177, 0.2129, 0.2128, 0.2227, 0.2176, 0.2102, 0.2205]

ax1.plot(epochs_frozen, loss_frozen, 'o-', color='#3498db', linewidth=2, markersize=6, label='Training loss')
ax1.axvline(65, color='red', linestyle='--', alpha=0.5, label='Best model (epoch 65)')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold', color='#3498db')
ax1.tick_params(axis='y', labelcolor='#3498db')
ax1.set_title('Frozen Encoder (22.05% best)', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')

ax1_twin = ax1.twinx()
ax1_twin.plot(epochs_frozen, bal_acc_frozen, 's-', color='#f39c12', linewidth=2, markersize=6, label='Balanced accuracy')
ax1_twin.axhline(0.2258, color='red', linestyle='--', alpha=0.3)
ax1_twin.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold', color='#f39c12')
ax1_twin.tick_params(axis='y', labelcolor='#f39c12')
ax1_twin.set_ylim(0.15, 0.25)
ax1_twin.legend(loc='center right')

# Fine-tuned encoder
epochs_finetuned = [1, 5, 10, 15, 20, 25, 30, 32]
loss_finetuned = [2.4689, 2.3293, 2.2128, 2.1277, 2.0700, 2.0163, 1.9816, 1.9816]
bal_acc_finetuned = [0.0919, 0.1666, 0.1783, 0.1837, 0.1825, 0.1774, 0.1751, 0.1760]

ax2.plot(epochs_finetuned, loss_finetuned, 'o-', color='#3498db', linewidth=2, markersize=6, label='Training loss')
ax2.axvline(15, color='red', linestyle='--', alpha=0.5, label='Best model (epoch 15)')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold', color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2.set_title('Fine-Tuned Encoder (17.60% best)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, linestyle='--')
ax2.legend(loc='upper right')

ax2_twin = ax2.twinx()
ax2_twin.plot(epochs_finetuned, bal_acc_finetuned, 's-', color='#e74c3c', linewidth=2, markersize=6, label='Balanced accuracy')
ax2_twin.axhline(0.1845, color='red', linestyle='--', alpha=0.3)
ax2_twin.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold', color='#e74c3c')
ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax2_twin.set_ylim(0.08, 0.20)
ax2_twin.legend(loc='center right')

plt.tight_layout()
plt.savefig('paper_draft/fig_clean_semisupervised_training.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_clean_semisupervised_training.png")
plt.close()

# Figure 4: Balanced accuracy penalty from class imbalance
fig, ax = plt.subplots(figsize=(10, 6))

lithologies_sorted = ['Carbonate', 'Clay/Mud', 'Mafic\nIgneous', 'Volcaniclastic',
                      'Biogenic\nSilica', 'Sand', 'Silt', 'Conglomerate/\nBreccia',
                      'Other', 'Evaporite', 'Metamorphic', 'Intermediate/\nFelsic Igneous']
sample_counts_sorted = [26276, 27186, 2377, 1774, 1512, 6561, 2237, 875, 319, 176, 162, 284]
direct_acc_sorted = [89.78, 78.29, 51.58, 33.54, 24.67, 23.67, 4.07, 1.60, 0.00, 0.00, 0.00, 0.00]

# Create scatter plot: sample count vs accuracy
scatter = ax.scatter(sample_counts_sorted, direct_acc_sorted, s=200, c=direct_acc_sorted,
                     cmap='RdYlGn', alpha=0.7, edgecolor='black', linewidth=1.5)

# Add labels
for i, (count, acc, lith) in enumerate(zip(sample_counts_sorted, direct_acc_sorted, lithologies_sorted)):
    ax.annotate(lith.replace('\n', ' '), (count, acc), fontsize=8,
                xytext=(5, 5), textcoords='offset points')

ax.set_xscale('log')
ax.set_xlabel('Test Samples (log scale)', fontsize=12, fontweight='bold')
ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Class Imbalance Effect: Sample Count vs Accuracy (Direct Classifier)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(alpha=0.3, linestyle='--')

# Add vertical line at 1000 samples
ax.axvline(1000, color='red', linestyle='--', alpha=0.5, linewidth=2, label='1000 samples')

# Add text annotation
ax.text(2000, 80, 'Classes >1K samples:\nReasonable accuracy\n(23-90%)',
        fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax.text(300, 50, 'Classes <1K samples:\nPoor/zero accuracy\n(0-34%)',
        fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

ax.legend(loc='lower right', framealpha=0.9)

plt.colorbar(scatter, ax=ax, label='Accuracy (%)')
plt.tight_layout()
plt.savefig('paper_draft/fig_clean_class_imbalance_effect.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_clean_class_imbalance_effect.png")
plt.close()

# Figure 5: Comparison of original (bad split) vs clean (entropy-balanced)
fig, ax = plt.subplots(figsize=(10, 6))

methods_comparison = ['Direct\nClassifier', 'Semi-supervised\n(frozen)', 'Semi-supervised\n(fine-tuned)']
original_bad_split = [42.32, 16.96, 16.24]
clean_entropy_split = [25.60, 22.05, 17.60]

x = np.arange(len(methods_comparison))
width = 0.35

bars1 = ax.bar(x - width/2, original_bad_split, width, label='Original (lucky split)',
               color='#95a5a6', alpha=0.8, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width/2, clean_entropy_split, width, label='Clean (entropy-balanced)',
               color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Impact of Train/Test Split Quality on Performance',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(methods_comparison)
ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_ylim(0, 50)

# Add annotations
ax.annotate('Direct classifier massively\ninflated by lucky split\n(-39.5%)',
            xy=(0+width/2, clean_entropy_split[0]), xytext=(0.5, 35),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'),
            fontsize=10, style='italic', ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.annotate('Semi-supervised frozen\nimproved with fair split\n(+30.0%)',
            xy=(1+width/2, clean_entropy_split[1]), xytext=(1.5, 10),
            arrowprops=dict(arrowstyle='->', lw=2, color='green'),
            fontsize=10, style='italic', ha='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.savefig('paper_draft/fig_clean_split_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_clean_split_comparison.png")
plt.close()

print("\n✓ All clean figures generated successfully!")
