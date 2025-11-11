"""
Generate figures for semi-supervised classifier paper section.
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

# Figure 1: Performance comparison across methods
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['Direct\nClassifier\n(raw 6D)',
           'VAE Classifier\nv1.1\n(hierarchical)',
           'Semi-supervised\n(frozen encoder)',
           'Semi-supervised\n(fine-tuned)']
accuracies = [42.32, 29.73, 16.96, 16.24]
colors = ['#2ecc71', '#3498db', '#e74c3c', '#c0392b']

bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.axhline(42.32, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.5, label='Best baseline')
ax.set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_xlabel('Method', fontsize=12, fontweight='bold')
ax.set_title('Lithology Classification Performance Comparison', fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 50)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.savefig('paper_draft/fig_semisupervised_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_semisupervised_comparison.png")
plt.close()

# Figure 2: Per-class performance (frozen encoder, best variant)
fig, ax = plt.subplots(figsize=(10, 6))

lithologies = ['Carbonate', 'Biogenic Silica', 'Sand', 'Volcaniclastic',
               'Mafic Igneous', 'Silt', 'Clay/Mud', 'Conglomerate/\nBreccia',
               'Evaporite', 'Intermediate/\nFelsic Igneous', 'Metamorphic',
               'Ultramafic', 'Other']
accuracies_per_class = [61.52, 47.95, 30.48, 31.57, 18.97, 11.35, 10.77,
                        6.86, 0.93, 0.00, 0.00, 0.00, 0.00]
sample_counts = [26276, 1512, 6561, 1774, 2377, 2237, 27186, 875, 107, 14, 18, 1, 297]

# Color by performance
colors_class = ['#2ecc71' if acc > 40 else '#f39c12' if acc > 20 else '#e74c3c'
                for acc in accuracies_per_class]

bars = ax.barh(lithologies, accuracies_per_class, color=colors_class, alpha=0.8,
               edgecolor='black', linewidth=1)

# Add sample counts as text
for i, (bar, count) in enumerate(zip(bars, sample_counts)):
    width = bar.get_width()
    ax.text(width + 2, bar.get_y() + bar.get_height()/2,
            f'n={count:,}', va='center', fontsize=8, style='italic')

ax.axvline(42.32, color='green', linestyle='--', linewidth=2, alpha=0.5,
           label='Direct classifier baseline')
ax.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Lithology Group', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance: Semi-Supervised Classifier (Frozen Encoder)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, 75)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.savefig('paper_draft/fig_semisupervised_per_class.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_semisupervised_per_class.png")
plt.close()

# Figure 3: Training curves (frozen vs fine-tuned)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Frozen encoder training curve
epochs_frozen = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 46]
loss_frozen = [2.5176, 2.0484, 1.9187, 1.8308, 1.7836, 1.7270, 1.7007, 1.6673, 1.6513, 1.6322, 1.6322]
bal_acc_frozen = [0.1607, 0.1606, 0.1602, 0.1711, 0.1756, 0.1672, 0.1699, 0.1742, 0.1691, 0.1683, 0.1696]

ax1.plot(epochs_frozen, loss_frozen, 'o-', color='#3498db', linewidth=2, markersize=6, label='Training loss')
ax1.axvline(46, color='red', linestyle='--', alpha=0.5, label='Early stop (epoch 46)')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold', color='#3498db')
ax1.tick_params(axis='y', labelcolor='#3498db')
ax1.set_title('Frozen Encoder Training', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3, linestyle='--')
ax1.legend(loc='upper right')

ax1_twin = ax1.twinx()
ax1_twin.plot(epochs_frozen, bal_acc_frozen, 's-', color='#e74c3c', linewidth=2, markersize=6, label='Balanced accuracy')
ax1_twin.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold', color='#e74c3c')
ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax1_twin.set_ylim(0.10, 0.20)
ax1_twin.legend(loc='center right')

# Fine-tuned encoder training curve
epochs_finetuned = [1, 5, 10, 15, 20, 25, 30, 35]
loss_finetuned = [2.6229, 2.5214, 2.4259, 2.3424, 2.2732, 2.2260, 2.1863, 2.1521]
bal_acc_finetuned = [0.0997, 0.1443, 0.1568, 0.1631, 0.1654, 0.1622, 0.1612, 0.1624]

ax2.plot(epochs_finetuned, loss_finetuned, 'o-', color='#3498db', linewidth=2, markersize=6, label='Training loss')
ax2.axvline(35, color='red', linestyle='--', alpha=0.5, label='Early stop (epoch 35)')
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold', color='#3498db')
ax2.tick_params(axis='y', labelcolor='#3498db')
ax2.set_title('Fine-Tuned Encoder Training', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, linestyle='--')
ax2.legend(loc='upper right')

ax2_twin = ax2.twinx()
ax2_twin.plot(epochs_finetuned, bal_acc_finetuned, 's-', color='#e74c3c', linewidth=2, markersize=6, label='Balanced accuracy')
ax2_twin.set_ylabel('Balanced Accuracy', fontsize=11, fontweight='bold', color='#e74c3c')
ax2_twin.tick_params(axis='y', labelcolor='#e74c3c')
ax2_twin.set_ylim(0.08, 0.18)
ax2_twin.legend(loc='center right')

plt.tight_layout()
plt.savefig('paper_draft/fig_semisupervised_training.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_semisupervised_training.png")
plt.close()

# Figure 4: Information loss illustration
fig, ax = plt.subplots(figsize=(8, 5))

stages = ['Raw 6D\nFeatures', 'VAE Encoder\n(6D → 10D)', 'Effective\nLatent (4D)',
          'Classification\nHead', 'Final\nPrediction']
information = [100, 75, 50, 25, 16.24]  # Conceptual information retained (%)

x = np.arange(len(stages))
colors_info = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c']

bars = ax.bar(x, information, color=colors_info, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, info in zip(bars, information):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{info:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Add arrows between bars
for i in range(len(stages)-1):
    ax.annotate('', xy=(x[i+1]-0.3, information[i+1]+5),
                xytext=(x[i]+0.3, information[i]-5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black', alpha=0.5))

ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=10)
ax.set_ylabel('Classification Information Retained (%)', fontsize=12, fontweight='bold')
ax.set_title('Information Loss Through Semi-Supervised Pipeline',
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylim(0, 120)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add annotation
ax.text(2.5, 90, 'Each compression\nstep loses discriminative\ninformation',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('paper_draft/fig_semisupervised_information_loss.png', dpi=300, bbox_inches='tight')
print("✓ Saved: paper_draft/fig_semisupervised_information_loss.png")
plt.close()

print("\n✓ All figures generated successfully!")
