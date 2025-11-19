#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot ROC Curves for Semi-Supervised VAE v2.14 Classification Head

Generates ROC curves for all 139 lithology classes using one-vs-rest approach.
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Set matplotlib font to monospace
plt.rcParams['font.family'] = 'monospace'

# Define model architecture directly
class DistributionAwareScaler:
    """Distribution-aware scaler for physical properties"""
    def __init__(self):
        self.medians = None
        self.iqrs = None

    def fit(self, X):
        """Fit scaler to data"""
        self.medians = np.median(X, axis=0)
        self.iqrs = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
        # Avoid division by zero
        self.iqrs = np.where(self.iqrs == 0, 1.0, self.iqrs)
        return self

    def transform(self, X):
        """Transform data"""
        return (X - self.medians) / self.iqrs

    def fit_transform(self, X):
        """Fit and transform data"""
        return self.fit(X).transform(X)

class SemiSupervisedVAE(nn.Module):
    """Semi-supervised VAE with classification head"""
    def __init__(self, input_dim=6, latent_dim=10, n_classes=139,
                 encoder_dims=[32, 16], decoder_dims=[16, 32], classifier_hidden=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder: 10D → [16, 32] → 6D (symmetric)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, (mu, logvar), logits

print("="*80)
print("SEMI-SUPERVISED VAE v2.14 - ROC CURVES")
print("="*80)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

# Get unique lithologies and create mapping
lithologies = sorted(df['Principal'].unique())
n_classes = len(lithologies)
lith_to_idx = {lith: idx for idx, lith in enumerate(lithologies)}
print(f"Number of classes: {n_classes}")

# Create train/test split (80/20 by borehole)
boreholes = df['Borehole_ID'].unique()
np.random.seed(42)
np.random.shuffle(boreholes)
split_idx = int(0.8 * len(boreholes))
train_boreholes = set(boreholes[:split_idx])

test_mask = ~df['Borehole_ID'].isin(train_boreholes)
test_df = df[test_mask].copy()
print(f"Test samples: {len(test_df):,}")

# Prepare test data
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X_test = test_df[feature_cols].values
y_test = test_df['Principal'].map(lith_to_idx).values

# Initialize scaler and model
print("\nInitializing model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

scaler = DistributionAwareScaler()
# Fit on test data (just for scaling, model already trained)
scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)

model = SemiSupervisedVAE(
    input_dim=6,
    latent_dim=10,
    n_classes=n_classes,
    encoder_dims=[32, 16],
    decoder_dims=[16, 32],
    classifier_hidden=32
).to(device)

# Load trained model
checkpoint_path = 'ml_models/checkpoints/semisup_vae_alpha0.1.pth'
print(f"Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded (epoch {checkpoint['epoch']})")

# Get predictions
print("\nGenerating predictions...")
X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)

with torch.no_grad():
    _, _, logits = model(X_test_tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()

print(f"Predictions shape: {probs.shape}")

# Binarize labels for one-vs-rest ROC
y_test_bin = label_binarize(y_test, classes=range(n_classes))
print(f"Binarized labels shape: {y_test_bin.shape}")

# Compute ROC curve and AUC for each class
print("\nComputing ROC curves for all classes...")
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), probs.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average AUC: {roc_auc['macro']:.4f}")

# ============================================================================
# PLOT 1: All ROC curves on one plot (semi-transparent)
# ============================================================================
print("\nCreating comprehensive ROC plot...")
fig, ax = plt.subplots(figsize=(12, 10))

# Plot all individual class ROC curves (semi-transparent)
for i in range(n_classes):
    ax.plot(fpr[i], tpr[i], alpha=0.25, linewidth=0.5, color='gray')

# Plot micro-average
ax.plot(fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='deeppink', linestyle='--', linewidth=2)

# Plot macro-average
ax.plot(fpr["macro"], tpr["macro"],
        label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
        color='navy', linestyle='--', linewidth=2)

# Plot diagonal
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random classifier')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
ax.set_title('Semi-Supervised VAE v2.14: ROC Curves (All 139 Lithology Classes)',
            fontsize=14, weight='bold')
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v2_14_roc_curves_all.png', dpi=300, bbox_inches='tight')
print("✓ Saved: v2_14_roc_curves_all.png")
plt.close()

# ============================================================================
# PLOT 2: Top 20 most common classes
# ============================================================================
print("\nCreating ROC plot for top 20 most common classes...")

# Get class frequencies
class_counts = pd.Series(y_test).value_counts()
top_20_classes = class_counts.head(20).index.tolist()

fig, ax = plt.subplots(figsize=(12, 10))

# Color map for top classes
colors = plt.cm.tab20(np.linspace(0, 1, 20))

for idx, class_idx in enumerate(top_20_classes):
    class_name = lithologies[class_idx]
    count = class_counts[class_idx]
    ax.plot(fpr[class_idx], tpr[class_idx],
           label=f'{class_name[:25]} (n={count}, AUC={roc_auc[class_idx]:.3f})',
           color=colors[idx], linewidth=1.5, alpha=0.8)

# Plot micro-average
ax.plot(fpr["micro"], tpr["micro"],
        label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
        color='black', linestyle='--', linewidth=2)

# Plot diagonal
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, weight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, weight='bold')
ax.set_title('Semi-Supervised VAE v2.14: ROC Curves (Top 20 Most Common Classes)',
            fontsize=14, weight='bold')
ax.legend(loc="lower right", fontsize=7, ncol=1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('v2_14_roc_curves_top20.png', dpi=300, bbox_inches='tight')
print("✓ Saved: v2_14_roc_curves_top20.png")
plt.close()

# ============================================================================
# PLOT 3: AUC distribution histogram
# ============================================================================
print("\nCreating AUC distribution histogram...")

auc_values = [roc_auc[i] for i in range(n_classes)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(auc_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(roc_auc["macro"], color='navy', linestyle='--', linewidth=2,
          label=f'Macro-average: {roc_auc["macro"]:.3f}')
ax.axvline(roc_auc["micro"], color='deeppink', linestyle='--', linewidth=2,
          label=f'Micro-average: {roc_auc["micro"]:.3f}')
ax.set_xlabel('AUC Score', fontsize=12, weight='bold')
ax.set_ylabel('Number of Classes', fontsize=12, weight='bold')
ax.set_title('Distribution of AUC Scores Across 139 Lithology Classes',
            fontsize=14, weight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('v2_14_auc_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: v2_14_auc_distribution.png")
plt.close()

# ============================================================================
# Save detailed AUC results
# ============================================================================
print("\nSaving detailed AUC results...")

results_df = pd.DataFrame({
    'Lithology': [lithologies[i] for i in range(n_classes)],
    'Class_Index': range(n_classes),
    'Test_Count': [class_counts.get(i, 0) for i in range(n_classes)],
    'AUC': [roc_auc[i] for i in range(n_classes)]
}).sort_values('AUC', ascending=False)

results_df.to_csv('v2_14_roc_auc_results.csv', index=False)
print("✓ Saved: v2_14_roc_auc_results.csv")

# Print summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Micro-average AUC: {roc_auc['micro']:.4f}")
print(f"Macro-average AUC: {roc_auc['macro']:.4f}")
print(f"\nAUC Statistics across {n_classes} classes:")
print(f"  Mean:   {np.mean(auc_values):.4f}")
print(f"  Median: {np.median(auc_values):.4f}")
print(f"  Std:    {np.std(auc_values):.4f}")
print(f"  Min:    {np.min(auc_values):.4f}")
print(f"  Max:    {np.max(auc_values):.4f}")

print("\nTop 10 classes by AUC:")
for idx, row in results_df.head(10).iterrows():
    print(f"  {row['Lithology'][:40]:40s}  AUC={row['AUC']:.4f}  n={int(row['Test_Count'])}")

print("\nBottom 10 classes by AUC:")
for idx, row in results_df.tail(10).iterrows():
    print(f"  {row['Lithology'][:40]:40s}  AUC={row['AUC']:.4f}  n={int(row['Test_Count'])}")

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print("Generated:")
print("  - v2_14_roc_curves_all.png (all 139 classes + averages)")
print("  - v2_14_roc_curves_top20.png (top 20 most common classes)")
print("  - v2_14_auc_distribution.png (histogram of AUC scores)")
print("  - v2_14_roc_auc_results.csv (detailed results)")
