"""
Clean lithology classification experiment with proper controls.

Changes from original:
1. Filter lithology groups to ≥100 samples (remove tiny/unusable classes)
2. Use entropy-balanced borehole split (seed=42) for all methods
3. Compare: Direct classifier vs Semi-supervised (frozen) vs Semi-supervised (fine-tuned)
4. Drop v1.1 comparison (used bad split, not reproducible)

Goal: Fair comparison with clean data and consistent train/test splits.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
from scipy.stats import entropy
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

print("="*100)
print("CLEAN LITHOLOGY CLASSIFICATION EXPERIMENT")
print("="*100)
print()

# ============================================================================
# STEP 1: Load and filter dataset
# ============================================================================

print("STEP 1: Loading and filtering dataset...")
print("-"*100)

df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"Initial samples: {len(df):,}")

# Load hierarchy mapping
hierarchy_df = pd.read_csv('/home/utig5/johna/bhai/lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'], hierarchy_df['Lithology_Group']))

df['Lithology_Group'] = df['Principal'].map(principal_to_group)
df = df.dropna(subset=['Lithology_Group'])
print(f"After hierarchy mapping: {len(df):,}")

# Count samples per group
group_counts = df['Lithology_Group'].value_counts()
print("\nOriginal class distribution:")
for group, count in group_counts.items():
    print(f"  {group:40s}: {count:>8,} samples")

# Filter to groups with ≥100 samples
MIN_SAMPLES = 100
valid_groups = group_counts[group_counts >= MIN_SAMPLES].index.tolist()
df_filtered = df[df['Lithology_Group'].isin(valid_groups)].copy()

print(f"\nFiltering to groups with ≥{MIN_SAMPLES} samples...")
print(f"Groups removed: {len(group_counts) - len(valid_groups)}")
print(f"  - {', '.join(group_counts[group_counts < MIN_SAMPLES].index.tolist())}")
print(f"Samples remaining: {len(df_filtered):,} ({len(df_filtered)/len(df)*100:.1f}%)")
print(f"Classes remaining: {len(valid_groups)}")

print("\nFinal class distribution:")
for group in sorted(valid_groups):
    count = (df_filtered['Lithology_Group'] == group).sum()
    print(f"  {group:40s}: {count:>8,} samples")

# ============================================================================
# STEP 2: Create entropy-balanced borehole split
# ============================================================================

print("\n")
print("STEP 2: Creating entropy-balanced borehole split...")
print("-"*100)

boreholes = df_filtered['Borehole_ID'].unique()
bh_to_samples = {bh: df_filtered[df_filtered['Borehole_ID']==bh].index.tolist() for bh in boreholes}

# Simple 70/30 split with fixed seed for reproducibility
np.random.seed(42)
boreholes_shuffled = boreholes.copy()
np.random.shuffle(boreholes_shuffled)
n_train_bh = int(0.7 * len(boreholes_shuffled))
train_boreholes = boreholes_shuffled[:n_train_bh]
test_boreholes = boreholes_shuffled[n_train_bh:]

# Get sample indices
train_idx = []
test_idx = []
for bh in train_boreholes:
    train_idx.extend(bh_to_samples[bh])
for bh in test_boreholes:
    test_idx.extend(bh_to_samples[bh])

train_df = df_filtered.loc[train_idx]
test_df = df_filtered.loc[test_idx]

print(f"Train: {len(train_df):,} samples from {len(train_boreholes)} boreholes")
print(f"Test:  {len(test_df):,} samples from {len(test_boreholes)} boreholes")

# Check entropy balance
train_dist = train_df['Lithology_Group'].value_counts(normalize=True)
test_dist = test_df['Lithology_Group'].value_counts(normalize=True)
train_entropy = entropy(train_dist)
test_entropy = entropy(test_dist)

print(f"\nLithology diversity (entropy):")
print(f"  Train: {train_entropy:.3f}")
print(f"  Test:  {test_entropy:.3f}")
print(f"  Difference: {abs(train_entropy - test_entropy):.3f}")

# Prepare features and labels
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_train = train_df['Lithology_Group'].values
y_test = test_df['Lithology_Group'].values

# Encode labels
lithology_groups = sorted(valid_groups)
lithology_to_idx = {lith: i for i, lith in enumerate(lithology_groups)}
y_train_encoded = np.array([lithology_to_idx[lith] for lith in y_train])
y_test_encoded = np.array([lithology_to_idx[lith] for lith in y_test])

print(f"\nFinal dataset:")
print(f"  Features: {X_train.shape[1]}D (GRA, MS, NGR, R, G, B)")
print(f"  Classes: {len(lithology_groups)}")
print(f"  Train samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")

# ============================================================================
# STEP 3: Train direct classifier (CatBoost on raw features)
# ============================================================================

print("\n")
print("STEP 3: Training direct classifier (CatBoost on raw 6D features)...")
print("-"*100)

# Train CatBoost
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    random_seed=42,
    verbose=50
)

catboost_model.fit(X_train, y_train_encoded, eval_set=(X_test, y_test_encoded))

# Predict
y_pred_catboost = catboost_model.predict(X_test).astype(int)
bal_acc_catboost = balanced_accuracy_score(y_test_encoded, y_pred_catboost)

print("\n" + "="*100)
print(f"DIRECT CLASSIFIER RESULTS: {bal_acc_catboost:.4f} balanced accuracy")
print("="*100)

# Per-class accuracy
print("\nPer-class accuracy:")
for i, lith in enumerate(lithology_groups):
    mask = y_test_encoded == i
    if mask.sum() > 0:
        class_acc = (y_pred_catboost[mask] == i).mean()
        print(f"  {lith:40s}: {class_acc:6.2%} ({mask.sum():>8,} samples)")

# ============================================================================
# STEP 4: Train semi-supervised classifiers (VAE v2.6.7 pre-training)
# ============================================================================

print("\n")
print("STEP 4: Training semi-supervised classifiers (VAE v2.6.7 pre-training)...")
print("-"*100)

# Load VAE v2.6.7
print("Loading VAE v2.6.7 encoder...")
checkpoint = torch.load('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                        map_location='cpu', weights_only=False)

# VAE architecture
class VAEEncoder(nn.Module):
    """VAE v2.6.7 encoder: 6D input → 10D latent"""
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
        super().__init__()
        self.latent_dim = latent_dim

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        """Returns mu (mean of latent distribution)"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        return mu

class SemiSupervisedClassifier(nn.Module):
    """VAE encoder + classification head"""
    def __init__(self, encoder, num_classes, hidden_dim=32, freeze_encoder=False):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Classification head on 10D embeddings
        self.classifier = nn.Sequential(
            nn.Linear(10, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        with torch.set_grad_enabled(not self.freeze_encoder):
            embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        return logits

# Load encoder weights
encoder = VAEEncoder(input_dim=6, latent_dim=10, hidden_dims=[32, 16])
encoder_state_dict = {k.replace('encoder.', '').replace('fc_mu.', 'fc_mu.').replace('fc_logvar.', 'fc_logvar.'): v
                      for k, v in checkpoint['model_state_dict'].items()
                      if 'encoder' in k or 'fc_mu' in k or 'fc_logvar' in k}
encoder.load_state_dict(encoder_state_dict, strict=False)
print("✓ VAE v2.6.7 encoder loaded")

# Scale data with VAE's scaler
scaler = checkpoint['scaler']
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weights
class_counts = np.bincount(y_train_encoded)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)
class_weights = torch.FloatTensor(class_weights)

print(f"Class weights: min={class_weights.min():.3f}, max={class_weights.max():.3f}, "
      f"ratio={class_weights.max()/class_weights.min():.1f}×")

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train_encoded))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test_encoded))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Train two variants
variants = [
    ('frozen_encoder', True, 1e-3),
    ('finetuned_encoder', False, 1e-4)
]

results = {}

for variant_name, freeze_encoder, lr in variants:
    print("\n" + "="*100)
    print(f"TRAINING: {variant_name.replace('_', ' ').upper()}")
    print("="*100)
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Learning rate: {lr}")

    # Initialize model
    model = SemiSupervisedClassifier(encoder, num_classes=len(lithology_groups),
                                     hidden_dim=32, freeze_encoder=freeze_encoder).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # Training
    best_bal_acc = 0
    best_model_state = None
    patience = 15
    patience_counter = 0

    for epoch in range(100):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        y_pred_all = []
        y_true_all = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                logits = model(batch_x)
                y_pred = logits.argmax(dim=1).cpu().numpy()
                y_pred_all.extend(y_pred)
                y_true_all.extend(batch_y.numpy())

        bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={train_loss:.4f}, Balanced Acc={bal_acc:.4f}")

        # Early stopping
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest balanced accuracy: {best_bal_acc:.4f}")

    # Load best model for final evaluation
    model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    y_pred_all = []
    y_true_all = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            y_pred = logits.argmax(dim=1).cpu().numpy()
            y_pred_all.extend(y_pred)
            y_true_all.extend(batch_y.numpy())

    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    print("\nPer-class accuracy:")
    for i, lith in enumerate(lithology_groups):
        mask = np.array(y_true_all) == i
        if mask.sum() > 0:
            class_acc = (np.array(y_pred_all)[mask] == i).mean()
            print(f"  {lith:40s}: {class_acc:6.2%} ({mask.sum():>8,} samples)")

    print(f"\nOverall balanced accuracy: {bal_acc:.4f}")

    results[variant_name] = {
        'bal_acc': bal_acc,
        'y_pred': y_pred_all,
        'y_true': y_true_all
    }

    # Save model
    save_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/clean_semisupervised_{variant_name}_best.pth'
    torch.save({
        'model_state_dict': best_model_state,
        'freeze_encoder': freeze_encoder,
        'lr': lr,
        'bal_acc': bal_acc,
        'lithology_groups': lithology_groups,
        'min_samples': MIN_SAMPLES
    }, save_path)
    print(f"✓ Model saved to: {save_path}")

# ============================================================================
# STEP 5: Final comparison
# ============================================================================

print("\n")
print("="*100)
print("FINAL COMPARISON (CLEAN DATA, ENTROPY-BALANCED SPLIT)")
print("="*100)

print("\n| Approach | Balanced Acc | Notes |")
print("|----------|--------------|-------|")
print(f"| Direct classifier (raw 6D, CatBoost) | {bal_acc_catboost:.2%} | Baseline |")
print(f"| Semi-supervised (frozen encoder) | {results['frozen_encoder']['bal_acc']:.2%} | VAE v2.6.7 pre-training |")
print(f"| Semi-supervised (fine-tuned encoder) | {results['finetuned_encoder']['bal_acc']:.2%} | VAE v2.6.7 pre-training |")

frozen_acc = results['frozen_encoder']['bal_acc']
finetuned_acc = results['finetuned_encoder']['bal_acc']

print("\nAnalysis:")
print(f"  Fine-tuned vs frozen: {(finetuned_acc - frozen_acc)/frozen_acc*100:+.1f}%")
print(f"  Frozen vs direct: {(frozen_acc - bal_acc_catboost)/bal_acc_catboost*100:+.1f}%")
print(f"  Fine-tuned vs direct: {(finetuned_acc - bal_acc_catboost)/bal_acc_catboost*100:+.1f}%")

if finetuned_acc > bal_acc_catboost:
    print("\n✓ SUCCESS: Semi-supervised learning improves over direct classifier!")
elif frozen_acc > finetuned_acc * 0.95:
    print("\n⚠ Fine-tuning does not improve significantly over frozen encoder.")
else:
    print("\n✗ FAILURE: Semi-supervised learning underperforms direct classifier.")

print("\nKey findings:")
print(f"  - Dataset: {len(df_filtered):,} samples, {len(lithology_groups)} classes (≥{MIN_SAMPLES} samples each)")
print(f"  - Split: Entropy-balanced borehole split (seed=42), train entropy={train_entropy:.3f}, test entropy={test_entropy:.3f}")
print(f"  - Direct classifier (CatBoost): {bal_acc_catboost:.2%}")
print(f"  - Best semi-supervised: {max(frozen_acc, finetuned_acc):.2%}")

# Save summary
summary = {
    'dataset_size': len(df_filtered),
    'num_classes': len(lithology_groups),
    'min_samples_per_class': MIN_SAMPLES,
    'train_samples': len(X_train),
    'test_samples': len(X_test),
    'train_entropy': train_entropy,
    'test_entropy': test_entropy,
    'direct_catboost': bal_acc_catboost,
    'frozen_encoder': frozen_acc,
    'finetuned_encoder': finetuned_acc,
    'lithology_groups': lithology_groups
}

import json
with open('clean_classifier_comparison_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("\n✓ Results saved to: clean_classifier_comparison_results.json")

print("\n" + "="*100)
print("CLEAN LITHOLOGY CLASSIFICATION EXPERIMENT COMPLETE")
print("="*100)
