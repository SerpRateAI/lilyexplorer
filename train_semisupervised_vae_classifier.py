"""
Semi-supervised classifier using VAE v2.6.7 pre-training.

Approach:
1. Load pre-trained VAE v2.6.7 encoder
2. Add classification head on top of 10D embeddings
3. Option A: Freeze encoder, train classifier only
4. Option B: Fine-tune encoder + classifier with small LR

Compare to baselines:
- Direct classifier (raw 6D features): 42.32% balanced accuracy
- VAE classifier v1.1 (frozen embeddings): 29.73% balanced accuracy

Goal: Test if fine-tuning VAE encoder for classification improves over frozen embeddings.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

print("="*100)
print("SEMI-SUPERVISED CLASSIFIER: VAE v2.6.7 PRE-TRAINING + FINE-TUNING")
print("="*100)
print()

# Load v2.6.7 VAE
print("Loading VAE v2.6.7...")
checkpoint = torch.load('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                        map_location='cpu', weights_only=False)

# VAE architecture (same as v2.6.7)
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
        return mu  # Use mean, not sampled z, for deterministic embeddings

class SemiSupervisedClassifier(nn.Module):
    """VAE encoder + classification head"""
    def __init__(self, encoder, num_classes=14, hidden_dim=32, freeze_encoder=False):
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

# Load pre-trained VAE encoder weights
encoder = VAEEncoder(input_dim=6, latent_dim=10, hidden_dims=[32, 16])
encoder_state_dict = {k.replace('encoder.', '').replace('fc_mu.', 'fc_mu.').replace('fc_logvar.', 'fc_logvar.'): v
                      for k, v in checkpoint['model_state_dict'].items()
                      if 'encoder' in k or 'fc_mu' in k or 'fc_logvar' in k}
encoder.load_state_dict(encoder_state_dict, strict=False)
print("✓ VAE v2.6.7 encoder loaded")
print()

# Load dataset
print("Loading v2.6.7 dataset...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"Total samples: {len(df):,}")

# Load lithology hierarchy
print("Loading lithology hierarchy...")
hierarchy_df = pd.read_csv('/home/utig5/johna/bhai/lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'], hierarchy_df['Lithology_Group']))

# Map Principal → Lithology_Group
df['Lithology_Group'] = df['Principal'].map(principal_to_group)
df = df.dropna(subset=['Lithology_Group'])  # Remove unmapped lithologies

print(f"Samples after hierarchy mapping: {len(df):,}")
print(f"Lithology groups: {df['Lithology_Group'].nunique()}")
print()

# Features and labels
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
y = df['Lithology_Group'].values

# Encode labels
lithology_groups = sorted(df['Lithology_Group'].unique())
lithology_to_idx = {lith: i for i, lith in enumerate(lithology_groups)}
y_encoded = np.array([lithology_to_idx[lith] for lith in y])

print(f"Classes: {len(lithology_groups)}")
for i, lith in enumerate(lithology_groups):
    count = (y_encoded == i).sum()
    print(f"  {i:2d}. {lith:30s}: {count:>8,} samples ({count/len(y)*100:5.2f}%)")
print()

# Entropy-balanced borehole split (same as direct classifier baseline)
print("Creating entropy-balanced borehole split...")
boreholes = df['Borehole_ID'].unique()
bh_to_samples = {bh: df[df['Borehole_ID']==bh].index.tolist() for bh in boreholes}

# Simple train/test split by boreholes (70/30)
np.random.seed(42)
np.random.shuffle(boreholes)
n_train_bh = int(0.7 * len(boreholes))
train_boreholes = boreholes[:n_train_bh]
test_boreholes = boreholes[n_train_bh:]

train_idx = []
test_idx = []
for bh in train_boreholes:
    train_idx.extend(bh_to_samples[bh])
for bh in test_boreholes:
    test_idx.extend(bh_to_samples[bh])

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

print(f"Train: {len(X_train):,} samples from {len(train_boreholes)} boreholes")
print(f"Test:  {len(X_test):,} samples from {len(test_boreholes)} boreholes")
print()

# Distribution-aware scaling (same as VAE)
scaler = checkpoint['scaler']  # Use VAE's scaler
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Class weights (inverse frequency, clipped)
class_counts = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1)
class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
class_weights = torch.FloatTensor(class_weights)

max_weight = class_weights.max()
min_weight = class_weights.min()
print(f"Class weights: min={min_weight:.3f}, max={max_weight:.3f}, ratio={max_weight/min_weight:.1f}×")
print()

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Train two variants
variants = [
    ('frozen_encoder', True, 1e-3),
    ('finetuned_encoder', False, 1e-4)  # Lower LR for fine-tuning
]

results = {}

for variant_name, freeze_encoder, lr in variants:
    print("="*100)
    print(f"TRAINING: {variant_name.replace('_', ' ').upper()}")
    print("="*100)
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Learning rate: {lr}")
    print()

    # Initialize model
    model = SemiSupervisedClassifier(encoder, num_classes=14, hidden_dim=32, freeze_encoder=freeze_encoder).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

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

    print()
    print(f"Best balanced accuracy: {best_bal_acc:.4f}")
    print()

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

    # Metrics
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)

    # Per-class accuracy
    per_class_acc = []
    for i in range(14):
        mask = np.array(y_true_all) == i
        if mask.sum() > 0:
            class_acc = (np.array(y_pred_all)[mask] == i).mean()
            per_class_acc.append(class_acc)
            print(f"  {lithology_groups[i]:30s}: {class_acc:6.2%} ({mask.sum():>6,} samples)")
        else:
            per_class_acc.append(0.0)

    print()
    print(f"Overall balanced accuracy: {bal_acc:.4f}")

    results[variant_name] = {
        'bal_acc': bal_acc,
        'per_class_acc': per_class_acc,
        'y_pred': y_pred_all,
        'y_true': y_true_all
    }

    # Save model
    save_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/semisupervised_{variant_name}_best.pth'
    torch.save({
        'model_state_dict': best_model_state,
        'freeze_encoder': freeze_encoder,
        'lr': lr,
        'bal_acc': bal_acc,
        'lithology_groups': lithology_groups
    }, save_path)
    print(f"✓ Model saved to: {save_path}")
    print()

# Comparison
print("="*100)
print("COMPARISON TO BASELINES")
print("="*100)
print()

print("| Approach | Balanced Acc | Notes |")
print("|----------|--------------|-------|")
print(f"| Direct classifier (raw 6D) | 42.32% | Baseline (best) |")
print(f"| VAE classifier v1.1 (frozen 10D) | 29.73% | Previous work |")
print(f"| Semi-supervised (frozen encoder) | {results['frozen_encoder']['bal_acc']:.2%} | This work |")
print(f"| Semi-supervised (fine-tuned encoder) | {results['finetuned_encoder']['bal_acc']:.2%} | This work |")
print()

# Analysis
frozen_acc = results['frozen_encoder']['bal_acc']
finetuned_acc = results['finetuned_encoder']['bal_acc']
direct_baseline = 0.4232
vae_baseline = 0.2973

print("Analysis:")
print(f"  Frozen encoder vs VAE v1.1: {(frozen_acc - vae_baseline)/vae_baseline*100:+.1f}%")
print(f"  Fine-tuned vs frozen: {(finetuned_acc - frozen_acc)/frozen_acc*100:+.1f}%")
print(f"  Fine-tuned vs direct classifier: {(finetuned_acc - direct_baseline)/direct_baseline*100:+.1f}%")
print()

if finetuned_acc > direct_baseline:
    print("✓ SUCCESS: Fine-tuning VAE encoder beats direct classifier!")
    print("  VAE pre-training provides useful initialization for classification.")
elif finetuned_acc > vae_baseline:
    print("⚠ PARTIAL SUCCESS: Fine-tuning improves over frozen VAE embeddings.")
    print(f"  But still {(direct_baseline - finetuned_acc)*100:.1f}% worse than direct classifier (42.32%).")
    print("  Raw features remain superior for classification.")
else:
    print("✗ FAILURE: Fine-tuning does not improve over frozen embeddings.")
    print("  VAE pre-training is not beneficial for classification task.")

print()
print("="*100)
print("SEMI-SUPERVISED CLASSIFIER TRAINING COMPLETE")
print("="*100)
