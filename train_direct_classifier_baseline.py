"""
Direct Baseline Classifier (no VAE)

Classify lithology groups directly from raw physical properties:
GRA + MS + NGR + RGB → Lithology Group

This establishes if ~30-40% accuracy is:
1. Fundamental limit of physical properties
2. Or VAE is losing discriminative information
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

# Simple feedforward classifier
class DirectClassifier(nn.Module):
    """
    Direct classifier from raw features to lithology groups.
    Input: 6D raw features (GRA, MS, NGR, RGB)
    Output: Lithology group probabilities
    """
    def __init__(self, input_dim=6, num_classes=14, hidden_dims=[64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = h_dim

        self.classifier = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        h = self.classifier(x)
        return self.fc_out(h)

def compute_class_weights(y_train, num_classes):
    """Compute inverse frequency class weights."""
    class_counts = np.bincount(y_train, minlength=num_classes)
    epsilon = 1.0
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights = class_weights / class_weights.sum() * num_classes
    return class_weights

def entropy_based_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    """Entropy-balanced borehole splitting."""
    np.random.seed(random_state)

    borehole_ids = df['Borehole_ID'].unique()
    borehole_stats = []

    for bh_id in borehole_ids:
        bh_data = df[df['Borehole_ID'] == bh_id]
        borehole_stats.append({
            'borehole_id': bh_id,
            'n_samples': len(bh_data)
        })

    borehole_stats = sorted(borehole_stats, key=lambda x: x['n_samples'], reverse=True)
    np.random.shuffle(borehole_stats)

    train_boreholes = []
    val_boreholes = []
    test_boreholes = []

    train_samples = 0
    val_samples = 0
    test_samples = 0

    for bh_stat in borehole_stats:
        bh_id = bh_stat['borehole_id']
        n_samples = bh_stat['n_samples']

        current_total = train_samples + val_samples + test_samples
        if current_total == 0:
            train_boreholes.append(bh_id)
            train_samples += n_samples
        else:
            train_prop = train_samples / current_total
            val_prop = val_samples / current_total
            test_prop = test_samples / current_total

            train_deficit = train_frac - train_prop
            val_deficit = val_frac - val_prop
            test_deficit = test_frac - test_prop

            max_deficit = max(train_deficit, val_deficit, test_deficit)

            if max_deficit == train_deficit:
                train_boreholes.append(bh_id)
                train_samples += n_samples
            elif max_deficit == val_deficit:
                val_boreholes.append(bh_id)
                val_samples += n_samples
            else:
                test_boreholes.append(bh_id)
                test_samples += n_samples

    return train_boreholes, val_boreholes, test_boreholes

def train_classifier(model, train_loader, val_loader, epochs, device, criterion, lr=1e-3):
    """Train classifier."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    best_val_loss = float('inf')
    best_val_balanced_acc = 0
    patience_counter = 0
    patience = 15

    print("Starting training...")
    print()

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(batch_y.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_balanced_acc = balanced_accuracy_score(val_labels, val_preds) * 100

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%, Val Balanced Acc={val_balanced_acc:.2f}%")

        # Early stopping
        if val_balanced_acc > best_val_balanced_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_balanced_acc = val_balanced_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'ml_models/checkpoints/direct_classifier_baseline_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print()
    print(f"Best validation accuracy (unweighted): {best_val_acc:.2f}%")
    print(f"Best validation accuracy (balanced): {best_val_balanced_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")

    model.load_state_dict(torch.load('ml_models/checkpoints/direct_classifier_baseline_best.pth'))

    return model

print("="*100)
print("DIRECT BASELINE CLASSIFIER (NO VAE)")
print("="*100)
print()

# Load data
print("Loading data...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

# Load lithology hierarchy
hierarchy_df = pd.read_csv('lithology_hierarchy_mapping.csv')
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'],
                             hierarchy_df['Lithology_Group']))

df['Lithology_Group'] = df['Principal'].map(principal_to_group)

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X = df[feature_cols].values
y = df['Lithology_Group'].values

print(f"Total samples: {len(df):,}")
print(f"Total boreholes: {df['Borehole_ID'].nunique()}")
print(f"Unique lithology groups: {len(np.unique(y))}")
print()

# Create label mapping
unique_labels = sorted(np.unique(y))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
y_encoded = np.array([label_to_idx[label] for label in y])

# Entropy-balanced split
print("Performing entropy-balanced borehole split...")
train_boreholes, val_boreholes, test_boreholes = entropy_based_split(
    df, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

X_train, y_train = X[train_mask], y_encoded[train_mask]
X_val, y_val = X[val_mask], y_encoded[val_mask]
X_test, y_test = X[test_mask], y_encoded[test_mask]

print(f"Train: {len(X_train):,} samples ({len(train_boreholes)} boreholes)")
print(f"Val:   {len(X_val):,} samples ({len(val_boreholes)} boreholes)")
print(f"Test:  {len(X_test):,} samples ({len(test_boreholes)} boreholes)")
print()

# Scale features (distribution-aware)
print("Scaling features (distribution-aware)...")
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")
print()

# Compute class weights
print("Computing class weights...")
class_weights = compute_class_weights(y_train, len(unique_labels))
print(f"Weight ratio (max/min): {class_weights.max() / class_weights.min():.1f}x")
print()

# Create dataloaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled),
                              torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled),
                            torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled),
                             torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize classifier
print("="*100)
print("TRAINING DIRECT CLASSIFIER")
print("="*100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

classifier = DirectClassifier(input_dim=6, num_classes=len(unique_labels),
                             hidden_dims=[64, 32]).to(device)

print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
print()

# Class-balanced loss
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Train
start_time = time.time()
classifier = train_classifier(classifier, train_loader, val_loader,
                             epochs=100, device=device, criterion=criterion, lr=1e-3)
train_time = time.time() - start_time

print(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
print()

# Evaluate on test set
print("="*100)
print("TEST SET EVALUATION")
print("="*100)

classifier.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        outputs = classifier(batch_x)
        _, predicted = outputs.max(1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(batch_y.numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)

test_acc = accuracy_score(test_labels, test_preds)
balanced_acc = balanced_accuracy_score(test_labels, test_preds)

print(f"Test Accuracy (unweighted): {test_acc*100:.2f}%")
print(f"Test Accuracy (balanced):   {balanced_acc*100:.2f}%")
print()

# Per-group metrics
print("Per-group accuracies (all groups):")
unique, counts = np.unique(test_labels, return_counts=True)
sorted_idx = counts.argsort()[::-1]

for idx in sorted_idx:
    class_idx = unique[idx]
    class_name = idx_to_label[class_idx]
    class_mask = test_labels == class_idx
    class_acc = (test_preds[class_mask] == test_labels[class_mask]).mean()
    print(f"  {class_name:30s}: {counts[idx]:6,d} samples, Acc={class_acc*100:.2f}%")

print()
print("="*100)
print("COMPARISON TO VAE CLASSIFIER v1.1")
print("="*100)
print()
print(f"Direct Classifier (raw features):   Unweighted={test_acc*100:.2f}%, Balanced={balanced_acc*100:.2f}%")
print(f"VAE Classifier v1.1 (10D embeddings): Unweighted=34.18%, Balanced=29.73%")
print()

if balanced_acc * 100 > 29.73:
    improvement = ((balanced_acc * 100) - 29.73) / 29.73 * 100
    print(f"✓ Direct classifier is {improvement:.1f}% better")
    print(f"  → VAE is losing discriminative information")
elif balanced_acc * 100 < 29.73:
    degradation = (29.73 - (balanced_acc * 100)) / 29.73 * 100
    print(f"✗ Direct classifier is {degradation:.1f}% worse")
    print(f"  → VAE embeddings are actually helping!")
else:
    print(f"≈ Similar performance")
    print(f"  → 30% balanced accuracy is the fundamental limit of physical properties")

print()
print("="*100)
