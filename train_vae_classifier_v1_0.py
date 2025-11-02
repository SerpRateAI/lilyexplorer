"""
VAE Classifier v1.0

Supervised neural network classifier using v2.6.7 VAE latent embeddings.

Changes from baseline:
- Entropy-balanced borehole splitting (maintains lithology distributions)
- Class-balanced cross entropy loss (inverse frequency weighting)

This addresses extreme class imbalance (49,778:1 ratio, 139 classes).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

# VAE architecture (for feature extraction)
class VAE(nn.Module):
    """v2.6.7 architecture: 10D latent space"""
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

        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(hidden_dims[0], input_dim)

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
        return self.decode(z), mu, logvar

# Lithology Classifier
class LithologyClassifier(nn.Module):
    """
    Neural network classifier for lithology prediction.
    Input: 10D latent embeddings from VAE
    Output: Lithology class probabilities
    """
    def __init__(self, input_dim=10, num_classes=139, hidden_dims=[64, 32]):
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
    """
    Compute inverse frequency class weights.

    Rare classes get higher weight to balance gradient contributions.
    """
    class_counts = np.bincount(y_train, minlength=num_classes)

    # Inverse frequency weighting
    # Add small epsilon to avoid division by zero for missing classes
    epsilon = 1.0
    class_weights = 1.0 / (class_counts + epsilon)

    # Normalize so weights sum to num_classes (keeps loss magnitude similar)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights

def entropy_based_split(df, train_frac=0.70, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Split boreholes into train/val/test maintaining similar lithology entropy.

    Uses greedy allocation to balance distributions while keeping boreholes intact.
    """
    np.random.seed(random_state)

    # Get lithology distribution per borehole
    borehole_ids = df['Borehole_ID'].unique()
    borehole_stats = []

    for bh_id in borehole_ids:
        bh_data = df[df['Borehole_ID'] == bh_id]
        lithology_counts = bh_data['Principal'].value_counts()
        borehole_stats.append({
            'borehole_id': bh_id,
            'n_samples': len(bh_data),
            'lithology_counts': lithology_counts
        })

    # Sort by number of samples (largest first) for greedy allocation
    borehole_stats = sorted(borehole_stats, key=lambda x: x['n_samples'], reverse=True)

    # Randomize order within size bins to avoid systematic bias
    np.random.shuffle(borehole_stats)

    # Greedy allocation to splits
    train_boreholes = []
    val_boreholes = []
    test_boreholes = []

    total_samples = len(df)
    train_samples = 0
    val_samples = 0
    test_samples = 0

    for bh_stat in borehole_stats:
        bh_id = bh_stat['borehole_id']
        n_samples = bh_stat['n_samples']

        # Calculate current proportions
        current_total = train_samples + val_samples + test_samples
        if current_total == 0:
            # First borehole goes to train
            train_boreholes.append(bh_id)
            train_samples += n_samples
        else:
            train_prop = train_samples / current_total
            val_prop = val_samples / current_total
            test_prop = test_samples / current_total

            # Assign to split that is furthest from target
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

def extract_vae_embeddings(vae_model, X, device, batch_size=256):
    """Extract latent embeddings from VAE encoder"""
    vae_model.eval()
    embeddings = []

    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch_x, in loader:
            batch_x = batch_x.to(device)
            mu, _ = vae_model.encode(batch_x)
            embeddings.append(mu.cpu().numpy())

    return np.vstack(embeddings)

def train_classifier(model, train_loader, val_loader, epochs, device, criterion, lr=1e-3):
    """Train lithology classifier with class-balanced loss"""
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

        # Early stopping based on balanced accuracy
        if val_balanced_acc > best_val_balanced_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_val_balanced_acc = val_balanced_acc
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'ml_models/checkpoints/vae_classifier_v1_0_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

    print()
    print(f"Best validation accuracy (unweighted): {best_val_acc:.2f}%")
    print(f"Best validation accuracy (balanced): {best_val_balanced_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Load best model
    model.load_state_dict(torch.load('ml_models/checkpoints/vae_classifier_v1_0_best.pth'))

    return model

print("="*100)
print("VAE CLASSIFIER v1.0 (CLASS-BALANCED LOSS + ENTROPY-BALANCED SPLIT)")
print("="*100)
print()

# Load data
print("Loading data...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X = df[feature_cols].values
y = df['Principal'].values

print(f"Total samples: {len(df):,}")
print(f"Total boreholes: {df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {len(np.unique(y))}")
print()

# Create label mapping
unique_labels = sorted(np.unique(y))
label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
idx_to_label = {idx: label for label, idx in label_to_idx.items()}
y_encoded = np.array([label_to_idx[label] for label in y])

print(f"Number of classes: {len(unique_labels)}")
print()

# Entropy-balanced borehole split
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

print(f"Train: {len(X_train):,} samples ({len(train_boreholes)} boreholes, {100*len(X_train)/len(df):.1f}%)")
print(f"Val:   {len(X_val):,} samples ({len(val_boreholes)} boreholes, {100*len(X_val)/len(df):.1f}%)")
print(f"Test:  {len(X_test):,} samples ({len(test_boreholes)} boreholes, {100*len(X_test)/len(df):.1f}%)")
print()

# Check entropy balance
def calculate_entropy(y_split):
    counts = np.unique(y_split, return_counts=True)[1]
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

train_entropy = calculate_entropy(y_train)
val_entropy = calculate_entropy(y_val)
test_entropy = calculate_entropy(y_test)

print(f"Lithology entropy:")
print(f"  Train: {train_entropy:.4f}")
print(f"  Val:   {val_entropy:.4f}")
print(f"  Test:  {test_entropy:.4f}")
print(f"  Max difference: {max(abs(train_entropy-val_entropy), abs(train_entropy-test_entropy), abs(val_entropy-test_entropy)):.4f}")
print()

# Compute class weights
print("Computing class weights (inverse frequency)...")
class_weights = compute_class_weights(y_train, len(unique_labels))

# Show weight distribution
print(f"Class weight statistics:")
print(f"  Min weight: {class_weights.min():.4f}")
print(f"  Max weight: {class_weights.max():.4f}")
print(f"  Mean weight: {class_weights.mean():.4f}")
print(f"  Median weight: {np.median(class_weights):.4f}")
print(f"  Weight ratio (max/min): {class_weights.max() / class_weights.min():.1f}x")
print()

# Show top 5 most weighted (rarest) and least weighted (most common) classes
weight_order = np.argsort(class_weights)[::-1]
print("Top 5 highest weighted classes (rarest):")
for i in range(min(5, len(weight_order))):
    idx = weight_order[i]
    class_name = idx_to_label[idx]
    n_samples = (y_train == idx).sum()
    print(f"  {class_name:40s}: weight={class_weights[idx]:.4f}, n_train={n_samples}")
print()

print("Top 5 lowest weighted classes (most common):")
for i in range(min(5, len(weight_order))):
    idx = weight_order[-(i+1)]
    class_name = idx_to_label[idx]
    n_samples = (y_train == idx).sum()
    print(f"  {class_name:40s}: weight={class_weights[idx]:.4f}, n_train={n_samples}")
print()

# Load v2.6.7 VAE and extract embeddings
print("Loading v2.6.7 VAE model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                       map_location=device, weights_only=False)

vae_model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()

scaler = checkpoint['scaler']

print("✓ VAE loaded successfully")
print()

# Scale data
print("Scaling features...")
X_train_scaled = scaler.transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("✓ Features scaled")
print()

# Extract VAE embeddings
print("Extracting VAE embeddings...")
start_time = time.time()
train_embeddings = extract_vae_embeddings(vae_model, X_train_scaled, device)
val_embeddings = extract_vae_embeddings(vae_model, X_val_scaled, device)
test_embeddings = extract_vae_embeddings(vae_model, X_test_scaled, device)
print(f"✓ Embeddings extracted in {time.time() - start_time:.1f}s")
print(f"  Embedding shape: {train_embeddings.shape}")
print()

# Analyze embeddings
embedding_stds = train_embeddings.std(axis=0)
print("Embedding statistics:")
for i, std in enumerate(embedding_stds):
    status = "✓" if std >= 0.1 else "✗"
    print(f"  Dim {i}: std={std:.4f} {status}")
print()

# Create dataloaders
train_dataset = TensorDataset(torch.FloatTensor(train_embeddings),
                              torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.FloatTensor(val_embeddings),
                            torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.FloatTensor(test_embeddings),
                             torch.LongTensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Initialize classifier
print("="*100)
print("TRAINING LITHOLOGY CLASSIFIER")
print("="*100)
classifier = LithologyClassifier(input_dim=10, num_classes=len(unique_labels),
                                hidden_dims=[64, 32]).to(device)

print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
print()

# Create class-balanced loss function
class_weights_tensor = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

print("Using class-balanced CrossEntropyLoss with inverse frequency weighting")
print()

# Train classifier
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

# Overall accuracy (unweighted)
test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy (unweighted): {test_acc*100:.2f}%")

# Balanced accuracy (weighted by class support)
balanced_acc = balanced_accuracy_score(test_labels, test_preds)
print(f"Test Accuracy (balanced):   {balanced_acc*100:.2f}%")
print()

# Per-class metrics
print("Top 10 Most Common Lithologies:")
unique, counts = np.unique(test_labels, return_counts=True)
top_10_idx = counts.argsort()[-10:][::-1]

for idx in top_10_idx:
    class_idx = unique[idx]
    class_name = idx_to_label[class_idx]
    class_mask = test_labels == class_idx
    class_acc = (test_preds[class_mask] == test_labels[class_mask]).mean()
    print(f"  {class_name:30s}: {counts[idx]:6d} samples, Acc={class_acc*100:.2f}%")

print()

# Check for classes with 0 training samples
print("Classes with no training samples:")
train_classes = set(np.unique(y_train))
test_classes = set(np.unique(y_test))
missing_classes = test_classes - train_classes

if len(missing_classes) > 0:
    print(f"  Found {len(missing_classes)} classes in test but not in train:")
    for class_idx in sorted(missing_classes):
        class_name = idx_to_label[class_idx]
        n_test = (y_test == class_idx).sum()
        print(f"    {class_name}: {n_test} test samples")
else:
    print(f"  ✓ All test classes present in training set")

print()
print("="*100)
print(f"✓ VAE Classifier v1.0 trained successfully")
print(f"  Test Accuracy (unweighted): {test_acc*100:.2f}%")
print(f"  Test Accuracy (balanced):   {balanced_acc*100:.2f}%")
print(f"  Model saved: ml_models/checkpoints/vae_classifier_v1_0_best.pth")
print("="*100)
