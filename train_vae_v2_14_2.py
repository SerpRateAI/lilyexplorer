#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE v2.14.2: Semi-Supervised VAE with Random Feature Masking

Architecture: Same as v2.14 (encoder [32,16], latent 10D, classifier head)
Key difference: Random feature masking during training (30% per feature)

Tests if masking mechanism works on complete data before applying to real missing data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
import time

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


class DistributionAwareScaler:
    """Custom scaler that applies distribution-specific transformations."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR can be negative
        self.log_indices = [3, 4, 5]  # R, G, B are always positive

    def signed_log_transform(self, x):
        """Log transform that preserves sign for data with negative values."""
        return np.sign(x) * np.log1p(np.abs(x))

    def fit_transform(self, X):
        """Apply distribution-specific transforms, then standard scale."""
        X_transformed = X.copy()

        # Apply signed log(|x| + 1) to features that can be negative (MS, NGR)
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply log(x + 1) to features that are always positive (RGB)
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Standard scale all features
        X_scaled = self.scaler.fit_transform(X_transformed)
        return X_scaled

    def transform(self, X):
        """Transform new data using fitted scaler."""
        X_transformed = X.copy()

        # Apply signed log to MS, NGR
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply regular log to RGB
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Standard scale
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled


class SemiSupervisedVAE_Masked(nn.Module):
    """
    Semi-Supervised VAE with Random Feature Masking

    Architecture (same as v2.14):
    - Encoder: Input → [32,16] → latent_dim (10D)
    - Decoder: latent_dim → [16,32] → Input
    - Classifier: latent_dim → [32] → n_classes

    New: Random masking during training
    """

    def __init__(self, input_dim=6, latent_dim=10, n_classes=209,
                 encoder_dims=[32, 16], classifier_hidden=32, mask_prob=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.mask_prob = mask_prob

        # Encoder (same as v2.14)
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder (same as v2.14)
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        # Classification head (same as v2.14)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def apply_mask(self, x):
        """
        Randomly mask features with probability mask_prob
        Returns: masked input, mask (1=present, 0=masked)
        """
        if not self.training:
            # No masking during evaluation
            return x, torch.ones_like(x)

        # Create random mask (1=keep, 0=mask)
        mask = (torch.rand_like(x) > self.mask_prob).float()

        # Ensure at least one feature is present per sample
        zero_mask_samples = (mask.sum(dim=1) == 0)
        if zero_mask_samples.any():
            # Randomly unmask one feature for each all-zero sample
            for idx in torch.where(zero_mask_samples)[0]:
                random_feature = torch.randint(0, self.input_dim, (1,))
                mask[idx, random_feature] = 1.0

        # Apply mask (set masked values to 0)
        x_masked = x * mask

        return x_masked, mask

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

    def classify(self, z):
        """Classify from latent space (no gradients through reparameterization)"""
        return self.classifier(z)

    def forward(self, x):
        # Apply random masking during training
        x_masked, mask = self.apply_mask(x)

        # Encode masked input
        mu, logvar = self.encode(x_masked)
        z = self.reparameterize(mu, logvar)

        # Reconstruct FULL input (not just masked features)
        x_recon = self.decode(z)

        # Classify from latent representation
        logits = self.classify(mu)

        return x_recon, mu, logvar, logits, mask


class LithologyDataset(Dataset):
    """Dataset with features and lithology labels"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def reconstruction_loss(x_recon, x_true, mask=None):
    """
    MSE reconstruction loss

    If mask provided: only compute loss on masked features
    (tests if model can impute missing values)
    """
    if mask is None:
        return torch.mean((x_recon - x_true) ** 2)
    else:
        # Loss only on masked features (0 in mask)
        masked_indices = (mask == 0)
        if masked_indices.sum() > 0:
            return torch.mean((x_recon[masked_indices] - x_true[masked_indices]) ** 2)
        else:
            # Fallback to full reconstruction if no masking
            return torch.mean((x_recon - x_true) ** 2)


def kl_divergence(mu, logvar):
    """KL divergence between latent distribution and N(0,1)"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, dataloader, optimizer, device, beta, alpha):
    """Train for one epoch with 3-part loss + masking"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_class = 0
    correct = 0
    total = 0
    total_mask_pct = 0
    n_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        # Forward pass (with random masking)
        x_recon, mu, logvar, logits, mask = model(x_batch)

        # 3-part loss
        recon = reconstruction_loss(x_recon, x_batch, mask)  # Loss on masked features
        kl = kl_divergence(mu, logvar)
        classification = nn.CrossEntropyLoss()(logits, y_batch)

        loss = recon + beta * kl + alpha * classification

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_class += classification.item()

        # Track masking percentage
        total_mask_pct += (1 - mask.mean()).item()
        n_batches += 1

        # Classification accuracy
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y_batch).sum().item()
        total += len(y_batch)

    n = len(dataloader)
    avg_mask_pct = total_mask_pct / n_batches

    return {
        'loss': total_loss / n,
        'recon': total_recon / n,
        'kl': total_kl / n,
        'class': total_class / n,
        'acc': 100.0 * correct / total,
        'mask_pct': 100.0 * avg_mask_pct
    }


def eval_epoch(model, dataloader, device):
    """Evaluate without masking"""
    model.eval()
    total_recon = 0
    total_kl = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass (no masking in eval mode)
            x_recon, mu, logvar, logits, _ = model(x_batch)

            # Metrics
            recon = reconstruction_loss(x_recon, x_batch)
            kl = kl_divergence(mu, logvar)

            total_recon += recon.item()
            total_kl += kl.item()

            # Classification accuracy
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)

    n = len(dataloader)
    return {
        'recon': total_recon / n,
        'kl': total_kl / n,
        'acc': 100.0 * correct / total
    }


def compute_ari(model, dataloader, device, n_clusters=18):
    """Compute Adjusted Rand Index using GMM on latent space"""
    model.eval()

    all_z = []
    all_labels = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            mu, _ = model.encode(x_batch)
            all_z.append(mu.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    z = np.concatenate(all_z, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Fit GMM on latent space
    gmm = GaussianMixture(n_components=n_clusters, random_state=42,
                         max_iter=100, n_init=3)
    clusters = gmm.fit_predict(z)

    # Compute ARI
    ari = adjusted_rand_score(labels, clusters)
    return ari


print("=" * 80)
print("VAE v2.14.2 TRAINING (Semi-Supervised + Random Masking)")
print("=" * 80)

# Load data
print("\nLoading dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')
print(f"Total samples: {len(df):,}")
print(f"Unique boreholes: {df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {df['Principal'].nunique()}")

# Prepare features and labels
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
y_raw = df['Principal'].values

# Encode lithology labels
le = LabelEncoder()
y = le.fit_transform(y_raw)
print(f"Encoded lithologies: {len(le.classes_)} classes")

# Scale features
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X)

# Train/val split (80/20) - no stratification due to rare classes
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"\nTrain samples: {len(X_train):,}")
print(f"Val samples: {len(X_val):,}")

# Create data loaders
train_dataset = LithologyDataset(X_train, y_train)
val_dataset = LithologyDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = SemiSupervisedVAE_Masked(
    input_dim=6,
    latent_dim=10,
    n_classes=len(le.classes_),
    encoder_dims=[32, 16],
    classifier_hidden=32,
    mask_prob=0.3  # 30% masking probability per feature
).to(device)

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Training settings
epochs = 100
beta_start = 1e-10
beta_end = 0.75
alpha = 0.1  # Classification weight

print(f"\nTraining settings:")
print(f"  Epochs: {epochs}")
print(f"  β annealing: {beta_start} → {beta_end}")
print(f"  α (classification): {alpha}")
print(f"  Mask probability: 30% per feature")
print(f"  Batch size: 256")
print(f"  Learning rate: 1e-3")

optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_ari = -1
best_epoch = 0

for epoch in range(1, epochs + 1):
    start_time = time.time()

    # Linear β annealing
    beta = beta_start + (beta_end - beta_start) * (epoch / epochs)

    # Train
    train_metrics = train_epoch(model, train_loader, optimizer, device, beta, alpha)

    # Validate
    val_metrics = eval_epoch(model, val_loader, device)

    # Compute ARI every 2 epochs
    ari = None
    if epoch % 2 == 0:
        ari = compute_ari(model, val_loader, device)
        if ari > best_ari:
            best_ari = ari
            best_epoch = epoch
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ari': ari,
                'scaler': scaler,
                'label_encoder': le
            }, 'ml_models/checkpoints/vae_v2_14_2_best.pth')

    epoch_time = time.time() - start_time

    # Print progress
    ari_str = f" | ARI={ari:.4f}" if ari is not None else ""
    print(f"Epoch {epoch:3d}/{epochs} | β={beta:.2e} | "
          f"Loss={train_metrics['loss']:.4f} "
          f"(Recon={train_metrics['recon']:.4f}, KL={train_metrics['kl']:.4f}, "
          f"Class={train_metrics['class']:.4f}) | "
          f"Mask={train_metrics['mask_pct']:.1f}% | {epoch_time:.1f}s{ari_str}")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"\nBest validation ARI: {best_ari:.4f} (epoch {best_epoch})")

# Save final model
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler': scaler,
    'label_encoder': le
}, 'ml_models/checkpoints/vae_v2_14_2_final.pth')

print("\n✓ Models saved:")
print("  - ml_models/checkpoints/vae_v2_14_2_best.pth")
print("  - ml_models/checkpoints/vae_v2_14_2_final.pth")

# Final clustering evaluation on full dataset
print("\n" + "=" * 80)
print("FINAL CLUSTERING EVALUATION")
print("=" * 80)

full_dataset = LithologyDataset(X_scaled, y)
full_loader = DataLoader(full_dataset, batch_size=256, shuffle=False)

train_ari = compute_ari(model, train_loader, device)
val_ari = compute_ari(model, val_loader, device)
full_ari = compute_ari(model, full_loader, device)

print(f"\nClustering Performance (GMM k=18):")
print(f"  Train ARI: {train_ari:.4f}")
print(f"  Val ARI: {val_ari:.4f}")
print(f"  Full dataset ARI: {full_ari:.4f}")
