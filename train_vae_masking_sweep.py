#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE Masking Hyperparameter Sweep
Train single model with specified mask_prob
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

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
    """Semi-Supervised VAE with Random Feature Masking"""

    def __init__(self, input_dim=6, latent_dim=10, n_classes=209,
                 encoder_dims=[32, 16], classifier_hidden=32, mask_prob=0.3):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.mask_prob = mask_prob

        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def apply_mask(self, x):
        """Randomly mask features with probability mask_prob"""
        if not self.training or self.mask_prob == 0:
            return x, torch.ones_like(x)

        # Create random mask (1=keep, 0=mask)
        mask = (torch.rand_like(x) > self.mask_prob).float()

        # Ensure at least one feature is present per sample
        zero_mask_samples = (mask.sum(dim=1) == 0)
        if zero_mask_samples.any():
            for idx in torch.where(zero_mask_samples)[0]:
                random_feature = torch.randint(0, self.input_dim, (1,))
                mask[idx, random_feature] = 1.0

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
        return self.classifier(z)

    def forward(self, x):
        x_masked, mask = self.apply_mask(x)
        mu, logvar = self.encode(x_masked)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
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
    """MSE reconstruction loss"""
    if mask is None:
        return torch.mean((x_recon - x_true) ** 2)
    else:
        masked_indices = (mask == 0)
        if masked_indices.sum() > 0:
            return torch.mean((x_recon[masked_indices] - x_true[masked_indices]) ** 2)
        else:
            return torch.mean((x_recon - x_true) ** 2)


def kl_divergence(mu, logvar):
    """KL divergence between latent distribution and N(0,1)"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, dataloader, optimizer, device, beta, alpha):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    n = len(dataloader)

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        x_recon, mu, logvar, logits, mask = model(x_batch)

        recon = reconstruction_loss(x_recon, x_batch, mask)
        kl = kl_divergence(mu, logvar)
        classification = nn.CrossEntropyLoss()(logits, y_batch)

        loss = recon + beta * kl + alpha * classification

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / n


def evaluate_reconstruction(model, dataloader, device):
    """Evaluate reconstruction R² scores per feature"""
    model.eval()

    all_true = []
    all_recon = []

    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            x_recon, mu, logvar, logits, _ = model(x_batch)
            all_true.append(x_batch.cpu().numpy())
            all_recon.append(x_recon.cpu().numpy())

    X_true = np.concatenate(all_true, axis=0)
    X_recon = np.concatenate(all_recon, axis=0)

    # Compute R² per feature
    r2_scores = []
    for i in range(X_true.shape[1]):
        r2 = r2_score(X_true[:, i], X_recon[:, i])
        r2_scores.append(r2)

    return r2_scores


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train_vae_masking_sweep.py <mask_prob>")
        sys.exit(1)

    mask_prob = float(sys.argv[1])
    print(f"Training with mask_prob={mask_prob:.2f}")

    # Load data
    df = pd.read_csv('vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                    'NGR total counts (cps)', 'R', 'G', 'B']
    X = df[feature_cols].values
    y_raw = df['Principal'].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Scale features
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Create data loaders
    train_dataset = LithologyDataset(X_train, y_train)
    val_dataset = LithologyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SemiSupervisedVAE_Masked(
        input_dim=6,
        latent_dim=10,
        n_classes=len(le.classes_),
        encoder_dims=[32, 16],
        classifier_hidden=32,
        mask_prob=mask_prob
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training settings
    epochs = 50  # Shorter training for sweep
    beta_start = 1e-10
    beta_end = 0.75
    alpha = 0.1

    # Training loop (minimal output)
    for epoch in range(1, epochs + 1):
        beta = beta_start + (beta_end - beta_start) * (epoch / epochs)
        loss = train_epoch(model, train_loader, optimizer, device, beta, alpha)

    # Final evaluation
    r2_scores = evaluate_reconstruction(model, val_loader, device)

    # Save results
    results = {
        'mask_prob': mask_prob,
        'r2_gra': r2_scores[0],
        'r2_ms': r2_scores[1],
        'r2_ngr': r2_scores[2],
        'r2_r': r2_scores[3],
        'r2_g': r2_scores[4],
        'r2_b': r2_scores[5]
    }

    # Save checkpoint
    torch.save({
        'mask_prob': mask_prob,
        'model_state_dict': model.state_dict(),
        'r2_scores': r2_scores,
        'scaler': scaler,
        'label_encoder': le
    }, f'ml_models/checkpoints/masking_sweep/mask_{int(mask_prob*100):03d}.pth')

    # Append to CSV
    import os
    csv_file = 'masking_sweep_results.csv'
    if not os.path.exists(csv_file):
        pd.DataFrame([results]).to_csv(csv_file, index=False)
    else:
        pd.DataFrame([results]).to_csv(csv_file, mode='a', header=False, index=False)

    print(f"COMPLETE: mask_prob={mask_prob:.2f}, R²=[{', '.join([f'{r:.4f}' for r in r2_scores])}]")
