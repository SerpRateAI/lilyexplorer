#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Semi-Supervised VAE with β AND α Annealing

Double annealing strategy:
- β: 1e-10 → 0.75 (KL regularization)
- α: 0 → α_end (classification guidance)

Tests multiple α_end values to find optimal classification weight.
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


class SemiSupervisedVAE(nn.Module):
    """Semi-Supervised VAE with Classification Head"""

    def __init__(self, input_dim=6, latent_dim=10, n_classes=139,
                 encoder_dims=[32, 16], classifier_hidden=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

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
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        logits = self.classify(mu)
        return x_recon, mu, logvar, logits


class LithologyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def reconstruction_loss(x_recon, x_true):
    return torch.mean((x_recon - x_true) ** 2)


def kl_divergence(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def train_epoch(model, dataloader, optimizer, device, beta, alpha):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_class = 0
    correct = 0
    total = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        x_recon, mu, logvar, logits = model(x_batch)

        recon = reconstruction_loss(x_recon, x_batch)
        kl = kl_divergence(mu, logvar)
        classification = nn.CrossEntropyLoss()(logits, y_batch)

        loss = recon + beta * kl + alpha * classification

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_class += classification.item()

        _, predicted = torch.max(logits, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

    n = len(dataloader)
    return (total_loss/n, total_recon/n, total_kl/n, total_class/n,
            100.0 * correct / total)


def validate(model, dataloader, device, beta, alpha):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_class = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            x_recon, mu, logvar, logits = model(x_batch)

            recon = reconstruction_loss(x_recon, x_batch)
            kl = kl_divergence(mu, logvar)
            classification = nn.CrossEntropyLoss()(logits, y_batch)

            loss = recon + beta * kl + alpha * classification

            total_loss += loss.item()
            total_recon += recon.item()
            total_kl += kl.item()
            total_class += classification.item()

            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    n = len(dataloader)
    return (total_loss/n, total_recon/n, total_kl/n, total_class/n,
            100.0 * correct / total)


def generate_latent_representations(model, X, device):
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        z = mu.cpu().numpy()

    return z


def evaluate_clustering(model, X, y_true, device, n_components=18):
    z = generate_latent_representations(model, X, device)

    z_std = np.std(z, axis=0)
    active_dims = np.sum(z_std > 0.01)

    z_scaled = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-8)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='diag',
        random_state=42,
        n_init=10,
        max_iter=200,
        reg_covar=1e-3
    )

    cluster_labels = gmm.fit_predict(z_scaled)
    ari = adjusted_rand_score(y_true, cluster_labels)

    return ari, active_dims, z_std


def load_data():
    print("Loading data...")
    df = pd.read_csv('vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                   'NGR total counts (cps)', 'R', 'G', 'B']
    X = df[feature_cols].values
    lithologies = df['Principal'].values

    le = LabelEncoder()
    y = le.fit_transform(lithologies)
    n_classes = len(le.classes_)

    print(f"  Samples: {len(X):,}")
    print(f"  Classes: {n_classes}")

    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Split by borehole (80/10/10)
    boreholes = df['Borehole_ID'].values
    unique_boreholes = np.unique(boreholes)
    np.random.shuffle(unique_boreholes)

    n_train = int(0.8 * len(unique_boreholes))
    n_val = int(0.1 * len(unique_boreholes))

    train_boreholes = unique_boreholes[:n_train]
    val_boreholes = unique_boreholes[n_train:n_train+n_val]
    test_boreholes = unique_boreholes[n_train+n_val:]

    train_mask = np.isin(boreholes, train_boreholes)
    val_mask = np.isin(boreholes, val_boreholes)
    test_mask = np.isin(boreholes, test_boreholes)

    X_train, y_train = X_scaled[train_mask], y[train_mask]
    X_val, y_val = X_scaled[val_mask], y[val_mask]
    X_test, y_test = X_scaled[test_mask], y[test_mask]
    lith_test = lithologies[test_mask]

    print(f"  Train: {len(X_train):,} samples, {len(train_boreholes)} boreholes")
    print(f"  Val:   {len(X_val):,} samples, {len(val_boreholes)} boreholes")
    print(f"  Test:  {len(X_test):,} samples, {len(test_boreholes)} boreholes")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            lith_test, n_classes, le)


def train_model(alpha_end, beta_start=1e-10, beta_end=0.75, n_epochs=100,
                batch_size=1024, lr=1e-3, alpha_anneal_epochs=50):
    """Train with β AND α annealing"""

    print(f"\n{'='*80}")
    print(f"TRAINING: α_end={alpha_end} (0→{alpha_end} annealing over {alpha_anneal_epochs} epochs)")
    print(f"{'='*80}")

    (X_train, y_train, X_val, y_val, X_test, y_test,
     lith_test, n_classes, le) = load_data()

    train_dataset = LithologyDataset(X_train, y_train)
    val_dataset = LithologyDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SemiSupervisedVAE(input_dim=6, latent_dim=10, n_classes=n_classes,
                              encoder_dims=[32, 16], classifier_hidden=32)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Device: {device}")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining with DOUBLE annealing:")
    print(f"  β: {beta_start}→{beta_end} (50 epochs)")
    print(f"  α: 0→{alpha_end} ({alpha_anneal_epochs} epochs)")
    start_time = time.time()

    best_val_loss = float('inf')
    best_epoch = 0

    for epoch in range(1, n_epochs + 1):
        # Anneal β (first 50 epochs)
        if epoch <= 50:
            beta = beta_start + (beta_end - beta_start) * (epoch / 50)
        else:
            beta = beta_end

        # Anneal α (configurable epochs)
        if epoch <= alpha_anneal_epochs:
            alpha = alpha_end * (epoch / alpha_anneal_epochs)
        else:
            alpha = alpha_end

        # Train
        train_loss, train_recon, train_kl, train_class, train_acc = \
            train_epoch(model, train_loader, optimizer, device, beta, alpha)

        # Validate
        val_loss, val_recon, val_kl, val_class, val_acc = \
            validate(model, val_loader, device, beta, alpha)

        # Track best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'alpha_end': alpha_end,
                'beta_end': beta_end,
                'n_classes': n_classes
            }, f'ml_models/checkpoints/semisup_vae_anneal_alpha{alpha_end}.pth')

        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Loss={train_loss:.4f}, "
                  f"Recon={train_recon:.4f}, KL={train_kl:.4f}, "
                  f"Class={train_class:.4f}, "
                  f"TrainAcc={train_acc:.1f}%, ValAcc={val_acc:.1f}%, "
                  f"β={beta:.6f}, α={alpha:.6f}")

    training_time = time.time() - start_time
    print(f"\nTraining time: {training_time:.1f}s")
    print(f"Best model: epoch {best_epoch}")

    # Load best model for evaluation
    checkpoint = torch.load(f'ml_models/checkpoints/semisup_vae_anneal_alpha{alpha_end}.pth',
                           weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate classification accuracy
    test_dataset = LithologyDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            _, mu, _, logits = model(x_batch)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    test_acc = 100.0 * correct / total
    print(f"\nTest Classification Accuracy: {test_acc:.2f}%")

    # Evaluate clustering
    print("\nEvaluating unsupervised clustering in latent space...")
    ari, active_dims, z_std = evaluate_clustering(model, X_test, lith_test,
                                                   device, n_components=18)

    print(f"  Active dimensions (std > 0.01): {active_dims}/10")
    print(f"  Dimension stds: {z_std}")
    print(f"  Adjusted Rand Index (GMM): {ari:.4f}")

    return {
        'alpha_end': alpha_end,
        'test_acc': test_acc,
        'ari': ari,
        'active_dims': active_dims,
        'training_time': training_time,
        'best_epoch': best_epoch
    }


if __name__ == "__main__":
    print("="*80)
    print("SEMI-SUPERVISED VAE WITH DOUBLE ANNEALING (β AND α)")
    print("="*80)
    print("\nStrategy:")
    print("  1. β: 1e-10→0.75 (KL regularization, first 50 epochs)")
    print("  2. α: 0→α_end (classification guidance, first 50 epochs)")
    print("\nGoal: Gradually introduce both regularization and supervision")
    print("="*80)

    # Test multiple α_end values
    alpha_ends = [0.01, 0.05, 0.1, 0.5, 1.0]
    results = []

    for alpha_end in alpha_ends:
        try:
            result = train_model(alpha_end, alpha_anneal_epochs=50)
            results.append(result)
        except Exception as e:
            print(f"\n⚠️  Training failed for α_end={alpha_end}: {e}")
            continue

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'α_end':>8s} {'Test Acc':>10s} {'GMM ARI':>10s} {'Active Dims':>12s} {'Time (s)':>10s}")
    print("-"*80)

    for r in results:
        print(f"{r['alpha_end']:8.2f} {r['test_acc']:9.2f}% {r['ari']:10.4f} "
              f"{r['active_dims']:12d}/10 {r['training_time']:10.1f}")

    # Find best α_end by ARI
    if results:
        best = max(results, key=lambda x: x['ari'])
        print("-"*80)
        print(f"Best α_end: {best['alpha_end']} (ARI={best['ari']:.4f}, Acc={best['test_acc']:.2f}%)")
        print("="*80)

        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv('semisup_vae_anneal_results.csv', index=False)
        print("\n✓ Results saved to 'semisup_vae_anneal_results.csv'")
