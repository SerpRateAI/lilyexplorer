#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VAE v2.13 - Multi-Decoder Architecture with Feature Weighting

Cross-validation with entropy-balanced folds.

Architecture changes from v2.6.7:
- Separate decoder for each of 6 features (vs single shared decoder)
- 2x weight on MS and NGR (poorly-reconstructed features)
- Same encoder and latent space (10D, [32,16] hidden)

Expected performance vs v2.6.7:
- ARI: ~0.21 (vs 0.196 ± 0.037)
- GRA R²: ~0.90 (vs 0.83)
- MS R²: ~0.84 (vs 0.44) - major improvement
- NGR R²: ~0.89 (vs 0.74)
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
import time

from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class MultiDecoderVAE(nn.Module):
    """VAE with separate decoder for each feature"""

    def __init__(self, input_dim=6, latent_dim=10, encoder_dims=[32, 16],
                 decoder_dims=[16, 32]):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_names = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']

        # Shared encoder
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Separate decoder per feature
        self.decoders = nn.ModuleDict()
        for name in self.feature_names:
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in decoder_dims:
                decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(decoder_dims[-1], 1))
            self.decoders[name] = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        outputs = []
        for name in self.feature_names:
            outputs.append(self.decoders[name](z))
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0, feature_weights=None):
        batch_size = x.size(0)

        if feature_weights is None:
            feature_weights = torch.ones(6, device=x.device)

        # Per-feature reconstruction loss
        recon_loss = 0
        for i in range(6):
            loss_i = F.mse_loss(recon_x[:, i], x[:, i], reduction='sum') / batch_size
            recon_loss += feature_weights[i] * loss_i

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_fold(X_train, X_val, y_train, y_val, device, fold_id):
    """Train one CV fold"""

    # Feature weights: 2x for MS and NGR
    feature_weights = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 1.0]).to(device)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.zeros(len(X_train))),
        batch_size=256, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.zeros(len(X_val))),
        batch_size=256, shuffle=False
    )

    model = MultiDecoderVAE(
        input_dim=6, latent_dim=10,
        encoder_dims=[32, 16], decoder_dims=[16, 32]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"  Training fold {fold_id}...")

    # β annealing: 1e-10 → 0.75
    beta_start = 1e-10
    beta_end = 0.75
    anneal_epochs = 50
    max_epochs = 100
    patience = 15

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        if epoch < anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
        else:
            beta = beta_end

        # Train
        model.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta, feature_weights)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon_x, mu, logvar = model(batch_x)
                loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta, feature_weights)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

    # Get embeddings
    model.eval()
    with torch.no_grad():
        z_list = []
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            mu, _ = model.encode(batch_x)
            z_list.append(mu.cpu().numpy())
    z = np.vstack(z_list)

    # GMM clustering
    gmm = GaussianMixture(n_components=15, random_state=42)
    cluster_labels = gmm.fit_predict(z)

    # ARI on common lithologies
    common_lith = pd.Series(y_val).value_counts()
    common_lith = common_lith[common_lith >= 100].index
    mask = np.isin(y_val, common_lith)

    if mask.sum() > 0:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        labels_encoded = le.fit_transform(y_val[mask])
        ari = adjusted_rand_score(labels_encoded, cluster_labels[mask])
    else:
        ari = 0.0

    return ari, model

def main():
    print("="*80)
    print("VAE v2.13 - Multi-Decoder with Feature Weighting")
    print("5-Fold Entropy-Balanced Cross-Validation")
    print("="*80)

    # Load data
    df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                    'NGR total counts (cps)', 'R', 'G', 'B']

    X = df[feature_cols].values
    lithology = df['Principal'].values
    borehole_ids = df['Borehole_ID'].values

    print(f"Dataset: {len(X):,} samples, {df['Borehole_ID'].nunique()} boreholes")

    # Scale
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Entropy-balanced 5-fold CV
    unique_boreholes = np.unique(borehole_ids)
    np.random.seed(42)

    # Calculate entropy per borehole
    borehole_entropies = []
    for bh in unique_boreholes:
        mask = borehole_ids == bh
        lith_counts = pd.Series(lithology[mask]).value_counts()
        probs = lith_counts / lith_counts.sum()
        ent = entropy(probs)
        borehole_entropies.append(ent)

    borehole_entropies = np.array(borehole_entropies)

    # Sort boreholes by entropy
    sorted_indices = np.argsort(borehole_entropies)
    sorted_boreholes = unique_boreholes[sorted_indices]

    # Split into 5 folds (stratified by entropy)
    n_folds = 5
    folds = [[] for _ in range(n_folds)]
    for i, bh in enumerate(sorted_boreholes):
        folds[i % n_folds].append(bh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Run CV
    results = []

    for fold_id in range(n_folds):
        print(f"\nFold {fold_id + 1}/{n_folds}")
        print("-" * 40)

        # Split
        test_bh = np.array(folds[fold_id])
        train_bh = np.concatenate([folds[j] for j in range(n_folds) if j != fold_id])

        train_mask = np.isin(borehole_ids, train_bh)
        test_mask = np.isin(borehole_ids, test_bh)

        X_train = X_scaled[train_mask]
        X_test = X_scaled[test_mask]
        y_train = lithology[train_mask]
        y_test = lithology[test_mask]

        # Calculate fold entropy
        lith_counts = pd.Series(y_test).value_counts()
        probs = lith_counts / lith_counts.sum()
        fold_entropy = entropy(probs)

        print(f"  Train: {len(X_train):,} samples, {len(train_bh)} boreholes")
        print(f"  Test:  {len(X_test):,} samples, {len(test_bh)} boreholes")
        print(f"  Test entropy: {fold_entropy:.4f}")

        # Train
        ari, model = train_fold(X_train, X_test, y_train, y_test, device, fold_id + 1)

        print(f"  Fold {fold_id + 1} ARI: {ari:.4f}")

        results.append({
            'fold': fold_id + 1,
            'ari': ari,
            'entropy': fold_entropy,
            'n_train': len(X_train),
            'n_test': len(X_test)
        })

    # Summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('v2_13_entropy_balanced_cv.csv', index=False)

    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))

    mean_ari = results_df['ari'].mean()
    std_ari = results_df['ari'].std()

    print(f"\nMean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
    print(f"Min ARI:  {results_df['ari'].min():.4f}")
    print(f"Max ARI:  {results_df['ari'].max():.4f}")

    print("\n" + "="*80)
    print("CROSS-VALIDATION COMPLETE")
    print("="*80)
    print(f"Results saved to: v2_13_entropy_balanced_cv.csv")

if __name__ == "__main__":
    main()
