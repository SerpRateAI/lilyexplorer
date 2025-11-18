#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Beta Grid Search for Multi-Decoder VAE v2.13

Tests β_end ∈ {0.5, 0.75, 1.0, 1.5, 2.0} with 5-fold entropy-balanced CV.

Fixed parameters:
- Architecture: Multi-decoder ([32,16] encoder → 10D → [16,32] decoders × 6)
- Feature weights: [1.0, 2.0, 2.0, 1.0, 1.0, 1.0]
- β_start: 1e-10
- anneal_epochs: 50
- max_epochs: 100

Variable:
- β_end: {0.5, 0.75, 1.0, 1.5, 2.0}
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
from pathlib import Path

from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class MultiDecoderVAE(nn.Module):
    """v2.13 Multi-decoder VAE"""

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

        # Separate decoders
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

        recon_loss = 0
        for i in range(6):
            loss_i = F.mse_loss(recon_x[:, i], x[:, i], reduction='sum') / batch_size
            recon_loss += feature_weights[i] * loss_i

        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_fold(X_train, X_val, y_train, y_val, device, beta_end, fold_id):
    """Train one fold with specific β_end"""

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

    beta_start = 1e-10
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
            break

    # Get embeddings and cluster
    model.eval()
    with torch.no_grad():
        z_list = []
        for batch_x, _ in val_loader:
            batch_x = batch_x.to(device)
            mu, _ = model.encode(batch_x)
            z_list.append(mu.cpu().numpy())
    z = np.vstack(z_list)

    gmm = GaussianMixture(n_components=15, random_state=42)
    cluster_labels = gmm.fit_predict(z)

    # ARI
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

    return ari

def main():
    print("="*80)
    print("PHASE 1: Beta Grid Search for Multi-Decoder VAE v2.13")
    print("="*80)
    print("Testing β_end ∈ {0.5, 0.75, 1.0, 1.5, 2.0}")
    print("5-fold entropy-balanced CV per β_end")
    print()

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

    # Entropy-balanced folds
    unique_boreholes = np.unique(borehole_ids)
    np.random.seed(42)

    borehole_entropies = []
    for bh in unique_boreholes:
        mask = borehole_ids == bh
        lith_counts = pd.Series(lithology[mask]).value_counts()
        probs = lith_counts / lith_counts.sum()
        ent = entropy(probs)
        borehole_entropies.append(ent)

    borehole_entropies = np.array(borehole_entropies)
    sorted_indices = np.argsort(borehole_entropies)
    sorted_boreholes = unique_boreholes[sorted_indices]

    n_folds = 5
    folds = [[] for _ in range(n_folds)]
    for i, bh in enumerate(sorted_boreholes):
        folds[i % n_folds].append(bh)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Test different β_end values
    beta_values = [0.5, 0.75, 1.0, 1.5, 2.0]
    all_results = []

    for beta_end in beta_values:
        print("\n" + "="*80)
        print(f"β_end = {beta_end}")
        print("="*80)

        beta_results = []

        for fold_id in range(n_folds):
            print(f"\nFold {fold_id + 1}/{n_folds}")

            test_bh = np.array(folds[fold_id])
            train_bh = np.concatenate([folds[j] for j in range(n_folds) if j != fold_id])

            train_mask = np.isin(borehole_ids, train_bh)
            test_mask = np.isin(borehole_ids, test_bh)

            X_train = X_scaled[train_mask]
            X_test = X_scaled[test_mask]
            y_train = lithology[train_mask]
            y_test = lithology[test_mask]

            print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

            start = time.time()
            ari = train_fold(X_train, X_test, y_train, y_test, device, beta_end, fold_id + 1)
            elapsed = time.time() - start

            print(f"  ARI: {ari:.4f} (time: {elapsed:.1f}s)")

            beta_results.append({
                'beta_end': beta_end,
                'fold': fold_id + 1,
                'ari': ari
            })

        # Summary for this β
        fold_aris = [r['ari'] for r in beta_results]
        mean_ari = np.mean(fold_aris)
        std_ari = np.std(fold_aris)

        print(f"\nβ_end={beta_end}: Mean ARI = {mean_ari:.4f} ± {std_ari:.4f}")

        all_results.extend(beta_results)

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('v2_13_beta_grid_search.csv', index=False)

    # Summary
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE - SUMMARY")
    print("="*80)

    summary = results_df.groupby('beta_end')['ari'].agg(['mean', 'std', 'min', 'max'])
    print("\nβ_end    Mean ARI   Std      Min      Max")
    print("-" * 60)
    for beta_end, row in summary.iterrows():
        print(f"{beta_end:5.2f}    {row['mean']:.4f}    {row['std']:.4f}   {row['min']:.4f}   {row['max']:.4f}")

    best_beta = summary['mean'].idxmax()
    best_ari = summary.loc[best_beta, 'mean']
    best_std = summary.loc[best_beta, 'std']

    print(f"\n**Best β_end: {best_beta} (ARI = {best_ari:.4f} ± {best_std:.4f})**")
    print(f"\nBaseline (β_end=0.75): ARI = {summary.loc[0.75, 'mean']:.4f} ± {summary.loc[0.75, 'std']:.4f}")

    if best_beta != 0.75:
        improvement = ((best_ari - summary.loc[0.75, 'mean']) / summary.loc[0.75, 'mean']) * 100
        print(f"Improvement: {improvement:+.1f}%")

    print(f"\nResults saved to: v2_13_beta_grid_search.csv")
    print("="*80)

if __name__ == "__main__":
    main()
