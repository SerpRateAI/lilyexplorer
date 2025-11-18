#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Semi-Supervised VAE Checkpoints

Load saved checkpoint files and compute ARI scores for clustering evaluation.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from pathlib import Path

np.random.seed(42)
torch.manual_seed(42)


class DistributionAwareScaler:
    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]
        self.log_indices = [3, 4, 5]

    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit_transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        X_scaled = self.scaler.fit_transform(X_transformed)
        return X_scaled

    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled


class SemiSupervisedVAE(nn.Module):
    def __init__(self, input_dim=6, latent_dim=10, n_classes=139,
                 encoder_dims=[32, 16], classifier_hidden=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


def load_data():
    df = pd.read_csv('vae_training_data_v2_20cm.csv')
    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                   'NGR total counts (cps)', 'R', 'G', 'B']
    X = df[feature_cols].values
    lithologies = df['Principal'].values

    le = LabelEncoder()
    y = le.fit_transform(lithologies)

    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    boreholes = df['Borehole_ID'].values
    unique_boreholes = np.unique(boreholes)
    np.random.shuffle(unique_boreholes)

    n_train = int(0.8 * len(unique_boreholes))
    n_val = int(0.1 * len(unique_boreholes))

    test_boreholes = unique_boreholes[n_train+n_val:]
    test_mask = np.isin(boreholes, test_boreholes)

    X_test = X_scaled[test_mask]
    lith_test = lithologies[test_mask]

    return X_test, lith_test, len(le.classes_)


def evaluate_checkpoint(checkpoint_path, X_test, lith_test, n_classes, device):
    """Evaluate a single checkpoint"""
    model = SemiSupervisedVAE(input_dim=6, latent_dim=10, n_classes=n_classes,
                              encoder_dims=[32, 16], classifier_hidden=32)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(device)

    # Generate latent representations
    X_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        mu, _ = model.encode(X_tensor)
        z = mu.cpu().numpy()

    # Check dimension collapse
    z_std = np.std(z, axis=0)
    active_dims = np.sum(z_std > 0.01)

    # Standardize and cluster
    z_scaled = (z - np.mean(z, axis=0)) / (np.std(z, axis=0) + 1e-8)

    gmm = GaussianMixture(
        n_components=18,
        covariance_type='diag',
        random_state=42,
        n_init=10,
        max_iter=200,
        reg_covar=1e-3
    )

    cluster_labels = gmm.fit_predict(z_scaled)
    ari = adjusted_rand_score(lith_test, cluster_labels)

    return ari, active_dims, z_std


if __name__ == "__main__":
    print("="*80)
    print("EVALUATING SEMI-SUPERVISED VAE CHECKPOINTS")
    print("="*80)

    # Load test data
    print("\nLoading test data...")
    X_test, lith_test, n_classes = load_data()
    print(f"Test samples: {len(X_test):,}")
    print(f"Classes: {n_classes}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Find all checkpoint files
    checkpoint_dir = Path('ml_models/checkpoints')
    alphas = [0.01, 0.1, 0.5, 1.0, 2.0]

    results = []
    print("\n" + "="*80)
    print("EVALUATING MODELS")
    print("="*80)

    for alpha in alphas:
        checkpoint_path = checkpoint_dir / f'semisup_vae_alpha{alpha}.pth'

        if not checkpoint_path.exists():
            print(f"\n⚠️  Checkpoint not found: {checkpoint_path}")
            continue

        print(f"\nα = {alpha}")
        try:
            ari, active_dims, z_std = evaluate_checkpoint(
                checkpoint_path, X_test, lith_test, n_classes, device
            )

            print(f"  Active dimensions: {active_dims}/10")
            print(f"  Dimension stds: {z_std}")
            print(f"  ✓ Adjusted Rand Index: {ari:.4f}")

            results.append({
                'alpha': alpha,
                'ari': ari,
                'active_dims': active_dims
            })
        except Exception as e:
            print(f"  ⚠️  Evaluation failed: {e}")

    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Alpha':>8s} {'GMM ARI':>10s} {'Active Dims':>12s}")
    print("-"*80)

    for r in results:
        print(f"{r['alpha']:8.2f} {r['ari']:10.4f} {r['active_dims']:12d}/10")

    if results:
        best = max(results, key=lambda x: x['ari'])
        print("-"*80)
        print(f"Best α: {best['alpha']} (ARI={best['ari']:.4f})")
        print("="*80)

        # Compare to v2.6.7
        v267_ari = 0.196
        improvement = (best['ari'] - v267_ari) / v267_ari * 100
        print(f"\nComparison to v2.6.7:")
        print(f"  v2.6.7 (unsupervised):        ARI = {v267_ari:.4f}")
        print(f"  Semi-supervised (best α):     ARI = {best['ari']:.4f}")
        print(f"  Change: {improvement:+.1f}%")

        # Save results
        df_results = pd.DataFrame(results)
        df_results.to_csv('semisup_vae_evaluation.csv', index=False)
        print("\n✓ Results saved to 'semisup_vae_evaluation.csv'")
