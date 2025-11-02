"""
Cross-validate v2.6.6 to measure ARI variance across different train/test splits.

Uses borehole-level k-fold CV to avoid data leakage.
Reports mean ARI ± std to understand if β schedule improvements are significant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class VAE(nn.Module):
    """v2.6.6 architecture: 10D latent, β annealing 0.001→0.5"""
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
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

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl_div, recon_loss, kl_div

def train_vae(model, train_loader, val_loader, epochs, device, beta_start=0.001, beta_end=0.5, anneal_epochs=50):
    """Train VAE with β annealing schedule"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # β annealing
        if epoch < anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
        else:
            beta = beta_end

        # Training
        model.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon_x, mu, logvar = model(batch_x)
                loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model

def evaluate_clustering(model, X_test, y_test, device):
    """Evaluate GMM clustering on latent space"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    # Latent space statistics
    latent_stds = latent.std(axis=0)
    collapsed_dims = (latent_stds < 0.1).sum()
    effective_dim = (latent_stds >= 0.1).sum()

    # GMM clustering for k∈{10,12,15,18}
    results = []
    for k in [10, 12, 15, 18]:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        clusters = gmm.fit_predict(latent)

        ari = adjusted_rand_score(y_test, clusters)
        sil = silhouette_score(latent, clusters)

        results.append({
            'k': k,
            'ari': ari,
            'sil': sil
        })

    # Find best k
    df_results = pd.DataFrame(results)
    best_idx = df_results['ari'].idxmax()
    best = df_results.iloc[best_idx]

    return {
        'collapsed_dims': collapsed_dims,
        'effective_dim': effective_dim,
        'best_k': best['k'],
        'best_ari': best['ari'],
        'best_sil': best['sil'],
        'all_results': results
    }

# ============================================================================
# CROSS-VALIDATION
# ============================================================================

print("="*100)
print("CROSS-VALIDATION: v2.6.6 (β: 0.001→0.5 over 50 epochs)")
print("="*100)
print()

# Load data
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

# Borehole-level k-fold CV
unique_boreholes = df['Borehole_ID'].unique()
n_folds = 5

print(f"Total boreholes: {len(unique_boreholes)}")
print(f"K-fold CV: {n_folds} folds")
print(f"Each fold: ~{len(unique_boreholes)//n_folds} boreholes in test set")
print()

kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

cv_results = []

for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(unique_boreholes)):
    print("="*100)
    print(f"FOLD {fold_idx + 1}/{n_folds}")
    print("="*100)

    # Split boreholes
    train_boreholes = unique_boreholes[train_idx]
    test_boreholes = unique_boreholes[test_idx]

    # Split data
    train_mask = df['Borehole_ID'].isin(train_boreholes)
    test_mask = df['Borehole_ID'].isin(test_boreholes)

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    # Further split train into train/val (85%/15% of train set)
    train_boreholes_sub = np.array(list(set(df_train['Borehole_ID'].unique())))
    n_val = int(len(train_boreholes_sub) * 0.15)
    val_boreholes = np.random.RandomState(42).choice(train_boreholes_sub, size=n_val, replace=False)

    val_mask = df_train['Borehole_ID'].isin(val_boreholes)
    df_val = df_train[val_mask].copy()
    df_train = df_train[~val_mask].copy()

    X_train = df_train[feature_cols].values
    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values
    y_test = df_test['Principal'].values

    print(f"Train: {len(df_train):,} samples ({len(df_train['Borehole_ID'].unique())} boreholes)")
    print(f"Val:   {len(df_val):,} samples ({len(df_val['Borehole_ID'].unique())} boreholes)")
    print(f"Test:  {len(df_test):,} samples ({len(df_test['Borehole_ID'].unique())} boreholes)")

    # Distribution-aware scaling
    scaler = DistributionAwareScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.zeros(len(X_train_scaled)))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.zeros(len(X_val_scaled)))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Train model
    model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)

    start_time = time.time()
    model = train_vae(model, train_loader, val_loader, epochs=100, device=device,
                      beta_start=0.001, beta_end=0.5, anneal_epochs=50)
    train_time = time.time() - start_time

    # Evaluate
    metrics = evaluate_clustering(model, X_test_scaled, y_test, device)

    print(f"\nResults:")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Collapsed dims: {metrics['collapsed_dims']}/10")
    print(f"  Effective dim: {metrics['effective_dim']}")
    print(f"  Best clustering: k={metrics['best_k']}, ARI={metrics['best_ari']:.4f}, Sil={metrics['best_sil']:.4f}")

    # Print all k results
    print(f"\n  Results for all k:")
    for result in metrics['all_results']:
        print(f"    k={result['k']:2d}: ARI={result['ari']:.4f}, Sil={result['sil']:.4f}")

    print()

    cv_results.append({
        'fold': fold_idx + 1,
        'train_samples': len(df_train),
        'test_samples': len(df_test),
        'train_time': train_time,
        'collapsed_dims': metrics['collapsed_dims'],
        'effective_dim': metrics['effective_dim'],
        'best_k': metrics['best_k'],
        'best_ari': metrics['best_ari'],
        'best_sil': metrics['best_sil'],
    })

# Summary
print("="*100)
print("CROSS-VALIDATION SUMMARY")
print("="*100)
print()

df_cv = pd.DataFrame(cv_results)
print(df_cv.to_string(index=False))
print()

# Statistics
mean_ari = df_cv['best_ari'].mean()
std_ari = df_cv['best_ari'].std()
min_ari = df_cv['best_ari'].min()
max_ari = df_cv['best_ari'].max()

print(f"ARI Statistics:")
print(f"  Mean: {mean_ari:.4f}")
print(f"  Std:  {std_ari:.4f}")
print(f"  Min:  {min_ari:.4f}")
print(f"  Max:  {max_ari:.4f}")
print(f"  95% CI: [{mean_ari - 1.96*std_ari:.4f}, {mean_ari + 1.96*std_ari:.4f}]")
print()

# Compare to original v2.6.6
print("="*100)
print("COMPARISON TO ORIGINAL v2.6.6")
print("="*100)
print(f"Original v2.6.6 (single split): ARI = 0.286")
print(f"Cross-validated mean:           ARI = {mean_ari:.4f} ± {std_ari:.4f}")
print()

if 0.286 < mean_ari - 1.96*std_ari or 0.286 > mean_ari + 1.96*std_ari:
    print("⚠️  Original v2.6.6 result is outside 95% CI - may be lucky/unlucky split")
else:
    print("✓ Original v2.6.6 result is within expected variance")

print("="*100)

# Save results
df_cv.to_csv('v2_6_6_cross_validation.csv', index=False)
print(f"\nResults saved to: v2_6_6_cross_validation.csv")
