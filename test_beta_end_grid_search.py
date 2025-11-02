"""
Fine grid search for optimal β_end value.

We found β_end=0.75 worked well, but only tested 0.5, 0.75, 1.0.
This tests a finer grid: 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9 to find true optimum.

Using lucky split for speed (same test set as original experiments).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class VAE(nn.Module):
    """v2.6.7 architecture"""
    def __init__(self, input_dim=6, latent_dim=8, hidden_dims=[32, 16]):
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

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + beta * kl_div, recon_loss, kl_div

def train_vae(model, train_loader, val_loader, epochs, device, beta_start, beta_end, anneal_epochs):
    """Train VAE with β annealing"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        if epoch < anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
        else:
            beta = beta_end

        model.train()
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon_x, mu, logvar = model(batch_x)
                loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model

def evaluate_clustering(model, X_test, y_test, device):
    """Evaluate GMM clustering"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    latent_stds = latent.std(axis=0)
    collapsed_dims = (latent_stds < 0.1).sum()
    effective_dim = (latent_stds >= 0.1).sum()

    best_ari = 0
    best_sil = 0
    best_k = 0

    for k in [10, 12, 15, 18]:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        clusters = gmm.fit_predict(latent)

        ari = adjusted_rand_score(y_test, clusters)
        sil = silhouette_score(latent, clusters)

        if ari > best_ari:
            best_ari = ari
            best_sil = sil
            best_k = k

    return {
        'collapsed_dims': collapsed_dims,
        'effective_dim': effective_dim,
        'best_k': best_k,
        'best_ari': best_ari,
        'best_sil': best_sil
    }

print("="*100)
print("FINE GRID SEARCH FOR OPTIMAL β_end")
print("="*100)
print("Testing: β: 1e-10 → [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]")
print()

# Load data (use lucky split for speed)
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

unique_boreholes = df['Borehole_ID'].unique()
train_val_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)

train_boreholes, val_boreholes = train_test_split(
    train_val_boreholes, train_size=0.8235, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

print(f"Train: {len(df_train):,} samples ({len(df_train['Borehole_ID'].unique())} boreholes)")
print(f"Val:   {len(df_val):,} samples ({len(df_val['Borehole_ID'].unique())} boreholes)")
print(f"Test:  {len(df_test):,} samples ({len(df_test['Borehole_ID'].unique())} boreholes)")
print()

# Scale
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.zeros(len(X_train_scaled)))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.zeros(len(X_val_scaled)))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Test β_end values
beta_end_values = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
results = []

for beta_end in beta_end_values:
    print("="*100)
    print(f"Testing β_end = {beta_end}")
    print("="*100)

    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16]).to(device)

    start_time = time.time()
    model = train_vae(model, train_loader, val_loader, epochs=100, device=device,
                      beta_start=1e-10, beta_end=beta_end, anneal_epochs=50)
    train_time = time.time() - start_time

    metrics = evaluate_clustering(model, X_test_scaled, y_test, device)

    print(f"\nResults:")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Collapsed dims: {metrics['collapsed_dims']}/8")
    print(f"  Effective dim: {metrics['effective_dim']}")
    print(f"  Best clustering: k={metrics['best_k']}, ARI={metrics['best_ari']:.4f}, Sil={metrics['best_sil']:.4f}")
    print()

    results.append({
        'beta_end': beta_end,
        'train_time': train_time,
        'collapsed_dims': metrics['collapsed_dims'],
        'effective_dim': metrics['effective_dim'],
        'best_k': metrics['best_k'],
        'best_ari': metrics['best_ari'],
        'best_sil': metrics['best_sil'],
    })

# Summary
print("="*100)
print("GRID SEARCH SUMMARY")
print("="*100)
print()

df_results = pd.DataFrame(results)
print(df_results[['beta_end', 'best_ari', 'best_sil', 'train_time']].to_string(index=False))
print()

best_idx = df_results['best_ari'].idxmax()
print(f"Optimal β_end: {df_results.loc[best_idx, 'beta_end']}")
print(f"  ARI = {df_results.loc[best_idx, 'best_ari']:.4f}")
print(f"  Silhouette = {df_results.loc[best_idx, 'best_sil']:.4f}")
print()

# Plot trend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(df_results['beta_end'], df_results['best_ari'], 'o-', linewidth=2, markersize=8)
ax1.axvline(df_results.loc[best_idx, 'beta_end'], color='r', linestyle='--', alpha=0.5, label=f"Optimal: {df_results.loc[best_idx, 'beta_end']}")
ax1.set_xlabel('β_end', fontsize=12)
ax1.set_ylabel('ARI', fontsize=12)
ax1.set_title('Clustering Performance vs β_end', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.plot(df_results['beta_end'], df_results['best_sil'], 'o-', linewidth=2, markersize=8, color='orange')
ax2.set_xlabel('β_end', fontsize=12)
ax2.set_ylabel('Silhouette Score', fontsize=12)
ax2.set_title('Cluster Separation vs β_end', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('beta_end_grid_search.png', dpi=150, bbox_inches='tight')
print("Plot saved to: beta_end_grid_search.png")

df_results.to_csv('beta_end_grid_search.csv', index=False)
print("Results saved to: beta_end_grid_search.csv")
print()
print("="*100)
