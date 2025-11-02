"""
Test extreme β annealing schedules.

Current best: β: 0.001 → 0.5, ARI = 0.258 (single split)
Hypothesis: Starting from β ≈ 0 (pure autoencoder) might help, but ending at β=10
           likely hurts (forces too much disentanglement, destroys feature correlations).
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
    """v2.6 architecture"""
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
        # β annealing
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
print("EXTREME β ANNEALING SCHEDULES TEST")
print("="*100)
print()

# Load data
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

# Use same test set as original v2.6 for fair comparison
unique_boreholes = df['Borehole_ID'].unique()
train_val_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)

train_boreholes, val_boreholes = train_test_split(
    train_val_boreholes, train_size=0.8235, random_state=42  # 0.85 * 0.8235 ≈ 0.70
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

# Test schedules
schedules = [
    {'name': 'v2.6 baseline', 'beta_start': 0.001, 'beta_end': 0.5, 'anneal_epochs': 50},
    {'name': 'Extreme: 1e-10 → 0.5', 'beta_start': 1e-10, 'beta_end': 0.5, 'anneal_epochs': 50},
    {'name': 'Extreme: 1e-10 → 1.0', 'beta_start': 1e-10, 'beta_end': 1.0, 'anneal_epochs': 50},
    {'name': 'Extreme: 1e-10 → 5.0', 'beta_start': 1e-10, 'beta_end': 5.0, 'anneal_epochs': 75},
    {'name': 'Extreme: 1e-10 → 10.0', 'beta_start': 1e-10, 'beta_end': 10.0, 'anneal_epochs': 100},
    {'name': 'Very slow: 1e-10 → 0.5', 'beta_start': 1e-10, 'beta_end': 0.5, 'anneal_epochs': 100},
]

results = []

for schedule in schedules:
    print("="*100)
    print(f"Testing: {schedule['name']}")
    print(f"  β: {schedule['beta_start']} → {schedule['beta_end']} over {schedule['anneal_epochs']} epochs")
    print("="*100)

    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16]).to(device)

    start_time = time.time()
    model = train_vae(model, train_loader, val_loader, epochs=100, device=device,
                      beta_start=schedule['beta_start'], beta_end=schedule['beta_end'],
                      anneal_epochs=schedule['anneal_epochs'])
    train_time = time.time() - start_time

    metrics = evaluate_clustering(model, X_test_scaled, y_test, device)

    print(f"\nResults:")
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Collapsed dims: {metrics['collapsed_dims']}/8")
    print(f"  Effective dim: {metrics['effective_dim']}")
    print(f"  Best clustering: k={metrics['best_k']}, ARI={metrics['best_ari']:.4f}, Sil={metrics['best_sil']:.4f}")
    print()

    results.append({
        'schedule': schedule['name'],
        'beta_start': schedule['beta_start'],
        'beta_end': schedule['beta_end'],
        'anneal_epochs': schedule['anneal_epochs'],
        'train_time': train_time,
        'collapsed_dims': metrics['collapsed_dims'],
        'effective_dim': metrics['effective_dim'],
        'best_k': metrics['best_k'],
        'best_ari': metrics['best_ari'],
        'best_sil': metrics['best_sil'],
    })

# Summary
print("="*100)
print("EXTREME β ANNEALING SUMMARY")
print("="*100)
print()

df_results = pd.DataFrame(results)
print(df_results[['schedule', 'beta_start', 'beta_end', 'anneal_epochs', 'best_ari', 'best_sil', 'train_time']].to_string(index=False))
print()

best_idx = df_results['best_ari'].idxmax()
baseline_idx = 0  # First is baseline

print(f"Baseline (0.001 → 0.5):        ARI = {df_results.loc[baseline_idx, 'best_ari']:.4f}")
print(f"Best schedule: {df_results.loc[best_idx, 'schedule']}")
print(f"  ARI = {df_results.loc[best_idx, 'best_ari']:.4f} ({100*(df_results.loc[best_idx, 'best_ari']/df_results.loc[baseline_idx, 'best_ari'] - 1):+.1f}%)")
print()

# Check if extreme endpoints help
extreme_10 = df_results[df_results['beta_end'] == 10.0]['best_ari'].values[0]
extreme_5 = df_results[df_results['beta_end'] == 5.0]['best_ari'].values[0]
extreme_1 = df_results[df_results['beta_end'] == 1.0]['best_ari'].values[0]

print("β endpoint analysis:")
print(f"  β_end = 0.5:  ARI = {df_results.loc[baseline_idx, 'best_ari']:.4f} (current best)")
print(f"  β_end = 1.0:  ARI = {extreme_1:.4f} ({100*(extreme_1/df_results.loc[baseline_idx, 'best_ari'] - 1):+.1f}%)")
print(f"  β_end = 5.0:  ARI = {extreme_5:.4f} ({100*(extreme_5/df_results.loc[baseline_idx, 'best_ari'] - 1):+.1f}%)")
print(f"  β_end = 10.0: ARI = {extreme_10:.4f} ({100*(extreme_10/df_results.loc[baseline_idx, 'best_ari'] - 1):+.1f}%)")
print()

if extreme_10 < df_results.loc[baseline_idx, 'best_ari']:
    print("⚠️  High β_end (≥1.0) degrades performance - too much disentanglement")
    print("   Clustering needs feature correlations ('dark + dense = basalt')")
else:
    print("✓ High β_end improves performance unexpectedly!")

print()
print("="*100)

df_results.to_csv('extreme_beta_schedules.csv', index=False)
print("Results saved to: extreme_beta_schedules.csv")
