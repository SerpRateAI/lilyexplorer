"""
Test different β annealing schedules for optimal clustering performance.

Research question: Does slower annealing (0.0001→1.0 over more epochs) improve beyond v2.6.6?

Schedules tested:
1. v2.6.6 baseline: 0.001→0.5 over 50 epochs
2. Lower start: 0.0001→0.5 over 50 epochs
3. Higher end: 0.001→1.0 over 50 epochs
4. Slower anneal: 0.0001→1.0 over 75 epochs
5. Very slow: 0.0001→0.5 over 100 epochs
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
    """Standard VAE with Gaussian N(0,I) prior"""
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

def train_vae(model, train_loader, val_loader, epochs, device, beta_start, beta_end, anneal_epochs):
    """Train VAE with β annealing schedule"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        # β annealing schedule
        if epoch < anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
        else:
            beta = beta_end

        # Training
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, _, _ = model.loss_function(recon_x, batch_x, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: β={beta:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return model

def evaluate_clustering(model, X_test, y_test, device):
    """Evaluate clustering performance with GMM"""
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

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

print("="*100)
print("TESTING β ANNEALING SCHEDULES FOR CLUSTERING")
print("="*100)
print()

# Load data
print("Loading data...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

# Borehole-level split
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, temp_boreholes = train_test_split(
    unique_boreholes, train_size=0.70, random_state=42
)
val_boreholes, test_boreholes = train_test_split(
    temp_boreholes, train_size=0.5, random_state=42
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

print(f"Train: {len(df_train):,} samples")
print(f"Val:   {len(df_val):,} samples")
print(f"Test:  {len(df_test):,} samples")
print()

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# β annealing schedules to test
schedules = [
    {'name': 'v2.6.6 baseline', 'beta_start': 0.001, 'beta_end': 0.5, 'anneal_epochs': 50, 'max_epochs': 100},
    {'name': 'Lower start', 'beta_start': 0.0001, 'beta_end': 0.5, 'anneal_epochs': 50, 'max_epochs': 100},
    {'name': 'Higher end', 'beta_start': 0.001, 'beta_end': 1.0, 'anneal_epochs': 50, 'max_epochs': 100},
    {'name': 'Slower anneal', 'beta_start': 0.0001, 'beta_end': 1.0, 'anneal_epochs': 75, 'max_epochs': 100},
    {'name': 'Very slow', 'beta_start': 0.0001, 'beta_end': 0.5, 'anneal_epochs': 100, 'max_epochs': 150},
]

results = []

for schedule in schedules:
    print("="*100)
    print(f"TRAINING: {schedule['name']}")
    print(f"  β: {schedule['beta_start']:.4f} → {schedule['beta_end']:.4f} over {schedule['anneal_epochs']} epochs")
    print("="*100)

    # Train model
    model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)

    start_time = time.time()
    model = train_vae(
        model, train_loader, val_loader,
        epochs=schedule['max_epochs'],
        device=device,
        beta_start=schedule['beta_start'],
        beta_end=schedule['beta_end'],
        anneal_epochs=schedule['anneal_epochs']
    )
    train_time = time.time() - start_time

    # Evaluate
    print(f"\nEvaluating clustering...")
    metrics = evaluate_clustering(model, X_test_scaled, y_test, device)

    print(f"\nResults:")
    print(f"  Collapsed dims: {metrics['collapsed_dims']}/10")
    print(f"  Effective dim: {metrics['effective_dim']}")
    print(f"  Best clustering: k={metrics['best_k']}, ARI={metrics['best_ari']:.4f}, Sil={metrics['best_sil']:.4f}")
    print(f"  Training time: {train_time:.1f}s")
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
        'best_sil': metrics['best_sil']
    })

# Results summary
print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()

df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
print()

# Save results
df_results.to_csv('beta_schedule_comparison.csv', index=False)

# Comparison to v2.6.6
print("="*100)
print("COMPARISON TO v2.6.6 BASELINE")
print("="*100)

baseline = df_results[df_results['schedule'] == 'v2.6.6 baseline'].iloc[0]
baseline_ari = baseline['best_ari']

print(f"v2.6.6 baseline: GMM ARI = {baseline_ari:.4f}")
print()

for _, row in df_results.iterrows():
    if row['schedule'] == 'v2.6.6 baseline':
        continue

    improvement = ((row['best_ari'] - baseline_ari) / baseline_ari) * 100
    symbol = "🎯" if improvement > 1.0 else "✓" if improvement > 0 else "✗"
    print(f"{symbol} {row['schedule']}: ARI = {row['best_ari']:.4f} ({improvement:+.1f}%)")

print()

# Winner
best_idx = df_results['best_ari'].idxmax()
best = df_results.iloc[best_idx]
if best['schedule'] != 'v2.6.6 baseline':
    improvement = ((best['best_ari'] - baseline_ari) / baseline_ari) * 100
    print(f"🏆 Winner: {best['schedule']} (+{improvement:.1f}% improvement)")
    print(f"   β: {best['beta_start']:.4f} → {best['beta_end']:.4f} over {best['anneal_epochs']} epochs")
else:
    print(f"🏆 v2.6.6 baseline remains best. Current β schedule is optimal.")

print("="*100)

print(f"\nResults saved to: beta_schedule_comparison.csv")
