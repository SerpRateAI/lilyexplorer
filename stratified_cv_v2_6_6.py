"""
Stratified cross-validation for v2.6.6 using geographic regions.

Strategy:
1. Parse expedition from Borehole_ID
2. Get lat/lon from GRA_DataLITH.csv
3. Cluster boreholes into ~5 geographic regions
4. Stratified 5-fold CV ensuring each fold has representation from all regions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class VAE(nn.Module):
    """v2.6.6 architecture"""
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
print("STRATIFIED CROSS-VALIDATION BY GEOGRAPHIC REGION")
print("="*100)
print()

# Load VAE training data
df_vae = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

# Parse expedition from Borehole_ID (keep as string, some have 'X' like 368X)
df_vae['Expedition'] = df_vae['Borehole_ID'].str.split('-').str[0]

# Load location data from GRA
print("Loading geographic data...")
gra = pd.read_csv('/home/utig5/johna/bhai/datasets/GRA_DataLITH.csv')
gra['Borehole_ID'] = gra['Exp'].astype(str) + '-' + gra['Site'] + '-' + gra['Hole']

borehole_locations = gra.groupby('Borehole_ID').agg({
    'Latitude (DD)': 'first',
    'Longitude (DD)': 'first',
    'Exp': 'first'
}).reset_index()

# Merge locations with VAE data
df_vae = df_vae.merge(borehole_locations, on='Borehole_ID', how='left')

print(f"Total boreholes: {df_vae['Borehole_ID'].nunique()}")
print(f"Latitude range: {df_vae['Latitude (DD)'].min():.1f}° to {df_vae['Latitude (DD)'].max():.1f}°")
print(f"Longitude range: {df_vae['Longitude (DD)'].min():.1f}° to {df_vae['Longitude (DD)'].max():.1f}°")
print()

# Cluster boreholes into geographic regions
unique_boreholes = df_vae.groupby('Borehole_ID').agg({
    'Latitude (DD)': 'first',
    'Longitude (DD)': 'first'
}).reset_index()

# Use k-means to create 5 geographic clusters
n_regions = 5
coords = unique_boreholes[['Latitude (DD)', 'Longitude (DD)']].values
kmeans = KMeans(n_clusters=n_regions, random_state=42)
unique_boreholes['region'] = kmeans.fit_predict(coords)

print(f"Created {n_regions} geographic regions:")
for region_id in range(n_regions):
    region_holes = unique_boreholes[unique_boreholes['region'] == region_id]
    print(f"  Region {region_id}: {len(region_holes)} boreholes, "
          f"Lat=[{region_holes['Latitude (DD)'].min():.1f}, {region_holes['Latitude (DD)'].max():.1f}], "
          f"Lon=[{region_holes['Longitude (DD)'].min():.1f}, {region_holes['Longitude (DD)'].max():.1f}]")
print()

# Stratified 5-fold CV
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")
print(f"K-fold CV: {n_folds} folds (stratified by region)")
print()

cv_results = []

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(unique_boreholes['Borehole_ID'], unique_boreholes['region'])):
    print("="*100)
    print(f"FOLD {fold_idx + 1}/{n_folds}")
    print("="*100)

    train_boreholes = unique_boreholes.iloc[train_idx]['Borehole_ID'].values
    test_boreholes = unique_boreholes.iloc[test_idx]['Borehole_ID'].values

    # Check region distribution
    train_regions = unique_boreholes.iloc[train_idx]['region'].value_counts().sort_index()
    test_regions = unique_boreholes.iloc[test_idx]['region'].value_counts().sort_index()
    print(f"Train regions: {train_regions.to_dict()}")
    print(f"Test regions:  {test_regions.to_dict()}")

    # Split data
    train_mask = df_vae['Borehole_ID'].isin(train_boreholes)
    test_mask = df_vae['Borehole_ID'].isin(test_boreholes)

    df_train = df_vae[train_mask].copy()
    df_test = df_vae[test_mask].copy()

    # Further split train into train/val
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

    # Train
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
print("STRATIFIED CROSS-VALIDATION SUMMARY")
print("="*100)
print()

df_cv = pd.DataFrame(cv_results)
print(df_cv.to_string(index=False))
print()

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

print("="*100)
print("COMPARISON TO ORIGINAL v2.6.6")
print("="*100)
print(f"Original v2.6.6 (random split):     ARI = 0.286")
print(f"Stratified CV mean:                 ARI = {mean_ari:.4f} ± {std_ari:.4f}")
print()

if 0.286 < mean_ari - 1.96*std_ari or 0.286 > mean_ari + 1.96*std_ari:
    print("⚠️  Original result outside 95% CI of stratified CV")
else:
    print("✓ Original result within expected variance")

print("="*100)

df_cv.to_csv('v2_6_6_stratified_cv.csv', index=False)
print(f"\nResults saved to: v2_6_6_stratified_cv.csv")
