"""
VAE GRA v2.6.3: RGB-Only Clustering

Test if RGB color alone is sufficient for lithology discrimination,
or if physical properties (GRA+MS+NGR) are necessary.

This isolates the RGB signal to understand its contribution to v2.6's performance.
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from vae_lithology_gra_v2_5_model import VAE, vae_loss

print("="*80)
print("VAE GRA v2.6.3 - RGB Only")
print("="*80)
print()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load v2.6 dataset but use RGB features only
print("Loading v2.6 dataset (RGB only)...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = ['R', 'G', 'B']

X = df[feature_cols].values
borehole_ids = df['Borehole_ID'].values
lithology = df['Principal'].values

print(f"Dataset: {len(X):,} samples from {len(np.unique(borehole_ids))} boreholes")
print(f"Features: {X.shape[1]}D (R, G, B)")
print(f"Lithologies: {len(np.unique(lithology))}")
print()

# Borehole-level split
print("Splitting data by borehole...")
unique_boreholes = np.unique(borehole_ids)
train_boreholes, temp_boreholes = train_test_split(unique_boreholes, test_size=0.3, random_state=42)
val_boreholes, test_boreholes = train_test_split(temp_boreholes, test_size=0.5, random_state=42)

train_mask = np.isin(borehole_ids, train_boreholes)
val_mask = np.isin(borehole_ids, val_boreholes)
test_mask = np.isin(borehole_ids, test_boreholes)

X_train = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]

print(f"Train: {len(X_train):,} samples ({len(train_boreholes)} boreholes)")
print(f"Val:   {len(X_val):,} samples ({len(val_boreholes)} boreholes)")
print(f"Test:  {len(X_test):,} samples ({len(test_boreholes)} boreholes)")
print()

# Log scaling for RGB (log-normal distribution)
print("Applying log scaling for RGB...")
X_train_log = np.log1p(X_train)
X_val_log = np.log1p(X_val)
X_test_log = np.log1p(X_test)

# Then StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_val_scaled = scaler.transform(X_val_log)
X_test_scaled = scaler.transform(X_test_log)
print()

# Create DataLoaders
print("Creating DataLoaders...")
batch_size = 256

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print()

# Train 8D latent model with β annealing
print("="*80)
print("Training RGB-only VAE with 8D latent space (β annealing 0.001 → 0.5)")
print("="*80)
print()

model = VAE(input_dim=3, latent_dim=8, hidden_dims=[32, 16]).to(device)

print(f"Model architecture:")
print(f"  Input: 3D (R, G, B)")
print(f"  Encoder: 3 → 32 → 16 → 8")
print(f"  Decoder: 8 → 16 → 32 → 3")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Training with β annealing
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = {
    'train_loss': [], 'train_recon': [], 'train_kl': [],
    'val_loss': [], 'val_recon': [], 'val_kl': [],
    'beta': []
}

beta_start = 0.001
beta_end = 0.5
anneal_epochs = 50
max_epochs = 100
patience = 20

best_val_loss = float('inf')
epochs_without_improvement = 0

start_time = time.time()

for epoch in range(max_epochs):
    # Calculate current β
    if epoch < anneal_epochs:
        progress = epoch / anneal_epochs
        current_beta = beta_start + (beta_end - beta_start) * progress
    else:
        current_beta = beta_end

    history['beta'].append(current_beta)

    # Training
    model.train()
    train_loss, train_recon, train_kl = 0, 0, 0

    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        recon, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=current_beta)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()

    train_loss /= len(train_loader)
    train_recon /= len(train_loader)
    train_kl /= len(train_loader)

    # Validation
    model.eval()
    val_loss, val_recon, val_kl = 0, 0, 0

    with torch.no_grad():
        for batch_idx, (data,) in enumerate(val_loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=current_beta)

            val_loss += loss.item()
            val_recon += recon_loss.item()
            val_kl += kl_loss.item()

    val_loss /= len(val_loader)
    val_recon /= len(val_loader)
    val_kl /= len(val_loader)

    # Store history
    history['train_loss'].append(train_loss)
    history['train_recon'].append(train_recon)
    history['train_kl'].append(train_kl)
    history['val_loss'].append(val_loss)
    history['val_recon'].append(val_recon)
    history['val_kl'].append(val_kl)

    print(f'Epoch {epoch+1:3d}/{max_epochs} | β={current_beta:.4f} | '
          f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | '
          f'Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        break

training_time = time.time() - start_time
print()
print(f"Training completed in {training_time:.1f} seconds ({len(history['train_loss'])} epochs)")
print()

# Save model
print("Saving RGB-only model...")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'history': history,
    'input_dim': 3,
    'latent_dim': 8,
    'hidden_dims': [32, 16],
    'training_time': training_time,
    'epochs': len(history['train_loss'])
}

torch.save(checkpoint, 'ml_models/checkpoints/vae_gra_v2_6_3_rgb_only.pth')
print("Saved: ml_models/checkpoints/vae_gra_v2_6_3_rgb_only.pth")
print()

# Test set evaluation
print("="*80)
print("Test Set Evaluation")
print("="*80)
print()

model.eval()
with torch.no_grad():
    test_loss = 0
    test_recon = 0
    test_kl = 0

    for (data,) in test_loader:
        data = data.to(device)
        recon, mu, logvar = model(data)
        loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=0.5)

        test_loss += loss.item()
        test_recon += recon_loss.item()
        test_kl += kl_loss.item()

    test_loss /= len(test_loader)
    test_recon /= len(test_loader)
    test_kl /= len(test_loader)

print(f"Test Loss: {test_loss:.4f}")
print(f"  Reconstruction: {test_recon:.4f}")
print(f"  KL Divergence: {test_kl:.4f}")
print()

print("="*80)
print("RGB-Only Training Complete")
print("="*80)
print()
print("Next step: Evaluate clustering performance and compare to:")
print("  - v2.6 (GRA+MS+NGR+RGB, 6D)")
print("  - v1 (GRA+MS+NGR, 3D)")
