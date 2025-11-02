"""
VAE GRA v2.6.2 - Stage 1: Pre-training on Physical Properties

Pre-train on GRA + MS + NGR from 524 boreholes (403K samples).
This learns robust physical property representations before adding RGB.
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

from vae_lithology_gra_v2_6_2_model import VAE, DistributionAwareScaler, train_vae_with_annealing

print("="*80)
print("VAE GRA v2.6.2 - Stage 1: Pre-training on Physical Properties")
print("="*80)
print()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load v1 dataset (GRA + MS + NGR)
print("Loading v1 dataset (GRA + MS + NGR)...")
df = pd.read_csv('vae_training_data_20cm.csv')

feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)'
]

X = df[feature_cols].values
borehole_ids = df['Borehole_ID'].values
lithology = df['Principal'].values

print(f"Dataset: {len(X):,} samples from {len(np.unique(borehole_ids))} boreholes")
print(f"Features: {X.shape[1]}D (GRA, MS, NGR)")
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

# Distribution-aware scaling
print("Applying distribution-aware scaling...")
scaler = DistributionAwareScaler(input_dim=3)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
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
print("Training VAE with 8D latent space (β annealing 0.001 → 0.5)")
print("="*80)
print()

model = VAE(input_dim=3, latent_dim=8, hidden_dims=[32, 16], output_dim=3).to(device)

print(f"Model architecture:")
print(f"  Input: 3D (GRA, MS, NGR)")
print(f"  Encoder: 3 → 32 → 16 → 8")
print(f"  Decoder: 8 → 16 → 32 → 3")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

start_time = time.time()

model, history = train_vae_with_annealing(
    model, train_loader, val_loader,
    epochs=100,
    learning_rate=1e-3,
    device=device,
    beta_start=0.001,
    beta_end=0.5,
    anneal_epochs=50,
    patience=20
)

training_time = time.time() - start_time
print()
print(f"Training completed in {training_time:.1f} seconds ({len(history['train_loss'])} epochs)")
print()

# Save model
print("Saving Stage 1 model...")
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

torch.save(checkpoint, 'ml_models/checkpoints/vae_gra_v2_6_2_stage1.pth')
print("Saved: ml_models/checkpoints/vae_gra_v2_6_2_stage1.pth")
print()

# Evaluate on test set
print("="*80)
print("Stage 1 Test Set Evaluation")
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

        from vae_lithology_gra_v2_6_2_model import vae_loss
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
print("Stage 1 Pre-training Complete")
print("="*80)
print()
print("Model captures physical property representations from 524 boreholes.")
print("Ready for Stage 2: Fine-tuning with RGB features.")
