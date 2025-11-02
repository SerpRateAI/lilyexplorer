"""
VAE GRA v2.6.2 - Stage 2: Fine-tuning with RGB Features

Load Stage 1 pre-trained model and fine-tune with RGB color features.
Transfer learning: GRA/MS/NGR pathways pre-trained, RGB pathways learned from scratch.
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

from vae_lithology_gra_v2_6_2_model import (
    VAE, DistributionAwareScaler, train_vae_with_annealing, expand_model_3d_to_6d
)

print("="*80)
print("VAE GRA v2.6.2 - Stage 2: Fine-tuning with RGB Features")
print("="*80)
print()

# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load Stage 1 checkpoint
print("Loading Stage 1 pre-trained model...")
checkpoint_stage1 = torch.load('ml_models/checkpoints/vae_gra_v2_6_2_stage1.pth', map_location=device, weights_only=False)

model_3d = VAE(input_dim=3, latent_dim=8, hidden_dims=[32, 16], output_dim=3).to(device)
model_3d.load_state_dict(checkpoint_stage1['model_state_dict'])

print(f"Stage 1 model loaded:")
print(f"  Training epochs: {checkpoint_stage1['epochs']}")
print(f"  Training time: {checkpoint_stage1['training_time']:.1f}s")
print()

# Expand model from 3D to 6D
print("Expanding model from 3D to 6D...")
model_6d = expand_model_3d_to_6d(model_3d, device=device)
print()

# Load v2.6 dataset (GRA + MS + NGR + RGB)
print("Loading v2.6 dataset (GRA + MS + NGR + RGB)...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'R',
    'G',
    'B'
]

X = df[feature_cols].values
borehole_ids = df['Borehole_ID'].values
lithology = df['Principal'].values

print(f"Dataset: {len(X):,} samples from {len(np.unique(borehole_ids))} boreholes")
print(f"Features: {X.shape[1]}D (GRA, MS, NGR, RGB)")
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

# Distribution-aware scaling for 6D
print("Applying distribution-aware scaling (6D)...")
scaler = DistributionAwareScaler(input_dim=6)
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

# Fine-tune with lower learning rate and shorter annealing
print("="*80)
print("Fine-tuning with RGB features")
print("="*80)
print()

print(f"Model architecture:")
print(f"  Input: 6D (GRA, MS, NGR, R, G, B)")
print(f"  Encoder: 6 → 32 → 16 → 8")
print(f"  Decoder: 8 → 16 → 32 → 6")
print(f"  Parameters: {sum(p.numel() for p in model_6d.parameters()):,}")
print()
print("Fine-tuning strategy:")
print("  - Lower learning rate (5e-4 vs 1e-3) to preserve pre-trained knowledge")
print("  - Shorter β annealing (25 epochs vs 50) since physical properties are pre-trained")
print("  - β range: 0.001 → 0.5 (same target as Stage 1)")
print()

start_time = time.time()

model_6d, history = train_vae_with_annealing(
    model_6d, train_loader, val_loader,
    epochs=100,
    learning_rate=5e-4,  # Lower LR to preserve pre-trained weights
    device=device,
    beta_start=0.001,
    beta_end=0.5,
    anneal_epochs=25,  # Faster annealing since we start from good initialization
    patience=20
)

training_time = time.time() - start_time
print()
print(f"Fine-tuning completed in {training_time:.1f} seconds ({len(history['train_loss'])} epochs)")
print()

# Save Stage 2 model
print("Saving Stage 2 fine-tuned model...")
checkpoint_stage2 = {
    'model_state_dict': model_6d.state_dict(),
    'scaler': scaler,
    'history': history,
    'input_dim': 6,
    'latent_dim': 8,
    'hidden_dims': [32, 16],
    'training_time': training_time,
    'epochs': len(history['train_loss']),
    'stage1_epochs': checkpoint_stage1['epochs'],
    'total_training_time': checkpoint_stage1['training_time'] + training_time
}

torch.save(checkpoint_stage2, 'ml_models/checkpoints/vae_gra_v2_6_2_stage2.pth')
print("Saved: ml_models/checkpoints/vae_gra_v2_6_2_stage2.pth")
print()

# Evaluate on test set
print("="*80)
print("Stage 2 Test Set Evaluation")
print("="*80)
print()

model_6d.eval()
with torch.no_grad():
    test_loss = 0
    test_recon = 0
    test_kl = 0

    for (data,) in test_loader:
        data = data.to(device)
        recon, mu, logvar = model_6d(data)

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
print("Stage 2 Fine-tuning Complete")
print("="*80)
print()
print(f"Total training time: {checkpoint_stage2['total_training_time']:.1f}s")
print(f"  Stage 1 (physical properties): {checkpoint_stage1['training_time']:.1f}s")
print(f"  Stage 2 (RGB fine-tuning): {training_time:.1f}s")
print()
print("Next step: Evaluate clustering performance and compare to v2.6")
