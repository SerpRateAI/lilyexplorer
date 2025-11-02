"""
VAE GRA v2.6.4 - Stage 2: Fusion Training

Load pre-trained physical and RGB encoders, concatenate their 4D latents to get 8D,
and train dual decoders that see the full 8D latent space to learn cross-modal patterns.
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

from vae_lithology_gra_v2_6_4_model import load_pretrained_encoders, train_vae_with_annealing

print("="*80)
print("VAE GRA v2.6.4 - Stage 2: Fusion Training")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load pre-trained encoders
print("Loading pre-trained encoders...")
model, phys_scaler, rgb_scaler = load_pretrained_encoders(
    'ml_models/checkpoints/vae_gra_v2_6_4_stage1a_physical.pth',
    'ml_models/checkpoints/vae_gra_v2_6_4_stage1b_rgb.pth',
    device=device
)
print()

# Load v2.6 dataset (full features)
print("Loading v2.6 dataset (GRA + MS + NGR + RGB)...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

phys_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)', 'NGR total counts (cps)']
rgb_cols = ['R', 'G', 'B']

X_phys = df[phys_cols].values
X_rgb = df[rgb_cols].values
borehole_ids = df['Borehole_ID'].values

print(f"Dataset: {len(X_phys):,} samples from {len(np.unique(borehole_ids))} boreholes")
print(f"Features: 6D (GRA, MS, NGR, R, G, B)")
print()

# Borehole-level split
print("Splitting data by borehole...")
unique_boreholes = np.unique(borehole_ids)
train_boreholes, temp_boreholes = train_test_split(unique_boreholes, test_size=0.3, random_state=42)
val_boreholes, test_boreholes = train_test_split(temp_boreholes, test_size=0.5, random_state=42)

train_mask = np.isin(borehole_ids, train_boreholes)
val_mask = np.isin(borehole_ids, val_boreholes)
test_mask = np.isin(borehole_ids, test_boreholes)

X_phys_train, X_rgb_train = X_phys[train_mask], X_rgb[train_mask]
X_phys_val, X_rgb_val = X_phys[val_mask], X_rgb[val_mask]
X_phys_test, X_rgb_test = X_phys[test_mask], X_rgb[test_mask]

print(f"Train: {len(X_phys_train):,} samples ({len(train_boreholes)} boreholes)")
print(f"Val:   {len(X_phys_val):,} samples ({len(val_boreholes)} boreholes)")
print(f"Test:  {len(X_phys_test):,} samples ({len(test_boreholes)} boreholes)")
print()

# Scale features using pre-trained scalers
print("Scaling features with pre-trained scalers...")
X_phys_train_scaled = phys_scaler.transform(X_phys_train)
X_phys_val_scaled = phys_scaler.transform(X_phys_val)
X_phys_test_scaled = phys_scaler.transform(X_phys_test)

X_rgb_train_scaled = rgb_scaler.transform(X_rgb_train)
X_rgb_val_scaled = rgb_scaler.transform(X_rgb_val)
X_rgb_test_scaled = rgb_scaler.transform(X_rgb_test)

# Concatenate to 6D input
X_train_scaled = np.concatenate([X_phys_train_scaled, X_rgb_train_scaled], axis=1)
X_val_scaled = np.concatenate([X_phys_val_scaled, X_rgb_val_scaled], axis=1)
X_test_scaled = np.concatenate([X_phys_test_scaled, X_rgb_test_scaled], axis=1)
print()

# Create DataLoaders
batch_size = 256
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print()

# Fine-tune fusion model
print("="*80)
print("Fine-tuning Dual Encoder VAE (8D latent = 4D physical + 4D RGB)")
print("="*80)
print()

print(f"Model architecture:")
print(f"  Physical encoder: 3D (GRA, MS, NGR) → 4D latent [PRE-TRAINED]")
print(f"  RGB encoder: 3D (R, G, B) → 4D latent [PRE-TRAINED]")
print(f"  Combined: 8D latent (4D + 4D)")
print(f"  Physical decoder: 8D → 3D (GRA, MS, NGR) [NEW - learns cross-modal]")
print(f"  RGB decoder: 8D → 3D (R, G, B) [NEW - learns cross-modal]")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

print("Fine-tuning strategy:")
print("  - Lower learning rate (5e-4) to preserve pre-trained encoder knowledge")
print("  - Shorter β annealing (25 epochs) since encoders are pre-trained")
print("  - Decoders see full 8D latent → learn physical↔RGB correlations")
print()

start_time = time.time()

model, history = train_vae_with_annealing(
    model, train_loader, val_loader,
    epochs=100,
    learning_rate=5e-4,  # Lower LR to preserve pre-trained weights
    device=device,
    beta_start=0.001,
    beta_end=0.5,
    anneal_epochs=25,  # Faster since encoders pre-trained
    patience=20
)

training_time = time.time() - start_time
print()
print(f"Fine-tuning completed in {training_time:.1f} seconds ({len(history['train_loss'])} epochs)")
print()

# Save Stage 2 model
print("Saving Stage 2 fusion model...")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'phys_scaler': phys_scaler,
    'rgb_scaler': rgb_scaler,
    'history': history,
    'training_time': training_time,
    'epochs': len(history['train_loss'])
}

torch.save(checkpoint, 'ml_models/checkpoints/vae_gra_v2_6_4_stage2_fusion.pth')
print("Saved: ml_models/checkpoints/vae_gra_v2_6_4_stage2_fusion.pth")
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

    from vae_lithology_gra_v2_6_4_model import vae_loss

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
print("Stage 2 Fusion Training Complete")
print("="*80)
print()
print("Total training time:")
print("  Stage 1a (physical): 271s")
print("  Stage 1b (RGB): 167s")
print(f"  Stage 2 (fusion): {training_time:.0f}s")
print(f"  Total: {271 + 167 + training_time:.0f}s")
print()
print("Next step: Evaluate clustering performance and compare to v2.6")
