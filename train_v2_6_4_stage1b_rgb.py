"""
VAE GRA v2.6.4 - Stage 1b: Pre-train RGB Encoder

Pre-train RGB encoder (R+G+B → 4D latent) on RGB data from v2.6 dataset.
This uses 296 boreholes with RGB imaging to learn visual appearance patterns.
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import time

from vae_lithology_gra_v2_6_4_model import SingleModalityVAE, DistributionAwareScaler, train_vae_with_annealing

print("="*80)
print("VAE GRA v2.6.4 - Stage 1b: Pre-train RGB Encoder")
print("="*80)
print()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print()

# Load v2.6 dataset (RGB only)
print("Loading v2.6 dataset (RGB only)...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

feature_cols = ['R', 'G', 'B']

X = df[feature_cols].values
borehole_ids = df['Borehole_ID'].values

print(f"Dataset: {len(X):,} samples from {len(np.unique(borehole_ids))} boreholes")
print(f"Features: 3D (R, G, B)")
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

# Distribution-aware scaling for RGB
print("Applying distribution-aware scaling for RGB...")
scaler = DistributionAwareScaler(feature_type='rgb')
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
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

# Train RGB VAE with 4D latent
print("="*80)
print("Training RGB VAE (3D → 4D latent, β annealing 0.001 → 0.5)")
print("="*80)
print()

model = SingleModalityVAE(input_dim=3, latent_dim=4, hidden_dims=[32, 16]).to(device)

print(f"Model architecture:")
print(f"  Input: 3D (R, G, B)")
print(f"  Encoder: 3 → 32 → 16 → 4")
print(f"  Decoder: 4 → 16 → 32 → 3")
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
print("Saving Stage 1b model...")
checkpoint = {
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'history': history,
    'input_dim': 3,
    'latent_dim': 4,
    'hidden_dims': [32, 16],
    'training_time': training_time,
    'epochs': len(history['train_loss'])
}

torch.save(checkpoint, 'ml_models/checkpoints/vae_gra_v2_6_4_stage1b_rgb.pth')
print("Saved: ml_models/checkpoints/vae_gra_v2_6_4_stage1b_rgb.pth")
print()

print("="*80)
print("Stage 1b Complete - RGB Encoder Pre-trained")
print("="*80)
print()
print(f"RGB encoder captures visual patterns from {len(unique_boreholes)} boreholes.")
print("Next: Stage 2 - Concatenate encoders and fine-tune fusion.")
