#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train final v2.13 model on ALL data (no splits).

Cross-validation validated the architecture:
- Multi-decoder (6 separate decoders)
- 2x weight on MS and NGR
- β annealing: 1e-10 → 0.75
- Expected performance: ARI = 0.187 ± 0.045

Now train production model on 100% of data (238,506 samples).
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import time

from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class MultiDecoderVAE(nn.Module):
    """v2.13 architecture: Multi-decoder with feature weighting"""

    def __init__(self, input_dim=6, latent_dim=10, encoder_dims=[32, 16],
                 decoder_dims=[16, 32]):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.feature_names = ['GRA', 'MS', 'NGR', 'R', 'G', 'B']

        # Shared encoder
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Separate decoder per feature
        self.decoders = nn.ModuleDict()
        for name in self.feature_names:
            decoder_layers = []
            prev_dim = latent_dim
            for h_dim in decoder_dims:
                decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
                prev_dim = h_dim
            decoder_layers.append(nn.Linear(decoder_dims[-1], 1))
            self.decoders[name] = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        outputs = []
        for name in self.feature_names:
            outputs.append(self.decoders[name](z))
        return torch.cat(outputs, dim=1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta=1.0, feature_weights=None):
        batch_size = x.size(0)

        if feature_weights is None:
            feature_weights = torch.ones(6, device=x.device)

        # Per-feature reconstruction loss
        recon_loss = 0
        for i in range(6):
            loss_i = F.mse_loss(recon_x[:, i], x[:, i], reduction='sum') / batch_size
            recon_loss += feature_weights[i] * loss_i

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        return recon_loss + beta * kl_loss, recon_loss, kl_loss

def train_vae_all_data(model, train_loader, epochs, device, beta_start=1e-10,
                       beta_end=0.75, anneal_epochs=50, feature_weights=None):
    """Train VAE on all data with β annealing"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training on ALL data...")
    print(f"β schedule: {beta_start} → {beta_end} over {anneal_epochs} epochs")
    print(f"Feature weights: {feature_weights}")
    print()

    for epoch in range(epochs):
        # β annealing schedule
        if epoch < anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
        else:
            beta = beta_end

        model.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            loss, recon_loss, kl_div = model.loss_function(
                recon_x, batch_x, mu, logvar, beta, feature_weights
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_div.item()

        epoch_loss /= len(train_loader)
        epoch_recon /= len(train_loader)
        epoch_kl /= len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss:.4f}, Recon={epoch_recon:.4f}, "
                  f"KL={epoch_kl:.4f}, β={beta:.6f}")

    print()
    print("Training complete!")
    return model

print("="*100)
print("TRAINING FINAL v2.13 MODEL ON ALL DATA")
print("="*100)
print("Architecture: Multi-decoder VAE with feature weighting")
print("β schedule: 1e-10 → 0.75 over 50 epochs")
print("Cross-validation: ARI = 0.187 ± 0.045")
print()

# Load ALL data
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X_all = df[feature_cols].values

print(f"Total samples: {len(df):,}")
print(f"Total boreholes: {df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {df['Principal'].nunique()}")
print()

# Scale
scaler = DistributionAwareScaler()
X_all_scaled = scaler.fit_transform(X_all)

# DataLoader (use all data for training)
train_dataset = TensorDataset(torch.FloatTensor(X_all_scaled), torch.zeros(len(X_all_scaled)))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Initialize model
model = MultiDecoderVAE(
    input_dim=6,
    latent_dim=10,
    encoder_dims=[32, 16],
    decoder_dims=[16, 32]
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Feature weights: 2x for MS and NGR
feature_weights = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0, 1.0]).to(device)

# Train
start_time = time.time()
model = train_vae_all_data(
    model, train_loader,
    epochs=100, device=device,
    beta_start=1e-10, beta_end=0.75, anneal_epochs=50,
    feature_weights=feature_weights
)
train_time = time.time() - start_time

print(f"Training time: {train_time:.1f}s")
print()

# Save
checkpoint_path = '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_13_final.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'latent_dim': 10,
    'input_dim': 6,
    'encoder_dims': [32, 16],
    'decoder_dims': [16, 32],
    'feature_weights': feature_weights.cpu(),
    'beta_schedule': {'start': 1e-10, 'end': 0.75, 'anneal_epochs': 50},
    'version': 'v2.13',
    'architecture': 'multi_decoder',
    'cv_results': {
        'mean_ari': 0.1865,
        'std_ari': 0.0445,
        'min_ari': 0.1357,
        'max_ari': 0.2566
    }
}, checkpoint_path)

print("="*100)
print(f"Model saved to: {checkpoint_path}")
print("="*100)
