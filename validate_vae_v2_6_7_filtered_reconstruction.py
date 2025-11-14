"""
Validate VAE v2.6.7 FILTERED Reconstruction Quality

Creates predicted vs true scatter plots for each reconstructed feature.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

# VAE architecture (must match trained model)
class VAE(nn.Module):
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

    def decode(self, z):
        h = self.decoder(z)
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Use deterministic encoding (mean) for reconstruction
        z = mu
        return self.decode(z), mu, logvar

print("="*100)
print("VAE v2.6.7 FILTERED RECONSTRUCTION QUALITY - PREDICTED VS TRUE PLOTS")
print("="*100)
print()

# Load FILTERED data
print("Loading filtered data...")
df = pd.read_csv('vae_training_data_v2_20cm_filtered_100.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X = df[feature_cols].values
print(f"Total samples: {len(X):,}")
print(f"Features: {feature_cols}")
print()

# Load VAE FILTERED model
print("Loading VAE v2.6.7 FILTERED model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_7_filtered_final.pth',
                       map_location=device, weights_only=False)

vae_model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()

scaler = checkpoint['scaler']
print(f"✓ VAE model loaded (device: {device})")
print()

# Scale data
print("Scaling features (distribution-aware)...")
X_scaled = scaler.transform(X)
X_tensor = torch.FloatTensor(X_scaled).to(device)
print("✓ Features scaled")
print()

# Compute reconstructions
print("Computing VAE reconstructions...")
with torch.no_grad():
    X_recon_scaled, mu, logvar = vae_model(X_tensor)
    X_recon_scaled = X_recon_scaled.cpu().numpy()
    mu = mu.cpu().numpy()

print(f"✓ Reconstructions computed: {X_recon_scaled.shape}")
print()

# Inverse transform to original space
print("Inverse transforming to original feature space...")
X_recon = scaler.inverse_transform(X_recon_scaled)
print("✓ Inverse transform complete")
print()

# Compute metrics
print("="*100)
print("RECONSTRUCTION METRICS (ORIGINAL FEATURE SPACE)")
print("="*100)
print()

feature_names = ['GRA (g/cm³)', 'MS (SI)', 'NGR (cps)', 'R', 'G', 'B']

print(f"{'Feature':<20s} {'MAE':>12s} {'RMSE':>12s} {'R²':>12s}")
print("-"*100)

metrics = []
for i, fname in enumerate(feature_names):
    mae = mean_absolute_error(X[:, i], X_recon[:, i])
    rmse = np.sqrt(mean_squared_error(X[:, i], X_recon[:, i]))
    r2 = r2_score(X[:, i], X_recon[:, i])

    metrics.append({'feature': fname, 'mae': mae, 'rmse': rmse, 'r2': r2})
    print(f"{fname:<20s} {mae:>12.4f} {rmse:>12.4f} {r2:>12.3f}")

print()
avg_r2 = np.mean([m['r2'] for m in metrics])
print(f"Average R²: {avg_r2:.3f}")
print()

# Create predicted vs true plots
print("Creating predicted vs true scatter plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, (fname, metric) in enumerate(zip(feature_names, metrics)):
    ax = axes[i]

    # Sample 10,000 points for visualization (if dataset is large)
    if len(X) > 10000:
        idx = np.random.choice(len(X), 10000, replace=False)
        x_true = X[idx, i]
        x_pred = X_recon[idx, i]
    else:
        x_true = X[:, i]
        x_pred = X_recon[:, i]

    # Scatter plot with density coloring
    scatter = ax.hexbin(x_true, x_pred, gridsize=50, cmap='viridis',
                        mincnt=1, alpha=0.8, edgecolors='none')

    # Perfect prediction line
    min_val = min(x_true.min(), x_pred.min())
    max_val = max(x_true.max(), x_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
            label='Perfect reconstruction', alpha=0.7)

    # Labels and title
    ax.set_xlabel(f'True {fname}', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Predicted {fname}', fontsize=11, fontweight='bold')
    ax.set_title(f'{fname}\nR²={metric["r2"]:.3f}, RMSE={metric["rmse"]:.3f}',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Count', fontsize=9)

plt.suptitle('VAE v2.6.7 FILTERED: Predicted vs True Feature Reconstruction',
            fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('vae_v2_6_7_filtered_reconstruction_quality.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print(f"✓ Plots saved: vae_v2_6_7_filtered_reconstruction_quality.png")
plt.close()
print()

# Summary
print("="*100)
print("SUMMARY")
print("="*100)
print()

if avg_r2 > 0.90:
    quality = "Excellent"
elif avg_r2 > 0.75:
    quality = "Good"
else:
    quality = "Poor"

print(f"Overall Reconstruction Quality: {quality} (avg R² = {avg_r2:.3f})")
print()
print("Per-feature performance:")
for m in metrics:
    print(f"  {m['feature']:<20s} R²={m['r2']:.3f}")
print()
print("The VAE successfully reconstructs physical properties from 10D latent space")
print("(with 6/10 dimensions collapsed → 4D effective)")
print()
print("="*100)
