"""
Train VAE v2.6.10: Expanded dataset with predicted RGB.

Novel approach: Combine real RGB (296 BH) + predicted RGB (228 BH)
- Total: 523 boreholes (+77% vs v2.6.7)
- RGB prediction: R²=0.72 using CatBoost on GRA+MS+NGR
- Composition: 60.3% real RGB, 39.7% predicted RGB

Same architecture as v2.6.7:
- 10D latent space, [32, 16] hidden layers
- β annealing: 1e-10 → 0.75 over 50 epochs
- Distribution-aware scaling

Hypothesis: More data + slight RGB noise → better or similar clustering
Baseline: v2.6.7 ARI = 0.196 ± 0.037
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class VAE(nn.Module):
    """v2.6.10 architecture: Same as v2.6.7 (10D latent space)"""
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
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

def train_vae_all_data(model, train_loader, epochs, device, beta_start=1e-10, beta_end=0.75, anneal_epochs=50):
    """Train VAE on all data with β annealing"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("Starting training on ALL data...")
    print(f"β schedule: {beta_start} → {beta_end} over {anneal_epochs} epochs")
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
            loss, recon_loss, kl_div = model.loss_function(recon_x, batch_x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kl += kl_div.item()

        epoch_loss /= len(train_loader)
        epoch_recon /= len(train_loader)
        epoch_kl /= len(train_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: Loss={epoch_loss:.4f}, Recon={epoch_recon:.4f}, KL={epoch_kl:.4f}, β={beta:.6f}")

    print()
    print("Training complete!")
    return model

print("="*100)
print("TRAINING VAE v2.6.10: EXPANDED DATASET WITH PREDICTED RGB")
print("="*100)
print()

# Load expanded dataset
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_6_10.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X_all = df[feature_cols].values

print(f"Dataset composition:")
print(f"  Total samples: {len(df):,}")
print(f"  Total boreholes: {df['Borehole_ID'].nunique()}")
print(f"  Unique lithologies: {df['Lithology_Group'].nunique()}")
print()
print(f"RGB breakdown:")
real_rgb_count = (df['RGB_Source'] == 'real').sum()
pred_rgb_count = (df['RGB_Source'] == 'predicted').sum()
print(f"  Real RGB:      {real_rgb_count:>8,} samples ({real_rgb_count/len(df)*100:5.1f}%)")
print(f"  Predicted RGB: {pred_rgb_count:>8,} samples ({pred_rgb_count/len(df)*100:5.1f}%)")
print()
print(f"Comparison to v2.6.7 baseline:")
print(f"  v2.6.7 samples:     238,506")
print(f"  v2.6.10 samples:    {len(df):,} (+{(len(df)-238506)/238506*100:.0f}%)")
print(f"  v2.6.7 boreholes:   296")
print(f"  v2.6.10 boreholes:  {df['Borehole_ID'].nunique()} (+{(df['Borehole_ID'].nunique()-296)/296*100:.0f}%)")
print()

# Scale
scaler = DistributionAwareScaler()
X_all_scaled = scaler.fit_transform(X_all)

# DataLoader
train_dataset = TensorDataset(torch.FloatTensor(X_all_scaled), torch.zeros(len(X_all_scaled)))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Initialize model
model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print()

# Train
start_time = time.time()
model = train_vae_all_data(model, train_loader, epochs=100, device=device,
                           beta_start=1e-10, beta_end=0.75, anneal_epochs=50)
train_time = time.time() - start_time

print(f"Total training time: {train_time:.1f}s ({train_time/60:.1f} min)")
print()

# Save model
checkpoint_path = '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_6_10.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'latent_dim': 10,
    'hidden_dims': [32, 16],
    'input_dim': 6,
    'scaler': scaler,
    'beta_schedule': {
        'beta_start': 1e-10,
        'beta_end': 0.75,
        'anneal_epochs': 50
    },
    'training_samples': len(df),
    'rgb_breakdown': {
        'real_rgb_samples': real_rgb_count,
        'predicted_rgb_samples': pred_rgb_count,
        'real_rgb_pct': real_rgb_count/len(df),
        'predicted_rgb_pct': pred_rgb_count/len(df)
    }
}, checkpoint_path)

print(f"✓ Model saved to: {checkpoint_path}")
print()

# Extract latent representations
print("="*100)
print("LATENT SPACE ANALYSIS")
print("="*100)
model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_all_scaled).to(device)
    mu, logvar = model.encode(X_tensor)
    latent = mu.cpu().numpy()

latent_stds = latent.std(axis=0)
collapsed_dims = (latent_stds < 0.1).sum()
effective_dim = (latent_stds >= 0.1).sum()

print(f"Latent space dimensionality:")
print(f"  Collapsed dims: {collapsed_dims}/10")
print(f"  Effective dims: {effective_dim}")
print()
print(f"Per-dimension std devs:")
for i, std in enumerate(latent_stds):
    status = "✓" if std >= 0.1 else "✗"
    print(f"  Dim {i}: {std:.4f} {status}")
print()

# Clustering evaluation
print("="*100)
print("CLUSTERING PERFORMANCE (GMM)")
print("="*100)

# Map lithology to labels
y_true = df['Lithology_Group'].values
lithology_to_idx = {lith: i for i, lith in enumerate(np.unique(y_true))}
y_true_encoded = np.array([lithology_to_idx[lith] for lith in y_true])

k_values = [10, 12, 15, 20]
results = []

for k in k_values:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    y_pred = gmm.fit_predict(latent)

    ari = adjusted_rand_score(y_true_encoded, y_pred)
    sil = silhouette_score(latent, y_pred, sample_size=min(10000, len(latent)))

    results.append({
        'k': k,
        'ARI': ari,
        'Silhouette': sil
    })

    print(f"k={k:2d}: ARI = {ari:.4f}, Silhouette = {sil:.4f}")

print()

# Average performance
avg_ari = np.mean([r['ARI'] for r in results])
avg_sil = np.mean([r['Silhouette'] for r in results])
print(f"Average: ARI = {avg_ari:.4f}, Silhouette = {avg_sil:.4f}")
print()

# Comparison to v2.6.7
v2_6_7_ari = 0.196
print(f"Comparison to v2.6.7 baseline:")
print(f"  v2.6.7 ARI:  {v2_6_7_ari:.4f} (238,506 samples, 296 BH, 100% real RGB)")
print(f"  v2.6.10 ARI: {avg_ari:.4f} ({len(df):,} samples, {df['Borehole_ID'].nunique()} BH, 60% real RGB)")
print()

if avg_ari >= v2_6_7_ari * 0.95:
    print("✓ SUCCESS: v2.6.10 maintains performance with +77% boreholes!")
    print("  More data + predicted RGB noise is a worthwhile trade-off.")
elif avg_ari >= v2_6_7_ari * 0.85:
    print("⚠ MODERATE: v2.6.10 slight degradation but more data coverage")
    print("  Trade-off: -10-15% ARI for +77% boreholes")
else:
    print("✗ FAILURE: v2.6.10 significant performance drop")
    print("  Predicted RGB noise degrades clustering too much")
print()

# Save results
results_df = pd.DataFrame(results)
results_df['v2_6_7_baseline'] = v2_6_7_ari
results_df['improvement'] = (results_df['ARI'] - v2_6_7_ari) / v2_6_7_ari * 100
results_df.to_csv('/home/utig5/johna/bhai/vae_v2_6_10_clustering_results.csv', index=False)
print(f"✓ Results saved to: vae_v2_6_10_clustering_results.csv")
print()

print("="*100)
print("v2.6.10 TRAINING COMPLETE")
print("="*100)
print(f"Checkpoint: {checkpoint_path}")
print(f"Training samples: {len(df):,}")
print(f"Performance: ARI = {avg_ari:.4f}")
print(f"Latent dims: {effective_dim}/10 active")
print()
print("Novel approach: First VAE model using predicted RGB features (60% real, 40% predicted)")
print("="*100)
