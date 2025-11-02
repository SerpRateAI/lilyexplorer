"""
Train VAE v2.6.1 with β annealing on GPU (A100)

Features: GRA + MS + NGR + RSC (L*a*b*) + MSP (7 features)
Architecture: Same as v2.6, scaled for 7 features
Device: GPU (cuda)
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import time

from vae_lithology_gra_v2_6_1_model import VAE, DistributionAwareScaler

def train_vae_with_annealing(model, train_loader, val_loader, epochs=100, device='cuda',
                              beta_start=0.001, beta_end=0.5, anneal_epochs=50, lr=0.001):
    """Train VAE with β annealing."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_recon': [],
        'train_kl': [],
        'beta_schedule': []
    }

    for epoch in range(epochs):
        # Compute current β
        if epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        history['beta_schedule'].append(current_beta)

        # Training
        model.train()
        train_loss = 0
        train_recon = 0
        train_kl = 0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)

            # Compute losses
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)

            loss = recon_loss + current_beta * kl_loss

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
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)

                recon_loss = F.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / data.size(0)

                loss = recon_loss + current_beta * kl_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: β={current_beta:.4f}, Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Recon={train_recon:.4f}, KL={train_kl:.4f}",
                  flush=True)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    return model, history


def evaluate_clustering(model, X_test, y_test, device='cuda'):
    """Evaluate clustering performance."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    results = {}
    for k in [10, 12, 15, 20]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent)
        ari = adjusted_rand_score(y_test, labels)
        sil = silhouette_score(latent, labels)

        results[k] = {'ari': ari, 'silhouette': sil}
        print(f"k={k}: ARI={ari:.3f}, Silhouette={sil:.3f}")

    return results


print("="*80)
print("VAE GRA v2.6.1 - RSC Color + MSP (7 features)")
print("="*80)
print()

# Check GPU availability
if torch.cuda.is_available():
    device = 'cuda'
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = 'cpu'
    print("WARNING: No GPU detected, using CPU")

print()

# Load data
print("Loading data...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_6_1_20cm.csv')

feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'Reflectance L*',
    'Reflectance a*',
    'Reflectance b*',
    'MSP (instr. units)'
]

X = df[feature_cols].values
lithology = df['Principal'].values
borehole_ids = df['Borehole_ID'].values

print(f"Dataset: {len(X):,} samples, {len(np.unique(borehole_ids))} boreholes")
print(f"Features: {len(feature_cols)}")
print()

# Borehole-level split
unique_boreholes = np.unique(borehole_ids)
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

train_mask = np.isin(borehole_ids, train_boreholes)
val_mask = np.isin(borehole_ids, val_boreholes)
test_mask = np.isin(borehole_ids, test_boreholes)

X_train, y_train = X[train_mask], lithology[train_mask]
X_val, y_val = X[val_mask], lithology[val_mask]
X_test, y_test = X[test_mask], lithology[test_mask]

print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")
print()

# Scale features
print("Applying distribution-aware scaling...")
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print("Done")
print()

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(X_train_scaled)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(X_val_scaled)
)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512)

# Train model
print("="*80)
print("Training VAE v2.6.1 with β annealing (0.001→0.5, 50 epochs)")
print("="*80)
print(f"Device: {device}")
print(f"Architecture: 7 → [32, 16] → 8 (latent) → [16, 32] → 7")
print()

model = VAE(input_dim=7, latent_dim=8, hidden_dims=[32, 16])

start_time = time.time()
model, history = train_vae_with_annealing(
    model, train_loader, val_loader,
    epochs=100, device=device,
    beta_start=0.001, beta_end=0.5, anneal_epochs=50,
    lr=0.001
)
train_time = time.time() - start_time

print(f"\nTraining completed in {train_time:.1f}s ({len(history['train_loss'])} epochs)")
print()

# Evaluate clustering
print("="*80)
print("Clustering Evaluation")
print("="*80)
results = evaluate_clustering(model, X_test_scaled, y_test, device=device)
print()

# Save model
save_path = Path('ml_models/checkpoints/vae_gra_v2_6_1_annealing.pth')
save_path.parent.mkdir(parents=True, exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'history': history,
    'results': results,
    'scaler': scaler,
    'train_time': train_time,
    'features': feature_cols,
    'n_samples': len(X),
    'n_boreholes': len(unique_boreholes)
}, save_path)

print(f"Model saved: {save_path}")
print()

# Compare to v2.6
print("="*80)
print("Comparison: v2.6.1 vs v2.6")
print("="*80)
print()
print("v2.6 (GRA+MS+NGR+RGB, 4 features):")
print("  - 238,506 samples, 296 boreholes")
print("  - ARI (k=12) = 0.258")
print()
print(f"v2.6.1 (GRA+MS+NGR+RSC+MSP, 7 features):")
print(f"  - {len(X):,} samples, {len(unique_boreholes)} boreholes")
print(f"  - ARI (k=12) = {results[12]['ari']:.3f}")
print()

sample_improvement = (len(X) - 238506) / 238506 * 100
bh_improvement = (len(unique_boreholes) - 296) / 296 * 100
ari_improvement = (results[12]['ari'] - 0.258) / 0.258 * 100

print(f"Improvements:")
print(f"  Samples: {sample_improvement:+.1f}%")
print(f"  Boreholes: {bh_improvement:+.1f}%")
print(f"  ARI (k=12): {ari_improvement:+.1f}%")
print()

if results[12]['ari'] > 0.258:
    print("✓ v2.6.1 BEATS v2.6!")
else:
    print(f"✗ v2.6.1 is {abs(ari_improvement):.1f}% worse than v2.6")

print("="*80)
