"""
Test β annealing: Start with low β, gradually increase to target.

Hypothesis: Starting with low β helps model learn good reconstructions first,
then gradually adding regularization may find better solutions than fixed β.

Annealing schedules to test:
1. Linear: β=0.01 → 0.5 over first 50% of epochs
2. Fixed β=0.5 (baseline for comparison)
3. Linear: β=0.001 → 0.5 over first 50% of epochs
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

def train_vae_with_annealing(model, train_loader, val_loader, epochs=100, device='cpu',
                              beta_start=0.01, beta_end=0.5, anneal_epochs=50):
    """
    Train VAE with β annealing.

    Args:
        beta_start: Initial β value
        beta_end: Final β value
        anneal_epochs: Number of epochs to anneal over (then stays at beta_end)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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
            # Linear annealing
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

        # Print progress every 5 epochs
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

print("="*80)
print("β ANNEALING EXPERIMENTS")
print("="*80)
print()

# Load and prepare data
print("Loading data...", flush=True)
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

print(f"Train: {len(train_boreholes)} boreholes, {len(df_train):,} samples", flush=True)
print(f"Test:  {len(test_boreholes)} boreholes, {len(df_test):,} samples", flush=True)
print()

# Prepare features
X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

# Scale features
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Create data loaders
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(X_train_scaled)
)
val_dataset = TensorDataset(
    torch.FloatTensor(X_val_scaled),
    torch.FloatTensor(X_val_scaled)
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Test different annealing schedules
schedules = [
    {'name': 'Fixed β=0.5', 'beta_start': 0.5, 'beta_end': 0.5, 'anneal_epochs': 0},
    {'name': 'Anneal 0.01→0.5 (50 epochs)', 'beta_start': 0.01, 'beta_end': 0.5, 'anneal_epochs': 50},
    {'name': 'Anneal 0.001→0.5 (50 epochs)', 'beta_start': 0.001, 'beta_end': 0.5, 'anneal_epochs': 50},
    {'name': 'Anneal 0.01→0.5 (25 epochs)', 'beta_start': 0.01, 'beta_end': 0.5, 'anneal_epochs': 25},
]

results = []

for schedule in schedules:
    print("="*80)
    print(f"Testing: {schedule['name']}")
    print("="*80)
    print()

    start_time = time.time()

    # Create and train model
    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
    model.to(device)

    model, history = train_vae_with_annealing(
        model, train_loader, val_loader,
        epochs=100, device=device,
        beta_start=schedule['beta_start'],
        beta_end=schedule['beta_end'],
        anneal_epochs=schedule['anneal_epochs']
    )

    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.1f}s", flush=True)
    print(f"Final β: {history['beta_schedule'][-1]:.4f}", flush=True)
    print(f"Final train loss: {history['train_loss'][-1]:.4f}", flush=True)
    print()

    # Evaluate on test set
    print("Evaluating on test set...", flush=True)
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    # Cluster and evaluate
    schedule_results = {
        'name': schedule['name'],
        'beta_start': schedule['beta_start'],
        'beta_end': schedule['beta_end'],
        'anneal_epochs': schedule['anneal_epochs'],
        'train_time': train_time,
        'final_train_loss': history['train_loss'][-1],
        'epochs_trained': len(history['train_loss'])
    }

    for k in [10, 12, 15, 20]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent)
        ari = adjusted_rand_score(y_test, labels)
        schedule_results[f'ari_k{k}'] = ari

    results.append(schedule_results)

    print(f"Results:")
    print(f"  k=10: ARI={schedule_results['ari_k10']:.3f}")
    print(f"  k=12: ARI={schedule_results['ari_k12']:.3f}")
    print(f"  k=15: ARI={schedule_results['ari_k15']:.3f}")
    print(f"  k=20: ARI={schedule_results['ari_k20']:.3f}")
    print()

    # Save model
    checkpoint_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_5_annealing_{schedule["name"].replace(" ", "_").replace("→", "to")}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'schedule': schedule,
        'history': history
    }, checkpoint_path)
    print(f"Saved: {checkpoint_path}", flush=True)
    print()

# Summary
print("="*80)
print("SUMMARY: β ANNEALING RESULTS")
print("="*80)
print()

df_results = pd.DataFrame(results)
print(df_results[['name', 'ari_k10', 'ari_k12', 'ari_k15', 'ari_k20']].to_string(index=False))
print()

# Compute average ARI
df_results['avg_ari'] = df_results[['ari_k10', 'ari_k12', 'ari_k15', 'ari_k20']].mean(axis=1)

print("Average ARI across k values:")
for _, row in df_results.iterrows():
    print(f"  {row['name']:40s}: {row['avg_ari']:.3f}")

print()

# Best schedule
best_idx = df_results['avg_ari'].idxmax()
best_schedule = df_results.loc[best_idx]

print("="*80)
print("BEST SCHEDULE")
print("="*80)
print()
print(f"Schedule: {best_schedule['name']}")
print(f"Average ARI: {best_schedule['avg_ari']:.3f}")
print(f"Training time: {best_schedule['train_time']:.1f}s")
print()

baseline_idx = df_results[df_results['name'] == 'Fixed β=0.5'].index[0]
baseline_ari = df_results.loc[baseline_idx, 'avg_ari']
improvement = (best_schedule['avg_ari'] - baseline_ari) / baseline_ari * 100

if best_schedule['name'] != 'Fixed β=0.5':
    print(f"Improvement over fixed β=0.5: {improvement:+.1f}%")
else:
    print("Fixed β=0.5 remains best schedule")

print()
print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("β annealing may help if:")
print("  - It improves average ARI over fixed β")
print("  - Training is more stable (fewer early stops)")
print("  - Final loss is lower")
print()
print("Otherwise, fixed β=0.5 is simpler and equally effective.")
print("="*80)

# Save results
df_results.to_csv('beta_annealing_results.csv', index=False)
print("\nResults saved to: beta_annealing_results.csv")
