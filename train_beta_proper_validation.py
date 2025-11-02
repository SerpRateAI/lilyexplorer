"""
Proper β hyperparameter tuning using VALIDATION set.

Workflow:
1. Train models with different β on TRAIN set
2. Evaluate on VALIDATION set to select best β
3. Report final performance on TEST set (unbiased estimate)
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

from vae_lithology_gra_v2_5_model import VAE, DistributionAwareScaler, train_vae
from torch.utils.data import DataLoader, TensorDataset

# Beta values to test
BETAS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
EPOCHS = 50

print("="*80)
print("PROPER β HYPERPARAMETER TUNING (Using Validation Set)")
print("="*80)
print()

# Load and prepare data
print("Loading data...", flush=True)
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

# Create same splits
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
print(f"Val:   {len(val_boreholes)} boreholes, {len(df_val):,} samples", flush=True)
print(f"Test:  {len(test_boreholes)} boreholes, {len(df_test):,} samples", flush=True)
print()

# Prepare features
X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values

y_val = df_val['Principal'].values
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

# Train models for each β
val_results = []

print("="*80)
print("STEP 1: Train models and evaluate on VALIDATION set")
print("="*80)
print()

for beta in BETAS:
    print(f"Training β={beta}...", flush=True)

    start_time = time.time()

    # Create and train model
    model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model, history = train_vae(
        model, train_loader, val_loader,
        epochs=EPOCHS, device=device, beta=beta
    )

    train_time = time.time() - start_time

    # Evaluate on VALIDATION set
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_val_scaled).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    # Cluster and evaluate on validation set
    beta_result = {'beta': beta, 'train_time': train_time}

    for k in [10, 12, 15, 20]:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent)
        ari = adjusted_rand_score(y_val, labels)
        beta_result[f'val_ari_k{k}'] = ari

    val_results.append(beta_result)

    print(f"  Val ARI: k=10:{beta_result['val_ari_k10']:.3f} | "
          f"k=15:{beta_result['val_ari_k15']:.3f} | "
          f"k=20:{beta_result['val_ari_k20']:.3f} | "
          f"Time: {train_time:.0f}s", flush=True)

    # Save checkpoint
    checkpoint_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_5_beta{beta}_latent8.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'beta': beta,
        'history': history,
        'epochs': EPOCHS
    }, checkpoint_path)

print()

# Select best β based on validation performance
df_val_results = pd.DataFrame(val_results)
df_val_results['avg_val_ari'] = df_val_results[[f'val_ari_k{k}' for k in [10, 12, 15, 20]]].mean(axis=1)

print("="*80)
print("STEP 2: Select best β based on VALIDATION performance")
print("="*80)
print()
print(df_val_results[['beta', 'val_ari_k10', 'val_ari_k15', 'val_ari_k20', 'avg_val_ari']].to_string(index=False))
print()

best_beta = df_val_results.loc[df_val_results['avg_val_ari'].idxmax(), 'beta']
print(f"Selected β: {best_beta} (best average validation ARI)")
print()

# Load best model and evaluate on TEST set (only once!)
print("="*80)
print("STEP 3: Evaluate selected β on TEST set (unbiased estimate)")
print("="*80)
print()

print(f"Loading model with β={best_beta}...", flush=True)
best_model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
checkpoint = torch.load(
    f'/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_5_beta{best_beta}_latent8.pth',
    map_location='cpu', weights_only=True
)
best_model.load_state_dict(checkpoint['model_state_dict'])
best_model.eval()

# Extract latent on test set
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    mu_test, _ = best_model.encode(X_test_tensor)
    latent_test = mu_test.numpy()

# Evaluate on test set
print(f"\nTest set performance (β={best_beta}):")
print("="*80)

test_results = []
for k in [10, 12, 15, 20]:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent_test)
    ari = adjusted_rand_score(y_test, labels)
    test_results.append({'k': k, 'test_ari': ari})
    print(f"k={k}: ARI = {ari:.3f}")

df_test_results = pd.DataFrame(test_results)
avg_test_ari = df_test_results['test_ari'].mean()

print(f"\nAverage test ARI: {avg_test_ari:.3f}")
print()

# Compare to v2.1 baseline
print("="*80)
print("COMPARISON TO BASELINE")
print("="*80)
print()

baseline_ari = {10: 0.192, 12: 0.167, 15: 0.179, 20: 0.166}
baseline_avg = np.mean(list(baseline_ari.values()))

print("v2.1 (β=1.0) - Test Set:")
for k, ari in baseline_ari.items():
    print(f"  k={k}: ARI = {ari:.3f}")
print(f"  Average: {baseline_avg:.3f}")
print()

print(f"v2.5 (β={best_beta}) - Test Set:")
for _, row in df_test_results.iterrows():
    k = int(row['k'])
    ari = row['test_ari']
    improvement = (ari - baseline_ari[k]) / baseline_ari[k] * 100
    print(f"  k={k}: ARI = {ari:.3f} ({improvement:+.1f}% vs v2.1)")
improvement_avg = (avg_test_ari - baseline_avg) / baseline_avg * 100
print(f"  Average: {avg_test_ari:.3f} ({improvement_avg:+.1f}% vs v2.1)")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print(f"Optimal β selected from validation: {best_beta}")
print(f"Unbiased test set performance: ARI = {avg_test_ari:.3f} (avg)")
print(f"Improvement over v2.1 baseline: {improvement_avg:+.1f}%")
print()
print("This is the CORRECT estimate (no test set leakage).")
print("="*80)

# Save results
df_val_results.to_csv('beta_validation_selection.csv', index=False)
df_test_results.to_csv('beta_final_test_performance.csv', index=False)

print("\nResults saved:")
print("  - beta_validation_selection.csv (β selection on validation)")
print("  - beta_final_test_performance.csv (final unbiased test performance)")
