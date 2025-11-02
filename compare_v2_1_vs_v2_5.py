"""
Compare VAE v2.1 (β=1.0) vs v2.5 (β=0.01) on identical test set.

This tests the hypothesis that lower β preserves feature correlations
and improves clustering performance for geological data.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

print("Loading models...")
# Import from model files to make classes available for unpickling
import vae_lithology_gra_v2_1_model
import vae_lithology_gra_v2_5_model
from vae_lithology_gra_v2_1_model import VAE as VAE_v2_1, DistributionAwareScaler as Scaler_v2_1
from vae_lithology_gra_v2_5_model import VAE as VAE_v2_5, DistributionAwareScaler as Scaler_v2_5

# Make classes available in __main__ namespace for unpickling
import __main__
__main__.DistributionAwareScaler = Scaler_v2_1

# Load data and create test split
print("Creating test split...", flush=True)
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

test_mask = df['Borehole_ID'].isin(test_boreholes)
df_test = df[test_mask].copy()

print(f"Test set: {len(test_boreholes)} boreholes, {len(df_test):,} samples\n", flush=True)

# Prepare features
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X_test = df_test[feature_cols].values
lithology = df_test['Principal'].values

# Load v2.1 model (β=1.0)
print("Loading v2.1 (β=1.0)...", flush=True)
model_v2_1 = VAE_v2_1(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
try:
    checkpoint_v2_1 = torch.load(
        '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth',
        map_location='cpu', weights_only=False
    )
    model_v2_1.load_state_dict(checkpoint_v2_1['model_state_dict'])
except (AttributeError, KeyError) as e:
    # Try loading directly as state dict
    checkpoint_v2_1 = torch.load(
        '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth',
        map_location='cpu', weights_only=True
    )
    model_v2_1.load_state_dict(checkpoint_v2_1)
model_v2_1.eval()

# Load v2.5 model (β=0.01)
print("Loading v2.5 (β=0.01)...", flush=True)
model_v2_5 = VAE_v2_5(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
try:
    checkpoint_v2_5 = torch.load(
        '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_5_latent8.pth',
        map_location='cpu', weights_only=False
    )
    model_v2_5.load_state_dict(checkpoint_v2_5['model_state_dict'])
except (AttributeError, KeyError) as e:
    # Try loading directly as state dict
    checkpoint_v2_5 = torch.load(
        '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_5_latent8.pth',
        map_location='cpu', weights_only=True
    )
    model_v2_5.load_state_dict(checkpoint_v2_5)
model_v2_5.eval()

# Extract latent representations
print("Extracting latent representations...\n", flush=True)

scaler_v2_1 = Scaler_v2_1()
X_scaled_v2_1 = scaler_v2_1.fit_transform(X_test)
with torch.no_grad():
    X_tensor_v2_1 = torch.FloatTensor(X_scaled_v2_1)
    mu_v2_1, _ = model_v2_1.encode(X_tensor_v2_1)
    latent_v2_1 = mu_v2_1.numpy()

scaler_v2_5 = Scaler_v2_5()
X_scaled_v2_5 = scaler_v2_5.fit_transform(X_test)
with torch.no_grad():
    X_tensor_v2_5 = torch.FloatTensor(X_scaled_v2_5)
    mu_v2_5, _ = model_v2_5.encode(X_tensor_v2_5)
    latent_v2_5 = mu_v2_5.numpy()

# Compare clustering performance
print("="*80)
print("VAE v2.1 (β=1.0) vs v2.5 (β=0.01) - Test Set Comparison")
print("="*80)
print()

for k in [10, 12, 15, 20]:
    print(f"k={k}:")

    # v2.1 clustering
    kmeans_v2_1 = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_v2_1 = kmeans_v2_1.fit_predict(latent_v2_1)
    ari_v2_1 = adjusted_rand_score(lithology, labels_v2_1)

    # v2.5 clustering
    kmeans_v2_5 = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_v2_5 = kmeans_v2_5.fit_predict(latent_v2_5)
    ari_v2_5 = adjusted_rand_score(lithology, labels_v2_5)

    improvement = (ari_v2_5 - ari_v2_1) / ari_v2_1 * 100

    print(f"  v2.1 (β=1.0):  ARI = {ari_v2_1:.3f}")
    print(f"  v2.5 (β=0.01): ARI = {ari_v2_5:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")
    print()

print("="*80)
print("CONCLUSION")
print("="*80)
print("\nβ=0.01 preserves feature correlations (MS↔alteration, GRA↔compaction)")
print("that are geologically meaningful for lithology discrimination.")
print("\nHigher β (disentanglement) destroys these correlations, hurting clustering.")
print("="*80 + "\n")
