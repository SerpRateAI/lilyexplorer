"""
Evaluate v2.1 on TEST SET (same split as training) with simplified labels.

This gives proper apples-to-apples comparison:
- Original 139 labels: ARI = 0.179 (from training log)
- Simplified 12 labels: ARI = ??? (what we want to find)
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_1_model import VAE, DistributionAwareScaler

print("Loading data and creating TEST split...", flush=True)
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm_simplified.csv')

# Recreate SAME borehole split as training (train=70%, val=15%, test=15%)
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

# Extract TEST set
test_mask = df['Borehole_ID'].isin(test_boreholes)
df_test = df[test_mask].copy()

print(f"Test set: {len(test_boreholes)} boreholes, {len(df_test):,} samples", flush=True)

# Load model
print("Loading model...", flush=True)
model_path = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth')
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare features
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X_test = df_test[feature_cols].values
lithology_simp = df_test['Lithology_Simplified'].values
lithology_orig = df_test['Principal'].values

# Scale and encode
print("Extracting latent representations...", flush=True)
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X_test)

with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    mu, _ = model.encode(X_tensor)
    latent = mu.numpy()

print("\n" + "="*80)
print("VAE v2.1 - TEST SET EVALUATION (apples-to-apples comparison)")
print("="*80)

for k in [10, 12, 15, 20]:
    print(f"\nk={k}:", flush=True)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent)

    ari_simp = adjusted_rand_score(lithology_simp, labels)
    ari_orig = adjusted_rand_score(lithology_orig, labels)

    improvement = (ari_simp - ari_orig) / ari_orig * 100

    print(f"  Simplified (12 types): ARI = {ari_simp:.3f}", flush=True)
    print(f"  Original (139 types):  ARI = {ari_orig:.3f}", flush=True)
    print(f"  Improvement: {improvement:+.1f}%", flush=True)

print("\n" + "="*80)
print("COMPARISON TO TRAINING LOG")
print("="*80)
print("\nFrom v2.1 training log (test set, original 139 labels):")
print("  k=10: ARI = 0.179")
print("  k=15: ARI = 0.166")
print("  k=20: ARI = 0.170")
print("\nAbove results show performance with simplified 12-type labels.")
print("="*80 + "\n")
