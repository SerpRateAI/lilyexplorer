"""Quick evaluation of v2.1 with simplified labels - get ARI numbers fast."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

print("Importing VAE classes...", flush=True)
from vae_lithology_gra_v2_1_model import VAE, DistributionAwareScaler

print("Loading VAE v2.1 model...", flush=True)
model_path = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_1_latent8.pth')
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
print("Checkpoint loaded.", flush=True)

print("Creating model...", flush=True)
model = VAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model ready.", flush=True)

print("Loading data with simplified labels...", flush=True)
t0 = time.time()
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm_simplified.csv')
print(f"Data loaded in {time.time()-t0:.1f}s", flush=True)

# Use FULL dataset
print(f"Using FULL dataset: {len(df):,} samples", flush=True)

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

print("Extracting features...", flush=True)
X = df[feature_cols].values
lithology_simp = df['Lithology_Simplified'].values
lithology_orig = df['Principal'].values

print("Scaling features...", flush=True)
t0 = time.time()
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X)
print(f"Scaled in {time.time()-t0:.1f}s", flush=True)

print("Extracting latent representations...", flush=True)
t0 = time.time()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X_scaled)
    mu, _ = model.encode(X_tensor)
    latent = mu.numpy()
print(f"Latent extraction done in {time.time()-t0:.1f}s", flush=True)

print("\n" + "="*70)
print("VAE v2.1 Performance Comparison")
print("="*70)

for k in [10, 12, 15]:
    print(f"\nRunning k={k}...", flush=True)
    t0 = time.time()

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(latent)
    print(f"  K-Means done in {time.time()-t0:.1f}s", flush=True)

    # Skip Silhouette (too slow for 238K samples - O(n²))
    ari_simp = adjusted_rand_score(lithology_simp, labels)
    ari_orig = adjusted_rand_score(lithology_orig, labels)

    improvement = (ari_simp - ari_orig) / ari_orig * 100

    print(f"\nk={k:2d}:", flush=True)
    print(f"  Simplified (12 types): ARI={ari_simp:.3f}", flush=True)
    print(f"  Original (139 types):  ARI={ari_orig:.3f}", flush=True)
    print(f"  Improvement: {improvement:+.1f}%", flush=True)

print("\n" + "="*70)
