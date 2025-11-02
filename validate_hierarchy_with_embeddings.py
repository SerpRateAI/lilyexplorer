"""
Validate lithology hierarchy by checking if groups cluster together in VAE embedding space.

Compares:
1. Expert-defined hierarchy (keyword-based)
2. Data-driven clustering (VAE embeddings of lithology centroids)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

# VAE architecture (full model needed for loading checkpoint)
class VAE(nn.Module):
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

        # Decoder (needed for checkpoint loading)
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

print("="*100)
print("VALIDATING LITHOLOGY HIERARCHY WITH VAE EMBEDDINGS")
print("="*100)
print()

# Load data
df = pd.read_csv('vae_training_data_v2_20cm.csv')
hierarchy_df = pd.read_csv('lithology_hierarchy_mapping.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

X = df[feature_cols].values
y = df['Principal'].values

# Load VAE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                       map_location=device, weights_only=False)

vae_model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16]).to(device)
vae_model.load_state_dict(checkpoint['model_state_dict'])
vae_model.eval()

scaler = checkpoint['scaler']

# Scale and extract embeddings
X_scaled = scaler.transform(X)
X_tensor = torch.FloatTensor(X_scaled).to(device)

print("Extracting VAE embeddings...")
with torch.no_grad():
    mu, _ = vae_model.encode(X_tensor)
    embeddings = mu.cpu().numpy()

print(f"✓ Embeddings extracted: {embeddings.shape}")
print()

# Compute lithology centroids (median embedding per lithology)
print("Computing lithology centroids...")
unique_lithologies = np.unique(y)
lithology_centroids = {}

for lith in unique_lithologies:
    mask = y == lith
    lith_embeddings = embeddings[mask]
    centroid = np.median(lith_embeddings, axis=0)  # Use median for robustness
    lithology_centroids[lith] = centroid

centroids_array = np.array([lithology_centroids[lith] for lith in unique_lithologies])
print(f"✓ Computed {len(unique_lithologies)} lithology centroids")
print()

# Get expert-defined groups
principal_to_group = dict(zip(hierarchy_df['Principal_Lithology'],
                             hierarchy_df['Lithology_Group']))

expert_labels = np.array([principal_to_group.get(lith, 'Unknown') for lith in unique_lithologies])
unique_expert_groups = np.unique(expert_labels)

print(f"Expert-defined groups: {len(unique_expert_groups)}")
print()

# Hierarchical clustering on centroids
print("Performing hierarchical clustering on lithology centroids...")
n_clusters = len(unique_expert_groups)  # Same number as expert groups

clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
data_driven_labels = clusterer.fit_predict(centroids_array)

print(f"✓ Clustered into {n_clusters} data-driven groups")
print()

# Compare expert vs data-driven
print("="*100)
print("COMPARISON: EXPERT-DEFINED vs DATA-DRIVEN CLUSTERING")
print("="*100)
print()

# Convert expert labels to integers for ARI
expert_label_to_idx = {label: idx for idx, label in enumerate(unique_expert_groups)}
expert_labels_int = np.array([expert_label_to_idx[label] for label in expert_labels])

ari = adjusted_rand_score(expert_labels_int, data_driven_labels)
print(f"Adjusted Rand Index (expert vs data-driven): {ari:.4f}")
print()

if ari > 0.5:
    print("✓ Strong agreement - expert groups align well with VAE embedding clusters")
elif ari > 0.2:
    print("⚠ Moderate agreement - some expert groups make sense, others don't cluster")
else:
    print("✗ Poor agreement - expert groups don't match natural VAE clustering")

print()

# Show which expert groups are coherent (low intra-group variance)
print("="*100)
print("EXPERT GROUP COHERENCE (intra-group variance in embedding space)")
print("="*100)
print()

group_variances = []
for group in unique_expert_groups:
    group_lithologies = [lith for lith, g in zip(unique_lithologies, expert_labels) if g == group]
    if len(group_lithologies) > 1:
        group_centroids = np.array([lithology_centroids[lith] for lith in group_lithologies])
        # Compute average pairwise distance
        distances = []
        for i in range(len(group_centroids)):
            for j in range(i+1, len(group_centroids)):
                dist = np.linalg.norm(group_centroids[i] - group_centroids[j])
                distances.append(dist)
        avg_dist = np.mean(distances) if distances else 0
        group_variances.append((group, len(group_lithologies), avg_dist))

group_variances.sort(key=lambda x: x[2])  # Sort by distance

print(f"{'Group':<30s} {'N_Lithologies':>15s} {'Avg_Distance':>15s} {'Coherence':>12s}")
print("-"*100)

for group, n_lith, avg_dist in group_variances:
    if avg_dist < 1.0:
        coherence = "✓ Tight"
    elif avg_dist < 2.0:
        coherence = "○ Moderate"
    else:
        coherence = "✗ Loose"

    print(f"{group:<30s} {n_lith:>15d} {avg_dist:>15.4f} {coherence:>12s}")

print()
print("="*100)
print("CONCLUSION")
print("="*100)
print()

if ari > 0.5:
    print("Expert hierarchy is well-aligned with VAE embeddings. Current groupings are good.")
elif ari > 0.2:
    print("Expert hierarchy is partially aligned. Consider revising loose groups.")
else:
    print("Expert hierarchy poorly aligned with VAE embeddings.")
    print("RECOMMENDATION: Use data-driven clustering instead of keyword matching.")

print()
print("="*100)
