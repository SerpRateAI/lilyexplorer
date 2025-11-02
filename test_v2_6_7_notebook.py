"""
Test v2.6.7 notebook cells to catch loading errors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
import sys

# Try to import umap, but continue without it if unavailable
try:
    import umap
    UMAP_AVAILABLE = True
except (ImportError, SystemError) as e:
    print(f"Warning: UMAP not available ({e})")
    print("Will skip UMAP projection but test PyTorch loading...")
    UMAP_AVAILABLE = False

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 80)
print("VAE v2.6.7 NOTEBOOK TEST")
print("=" * 80)

# VAE Architecture
class VAE(nn.Module):
    """v2.6.7 architecture: 10D latent space"""
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

print("\n✓ VAE architecture defined")

# Load data
print("\nLoading data...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

# Use same split as original
unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)

test_mask = df['Borehole_ID'].isin(test_boreholes)
df_test = df[test_mask].copy()

X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

print(f"✓ Test set: {len(df_test):,} samples, {len(np.unique(y_test))} unique lithologies")

# Load checkpoint (THIS IS WHERE ERRORS USUALLY OCCUR)
print("\nLoading checkpoint...")
try:
    checkpoint = torch.load('ml_models/checkpoints/vae_gra_v2_6_7_final.pth',
                           map_location='cpu', weights_only=False)
    print("✓ Checkpoint loaded successfully")
except Exception as e:
    print(f"✗ LOADING ERROR: {e}")
    raise

# Initialize model
print("\nInitializing model...")
try:
    model = VAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16])
    model.load_state_dict(checkpoint['model_state_dict'])
    scaler = checkpoint['scaler']
    print("✓ Model state loaded successfully")
except Exception as e:
    print(f"✗ STATE DICT ERROR: {e}")
    raise

# Scale test data
print("\nScaling data...")
try:
    X_test_scaled = scaler.transform(X_test)
    print("✓ Data scaled successfully")
except Exception as e:
    print(f"✗ SCALING ERROR: {e}")
    raise

# Move to device and evaluate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Extract latent representations
print("\nExtracting latent representations...")
try:
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        mu, logvar = model.encode(X_tensor)
        latent = mu.cpu().numpy()
    print("✓ Latent extraction successful")
except Exception as e:
    print(f"✗ INFERENCE ERROR: {e}")
    raise

print(f"\nLatent shape: {latent.shape}")
print(f"Device: {device}")
print(f"\nModel metadata:")
print(f"  Training samples: {checkpoint['training_samples']:,}")
print(f"  β schedule: {checkpoint['beta_schedule']['beta_start']} → {checkpoint['beta_schedule']['beta_end']} over {checkpoint['beta_schedule']['anneal_epochs']} epochs")
print(f"  CV performance: ARI = {checkpoint['cv_performance']['mean_ari']:.3f} ± {checkpoint['cv_performance']['std_ari']:.3f}")

# Latent space statistics
print("\n" + "=" * 80)
print("LATENT SPACE STATISTICS")
print("=" * 80)

latent_stds = latent.std(axis=0)
collapsed_dims = (latent_stds < 0.1).sum()
effective_dim = (latent_stds >= 0.1).sum()

stats_data = []
for i in range(latent.shape[1]):
    dim_data = latent[:, i]
    stats_data.append({
        'Dimension': i+1,
        'Mean': dim_data.mean(),
        'Std': dim_data.std(),
        'Min': dim_data.min(),
        'Max': dim_data.max(),
        'Range': dim_data.max() - dim_data.min(),
        'Skewness': stats.skew(dim_data),
        'Kurtosis': stats.kurtosis(dim_data)
    })

df_stats = pd.DataFrame(stats_data)
print(df_stats.to_string(index=False))
print(f"\nCollapsed dimensions (std<0.1): {collapsed_dims}/10")
print(f"Effective dimensionality: {effective_dim}")

# UMAP projection
print("\n" + "=" * 80)
print("UMAP PROJECTION")
print("=" * 80)

if UMAP_AVAILABLE:
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    latent_2d = reducer.fit_transform(latent)
    print(f"✓ UMAP projection complete: {latent_2d.shape}")

    # Create UMAP plot
    lithology_counts = pd.Series(y_test).value_counts()
    top_10_lithologies = lithology_counts.head(10).index.tolist()

    colors = plt.cm.tab20(np.linspace(0, 1, 10))
    lithology_colors = {lith: colors[i] for i, lith in enumerate(top_10_lithologies)}

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    for lith in top_10_lithologies:
        mask = y_test == lith
        ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1],
                   c=[lithology_colors[lith]], s=1, alpha=0.6, rasterized=True)

    for lith in top_10_lithologies:
        mask = y_test == lith
        if mask.sum() > 0:
            centroid_x = latent_2d[mask, 0].mean()
            centroid_y = latent_2d[mask, 1].mean()
            ax.text(centroid_x, centroid_y, lith,
                    fontsize=9, weight='bold',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                             edgecolor=lithology_colors[lith], linewidth=3))

    ax.set_xlabel('UMAP 1', fontsize=12)
    ax.set_ylabel('UMAP 2', fontsize=12)
    ax.set_title('v2.6.7 Latent Space by Lithology (Top 10)\\nUMAP projection of 10D latent space\\nβ: 1e-10→0.75, ARI=0.196±0.037', fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('v2_6_7_umap_test.png', dpi=150, bbox_inches='tight')
    print("✓ UMAP plot saved to: v2_6_7_umap_test.png")
else:
    print("⚠ UMAP not available - skipping projection")

print("\n" + "=" * 80)
print("✓ ALL TESTS PASSED - NOTEBOOK SHOULD WORK!")
print("=" * 80)
