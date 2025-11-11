"""
Train VAE v2.7: v2.6 features + MAD features (porosity, grain density, water content)

9 features: GRA, MS, NGR, RGB (R,G,B), Porosity, Grain Density, Water Content

Goal: Test if discriminative MAD features improve clustering despite -95% sample reduction.

Key hypothesis: Porosity distinguishes basalt/gabbro, grain density identifies carbonates,
water content tracks diagenesis → these should improve classification beyond 42% ceiling.
"""

print("Starting script...")
import torch
print("Torch imported")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import sys
import time

print("="*100)
print("VAE v2.7 TRAINING: v2.6 + MAD FEATURES")
print("="*100)
print()

# Load dataset
print("Loading v2.7 dataset...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_7_20cm.csv')
print(f"Total samples: {len(df):,}")
print(f"Total boreholes: {df['Borehole_ID'].nunique()}")
print(f"Unique lithologies: {df['Principal'].nunique()}")
print()

# Features
feature_cols = ['GRA_bulk_density', 'MS', 'NGR',
                'RGB_R', 'RGB_G', 'RGB_B',
                'Porosity', 'Grain_Density', 'Water_Content']

X = df[feature_cols].values
y = df['Principal'].values
boreholes = df['Borehole_ID'].values

print("Feature summary:")
for feat in feature_cols:
    print(f"  {feat:20s}: mean={X[:, feature_cols.index(feat)].mean():8.3f}, "
          f"std={X[:, feature_cols.index(feat)].std():8.3f}")
print()

# Borehole-level train/val/test split
print("Creating borehole-level train/val/test split...")
unique_boreholes = df['Borehole_ID'].unique()
np.random.seed(42)
np.random.shuffle(unique_boreholes)

n_train = int(0.70 * len(unique_boreholes))
n_val = int(0.15 * len(unique_boreholes))

train_boreholes = unique_boreholes[:n_train]
val_boreholes = unique_boreholes[n_train:n_train+n_val]
test_boreholes = unique_boreholes[n_train+n_val:]

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

print(f"Train: {len(X_train):>5,} samples from {len(train_boreholes):>3d} boreholes")
print(f"Val:   {len(X_val):>5,} samples from {len(val_boreholes):>3d} boreholes")
print(f"Test:  {len(X_test):>5,} samples from {len(test_boreholes):>3d} boreholes")
print()

# Distribution-aware scaling (same strategy as v2.6, adapted for 9 features)
print("Applying distribution-aware scaling...")

class DistributionAwareScaler_v27:
    """Custom scaler for v2.7 9D features."""
    def __init__(self):
        self.scaler = StandardScaler()
        # Feature indices:
        # 0: GRA (Gaussian)
        # 1: MS (Poisson/Bimodal - signed log)
        # 2: NGR (Bimodal - signed log)
        # 3: RGB_R (Log-normal - log)
        # 4: RGB_G (Log-normal - log)
        # 5: RGB_B (Log-normal - log)
        # 6: Porosity (Log-normal but has negatives - signed log)
        # 7: Grain_Density (Gaussian)
        # 8: Water_Content (Log-normal but has negatives - signed log)
        self.signed_log_indices = [1, 2, 6, 8]  # MS, NGR, Porosity, Water Content
        self.log_indices = [3, 4, 5]  # RGB only

    def fit_transform(self, X):
        X_transformed = X.copy()
        # Signed log for MS, NGR, Porosity, Water Content (handles negatives)
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = np.sign(X_transformed[:, idx]) * np.log1p(np.abs(X_transformed[:, idx]))
        # Regular log for RGB (always positive)
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X_transformed[:, idx])
        return self.scaler.fit_transform(X_transformed)

    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = np.sign(X_transformed[:, idx]) * np.log1p(np.abs(X_transformed[:, idx]))
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X_transformed[:, idx])
        return self.scaler.transform(X_transformed)

scaler = DistributionAwareScaler_v27()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Scaled features: {X_train_scaled.shape[1]}D")
print()

# VAE Model (same architecture as v2.6, scaled for 9D input)
class VAE(nn.Module):
    def __init__(self, input_dim=9, latent_dim=10, hidden_dims=[32, 16]):
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
        layers.append(nn.Linear(hidden_dims[0], input_dim))  # Output layer
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta):
    """VAE loss with β annealing"""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

model = VAE(input_dim=9, latent_dim=10, hidden_dims=[32, 16]).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params:,}")
print()

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# β annealing schedule (same as v2.6: 1e-10 → 0.75 over 50 epochs)
def get_beta(epoch, beta_start=1e-10, beta_end=0.75, anneal_epochs=50):
    if epoch < anneal_epochs:
        return beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
    return beta_end

# Training loop
print("="*100)
print("TRAINING")
print("="*100)
print()

best_val_loss = float('inf')
best_model_state = None
patience = 15
patience_counter = 0
history = {'train_loss': [], 'val_loss': [], 'beta': []}

start_time = time.time()

for epoch in range(100):
    beta = get_beta(epoch)

    # Train
    model.train()
    train_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)

        optimizer.zero_grad()
        recon_x, mu, logvar = model(x)
        loss, recon_loss, kl_loss = vae_loss(recon_x, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            recon_x, mu, logvar = model(x)
            loss, _, _ = vae_loss(recon_x, x, mu, logvar, beta)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['beta'].append(beta)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, β={beta:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

elapsed = time.time() - start_time
print(f"\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Best validation loss: {best_val_loss:.4f}")
print()

# Load best model
model.load_state_dict(best_model_state)

# Generate embeddings
print("="*100)
print("GENERATING EMBEDDINGS")
print("="*100)
print()

model.eval()
with torch.no_grad():
    embeddings_train = []
    for batch in train_loader:
        x = batch[0].to(device)
        mu, _ = model.encode(x)
        embeddings_train.append(mu.cpu().numpy())
    embeddings_train = np.vstack(embeddings_train)

    embeddings_test = []
    for batch in test_loader:
        x = batch[0].to(device)
        mu, _ = model.encode(x)
        embeddings_test.append(mu.cpu().numpy())
    embeddings_test = np.vstack(embeddings_test)

print(f"Train embeddings: {embeddings_train.shape}")
print(f"Test embeddings: {embeddings_test.shape}")
print()

# Clustering evaluation
print("="*100)
print("CLUSTERING EVALUATION")
print("="*100)
print()

results = []

for k in [10, 12, 15, 20]:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    gmm.fit(embeddings_train)

    # Predict on test
    test_clusters = gmm.predict(embeddings_test)

    # Metrics
    ari = adjusted_rand_score(y_test, test_clusters)
    sil = silhouette_score(embeddings_test, test_clusters)

    print(f"k={k:2d}: ARI={ari:.3f}, Silhouette={sil:.3f}")

    results.append({'k': k, 'ARI': ari, 'Silhouette': sil})

print()

# Average performance
avg_ari = np.mean([r['ARI'] for r in results])
avg_sil = np.mean([r['Silhouette'] for r in results])

print(f"Average: ARI={avg_ari:.3f}, Silhouette={avg_sil:.3f}")
print()

# Comparison to v2.6
print("="*100)
print("COMPARISON TO v2.6")
print("="*100)
print()

v2_6_ari = 0.196  # v2.6 average ARI
change = ((avg_ari - v2_6_ari) / v2_6_ari) * 100

print(f"v2.6 (239K samples, 6 features): ARI = 0.196")
print(f"v2.7 (12K samples, 9 features):  ARI = {avg_ari:.3f} ({change:+.1f}%)")
print()

if avg_ari > v2_6_ari:
    print("✓ SUCCESS: MAD features improve clustering despite -95% sample reduction!")
    print("  Porosity/grain density/water content are highly discriminative.")
elif avg_ari > 0.15:
    print("⚠ PARTIAL SUCCESS: Performance similar to v2.6 with much less data.")
    print(f"  MAD features partially compensate for sample reduction.")
else:
    print("✗ FAILURE: -95% sample reduction too severe.")
    print("  Need more data or different approach.")

print()

# Save model
checkpoint_path = '/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_7.pth'
torch.save({
    'model_state_dict': best_model_state,
    'scaler': scaler,
    'latent_dim': 10,
    'history': history,
    'results': results,
    'avg_ari': avg_ari
}, checkpoint_path)

print(f"✓ Model saved to: {checkpoint_path}")
print()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('/home/utig5/johna/bhai/vae_v2_7_clustering_results.csv', index=False)
print("✓ Clustering results saved to: vae_v2_7_clustering_results.csv")
print()

print("="*100)
print("VAE v2.7 TRAINING COMPLETE")
print("="*100)
