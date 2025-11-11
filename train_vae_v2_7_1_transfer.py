"""
VAE v2.7.1: Transfer learning from MAD-rich subset to full dataset

Strategy:
1. Use v2.7 encoder (trained on 12K samples with 9 features including MAD)
2. Train new decoder for 6 features (GRA+MS+NGR+RGB only) on full 239K samples
3. Hypothesis: Encoder learned porosity/grain density relationships from MAD subset,
   can apply that knowledge to infer those properties from GRA+RGB alone on full dataset

Key difference from v2.6.2 failure:
- v2.6.2: Pre-trained on subset of features (GRA+MS+NGR), added RGB later
- v2.7.1: Pre-trained on superset of features (all 9), apply to subset (6)
- Direction matters: rich → sparse works, sparse → rich doesn't
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
import time

print("="*100)
print("VAE v2.7.1 TRANSFER LEARNING: MAD Pre-training → Full Dataset")
print("="*100)
print()

# Load v2.7 checkpoint (trained on 12K samples with 9 features)
print("Loading v2.7 encoder...")

# Define dummy scaler class to enable unpickling
class DistributionAwareScaler_v27:
    pass

v2_7_checkpoint = torch.load('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_7.pth',
                              map_location='cpu', weights_only=False)
print("✓ v2.7 checkpoint loaded")
print(f"  v2.7 ARI: {v2_7_checkpoint['avg_ari']:.3f}")
print()

# Load v2.6 dataset (239K samples, 6 features)
print("Loading v2.6 full dataset (239K samples, 6 features)...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
print(f"Total samples: {len(df):,}")
print(f"Total boreholes: {df['Borehole_ID'].nunique()}")
print()

# Features (6D: GRA + MS + NGR + RGB)
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
y = df['Principal'].values
boreholes = df['Borehole_ID'].values

# Borehole-level split
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

print(f"Train: {len(X_train):>6,} samples from {len(train_boreholes):>3d} boreholes")
print(f"Val:   {len(X_val):>6,} samples from {len(val_boreholes):>3d} boreholes")
print(f"Test:  {len(X_test):>6,} samples from {len(test_boreholes):>3d} boreholes")
print()

# Distribution-aware scaling (same as v2.6: 6 features)
print("Applying distribution-aware scaling (6D)...")

class DistributionAwareScaler_v26:
    """Scaler for v2.6 6D features."""
    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR
        self.log_indices = [3, 4, 5]  # RGB

    def fit_transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = np.sign(X_transformed[:, idx]) * np.log1p(np.abs(X_transformed[:, idx]))
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

scaler = DistributionAwareScaler_v26()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
print()

# VAE architecture
class VAE_Transfer(nn.Module):
    """VAE with pre-trained 10D encoder, new 6D decoder."""
    def __init__(self, pretrained_encoder_state):
        super().__init__()
        self.latent_dim = 10

        # Load pre-trained encoder (9D → 10D latent)
        # We'll use the first 6 dimensions of the 9D input (GRA, MS, NGR, RGB)
        # The encoder expects 9D, but we'll pad with zeros for the missing MAD features

        # Encoder from v2.7 (9D input)
        self.encoder = nn.Sequential(
            nn.Linear(9, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc_mu = nn.Linear(16, 10)
        self.fc_logvar = nn.Linear(16, 10)

        # Load pre-trained weights
        # Map v2.7 state dict keys
        encoder_state = {}
        for k, v in pretrained_encoder_state.items():
            if 'encoder.0' in k or 'encoder.2' in k:
                encoder_state[k] = v
            elif 'fc_mu' in k:
                encoder_state[k] = v
            elif 'fc_logvar' in k:
                encoder_state[k] = v

        self.encoder.load_state_dict({k.replace('encoder.', ''): v for k, v in encoder_state.items()
                                       if 'encoder' in k}, strict=False)
        self.fc_mu.load_state_dict({k.replace('fc_mu.', ''): v for k, v in encoder_state.items()
                                     if 'fc_mu' in k})
        self.fc_logvar.load_state_dict({k.replace('fc_logvar.', ''): v for k, v in encoder_state.items()
                                         if 'fc_logvar' in k})

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.fc_mu.parameters():
            param.requires_grad = False
        for param in self.fc_logvar.parameters():
            param.requires_grad = False

        # New decoder (10D latent → 6D output)
        self.decoder = nn.Sequential(
            nn.Linear(10, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 6)  # Output 6 features
        )

    def encode(self, x):
        """Pad 6D input to 9D (zeros for missing MAD features), then encode."""
        # Pad with zeros for MAD features (indices 6, 7, 8)
        batch_size = x.size(0)
        x_padded = torch.zeros(batch_size, 9, device=x.device)
        x_padded[:, :6] = x  # Copy GRA, MS, NGR, RGB
        # x_padded[:, 6:9] remain zeros (Porosity, Grain Density, Water Content)

        h = self.encoder(x_padded)
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
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

model = VAE_Transfer(v2_7_checkpoint['model_state_dict']).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} (decoder only)")
print()

# DataLoaders
train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled))
test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# β annealing (same as v2.7)
def get_beta(epoch, beta_start=1e-10, beta_end=0.75, anneal_epochs=50):
    if epoch < anneal_epochs:
        return beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
    return beta_end

# Training
print("="*100)
print("TRAINING: Frozen Encoder + New Decoder")
print("="*100)
print()

best_val_loss = float('inf')
best_model_state = None
patience = 15
patience_counter = 0

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
        loss, _, _ = vae_loss(recon_x, x, mu, logvar, beta)
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
print(f"\nTraining time: {elapsed:.1f}s")
print()

# Load best model
model.load_state_dict(best_model_state)

# Generate embeddings
print("="*100)
print("CLUSTERING EVALUATION")
print("="*100)
print()

model.eval()
with torch.no_grad():
    embeddings_test = []
    for batch in test_loader:
        x = batch[0].to(device)
        mu, _ = model.encode(x)
        embeddings_test.append(mu.cpu().numpy())
    embeddings_test = np.vstack(embeddings_test)

    embeddings_train = []
    for batch in train_loader:
        x = batch[0].to(device)
        mu, _ = model.encode(x)
        embeddings_train.append(mu.cpu().numpy())
    embeddings_train = np.vstack(embeddings_train)

results = []
for k in [10, 12, 15, 20]:
    gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
    gmm.fit(embeddings_train)
    test_clusters = gmm.predict(embeddings_test)
    ari = adjusted_rand_score(y_test, test_clusters)
    sil = silhouette_score(embeddings_test, test_clusters)
    print(f"k={k:2d}: ARI={ari:.3f}, Silhouette={sil:.3f}")
    results.append({'k': k, 'ARI': ari, 'Silhouette': sil})

avg_ari = np.mean([r['ARI'] for r in results])
print(f"\nAverage: ARI={avg_ari:.3f}")
print()

# Comparison
print("="*100)
print("COMPARISON")
print("="*100)
print()

v2_6_ari = 0.196  # v2.6 baseline (239K samples, 6 features)
v2_7_ari = 0.268  # v2.7 (12K samples, 9 features)

print(f"v2.6 (239K samples, 6 features):        ARI = {v2_6_ari:.3f}")
print(f"v2.7 (12K samples, 9 features):         ARI = {v2_7_ari:.3f} (+{((v2_7_ari-v2_6_ari)/v2_6_ari*100):+.1f}%)")
print(f"v2.7.1 (239K samples, 6 features, transfer): ARI = {avg_ari:.3f} ({((avg_ari-v2_6_ari)/v2_6_ari*100):+.1f}% vs v2.6, {((avg_ari-v2_7_ari)/v2_7_ari*100):+.1f}% vs v2.7)")
print()

if avg_ari > v2_7_ari:
    print("✓ BEST RESULT: Transfer learning combines v2.7 MAD knowledge with v2.6 data scale!")
    print("  Encoder learned porosity/grain density relationships, applied to full dataset.")
elif avg_ari > v2_6_ari:
    print("✓ SUCCESS: Transfer improves over v2.6 baseline but doesn't reach v2.7 performance.")
    print("  MAD pre-training helps, but smaller dataset limits v2.7's advantage.")
else:
    print("✗ FAILURE: Transfer learning doesn't improve over v2.6 baseline.")
    print("  MAD-learned representations don't transfer to GRA-only samples.")

print()
print("="*100)
print("VAE v2.7.1 TRANSFER LEARNING COMPLETE")
print("="*100)
