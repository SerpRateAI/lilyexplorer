"""
VaDE (Variational Deep Embedding) Lithology Model v2.4

Joint optimization of VAE and Gaussian Mixture Model for cluster-aware representation learning.

Key innovation: Replace standard Gaussian prior p(z) = N(0,I) with GMM prior p(z) = Σ π_k N(μ_k, σ_k)

This learns a latent space specifically optimized for clustering, not just reconstruction.

Dataset: 238K+ samples from 296 boreholes
Features: 6-dimensional (GRA, MS, NGR, R, G, B) with distribution-aware scaling
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA
from collections import Counter
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class LithologyDataset(Dataset):
    """Dataset for VaDE training."""

    def __init__(self, features, lithology_labels):
        self.features = torch.FloatTensor(features)
        self.lithology_labels = lithology_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.lithology_labels[idx]

class VaDE(nn.Module):
    """Variational Deep Embedding - VAE with GMM prior for joint clustering."""

    def __init__(self, input_dim=6, latent_dim=8, n_clusters=20, hidden_dims=[32, 16]):
        super(VaDE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Encoder (same as standard VAE)
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (same as standard VAE)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # GMM parameters (learned during training)
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)  # Mixture weights
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))  # Cluster centers
        self.log_sigma_c = nn.Parameter(torch.randn(n_clusters, latent_dim))  # Cluster log-variances

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
        return recon_x, mu, logvar, z

    def get_gamma(self, z):
        """
        Compute cluster assignment probabilities γ_ik = p(c=k|z_i)

        Using Bayes rule:
        γ_ik ∝ π_k * N(z_i | μ_k, σ_k)
        """
        batch_size = z.shape[0]

        # Get GMM parameters
        pi = torch.softmax(self.pi, dim=0)  # Normalize to probabilities
        mu_c = self.mu_c
        sigma_c = torch.exp(self.log_sigma_c)  # Ensure positive variance

        # Compute log p(z|c) for each cluster
        # z: [batch, latent_dim]
        # mu_c: [n_clusters, latent_dim]
        # sigma_c: [n_clusters, latent_dim]

        z_expanded = z.unsqueeze(1)  # [batch, 1, latent_dim]
        mu_c_expanded = mu_c.unsqueeze(0)  # [1, n_clusters, latent_dim]
        sigma_c_expanded = sigma_c.unsqueeze(0)  # [1, n_clusters, latent_dim]

        # Log probability under each Gaussian component
        # log N(z|μ,σ) = -0.5 * [log(2π) + log(σ²) + (z-μ)²/σ²]
        log_prob = -0.5 * (
            torch.log(2 * np.pi * sigma_c_expanded + 1e-10) +
            ((z_expanded - mu_c_expanded) ** 2) / (sigma_c_expanded + 1e-10)
        )
        log_prob = log_prob.sum(dim=2)  # Sum over latent dimensions: [batch, n_clusters]

        # Add log prior π_k
        log_pi = torch.log(pi + 1e-10)  # [n_clusters]
        log_prob = log_prob + log_pi.unsqueeze(0)  # [batch, n_clusters]

        # Normalize to get probabilities (softmax in log space)
        gamma = torch.softmax(log_prob, dim=1)  # [batch, n_clusters]

        return gamma

def vade_loss(recon_x, x, mu, logvar, z, gamma, model, beta=1.0):
    """
    VaDE loss = reconstruction loss + cluster-aware KL divergence

    Standard VAE: KL(q(z|x) || p(z)) where p(z) = N(0,I)
    VaDE: KL(q(z|x) || p(z|c)) where p(z|c) = GMM
    """
    batch_size = x.shape[0]

    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # Get GMM parameters
    pi = torch.softmax(model.pi, dim=0)
    mu_c = model.mu_c
    sigma_c = torch.exp(model.log_sigma_c)

    # Cluster-aware KL divergence
    # KL = Σ_k γ_k [log(γ_k/π_k) + KL(N(μ,σ²) || N(μ_k,σ_k²))]

    # 1. Categorical divergence: Σ_k γ_k log(γ_k/π_k)
    log_pi = torch.log(pi + 1e-10)
    log_gamma = torch.log(gamma + 1e-10)
    categorical_div = (gamma * (log_gamma - log_pi.unsqueeze(0))).sum(dim=1).sum()

    # 2. Gaussian KL divergence for each cluster: KL(N(μ,σ²) || N(μ_k,σ_k²))
    # KL(N(μ,σ²) || N(μ_k,σ_k²)) = 0.5 * [log(σ_k²/σ²) - 1 + σ²/σ_k² + (μ-μ_k)²/σ_k²]

    mu_expanded = mu.unsqueeze(1)  # [batch, 1, latent_dim]
    logvar_expanded = logvar.unsqueeze(1)  # [batch, 1, latent_dim]

    mu_c_expanded = mu_c.unsqueeze(0)  # [1, n_clusters, latent_dim]
    log_sigma_c_expanded = model.log_sigma_c.unsqueeze(0)  # [1, n_clusters, latent_dim]
    sigma_c_expanded = sigma_c.unsqueeze(0)

    gaussian_kl = 0.5 * (
        log_sigma_c_expanded * 2 - logvar_expanded - 1 +
        torch.exp(logvar_expanded) / sigma_c_expanded +
        ((mu_expanded - mu_c_expanded) ** 2) / sigma_c_expanded
    )
    gaussian_kl = gaussian_kl.sum(dim=2)  # Sum over latent dims: [batch, n_clusters]

    # Weight by cluster assignments
    weighted_gaussian_kl = (gamma * gaussian_kl).sum(dim=1).sum()

    # Total KL
    kl_loss = categorical_div + weighted_gaussian_kl

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

class DistributionAwareScaler:
    """Custom scaler that applies distribution-specific transformations."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR can be negative
        self.log_indices = [3, 4, 5]  # R, G, B are always positive

    def signed_log_transform(self, x):
        """Log transform that preserves sign for data with negative values."""
        return np.sign(x) * np.log1p(np.abs(x))

    def inverse_signed_log_transform(self, x):
        """Inverse of signed log transform."""
        return np.sign(x) * (np.exp(np.abs(x)) - 1)

    def fit_transform(self, X):
        """Apply distribution-specific transforms, then standard scale."""
        X_transformed = X.copy()

        # Apply signed log(|x| + 1) to features that can be negative (MS, NGR)
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply log(x + 1) to features that are always positive (RGB)
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Standard scale all features
        X_scaled = self.scaler.fit_transform(X_transformed)
        return X_scaled

    def transform(self, X):
        """Transform new data using fitted scaler."""
        X_transformed = X.copy()

        # Apply signed log to MS, NGR
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])

        # Apply regular log to RGB
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])

        # Standard scale
        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled

def load_and_prepare_data(data_path):
    """Load and prepare data for training with distribution-aware scaling."""
    print("Loading data...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

    # Extract features (6D: GRA, MS, NGR, R, G, B)
    feature_cols = [
        'Bulk density (GRA)',
        'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)',
        'R',
        'G',
        'B'
    ]

    X = df[feature_cols].values
    lithology = df['Principal'].values
    borehole_ids = df['Borehole_ID'].values

    # Remove any remaining NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    lithology = lithology[valid_mask]
    borehole_ids = borehole_ids[valid_mask]

    print(f"After removing NaN: {len(X):,} samples")

    # Normalize features with distribution-aware scaling
    scaler = DistributionAwareScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode lithology labels
    label_encoder = LabelEncoder()
    lithology_encoded = label_encoder.fit_transform(lithology)

    print(f"Found {len(label_encoder.classes_)} unique lithologies")

    return X_scaled, lithology_encoded, lithology, borehole_ids, scaler, label_encoder

def split_by_borehole(X, y, lithology, borehole_ids, train_size=0.7, val_size=0.15):
    """Split data by borehole to prevent data leakage."""
    unique_boreholes = np.unique(borehole_ids)

    # Split boreholes
    train_boreholes, test_boreholes = train_test_split(
        unique_boreholes, train_size=train_size + val_size, random_state=42
    )
    train_boreholes, val_boreholes = train_test_split(
        train_boreholes, train_size=train_size/(train_size+val_size), random_state=42
    )

    # Create masks
    train_mask = np.isin(borehole_ids, train_boreholes)
    val_mask = np.isin(borehole_ids, val_boreholes)
    test_mask = np.isin(borehole_ids, test_boreholes)

    # Split data
    X_train, y_train, lith_train = X[train_mask], y[train_mask], lithology[train_mask]
    X_val, y_val, lith_val = X[val_mask], y[val_mask], lithology[val_mask]
    X_test, y_test, lith_test = X[test_mask], y[test_mask], lithology[test_mask]

    print(f"\nTrain: {len(train_boreholes)} boreholes, {len(X_train):,} samples")
    print(f"Val:   {len(val_boreholes)} boreholes, {len(X_val):,} samples")
    print(f"Test:  {len(test_boreholes)} boreholes, {len(X_test):,} samples")

    return (X_train, y_train, lith_train), (X_val, y_val, lith_val), (X_test, y_test, lith_test)

def pretrain_vae(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """Pre-train as standard VAE to initialize encoder/decoder."""
    print("\nPre-training VAE (standard Gaussian prior)...")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)

            optimizer.zero_grad()
            recon_X, mu, logvar, z = model(batch_X)

            # Standard VAE loss
            recon_loss = nn.functional.mse_loss(recon_X, batch_X, reduction='sum')
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:2d}: Loss={train_loss/len(train_loader.dataset):.3f}")

    print("Pre-training complete.\n")
    return model

def initialize_gmm_params(model, train_loader, device='cpu'):
    """Initialize GMM parameters using K-Means on latent representations."""
    print("Initializing GMM parameters with K-Means...")

    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            mu, _ = model.encode(batch_X)
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)

    # Run K-Means
    kmeans = KMeans(n_clusters=model.n_clusters, random_state=42, n_init=10)
    kmeans.fit(latent_vectors)

    # Initialize cluster centers
    model.mu_c.data = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    # Initialize cluster variances (use variance within each cluster)
    cluster_vars = []
    for k in range(model.n_clusters):
        cluster_points = latent_vectors[kmeans.labels_ == k]
        if len(cluster_points) > 1:
            var = np.var(cluster_points, axis=0)
        else:
            var = np.ones(model.latent_dim)
        cluster_vars.append(var)

    model.log_sigma_c.data = torch.FloatTensor(np.log(np.array(cluster_vars) + 1e-6)).to(device)

    # Initialize mixture weights uniformly
    model.pi.data = torch.ones(model.n_clusters).to(device) / model.n_clusters

    print(f"Initialized {model.n_clusters} clusters\n")

def train_vade(model, train_loader, val_loader, epochs=100, lr=0.001, device='cpu', beta=1.0):
    """Train VaDE model with cluster-aware loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'train_kl': []}

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    print(f"Training VaDE with {model.n_clusters} clusters, latent_dim={model.latent_dim}")
    print("="*60)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_epoch = 0
        train_recon_epoch = 0
        train_kl_epoch = 0

        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)

            optimizer.zero_grad()
            recon_X, mu, logvar, z = model(batch_X)

            # Compute cluster assignments
            gamma = model.get_gamma(z)

            # VaDE loss
            loss, recon_loss, kl_loss = vade_loss(recon_X, batch_X, mu, logvar, z, gamma, model, beta)

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_recon_epoch += recon_loss.item()
            train_kl_epoch += kl_loss.item()

        train_loss_epoch /= len(train_loader.dataset)
        train_recon_epoch /= len(train_loader.dataset)
        train_kl_epoch /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss_epoch = 0

        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                recon_X, mu, logvar, z = model(batch_X)
                gamma = model.get_gamma(z)
                loss, _, _ = vade_loss(recon_X, batch_X, mu, logvar, z, gamma, model, beta)
                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader.dataset)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_recon'].append(train_recon_epoch)
        history['train_kl'].append(train_kl_epoch)

        scheduler.step(val_loss_epoch)

        # Early stopping
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: Train={train_loss_epoch:.3f}, Val={val_loss_epoch:.3f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            print(f"Best validation loss: {best_val_loss:.3f}\n")
            break

    return model, history

def get_cluster_assignments(model, data_loader, device='cpu'):
    """Get cluster assignments and latent representations."""
    model.eval()
    latent_vectors = []
    cluster_probs = []
    labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            mu, _ = model.encode(batch_X)
            gamma = model.get_gamma(mu)

            latent_vectors.append(mu.cpu().numpy())
            cluster_probs.append(gamma.cpu().numpy())
            labels.append(batch_y.numpy())

    latent_vectors = np.vstack(latent_vectors)
    cluster_probs = np.vstack(cluster_probs)
    labels = np.concatenate(labels)

    # Hard assignments (argmax)
    cluster_assignments = np.argmax(cluster_probs, axis=1)

    return latent_vectors, cluster_assignments, cluster_probs, labels

def evaluate_clustering(cluster_assignments, lithology, cluster_probs):
    """Evaluate clustering performance."""
    # ARI (supervised metric)
    ari = adjusted_rand_score(lithology, cluster_assignments)

    # Cluster purity (unsupervised metric)
    n_clusters = cluster_probs.shape[1]
    cluster_purities = []

    for k in range(n_clusters):
        cluster_mask = cluster_assignments == k
        if cluster_mask.sum() > 0:
            cluster_lithologies = lithology[cluster_mask]
            most_common = Counter(cluster_lithologies).most_common(1)
            purity = most_common[0][1] / len(cluster_lithologies)
            cluster_purities.append((k, purity, len(cluster_lithologies), most_common[0][0]))

    avg_purity = np.mean([p[1] for p in cluster_purities])

    return ari, avg_purity, cluster_purities

def main():
    """Main training function."""
    print("="*80)
    print("VaDE (Variational Deep Embedding) v2.4")
    print("Cluster-Aware Representation Learning")
    print("="*80)

    # Configuration
    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load data
    X, y, lithology, borehole_ids, scaler, label_encoder = load_and_prepare_data(data_path)

    # Split data
    (X_train, y_train, lith_train), (X_val, y_val, lith_val), (X_test, y_test, lith_test) = \
        split_by_borehole(X, y, lithology, borehole_ids)

    # Create data loaders
    train_dataset = LithologyDataset(X_train, y_train)
    val_dataset = LithologyDataset(X_val, y_val)
    test_dataset = LithologyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Train VaDE with different n_clusters (unsupervised model selection)
    n_clusters_list = [10, 15, 20, 30]
    results = []

    for n_clusters in n_clusters_list:
        print(f"\n{'='*80}")
        print(f"Training VaDE with n_clusters={n_clusters}")
        print(f"{'='*80}\n")

        # Create model
        model = VaDE(input_dim=6, latent_dim=8, n_clusters=n_clusters, hidden_dims=[32, 16]).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Pre-train as standard VAE
        model = pretrain_vae(model, train_loader, val_loader, epochs=20, device=device)

        # Initialize GMM parameters
        initialize_gmm_params(model, train_loader, device=device)

        # Train VaDE
        start_time = time.time()
        model, history = train_vade(model, train_loader, val_loader, epochs=100, device=device)
        train_time = time.time() - start_time
        print(f"Training completed in {train_time:.1f} seconds")

        # Save model
        model_path = checkpoint_dir / f'vade_gra_v2_4_k{n_clusters}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'n_clusters': n_clusters,
            'latent_dim': 8,
            'input_dim': 6,
            'history': history,
            'version': 'v2.4_vade'
        }, model_path)
        print(f"✓ Saved checkpoint to {model_path}")

        # Evaluate
        latent_test, cluster_test, probs_test, labels_test = get_cluster_assignments(model, test_loader, device)
        ari, avg_purity, purities = evaluate_clustering(cluster_test, lith_test, probs_test)

        print(f"\n{'='*60}")
        print(f"RESULTS (n_clusters={n_clusters})")
        print(f"{'='*60}")
        print(f"ARI:          {ari:.3f}")
        print(f"Avg Purity:   {avg_purity:.3f}")
        print(f"\nHigh-purity clusters (>70%):")
        high_purity = [p for p in purities if p[1] > 0.7]
        for k, purity, size, lith in sorted(high_purity, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  Cluster {k:2d} (n={size:5d}): {lith:30s} ({purity*100:.1f}%)")

        results.append({
            'n_clusters': n_clusters,
            'ari': ari,
            'avg_purity': avg_purity,
            'train_time': train_time
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY ACROSS ALL n_clusters")
    print(f"{'='*80}")
    print(f"{'k':>4} | {'ARI':>8} | {'Avg Purity':>12} | {'Train Time (s)':>15}")
    print("-"*80)
    for r in results:
        print(f"{r['n_clusters']:4d} | {r['ari']:8.3f} | {r['avg_purity']:12.3f} | {r['train_time']:15.1f}")

    best = max(results, key=lambda x: x['ari'])
    print(f"\n✓ Best ARI: k={best['n_clusters']} with ARI={best['ari']:.3f}")

    print(f"\n{'='*80}")
    print("VaDE v2.4 Training Complete!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
