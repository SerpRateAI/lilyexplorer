"""
VAE Lithology Model v2.7 - VaDE (Variational Deep Embedding) Loss

Key innovation: Cluster-aware loss function with Gaussian Mixture Model prior.

VaDE combines VAE with GMM for joint representation learning and clustering:
- Standard VAE: p(z) ~ N(0, I) (single Gaussian prior)
- VaDE: p(z) = Σ π_k * N(μ_k, σ²_k) (mixture of Gaussians)

Advantages over v2.6:
- Explicit cluster structure in prior distribution
- Cluster assignment probabilities γ guide representation learning
- Joint optimization of reconstruction + clustering + regularization

Same preprocessing as v2.1/v2.5/v2.6:
- GRA bulk density: Gaussian → StandardScaler
- Magnetic susceptibility: Poisson → sign(x)*log(|x|+1) + StandardScaler
- NGR: Bimodal → sign(x)*log(|x|+1) + StandardScaler
- R, G, B: Log-normal → log(x+1) + StandardScaler

Plus β annealing from v2.6:
- Start with low β (focus on reconstruction)
- Gradually increase to target β (add regularization)
- Better training dynamics than fixed β

Dataset: 238K+ samples from 296 boreholes
Features: 6-dimensional input space with distribution-aware scaling
Latent spaces: 2D and 8D models for comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
try:
    import umap
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available, will use PCA only for 8D visualization")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import json
import pickle

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

class LithologyDataset(Dataset):
    """Dataset for VAE training."""

    def __init__(self, features, lithology_labels):
        self.features = torch.FloatTensor(features)
        self.lithology_labels = lithology_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.lithology_labels[idx]

class VaDE(nn.Module):
    """Variational Deep Embedding (VaDE) for joint clustering and representation learning."""

    def __init__(self, input_dim=6, latent_dim=8, n_clusters=10, hidden_dims=[32, 16]):
        super(VaDE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        # Encoder
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

        # Decoder
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

        # GMM parameters (learnable)
        # π_k: cluster weights (mixture coefficients)
        self.pi = nn.Parameter(torch.ones(n_clusters) / n_clusters)

        # μ_k: cluster means in latent space
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))

        # log(σ²_k): cluster variances in latent space
        self.logvar_c = nn.Parameter(torch.randn(n_clusters, latent_dim))

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
        return recon_x, z, mu, logvar

    def get_gamma(self, z):
        """
        Compute cluster assignment probabilities γ_ik using Bayes rule:
        γ_ik = p(c=k|z_i) ∝ π_k * N(z_i | μ_k, σ²_k)

        Args:
            z: latent representations [batch_size, latent_dim]

        Returns:
            gamma: cluster probabilities [batch_size, n_clusters]
        """
        # Normalize pi to ensure valid probabilities
        pi = F.softmax(self.pi, dim=0)  # [n_clusters]

        # Compute log p(z|c=k) for each cluster k
        # log N(z|μ_k, σ²_k) = -0.5 * [log(2π) + log(σ²_k) + (z-μ_k)²/σ²_k]

        z_expand = z.unsqueeze(1)  # [batch, 1, latent_dim]
        mu_c_expand = self.mu_c.unsqueeze(0)  # [1, n_clusters, latent_dim]
        logvar_c_expand = self.logvar_c.unsqueeze(0)  # [1, n_clusters, latent_dim]

        # Log probability for each cluster
        log_p_z_given_c = -0.5 * torch.sum(
            logvar_c_expand + torch.exp(-logvar_c_expand) * (z_expand - mu_c_expand) ** 2,
            dim=2
        )  # [batch, n_clusters]

        # Add log π_k
        log_p_c = torch.log(pi + 1e-10)  # [n_clusters]
        log_p = log_p_z_given_c + log_p_c.unsqueeze(0)  # [batch, n_clusters]

        # Normalize to get probabilities (softmax in log space)
        gamma = F.softmax(log_p, dim=1)  # [batch, n_clusters]

        return gamma

def vade_loss(recon_x, x, z, mu, logvar, model, beta=0.1):
    """
    VaDE loss function with cluster-aware KL divergence.

    Loss = Reconstruction + β * KL_divergence

    KL divergence is computed as weighted sum over clusters:
    KL = Σ_k γ_k * KL(q(z|x) || p(z|c=k))

    Args:
        recon_x: reconstructed input [batch, input_dim]
        x: original input [batch, input_dim]
        z: sampled latent code [batch, latent_dim]
        mu: encoder mean [batch, latent_dim]
        logvar: encoder log variance [batch, latent_dim]
        model: VaDE model (for GMM parameters)
        beta: weight for KL divergence term

    Returns:
        total_loss, recon_loss, kl_loss
    """
    batch_size = x.size(0)

    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # Get cluster assignment probabilities
    gamma = model.get_gamma(z)  # [batch, n_clusters]

    # Normalize pi
    pi = F.softmax(model.pi, dim=0)

    # Cluster-aware KL divergence
    # KL(q(z|x) || p(z|c=k)) for each cluster k
    kl_loss = 0

    mu_expand = mu.unsqueeze(1)  # [batch, 1, latent_dim]
    logvar_expand = logvar.unsqueeze(1)  # [batch, 1, latent_dim]
    mu_c_expand = model.mu_c.unsqueeze(0)  # [1, n_clusters, latent_dim]
    logvar_c_expand = model.logvar_c.unsqueeze(0)  # [1, n_clusters, latent_dim]

    # KL(q(z|x) || p(z|c=k)) for all k
    # = 0.5 * Σ[log(σ²_k) - log(σ²) + σ²/σ²_k + (μ-μ_k)²/σ²_k - 1]
    kl_per_cluster = 0.5 * torch.sum(
        logvar_c_expand - logvar_expand +
        torch.exp(logvar_expand - logvar_c_expand) +
        torch.exp(-logvar_c_expand) * (mu_expand - mu_c_expand) ** 2 - 1,
        dim=2
    )  # [batch, n_clusters]

    # Weight by cluster assignment probabilities
    kl_loss = torch.sum(gamma * kl_per_cluster)

    # Total loss
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss

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

    def inverse_transform(self, X_scaled):
        """Inverse transform back to original scale."""
        # First inverse standard scaling
        X_transformed = self.scaler.inverse_transform(X_scaled)

        # Then inverse log transforms
        X_original = X_transformed.copy()

        # Inverse signed log for MS, NGR
        for idx in self.signed_log_indices:
            X_original[:, idx] = self.inverse_signed_log_transform(X_transformed[:, idx])

        # Inverse regular log for RGB
        for idx in self.log_indices:
            X_original[:, idx] = np.expm1(X_transformed[:, idx])

        return X_original

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

    print("\nApplying distribution-aware scaling:")
    print("  GRA bulk density:         Gaussian      → StandardScaler")
    print("  Magnetic susceptibility:  Poisson       → sign(x)*log(|x|+1) + StandardScaler")
    print("  NGR:                      Bimodal       → sign(x)*log(|x|+1) + StandardScaler")
    print("  R, G, B:                  Log-normal    → log(x+1) + StandardScaler")

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

    print(f"\nData split by borehole:")
    print(f"  Train: {len(train_boreholes)} boreholes, {len(X_train):,} samples")
    print(f"  Val:   {len(val_boreholes)} boreholes, {len(X_val):,} samples")
    print(f"  Test:  {len(test_boreholes)} boreholes, {len(X_test):,} samples")

    return (X_train, y_train, lith_train), (X_val, y_val, lith_val), (X_test, y_test, lith_test)

def train_vade(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda',
               beta_start=0.001, beta_end=0.5, anneal_epochs=50, use_annealing=True):
    """Train VaDE model with optional β annealing."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'train_kl': [], 'beta': []}

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    if use_annealing:
        print(f"\nTraining VaDE with latent dim={model.latent_dim}, n_clusters={model.n_clusters}")
        print(f"β annealing: {beta_start:.3f} → {beta_end:.3f} over {anneal_epochs} epochs")
    else:
        print(f"\nTraining VaDE with latent dim={model.latent_dim}, n_clusters={model.n_clusters}, β={beta_end}")
    print("="*60)

    for epoch in range(epochs):
        # Compute current β (with annealing if enabled)
        if use_annealing and epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        # Training
        model.train()
        train_loss_epoch = 0
        train_recon_epoch = 0
        train_kl_epoch = 0

        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)

            optimizer.zero_grad()
            recon_X, z, mu, logvar = model(batch_X)
            loss, recon_loss, kl_loss = vade_loss(recon_X, batch_X, z, mu, logvar, model, current_beta)

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
                recon_X, z, mu, logvar = model(batch_X)
                loss, _, _ = vade_loss(recon_X, batch_X, z, mu, logvar, model, current_beta)
                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader.dataset)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_recon'].append(train_recon_epoch)
        history['train_kl'].append(train_kl_epoch)
        history['beta'].append(current_beta)

        scheduler.step(val_loss_epoch)

        # Early stopping
        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss_epoch:.4f}, "
                  f"Val Loss={val_loss_epoch:.4f}, "
                  f"Recon={train_recon_epoch:.4f}, KL={train_kl_epoch:.4f}, β={current_beta:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, history

def get_latent_representations(model, data_loader, device='cuda'):
    """Extract latent representations from trained VaDE."""
    model.eval()
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            mu, _ = model.encode(batch_X)
            latent_vectors.append(mu.cpu().numpy())
            labels.append(batch_y.numpy())

    return np.vstack(latent_vectors), np.concatenate(labels)

def get_vade_clusters(model, data_loader, device='cuda'):
    """Get VaDE cluster assignments using γ probabilities."""
    model.eval()
    cluster_probs = []

    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            mu, logvar = model.encode(batch_X)
            z = model.reparameterize(mu, logvar)
            gamma = model.get_gamma(z)
            cluster_probs.append(gamma.cpu().numpy())

    cluster_probs = np.vstack(cluster_probs)
    cluster_assignments = np.argmax(cluster_probs, axis=1)

    return cluster_assignments, cluster_probs

def cluster_analysis(latent_vectors, true_labels, lithology_names, n_clusters_list=[5, 10, 15, 20]):
    """Perform K-Means clustering analysis on latent space."""
    print(f"\n{'='*60}")
    print("K-MEANS CLUSTERING ANALYSIS")
    print(f"{'='*60}")

    results = []

    for n_clusters in n_clusters_list:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(latent_vectors)

        silhouette = silhouette_score(latent_vectors, cluster_labels)
        ari = adjusted_rand_score(true_labels, cluster_labels)

        print(f"\nn_clusters={n_clusters:2d}: Silhouette={silhouette:.3f}, ARI={ari:.3f}")

        # Analyze cluster composition
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_lithologies = lithology_names[cluster_mask]
            cluster_size = len(cluster_lithologies)

            if cluster_size > 0:
                most_common = Counter(cluster_lithologies).most_common(3)
                top_lith = most_common[0][0]
                top_pct = most_common[0][1] / cluster_size * 100
                print(f"  Cluster {cluster_id:2d} (n={cluster_size:5d}): {top_lith} ({top_pct:.1f}%)")

        results.append({
            'n_clusters': n_clusters,
            'silhouette': silhouette,
            'ari': ari,
            'cluster_labels': cluster_labels
        })

    return results

def vade_cluster_analysis(model, data_loader, true_labels, lithology_names, device='cuda'):
    """Analyze VaDE's built-in cluster assignments."""
    print(f"\n{'='*60}")
    print("VaDE CLUSTER ANALYSIS (using γ probabilities)")
    print(f"{'='*60}")

    cluster_assignments, cluster_probs = get_vade_clusters(model, data_loader, device)
    latent_vectors, _ = get_latent_representations(model, data_loader, device)

    n_clusters = model.n_clusters
    silhouette = silhouette_score(latent_vectors, cluster_assignments)
    ari = adjusted_rand_score(true_labels, cluster_assignments)

    print(f"\nn_clusters={n_clusters:2d}: Silhouette={silhouette:.3f}, ARI={ari:.3f}")

    # Analyze cluster composition
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_assignments == cluster_id
        cluster_lithologies = lithology_names[cluster_mask]
        cluster_size = len(cluster_lithologies)

        if cluster_size > 0:
            most_common = Counter(cluster_lithologies).most_common(3)
            top_lith = most_common[0][0]
            top_pct = most_common[0][1] / cluster_size * 100
            avg_confidence = cluster_probs[cluster_mask, cluster_id].mean()
            print(f"  Cluster {cluster_id:2d} (n={cluster_size:5d}): {top_lith} ({top_pct:.1f}%), confidence={avg_confidence:.3f}")

    return {
        'n_clusters': n_clusters,
        'silhouette': silhouette,
        'ari': ari,
        'cluster_assignments': cluster_assignments,
        'cluster_probs': cluster_probs
    }

def main():
    """Main training function."""
    print("="*80)
    print("VAE LITHOLOGY MODEL v2.7 - VaDE (Variational Deep Embedding)")
    print("="*80)

    # Configuration
    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    output_dir = Path('/home/utig5/johna/bhai/vae_v2_7_outputs')
    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')

    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # VaDE configuration
    n_clusters = 12  # Use k=12 (best from v2.6 experiments)
    beta_start = 0.001
    beta_end = 0.5
    anneal_epochs = 50
    use_annealing = True

    print(f"\nVaDE Configuration:")
    print(f"  Number of clusters: {n_clusters}")
    if use_annealing:
        print(f"  β annealing: {beta_start:.3f} → {beta_end:.3f} over {anneal_epochs} epochs")
    else:
        print(f"  Fixed β: {beta_end}")

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

    # Train models with different latent dimensions
    latent_dims = [8]  # Focus on 8D (best from previous experiments)

    for latent_dim in latent_dims:
        print(f"\n{'='*80}")
        print(f"TRAINING VaDE WITH LATENT DIM = {latent_dim}, N_CLUSTERS = {n_clusters}")
        print(f"{'='*80}")

        # Create model
        model = VaDE(
            input_dim=6,
            latent_dim=latent_dim,
            n_clusters=n_clusters,
            hidden_dims=[32, 16]
        ).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train with β annealing
        start_time = time.time()
        model, history = train_vade(
            model, train_loader, val_loader,
            epochs=100, device=device,
            beta_start=beta_start, beta_end=beta_end,
            anneal_epochs=anneal_epochs, use_annealing=use_annealing
        )
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.1f}s")

        # Save model
        model_name = f'vae_gra_v2_7_latent{latent_dim}_k{n_clusters}'
        if use_annealing:
            model_name += f'_anneal'
        model_path = checkpoint_dir / f'{model_name}.pth'

        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'label_encoder': label_encoder,
            'history': history,
            'latent_dim': latent_dim,
            'n_clusters': n_clusters,
            'input_dim': 6,
            'beta_start': beta_start,
            'beta_end': beta_end,
            'anneal_epochs': anneal_epochs,
            'use_annealing': use_annealing,
            'version': 'v2.7'
        }, model_path)
        print(f"Model saved to: {model_path}")

        # Get latent representations
        latent_test, labels_test = get_latent_representations(model, test_loader, device)

        # VaDE cluster analysis (using built-in γ)
        vade_results = vade_cluster_analysis(model, test_loader, labels_test, lith_test, device)

        # K-Means clustering analysis (for comparison)
        kmeans_results = cluster_analysis(
            latent_test, labels_test, lith_test, n_clusters_list=[10, 12, 15, 20]
        )

        # Visualizations
        print(f"\nGenerating visualizations...")

        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Loss curves
        axes[0, 0].plot(history['train_loss'], label='Train')
        axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title(f'VaDE v2.7 Loss (Latent={latent_dim}, K={n_clusters})')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Loss components
        axes[0, 1].plot(history['train_recon'], label='Reconstruction')
        axes[0, 1].plot(history['train_kl'], label='KL Divergence')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Component')
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # β schedule
        axes[1, 0].plot(history['beta'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('β')
        axes[1, 0].set_title('β Annealing Schedule')
        axes[1, 0].grid(True)

        # ARI comparison
        kmeans_aris = [r['ari'] for r in kmeans_results]
        k_values = [r['n_clusters'] for r in kmeans_results]
        axes[1, 1].plot(k_values, kmeans_aris, 'o-', label='K-Means')
        axes[1, 1].axhline(vade_results['ari'], color='red', linestyle='--', label=f'VaDE (k={n_clusters})')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('ARI')
        axes[1, 1].set_title('Clustering Performance Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / f'training_summary_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Latent space visualization with UMAP
        if UMAP_AVAILABLE:
            print("  Computing UMAP projection...")
            umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            latent_umap = umap_model.fit_transform(latent_test)

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Color by lithology (top 10)
            top_lithologies = pd.Series(lith_test).value_counts().head(10).index

            for lith in list(top_lithologies) + ['Other']:
                mask = [(l == lith if lith != 'Other' else l not in top_lithologies.values)
                        for l in lith_test]
                if sum(mask) > 0:
                    axes[0].scatter(latent_umap[mask, 0], latent_umap[mask, 1],
                                   label=lith, alpha=0.5, s=1)

            axes[0].set_xlabel('UMAP 1')
            axes[0].set_ylabel('UMAP 2')
            axes[0].set_title('VaDE Latent Space (colored by lithology)')
            axes[0].legend(markerscale=3, fontsize=8)
            axes[0].grid(True, alpha=0.3)

            # Color by VaDE cluster
            cluster_assignments = vade_results['cluster_assignments']
            for cluster_id in range(n_clusters):
                mask = cluster_assignments == cluster_id
                axes[1].scatter(latent_umap[mask, 0], latent_umap[mask, 1],
                               label=f'C{cluster_id}', alpha=0.5, s=1)

            axes[1].set_xlabel('UMAP 1')
            axes[1].set_ylabel('UMAP 2')
            axes[1].set_title('VaDE Latent Space (colored by VaDE cluster)')
            axes[1].legend(markerscale=3, fontsize=8, ncol=2)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f'latent_space_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to: {output_dir}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
