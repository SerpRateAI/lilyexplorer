"""
VAE Lithology Model v2.8 - Contrastive Learning with Pseudo-Labels

Key innovation: Adds contrastive loss using k-means pseudo-labels for discrimination.

Combines three complementary objectives:
1. Reconstruction loss: Learn to decode latent codes back to input
2. KL divergence (with β annealing): Regularize latent space
3. Contrastive loss (InfoNCE): Pull together samples with same pseudo-label, push apart different ones

Contrastive approach:
- Every 10 epochs: Run k-means on latent codes → pseudo-labels
- Positive pairs: Samples with same pseudo-label (similar lithology)
- Negative pairs: Samples with different pseudo-labels
- InfoNCE loss: -log(exp(sim(z_i, z_+)) / Σ_j exp(sim(z_i, z_j)))

This creates a self-supervised clustering signal that should improve discrimination.

Same preprocessing as v2.1/v2.5/v2.6:
- GRA bulk density: Gaussian → StandardScaler
- Magnetic susceptibility: Poisson → sign(x)*log(|x|+1) + StandardScaler
- NGR: Bimodal → sign(x)*log(|x|+1) + StandardScaler
- R, G, B: Log-normal → log(x+1) + StandardScaler

Plus β annealing from v2.6:
- Start with low β (focus on reconstruction)
- Gradually increase to target β (add regularization)

Dataset: 238K+ samples from 296 boreholes
Features: 6-dimensional input space with distribution-aware scaling
Latent spaces: 8D (best from previous experiments)
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

class VAE(nn.Module):
    """Variational Autoencoder for lithology representation learning."""

    def __init__(self, input_dim=6, latent_dim=8, hidden_dims=[32, 16]):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

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

def contrastive_vae_loss(recon_x, x, mu, logvar, z, pseudo_labels, beta=0.5, gamma=0.1, tau=0.5):
    """
    VAE loss with contrastive learning using pseudo-labels.

    Loss = Reconstruction + β * KL + γ * Contrastive

    Args:
        recon_x: reconstructed input [batch, input_dim]
        x: original input [batch, input_dim]
        mu: encoder mean [batch, latent_dim]
        logvar: encoder log variance [batch, latent_dim]
        z: sampled latent code [batch, latent_dim]
        pseudo_labels: cluster assignments from k-means [batch]
        beta: weight for KL divergence
        gamma: weight for contrastive loss
        tau: temperature for InfoNCE

    Returns:
        total_loss, recon_loss, kl_loss, contrastive_loss
    """
    batch_size = x.size(0)

    # Standard VAE losses
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Contrastive loss (InfoNCE with pseudo-labels)
    if pseudo_labels is not None:
        # Normalize latent codes for cosine similarity
        z_norm = F.normalize(z, dim=1)  # [batch, latent_dim]

        # Compute similarity matrix (cosine similarity / temperature)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / tau  # [batch, batch]

        # Create positive and negative masks based on pseudo-labels
        pseudo_labels_tensor = torch.tensor(pseudo_labels, device=z.device)
        pos_mask = (pseudo_labels_tensor.unsqueeze(0) == pseudo_labels_tensor.unsqueeze(1)).float()
        pos_mask.fill_diagonal_(0)  # Don't use self as positive

        neg_mask = 1 - pos_mask
        neg_mask.fill_diagonal_(0)  # Don't use self as negative

        # InfoNCE loss
        # For each anchor, compute log probability of positive pairs
        exp_sim = torch.exp(sim_matrix)

        # Denominator: sum of similarities to all negatives
        neg_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True) + 1e-8

        # Numerator: similarities to positives
        log_prob = sim_matrix - torch.log(neg_sum)

        # Average over positive pairs for each anchor
        num_positives = pos_mask.sum(dim=1) + 1e-8
        contrastive_loss = -(log_prob * pos_mask).sum(dim=1) / num_positives
        contrastive_loss = contrastive_loss.mean()
    else:
        contrastive_loss = torch.tensor(0.0, device=z.device)

    # Total loss
    total_loss = recon_loss + beta * kl_loss + gamma * contrastive_loss

    return total_loss, recon_loss, kl_loss, contrastive_loss

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

def update_pseudo_labels(model, data_loader, n_clusters=12, device='cuda'):
    """
    Update pseudo-labels by running k-means on current latent representations.

    Returns:
        pseudo_labels: array of cluster assignments [n_samples]
    """
    model.eval()
    latent_vectors = []

    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            mu, _ = model.encode(batch_X)
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.vstack(latent_vectors)

    # Run k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pseudo_labels = kmeans.fit_predict(latent_vectors)

    return pseudo_labels

def train_contrastive_vae(model, train_loader, val_loader, epochs=100, lr=0.001, device='cuda',
                          beta_start=0.001, beta_end=0.5, anneal_epochs=50,
                          gamma=0.1, tau=0.5, n_clusters=12, update_interval=10):
    """Train VAE with contrastive loss and β annealing."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_recon': [], 'train_kl': [], 'train_contrastive': [],
        'beta': [], 'gamma': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    print(f"\nTraining Contrastive VAE:")
    print(f"  Latent dim: {model.latent_dim}")
    print(f"  β annealing: {beta_start:.3f} → {beta_end:.3f} over {anneal_epochs} epochs")
    print(f"  γ (contrastive weight): {gamma:.3f}")
    print(f"  τ (temperature): {tau:.3f}")
    print(f"  Pseudo-label clusters: {n_clusters}")
    print(f"  Update interval: {update_interval} epochs")
    print("="*60)

    # Initialize pseudo-labels
    print("Initializing pseudo-labels...")
    pseudo_labels = update_pseudo_labels(model, train_loader, n_clusters, device)

    for epoch in range(epochs):
        # Update pseudo-labels periodically
        if epoch % update_interval == 0 and epoch > 0:
            print(f"  Updating pseudo-labels at epoch {epoch}...")
            pseudo_labels = update_pseudo_labels(model, train_loader, n_clusters, device)

        # Compute current β
        if epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        # Training
        model.train()
        train_loss_epoch = 0
        train_recon_epoch = 0
        train_kl_epoch = 0
        train_contrastive_epoch = 0

        batch_start_idx = 0
        for batch_X, _ in train_loader:
            batch_X = batch_X.to(device)
            batch_size = batch_X.size(0)

            # Get pseudo-labels for this batch
            batch_pseudo_labels = pseudo_labels[batch_start_idx:batch_start_idx + batch_size]
            batch_start_idx += batch_size

            optimizer.zero_grad()
            recon_X, mu, logvar, z = model(batch_X)
            loss, recon_loss, kl_loss, contrastive_loss = contrastive_vae_loss(
                recon_X, batch_X, mu, logvar, z, batch_pseudo_labels,
                beta=current_beta, gamma=gamma, tau=tau
            )

            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item()
            train_recon_epoch += recon_loss.item()
            train_kl_epoch += kl_loss.item()
            train_contrastive_epoch += contrastive_loss.item() * batch_size

        train_loss_epoch /= len(train_loader.dataset)
        train_recon_epoch /= len(train_loader.dataset)
        train_kl_epoch /= len(train_loader.dataset)
        train_contrastive_epoch /= len(train_loader.dataset)

        # Validation (without contrastive loss since we don't have pseudo-labels for val)
        model.eval()
        val_loss_epoch = 0

        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                recon_X, mu, logvar, z = model(batch_X)
                # Validation uses only VAE loss (no contrastive)
                recon_loss = F.mse_loss(recon_X, batch_X, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + current_beta * kl_loss
                val_loss_epoch += loss.item()

        val_loss_epoch /= len(val_loader.dataset)

        history['train_loss'].append(train_loss_epoch)
        history['val_loss'].append(val_loss_epoch)
        history['train_recon'].append(train_recon_epoch)
        history['train_kl'].append(train_kl_epoch)
        history['train_contrastive'].append(train_contrastive_epoch)
        history['beta'].append(current_beta)
        history['gamma'].append(gamma)

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
                  f"Recon={train_recon_epoch:.4f}, KL={train_kl_epoch:.4f}, "
                  f"Contr={train_contrastive_epoch:.4f}, β={current_beta:.4f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, history

def get_latent_representations(model, data_loader, device='cuda'):
    """Extract latent representations from trained VAE."""
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

def cluster_analysis(latent_vectors, true_labels, lithology_names, n_clusters_list=[10, 12, 15, 20]):
    """Perform clustering analysis on latent space."""
    print(f"\n{'='*60}")
    print("CLUSTERING ANALYSIS")
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

def main():
    """Main training function."""
    print("="*80)
    print("VAE LITHOLOGY MODEL v2.8 - Contrastive Learning with Pseudo-Labels")
    print("="*80)

    # Configuration
    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    output_dir = Path('/home/utig5/johna/bhai/vae_v2_8_outputs')
    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')

    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    beta_start = 0.001
    beta_end = 0.5
    anneal_epochs = 50
    gamma = 0.1  # Contrastive loss weight
    tau = 0.5    # Temperature
    n_clusters = 12  # Pseudo-label clusters
    update_interval = 10  # Update pseudo-labels every N epochs

    print(f"\nContrastive VAE Configuration:")
    print(f"  β annealing: {beta_start:.3f} → {beta_end:.3f} over {anneal_epochs} epochs")
    print(f"  γ (contrastive weight): {gamma}")
    print(f"  τ (temperature): {tau}")
    print(f"  Pseudo-label clusters: {n_clusters}")
    print(f"  Update interval: {update_interval} epochs")

    # Load data
    X, y, lithology, borehole_ids, scaler, label_encoder = load_and_prepare_data(data_path)

    # Split data
    (X_train, y_train, lith_train), (X_val, y_val, lith_val), (X_test, y_test, lith_test) = \
        split_by_borehole(X, y, lithology, borehole_ids)

    # Create data loaders
    train_dataset = LithologyDataset(X_train, y_train)
    val_dataset = LithologyDataset(X_val, y_val)
    test_dataset = LithologyDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)  # Don't shuffle for pseudo-label indexing
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Train 8D model (best from previous experiments)
    latent_dim = 8

    print(f"\n{'='*80}")
    print(f"TRAINING CONTRASTIVE VAE WITH LATENT DIM = {latent_dim}")
    print(f"{'='*80}")

    # Create model
    model = VAE(input_dim=6, latent_dim=latent_dim, hidden_dims=[32, 16]).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train with contrastive loss + β annealing
    start_time = time.time()
    model, history = train_contrastive_vae(
        model, train_loader, val_loader,
        epochs=100, device=device,
        beta_start=beta_start, beta_end=beta_end, anneal_epochs=anneal_epochs,
        gamma=gamma, tau=tau, n_clusters=n_clusters, update_interval=update_interval
    )
    train_time = time.time() - start_time
    print(f"Training time: {train_time:.1f}s")

    # Save model
    model_path = checkpoint_dir / f'vae_gra_v2_8_latent{latent_dim}_contrastive.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'label_encoder': label_encoder,
        'history': history,
        'latent_dim': latent_dim,
        'input_dim': 6,
        'beta_start': beta_start,
        'beta_end': beta_end,
        'anneal_epochs': anneal_epochs,
        'gamma': gamma,
        'tau': tau,
        'n_clusters': n_clusters,
        'update_interval': update_interval,
        'version': 'v2.8'
    }, model_path)
    print(f"Model saved to: {model_path}")

    # Get latent representations
    latent_test, labels_test = get_latent_representations(model, test_loader, device)

    # Clustering analysis
    cluster_results = cluster_analysis(
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
    axes[0, 0].set_title(f'Contrastive VAE v2.8 Loss (Latent={latent_dim})')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss components
    axes[0, 1].plot(history['train_recon'], label='Reconstruction')
    axes[0, 1].plot(history['train_kl'], label='KL Divergence')
    axes[0, 1].plot(history['train_contrastive'], label='Contrastive')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss Component')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # β and γ schedules
    axes[1, 0].plot(history['beta'], label='β')
    axes[1, 0].plot(history['gamma'], label='γ')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Loss Weights Schedule')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # ARI by k
    k_values = [r['n_clusters'] for r in cluster_results]
    ari_values = [r['ari'] for r in cluster_results]
    axes[1, 1].plot(k_values, ari_values, 'o-')
    axes[1, 1].set_xlabel('Number of Clusters')
    axes[1, 1].set_ylabel('ARI')
    axes[1, 1].set_title('Clustering Performance')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / f'training_summary_v2_8_latent{latent_dim}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # UMAP visualization
    if UMAP_AVAILABLE:
        print("  Computing UMAP projection...")
        umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        latent_umap = umap_model.fit_transform(latent_test)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Color by lithology (top 10)
        top_lithologies = pd.Series(lith_test).value_counts().head(10).index

        for lith in list(top_lithologies) + ['Other']:
            mask = [(l == lith if lith != 'Other' else l not in top_lithologies.values)
                    for l in lith_test]
            if sum(mask) > 0:
                ax.scatter(latent_umap[mask, 0], latent_umap[mask, 1],
                          label=lith, alpha=0.5, s=1)

        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_title('Contrastive VAE Latent Space (v2.8)')
        ax.legend(markerscale=3, fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'latent_space_v2_8_latent{latent_dim}.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"Visualizations saved to: {output_dir}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Model saved to: {checkpoint_dir}")
    print(f"Visualizations saved to: {output_dir}")

    # Print summary
    print(f"\nBest Results:")
    for result in cluster_results:
        if result['n_clusters'] == 12:
            print(f"  k={result['n_clusters']}: ARI={result['ari']:.3f}, Silhouette={result['silhouette']:.3f}")

if __name__ == "__main__":
    main()
