"""
VAE Lithology Model v2 - GRA + MS + NGR + RGB measurements with depth binning.

This model extends v1 by adding RGB color features:
- GRA bulk density (g/cm³)
- Magnetic susceptibility (instrument units)
- Natural Gamma Radiation (counts per second)
- R (red channel, 0-255)
- G (green channel, 0-255)
- B (blue channel, 0-255)

Dataset: 238K+ samples from 296 boreholes
Features: 6-dimensional input space
Latent spaces: 2D and 8D models for comparison
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
        return recon_x, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence."""
    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss, recon_loss, kl_loss

def load_and_prepare_data(data_path):
    """Load and prepare data for training."""
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

    # Normalize features
    scaler = StandardScaler()
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

def train_vae(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda', beta=1.0):
    """Train VAE model."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    history = {'train_loss': [], 'val_loss': [], 'train_recon': [], 'train_kl': []}

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    print(f"\nTraining VAE with latent dim={model.latent_dim}, beta={beta}")
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
            recon_X, mu, logvar = model(batch_X)
            loss, recon_loss, kl_loss = vae_loss(recon_X, batch_X, mu, logvar, beta)

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
                recon_X, mu, logvar = model(batch_X)
                loss, _, _ = vae_loss(recon_X, batch_X, mu, logvar, beta)
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

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:3d}: Train Loss={train_loss_epoch:.4f}, "
                  f"Val Loss={val_loss_epoch:.4f}, "
                  f"Recon={train_recon_epoch:.4f}, KL={train_kl_epoch:.4f}")

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

def cluster_analysis(latent_vectors, true_labels, lithology_names, n_clusters_list=[5, 10, 15, 20]):
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
    print("VAE LITHOLOGY MODEL v2 - GRA + MS + NGR + RGB")
    print("="*80)

    # Configuration
    data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')
    output_dir = Path('/home/utig5/johna/bhai/vae_v2_outputs')
    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')

    output_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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
    latent_dims = [2, 8]

    for latent_dim in latent_dims:
        print(f"\n{'='*80}")
        print(f"TRAINING VAE WITH LATENT DIM = {latent_dim}")
        print(f"{'='*80}")

        # Create model (6D input, larger hidden layers for 6D)
        model = VAE(input_dim=6, latent_dim=latent_dim, hidden_dims=[32, 16]).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Train
        start_time = time.time()
        model, history = train_vae(model, train_loader, val_loader, epochs=100, device=device)
        train_time = time.time() - start_time
        print(f"Training time: {train_time:.1f}s")

        # Save model
        model_path = checkpoint_dir / f'vae_gra_v2_latent{latent_dim}.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'scaler': scaler,
            'label_encoder': label_encoder,
            'history': history,
            'latent_dim': latent_dim,
            'input_dim': 6
        }, model_path)
        print(f"Model saved to: {model_path}")

        # Get latent representations
        latent_train, labels_train = get_latent_representations(model, train_loader, device)
        latent_test, labels_test = get_latent_representations(model, test_loader, device)

        # Clustering analysis
        cluster_results = cluster_analysis(
            latent_test, labels_test, lith_test, n_clusters_list=[5, 10, 15, 20]
        )

        # Visualizations
        print(f"\nGenerating visualizations...")

        # Plot training history
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'VAE v2 Loss (Latent Dim={latent_dim})')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(history['train_recon'], label='Reconstruction')
        axes[1].plot(history['train_kl'], label='KL Divergence')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss Component')
        axes[1].set_title('Loss Components')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(output_dir / f'training_history_v2_latent{latent_dim}.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Latent space visualization
        if latent_dim == 2:
            # Direct 2D visualization
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Color by lithology (top 10)
            top_lithologies = pd.Series(lith_test).value_counts().head(10).index

            for ax, data, title in [(axes[0], (latent_train, lith_train), 'Train'),
                                    (axes[1], (latent_test, lith_test), 'Test')]:
                lat, liths = data
                top_liths = pd.Series(liths).value_counts().head(10).index
                cols = [lith if lith in top_liths.values else 'Other' for lith in liths]

                for lith in list(top_liths) + ['Other']:
                    mask = np.array([c == lith for c in cols])
                    if mask.sum() > 0:
                        ax.scatter(lat[mask, 0], lat[mask, 1], label=lith, alpha=0.5, s=1)

                ax.set_xlabel('Latent Dim 1')
                ax.set_ylabel('Latent Dim 2')
                ax.set_title(f'2D Latent Space v2 - {title}')
                ax.legend(markerscale=3, fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'latent_space_v2_2d.png', dpi=300, bbox_inches='tight')
            plt.close()

        elif latent_dim == 8:
            # UMAP and PCA for 8D
            projections = []
            titles = []

            if UMAP_AVAILABLE:
                print("  Computing UMAP projection...")
                umap_model = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
                latent_umap = umap_model.fit_transform(latent_test)
                projections.append(latent_umap)
                titles.append('UMAP')

            print("  Computing PCA projection...")
            pca_model = PCA(n_components=2)
            latent_pca = pca_model.fit_transform(latent_test)
            projections.append(latent_pca)
            titles.append('PCA')

            # Create visualization
            n_plots = len(projections)
            fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 6))
            if n_plots == 1:
                axes = [axes]

            top_lithologies = pd.Series(lith_test).value_counts().head(10).index

            for ax, data, title in zip(axes, projections, titles):
                for lith in list(top_lithologies) + ['Other']:
                    mask = [(l == lith if lith != 'Other' else l not in top_lithologies.values)
                            for l in lith_test]
                    if sum(mask) > 0:
                        ax.scatter(data[mask, 0], data[mask, 1], label=lith, alpha=0.5, s=1)

                ax.set_xlabel(f'{title} 1')
                ax.set_ylabel(f'{title} 2')
                ax.set_title(f'8D Latent Space v2 - {title}')
                ax.legend(markerscale=3, fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / 'latent_space_v2_8d.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizations saved to: {output_dir}")

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Models saved to: {checkpoint_dir}")
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
