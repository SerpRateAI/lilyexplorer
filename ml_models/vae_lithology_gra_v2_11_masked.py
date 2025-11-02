"""
VAE GRA v2.11 - Masked Encoding for Missing Data Imputation

Key innovation: Train with random feature masking (like BERT/MAE) to learn:
1. Robust representations despite missing features
2. Feature dependencies for imputation (e.g., predict NGR+RGB from GRA+MS)
3. Better generalization

Architecture: Same as v2.6 (distribution-aware scaling, β annealing)
Training: Random feature masking during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import time
from pathlib import Path


class DistributionAwareScaler:
    """Feature-specific scaling based on observed distributions"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR (Poisson/bimodal)
        self.log_indices = [3, 4, 5]      # R, G, B (log-normal)

    def signed_log_transform(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def fit_transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        return self.scaler.fit_transform(X_transformed)

    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        return self.scaler.transform(X_transformed)

    def inverse_transform(self, X_scaled):
        X = self.scaler.inverse_transform(X_scaled)
        # Inverse transforms
        for idx in self.signed_log_indices:
            X[:, idx] = np.sign(X[:, idx]) * (np.exp(np.abs(X[:, idx])) - 1)
        for idx in self.log_indices:
            X[:, idx] = np.exp(X[:, idx]) - 1
        return X


class MaskedVAE(nn.Module):
    """VAE with masked encoding support"""
    def __init__(self, input_dim=6, latent_dim=8, hidden_dims=[32, 16]):
        super(MaskedVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(0.1)

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.bn4 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout4 = nn.Dropout(0.1)

        self.fc5 = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = self.dropout1(h)
        h = F.relu(self.bn2(self.fc2(h)))
        h = self.dropout2(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = self.dropout3(h)
        h = F.relu(self.bn4(self.fc4(h)))
        h = self.dropout4(h)
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def apply_random_masking(X, mask_prob=0.15, mask_value=0.0):
    """
    Apply random feature masking to input batch.

    Args:
        X: Input tensor [batch_size, features]
        mask_prob: Probability of masking each feature
        mask_value: Value to use for masked features (0.0 in scaled space)

    Returns:
        X_masked: Masked input
        mask: Boolean mask (True = keep, False = masked)
    """
    mask = torch.rand_like(X) > mask_prob
    X_masked = X * mask + mask_value * (~mask)
    return X_masked, mask


def apply_block_masking(X, mask_strategy='random'):
    """
    Apply block masking strategies.

    Args:
        X: Input tensor [batch_size, 6]
        mask_strategy:
            - 'random': random 15% per feature
            - 'rgb': mask all RGB (features 3,4,5) with 30% probability
            - 'ngr_rgb': mask NGR+RGB (features 2,3,4,5) with 30% probability
            - 'physical': mask all physical (0,1,2) with 20% probability

    Returns:
        X_masked, mask
    """
    batch_size = X.shape[0]
    device = X.device

    if mask_strategy == 'random':
        return apply_random_masking(X, mask_prob=0.15)

    elif mask_strategy == 'rgb':
        # Mask all RGB channels together
        mask = torch.ones_like(X, dtype=torch.bool)
        block_mask = torch.rand(batch_size) > 0.3  # 30% chance to mask RGB
        for i in range(batch_size):
            if not block_mask[i]:
                mask[i, 3:6] = False  # Mask R, G, B
        X_masked = X * mask
        return X_masked, mask

    elif mask_strategy == 'ngr_rgb':
        # Mask NGR+RGB together (simulate "only have GRA+MS" scenario)
        mask = torch.ones_like(X, dtype=torch.bool)
        block_mask = torch.rand(batch_size) > 0.3  # 30% chance
        for i in range(batch_size):
            if not block_mask[i]:
                mask[i, 2:6] = False  # Mask NGR, R, G, B
        X_masked = X * mask
        return X_masked, mask

    elif mask_strategy == 'physical':
        # Mask physical properties together
        mask = torch.ones_like(X, dtype=torch.bool)
        block_mask = torch.rand(batch_size) > 0.2  # 20% chance
        for i in range(batch_size):
            if not block_mask[i]:
                mask[i, 0:3] = False  # Mask GRA, MS, NGR
        X_masked = X * mask
        return X_masked, mask

    else:
        return apply_random_masking(X, mask_prob=0.15)


def masked_vae_loss(recon_x, x, mu, logvar, beta=0.5, mask=None):
    """
    VAE loss with optional masking.

    Args:
        recon_x: Reconstructed input [batch, features]
        x: Original input [batch, features]
        mu, logvar: Latent distribution parameters
        beta: KL weight
        mask: Optional boolean mask. If provided, loss only on masked features.
              If None, loss on all features.
    """
    # Reconstruction loss
    if mask is not None:
        # Loss only on masked features
        recon_loss = F.mse_loss(recon_x * (~mask).float(),
                                 x * (~mask).float(),
                                 reduction='sum')
    else:
        # Loss on all features (standard VAE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * kl_loss


def train_masked_vae(model, train_loader, val_loader, n_epochs=100, lr=0.001,
                     device='cpu', mask_strategy='random', loss_on_masked_only=False):
    """
    Train VAE with masked encoding.

    Args:
        mask_strategy: 'random', 'rgb', 'ngr_rgb', 'physical'
        loss_on_masked_only: If True, reconstruction loss only on masked features.
                            If False, reconstruct all features (learn dependencies).
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'beta': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(n_epochs):
        # β annealing schedule (same as v2.6)
        if epoch < 50:
            beta = 0.001 + (0.5 - 0.001) * (epoch / 50)
        else:
            beta = 0.5

        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)

            # Apply masking
            data_masked, mask = apply_block_masking(data, mask_strategy=mask_strategy)

            optimizer.zero_grad()

            # Forward pass with masked input
            recon, mu, logvar = model(data_masked)

            # Loss: reconstruct original (unmasked) data
            if loss_on_masked_only:
                loss = masked_vae_loss(recon, data, mu, logvar, beta=beta, mask=mask)
            else:
                loss = masked_vae_loss(recon, data, mu, logvar, beta=beta, mask=None)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        # Validation (no masking)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss = masked_vae_loss(recon, data, mu, logvar, beta=beta, mask=None)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['beta'].append(beta)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: β={beta:.4f}, Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    return model, history


def evaluate_clustering(model, X_test, y_test, device='cpu', k_values=[10, 12, 15, 20]):
    """Evaluate clustering performance"""
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        mu, _ = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    results = {}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent)

        ari = adjusted_rand_score(y_test, labels)
        sil = silhouette_score(latent, labels)

        results[k] = {
            'ari': ari,
            'silhouette': sil,
            'labels': labels
        }

        print(f"k={k:2d}: ARI={ari:.3f}, Silhouette={sil:.3f}")

    return results


def test_imputation(model, scaler, X_test, device='cpu', n_samples=5):
    """
    Test missing data imputation.

    Scenario: Given GRA + MS, predict NGR + RGB
    """
    print("\n" + "="*80)
    print("IMPUTATION TEST: Predict NGR+RGB from GRA+MS")
    print("="*80)

    # Select test samples
    test_samples = X_test[:n_samples].copy()

    # Scale
    test_scaled = scaler.transform(test_samples)

    # Create masked version (mask NGR + RGB)
    test_masked = test_scaled.copy()
    test_masked[:, 2:6] = 0  # Mask NGR, R, G, B

    # Impute
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(test_masked).to(device)
        recon, _, _ = model(X_tensor)
        imputed_scaled = recon.cpu().numpy()

    # Inverse transform
    true_values = scaler.inverse_transform(test_scaled)
    imputed_values = scaler.inverse_transform(imputed_scaled)

    print("\nTrue values:")
    print("     GRA       MS       NGR       R        G        B")
    for i in range(n_samples):
        print(f"{i+1}: {true_values[i,0]:6.3f}  {true_values[i,1]:7.2f}  "
              f"{true_values[i,2]:6.2f}  {true_values[i,3]:6.1f}  "
              f"{true_values[i,4]:6.1f}  {true_values[i,5]:6.1f}")

    print("\nImputed values (given GRA+MS):")
    print("     GRA       MS       NGR       R        G        B")
    for i in range(n_samples):
        print(f"{i+1}: {imputed_values[i,0]:6.3f}  {imputed_values[i,1]:7.2f}  "
              f"{imputed_values[i,2]:6.2f}  {imputed_values[i,3]:6.1f}  "
              f"{imputed_values[i,4]:6.1f}  {imputed_values[i,5]:6.1f}")

    # Calculate errors on imputed features
    mse_ngr = np.mean((true_values[:, 2] - imputed_values[:, 2])**2)
    mse_rgb = np.mean((true_values[:, 3:6] - imputed_values[:, 3:6])**2)

    print(f"\nImputation MSE:")
    print(f"  NGR: {mse_ngr:.2f}")
    print(f"  RGB: {mse_rgb:.2f}")

    return true_values, imputed_values


if __name__ == "__main__":
    print("="*80)
    print("VAE GRA v2.11 - Masked Encoding")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                    'NGR total counts (cps)', 'R', 'G', 'B']

    X = df[feature_cols].values
    lithology = df['Principal'].values
    borehole_ids = df['Borehole_ID'].values

    # Remove NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    lithology = lithology[valid_mask]
    borehole_ids = borehole_ids[valid_mask]

    print(f"Dataset: {len(X):,} samples")

    # Borehole-level split
    unique_boreholes = np.unique(borehole_ids)
    train_boreholes, test_boreholes = train_test_split(
        unique_boreholes, train_size=0.85, random_state=42
    )
    train_boreholes, val_boreholes = train_test_split(
        train_boreholes, train_size=0.7/0.85, random_state=42
    )

    train_mask = np.isin(borehole_ids, train_boreholes)
    val_mask = np.isin(borehole_ids, val_boreholes)
    test_mask = np.isin(borehole_ids, test_boreholes)

    X_train, y_train = X[train_mask], lithology[train_mask]
    X_val, y_val = X[val_mask], lithology[val_mask]
    X_test, y_test = X[test_mask], lithology[test_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Scale
    scaler = DistributionAwareScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.FloatTensor(X_train_scaled)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_scaled),
        torch.FloatTensor(X_val_scaled)
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # Train model
    print("\n" + "="*80)
    print("Training with masked encoding (NGR+RGB block masking)")
    print("="*80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = MaskedVAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16]).to(device)

    start_time = time.time()
    model, history = train_masked_vae(
        model, train_loader, val_loader,
        n_epochs=100, lr=0.001, device=device,
        mask_strategy='ngr_rgb',  # Mask NGR+RGB to learn GRA+MS → NGR+RGB
        loss_on_masked_only=False  # Reconstruct all features
    )
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.1f}s ({len(history['train_loss'])} epochs)")

    # Evaluate clustering
    print("\n" + "="*80)
    print("Clustering Evaluation")
    print("="*80)
    results = evaluate_clustering(model, X_test_scaled, y_test, device=device)

    # Test imputation
    true_vals, imputed_vals = test_imputation(model, scaler, X_test, device=device, n_samples=10)

    # Save model
    save_path = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_11_masked_ngr_rgb.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'results': results,
        'scaler': scaler,
        'mask_strategy': 'ngr_rgb',
        'train_time': train_time
    }, save_path)

    print(f"\nModel saved: {save_path}")

    # Compare to v2.6
    print("\n" + "="*80)
    print("Comparison to v2.6 Baseline")
    print("="*80)
    print(f"v2.6 (no masking):     ARI (k=12) = 0.258")
    print(f"v2.11 (masked NGR+RGB): ARI (k=12) = {results[12]['ari']:.3f}")

    improvement = (results[12]['ari'] - 0.258) / 0.258 * 100
    print(f"Improvement: {improvement:+.1f}%")
