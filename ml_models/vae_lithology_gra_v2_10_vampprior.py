"""
VAE GRA v2.10 - VampPrior Implementation

Key innovations:
- VampPrior: Mixture of posteriors instead of N(0,I)
- Better generative modeling (realistic synthetic lithologies)
- Missing data imputation (predict NGR/RGB from GRA/MS)
- Same architecture as v2.6 for fair comparison

References:
- Tomczak & Welling (2018) "VAE with a VampPrior" https://arxiv.org/abs/1705.07120
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import time

class DistributionAwareScaler:
    """Same preprocessing as v2.6"""
    def __init__(self):
        self.scaler = StandardScaler()
        self.signed_log_indices = [1, 2]  # MS, NGR
        self.log_indices = [3, 4, 5]      # R, G, B

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
        """Reverse scaling for generation"""
        X = self.scaler.inverse_transform(X_scaled)
        X_original = X.copy()

        # Reverse log transforms
        for idx in self.log_indices:
            X_original[:, idx] = np.expm1(X[:, idx])
        for idx in self.signed_log_indices:
            X_original[:, idx] = np.sign(X[:, idx]) * np.expm1(np.abs(X[:, idx]))

        return X_original


class VampPriorVAE(nn.Module):
    """VAE with VampPrior (mixture of posteriors)"""

    def __init__(self, input_dim=6, latent_dim=8, hidden_dims=[32, 16], n_components=50):
        super(VampPriorVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_components = n_components

        # Encoder (same as v2.6)
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

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder (same as v2.6)
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

        # VampPrior: K learnable pseudo-inputs
        # Initialize with small random values in scaled space
        self.pseudo_inputs = nn.Parameter(torch.randn(n_components, input_dim) * 0.01)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_vamp_prior(self):
        """
        Compute VampPrior: mixture of posteriors at pseudo-inputs
        Returns: (prior_mu, prior_logvar) for each component
        """
        # Encode all pseudo-inputs
        prior_mu, prior_logvar = self.encode(self.pseudo_inputs)
        return prior_mu, prior_logvar

    def log_p_z(self, z):
        """
        Compute log p(z) under VampPrior
        p(z) = (1/K) * sum_k N(z | mu_k, sigma_k^2)
        """
        # Get prior components
        prior_mu, prior_logvar = self.get_vamp_prior()  # [K, latent_dim]

        # Expand z to compare with all components
        z_expand = z.unsqueeze(1)  # [batch, 1, latent_dim]
        prior_mu = prior_mu.unsqueeze(0)  # [1, K, latent_dim]
        prior_logvar = prior_logvar.unsqueeze(0)  # [1, K, latent_dim]

        # Compute log probability for each component
        log_p_z_given_c = -0.5 * (
            prior_logvar +
            (z_expand - prior_mu)**2 / torch.exp(prior_logvar)
        )  # [batch, K, latent_dim]

        log_p_z_given_c = torch.sum(log_p_z_given_c, dim=2)  # [batch, K]

        # Log-sum-exp for mixture
        log_p_z = torch.logsumexp(log_p_z_given_c, dim=1) - np.log(self.n_components)

        return log_p_z

    def sample_from_prior(self, n_samples):
        """
        Sample from VampPrior for generation
        """
        # Sample component indices
        component_indices = torch.randint(0, self.n_components, (n_samples,))

        # Get component parameters
        prior_mu, prior_logvar = self.get_vamp_prior()

        # Sample from selected components
        selected_mu = prior_mu[component_indices]
        selected_logvar = prior_logvar[component_indices]

        # Reparameterization
        std = torch.exp(0.5 * selected_logvar)
        eps = torch.randn_like(std)
        z = selected_mu + eps * std

        return z


def vampprior_loss(recon_x, x, mu, logvar, z, model, beta=0.5):
    """
    VampPrior VAE loss

    Loss = Reconstruction + β * (log q(z|x) - log p(z))
    where p(z) = VampPrior (mixture of posteriors)
    """
    # Reconstruction loss
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')

    # Log q(z|x) - standard Gaussian with learned params
    log_q_z_given_x = -0.5 * torch.sum(
        logvar + (z - mu)**2 / torch.exp(logvar),
        dim=1
    )

    # Log p(z) - VampPrior (mixture of posteriors)
    log_p_z = model.log_p_z(z)

    # KL divergence = log q(z|x) - log p(z)
    kl_loss = torch.sum(log_q_z_given_x - log_p_z)

    return recon_loss + beta * kl_loss


class LithologyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_vampprior_vae(model, train_loader, val_loader, n_epochs=100,
                        lr=0.001, device='cpu', beta_schedule='anneal'):
    """
    Train VampPrior VAE with β annealing (same schedule as v2.6)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    model = model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'beta': []}

    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(n_epochs):
        # β annealing (same as v2.6)
        if beta_schedule == 'anneal':
            if epoch < 50:
                beta = 0.001 + (0.5 - 0.001) * (epoch / 50)
            else:
                beta = 0.5
        else:
            beta = 0.5

        # Training
        model.train()
        train_loss = 0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = model(batch_x)
            z = model.reparameterize(mu, logvar)
            loss = vampprior_loss(recon_x, batch_x, mu, logvar, z, model, beta)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.to(device)
                recon_x, mu, logvar = model(batch_x)
                z = model.reparameterize(mu, logvar)
                loss = vampprior_loss(recon_x, batch_x, mu, logvar, z, model, beta)
                val_loss += loss.item()

        val_loss /= len(val_loader.dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['beta'].append(beta)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: β={beta:.3f}, Train Loss={train_loss:.3f}, "
                  f"Val Loss={val_loss:.3f}")

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    return model, history


def evaluate_clustering(model, X_test, y_test, k_values=[10, 12, 15, 20], device='cpu'):
    """Evaluate clustering performance"""
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        mu, _ = model.encode(X_tensor)
        z = mu.cpu().numpy()

    results = {}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(z)

        sil = silhouette_score(z, labels)
        ari = adjusted_rand_score(y_test, labels)

        results[k] = {'ari': ari, 'silhouette': sil}
        print(f"k={k:2d}: ARI={ari:.3f}, Silhouette={sil:.3f}")

    return results


def generate_synthetic_lithologies(model, scaler, n_samples=100, device='cpu'):
    """
    Generate synthetic lithology profiles by sampling from VampPrior

    Returns:
        synthetic_data: [n_samples, 6] array of (GRA, MS, NGR, R, G, B)
    """
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        # Sample from VampPrior
        z = model.sample_from_prior(n_samples).to(device)

        # Decode
        synthetic_scaled = model.decode(z).cpu().numpy()

        # Inverse transform to original scale
        synthetic_original = scaler.inverse_transform(synthetic_scaled)

    return synthetic_original


def impute_missing_features(model, scaler, partial_data, missing_mask, device='cpu'):
    """
    Impute missing features

    Args:
        partial_data: [n_samples, 6] with observed values, 0 for missing
        missing_mask: [n_samples, 6] boolean, True = missing

    Returns:
        imputed_data: [n_samples, 6] with missing values filled in
    """
    model.eval()
    model = model.to(device)

    # Scale partial data
    partial_scaled = scaler.transform(partial_data)

    with torch.no_grad():
        # Encode (uses all values, even missing=0)
        X_tensor = torch.FloatTensor(partial_scaled).to(device)
        mu, _ = model.encode(X_tensor)

        # Decode to get predictions
        predicted_scaled = model.decode(mu).cpu().numpy()

        # Combine observed + predicted
        imputed_scaled = partial_scaled.copy()
        imputed_scaled[missing_mask] = predicted_scaled[missing_mask]

        # Inverse transform
        imputed_original = scaler.inverse_transform(imputed_scaled)

    return imputed_original


if __name__ == "__main__":
    print("="*80)
    print("VAE GRA v2.10 - VampPrior Training")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                    'NGR total counts (cps)', 'R', 'G', 'B']

    X = df[feature_cols].values
    lithology = df['Principal'].values
    borehole_ids = df['Borehole_ID'].values

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    lithology = lithology[valid_mask]
    borehole_ids = borehole_ids[valid_mask]

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(lithology)

    # Borehole-level split
    unique_boreholes = np.unique(borehole_ids)
    train_boreholes, temp_boreholes = train_test_split(
        unique_boreholes, train_size=0.70, random_state=42
    )
    val_boreholes, test_boreholes = train_test_split(
        temp_boreholes, train_size=0.5, random_state=42
    )

    train_mask = np.isin(borehole_ids, train_boreholes)
    val_mask = np.isin(borehole_ids, val_boreholes)
    test_mask = np.isin(borehole_ids, test_boreholes)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Scale data
    scaler = DistributionAwareScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Create dataloaders
    train_dataset = LithologyDataset(X_train_scaled, y_train)
    val_dataset = LithologyDataset(X_val_scaled, y_val)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

    # Create model
    print("\nInitializing VampPrior VAE...")
    model = VampPriorVAE(
        input_dim=6,
        latent_dim=8,
        hidden_dims=[32, 16],
        n_components=50  # K=50 pseudo-inputs
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Model: {sum(p.numel() for n, p in model.named_parameters() if 'pseudo' not in n):,}")
    print(f"  Pseudo-inputs: {model.pseudo_inputs.numel()}")

    # Train
    print("\nTraining with β annealing (0.001→0.5 over 50 epochs)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    start_time = time.time()
    model, history = train_vampprior_vae(
        model, train_loader, val_loader,
        n_epochs=100, lr=0.001, device=device, beta_schedule='anneal'
    )
    train_time = time.time() - start_time

    print(f"\nTraining completed in {train_time:.1f}s")

    # Evaluate
    print("\n" + "="*80)
    print("Clustering Evaluation")
    print("="*80)
    results = evaluate_clustering(model, X_test_scaled, y_test, device=device)

    # Save model
    save_path = Path('/home/utig5/johna/bhai/ml_models/checkpoints/vae_gra_v2_10_vampprior_K50.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'results': results,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'n_components': 50
    }, save_path)

    print(f"\nModel saved: {save_path}")

    # Demo: Generate synthetic lithologies
    print("\n" + "="*80)
    print("Synthetic Generation Demo")
    print("="*80)
    synthetic = generate_synthetic_lithologies(model, scaler, n_samples=10, device=device)
    print("Generated 10 synthetic lithology samples:")
    print("     GRA       MS       NGR       R        G        B")
    for i, sample in enumerate(synthetic[:5]):
        print(f"{i+1}: {sample[0]:6.3f}  {sample[1]:7.2f}  {sample[2]:6.2f}  "
              f"{sample[3]:6.1f}  {sample[4]:6.1f}  {sample[5]:6.1f}")
    print("...")

    # Demo: Missing data imputation
    print("\n" + "="*80)
    print("Missing Data Imputation Demo")
    print("="*80)
    # Take real samples, mask NGR+RGB, try to predict
    test_samples = X_test[:5].copy()
    partial = test_samples.copy()
    partial[:, 2:] = 0  # Mask NGR, R, G, B

    missing_mask = np.zeros_like(test_samples, dtype=bool)
    missing_mask[:, 2:] = True

    imputed = impute_missing_features(model, scaler, partial, missing_mask, device=device)

    print("Given GRA + MS, predict NGR + RGB:")
    print("\nTrue values:")
    print("     GRA       MS       NGR       R        G        B")
    for i, sample in enumerate(test_samples):
        print(f"{i+1}: {sample[0]:6.3f}  {sample[1]:7.2f}  {sample[2]:6.2f}  "
              f"{sample[3]:6.1f}  {sample[4]:6.1f}  {sample[5]:6.1f}")

    print("\nImputed values:")
    print("     GRA       MS       NGR       R        G        B")
    for i, sample in enumerate(imputed):
        print(f"{i+1}: {sample[0]:6.3f}  {sample[1]:7.2f}  {sample[2]:6.2f}  "
              f"{sample[3]:6.1f}  {sample[4]:6.1f}  {sample[5]:6.1f}")

    print("\n" + "="*80)
    print("VampPrior VAE v2.10 Complete!")
    print("="*80)
