"""
Test flexible VAE priors for improved clustering.

Compares v2.6.6 (Gaussian prior) against:
1. Mixture of Gaussians prior (VaDE-style)
2. Student's t prior (heavy tails)
3. Learnable Gaussian prior (learn μ, σ)

Research question: Does prior flexibility improve clustering beyond β annealing?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import sys
import time

sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')
from vae_lithology_gra_v2_5_model import DistributionAwareScaler

class GaussianPriorVAE(nn.Module):
    """Baseline: Standard Gaussian prior N(0,I)"""
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16]):
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
        layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*layers)

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

    def kl_divergence(self, mu, logvar):
        """KL(q(z|x) || N(0,I))"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()


class MixtureOfGaussiansPriorVAE(nn.Module):
    """Flexible: Mixture of K Gaussians prior"""
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16], n_components=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_components = n_components

        # Same encoder/decoder as baseline
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
        layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*layers)

        # Learnable prior parameters
        self.pi_logits = nn.Parameter(torch.zeros(n_components))  # Mixture weights
        self.mu_prior = nn.Parameter(torch.randn(n_components, latent_dim) * 0.1)
        self.logvar_prior = nn.Parameter(torch.zeros(n_components, latent_dim))

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

    def kl_divergence(self, mu, logvar):
        """KL(q(z|x) || Σ πₖ N(μₖ, Σₖ)) - lower bound"""
        batch_size = mu.size(0)

        # Mixture weights
        pi = F.softmax(self.pi_logits, dim=0)  # (n_components,)

        # Compute log q(z|x) for each component
        # z ~ N(mu, exp(logvar))
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # Expand for broadcasting
        z_expanded = z.unsqueeze(1)  # (batch, 1, latent_dim)
        mu_prior_expanded = self.mu_prior.unsqueeze(0)  # (1, n_components, latent_dim)
        logvar_prior_expanded = self.logvar_prior.unsqueeze(0)

        # Log probability of z under each mixture component
        log_p_z_given_c = -0.5 * (
            self.latent_dim * np.log(2 * np.pi)
            + torch.sum(logvar_prior_expanded, dim=2)
            + torch.sum((z_expanded - mu_prior_expanded).pow(2) / torch.exp(logvar_prior_expanded), dim=2)
        )  # (batch, n_components)

        # Log probability of z under mixture
        log_pi = torch.log(pi + 1e-10)  # (n_components,)
        log_p_z = torch.logsumexp(log_pi + log_p_z_given_c, dim=1)  # (batch,)

        # Log probability of z under q(z|x)
        log_q_z_given_x = -0.5 * torch.sum(
            np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
            dim=1
        )  # (batch,)

        # KL divergence
        return (log_q_z_given_x - log_p_z).mean()


class StudentTPriorVAE(nn.Module):
    """Flexible: Student's t prior (heavy tails)"""
    def __init__(self, input_dim=6, latent_dim=10, hidden_dims=[32, 16], df=3.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.df = df  # Degrees of freedom (df=3 is common)

        # Same encoder/decoder as baseline
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
        layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*layers)

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

    def kl_divergence(self, mu, logvar):
        """KL(N(μ,σ²) || Student-t(ν,0,1)) - approximation"""
        # Closed form doesn't exist, use Monte Carlo estimate
        # Sample z ~ q(z|x)
        z = self.reparameterize(mu, logvar)

        # Log q(z|x)
        log_q = -0.5 * torch.sum(
            np.log(2 * np.pi) + logvar + (z - mu).pow(2) / torch.exp(logvar),
            dim=1
        )

        # Log p(z) - Student's t
        from torch.distributions import StudentT
        t_dist = StudentT(self.df)
        log_p = torch.sum(t_dist.log_prob(z), dim=1)

        return (log_q - log_p).mean()


def train_vae(model, train_loader, val_loader, epochs=100, device='cpu',
              beta_start=0.001, beta_end=0.5, anneal_epochs=50, model_name="VAE"):
    """Train VAE with β annealing"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        # β annealing schedule
        if epoch < anneal_epochs:
            beta = beta_start + (beta_end - beta_start) * (epoch / anneal_epochs)
        else:
            beta = beta_end

        # Training
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
            kl_loss = model.kl_divergence(mu, logvar)

            loss = recon_loss + beta * kl_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum') / data.size(0)
                kl_loss = model.kl_divergence(mu, logvar)
                loss = recon_loss + beta * kl_loss
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: β={beta:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    return model


def evaluate_clustering(model, X_test_scaled, y_test, device):
    """Evaluate clustering performance"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        mu, logvar = model.encode(X_tensor)
        latent = mu.cpu().numpy()

    # Analyze latent space
    latent_stds = latent.std(axis=0)
    collapsed_dims = (latent_stds < 0.1).sum()
    effective_dim = (latent_stds >= 0.1).sum()

    # Clustering with GMM
    results = []
    for k in [10, 12, 15, 18]:
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
        labels = gmm.fit_predict(latent)
        ari = adjusted_rand_score(y_test, labels)
        sil = silhouette_score(latent, labels, sample_size=10000)
        results.append({'k': k, 'ARI': ari, 'Silhouette': sil})

    best_result = max(results, key=lambda x: x['ARI'])

    return {
        'collapsed_dims': collapsed_dims,
        'effective_dim': effective_dim,
        'best_k': best_result['k'],
        'best_ari': best_result['ARI'],
        'best_sil': best_result['Silhouette'],
        'all_results': results
    }


print("="*100)
print("TESTING FLEXIBLE VAE PRIORS FOR CLUSTERING")
print("="*100)
print()

# Load data
print("Loading data...")
df = pd.read_csv('/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv')

feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']

unique_boreholes = df['Borehole_ID'].unique()
train_boreholes, test_boreholes = train_test_split(
    unique_boreholes, train_size=0.85, random_state=42
)
train_boreholes, val_boreholes = train_test_split(
    train_boreholes, train_size=0.7/0.85, random_state=42
)

train_mask = df['Borehole_ID'].isin(train_boreholes)
val_mask = df['Borehole_ID'].isin(val_boreholes)
test_mask = df['Borehole_ID'].isin(test_boreholes)

df_train = df[train_mask].copy()
df_val = df[val_mask].copy()
df_test = df[test_mask].copy()

X_train = df_train[feature_cols].values
X_val = df_val[feature_cols].values
X_test = df_test[feature_cols].values
y_test = df_test['Principal'].values

print(f"Train: {len(df_train):,} samples")
print(f"Val:   {len(df_val):,} samples")
print(f"Test:  {len(df_test):,} samples")
print()

# Scale
scaler = DistributionAwareScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(X_train_scaled))
val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(X_val_scaled))

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
print()

# Test 3 priors
models_to_test = [
    ("Gaussian N(0,I)", GaussianPriorVAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16])),
    ("Mixture of 10 Gaussians", MixtureOfGaussiansPriorVAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16], n_components=10)),
    ("Student's t (df=3)", StudentTPriorVAE(input_dim=6, latent_dim=10, hidden_dims=[32, 16], df=3.0))
]

all_results = []

for model_name, model in models_to_test:
    print("="*100)
    print(f"TRAINING: {model_name}")
    print("="*100)

    start_time = time.time()
    model = train_vae(model, train_loader, val_loader, epochs=100, device=device,
                      beta_start=0.001, beta_end=0.5, anneal_epochs=50, model_name=model_name)
    train_time = time.time() - start_time

    print(f"\nEvaluating clustering...")
    metrics = evaluate_clustering(model, X_test_scaled, y_test, device)

    result = {
        'model': model_name,
        'train_time': train_time,
        'collapsed_dims': metrics['collapsed_dims'],
        'effective_dim': metrics['effective_dim'],
        'best_k': metrics['best_k'],
        'best_ari': metrics['best_ari'],
        'best_sil': metrics['best_sil']
    }
    all_results.append(result)

    print(f"\nResults:")
    print(f"  Collapsed dims: {metrics['collapsed_dims']}/10")
    print(f"  Effective dim: {metrics['effective_dim']}")
    print(f"  Best clustering: k={metrics['best_k']}, ARI={metrics['best_ari']:.4f}, Sil={metrics['best_sil']:.4f}")
    print(f"  Training time: {train_time:.1f}s")
    print()

# Summary
print("="*100)
print("RESULTS SUMMARY")
print("="*100)
print()

df_results = pd.DataFrame(all_results)
print(df_results.to_string(index=False))
print()

# Compare to v2.6.6 baseline
baseline_ari = 0.286  # GMM k=18 from v2.6.6
print("="*100)
print("COMPARISON TO v2.6.6 BASELINE")
print("="*100)
print(f"v2.6.6 (Gaussian prior, latent_dim=10): GMM ARI = {baseline_ari:.4f}")
print()

for _, row in df_results.iterrows():
    improvement = (row['best_ari'] - baseline_ari) / baseline_ari * 100
    symbol = "✓" if improvement > 0 else "✗"
    print(f"{symbol} {row['model']}: ARI = {row['best_ari']:.4f} ({improvement:+.1f}%)")

print()

best_model = df_results.loc[df_results['best_ari'].idxmax()]
if best_model['best_ari'] > baseline_ari:
    improvement = (best_model['best_ari'] - baseline_ari) / baseline_ari * 100
    print(f"🎯 Winner: {best_model['model']} (+{improvement:.1f}% improvement)")
    print(f"   Recommend training v2.6.7 with this prior!")
else:
    print(f"🏆 v2.6.6 remains best. Flexible priors don't improve clustering.")
    print(f"   Low β (0.5) already provides sufficient flexibility.")

print("="*100)

# Save results
df_results.to_csv('flexible_prior_comparison.csv', index=False)
print("\nResults saved to: flexible_prior_comparison.csv")
