"""
VAE GRA v2.6.4: Dual Pre-training with Fusion

Stage 1a: Pre-train physical encoder (GRA+MS+NGR → 4D latent)
Stage 1b: Pre-train RGB encoder (R+G+B → 4D latent)
Stage 2: Concatenate encoders (4D + 4D = 8D), fine-tune fusion

Key difference from v2.6.2: Each modality gets its own latent space first,
then we learn how to combine them (instead of forcing RGB into pre-trained
physical latent space).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler


class DistributionAwareScaler:
    """Custom scaler for physical or RGB features."""
    def __init__(self, feature_type='physical'):
        """
        Args:
            feature_type: 'physical' (GRA+MS+NGR) or 'rgb' (R+G+B)
        """
        self.scaler = StandardScaler()
        self.feature_type = feature_type

    def fit_transform(self, X):
        X_transformed = X.copy()

        if self.feature_type == 'physical':
            # Signed log for MS (idx 1) and NGR (idx 2)
            X_transformed[:, 1] = np.sign(X[:, 1]) * np.log1p(np.abs(X[:, 1]))
            X_transformed[:, 2] = np.sign(X[:, 2]) * np.log1p(np.abs(X[:, 2]))
        elif self.feature_type == 'rgb':
            # Regular log for RGB (all channels)
            X_transformed = np.log1p(X)

        X_scaled = self.scaler.fit_transform(X_transformed)
        return X_scaled

    def transform(self, X):
        X_transformed = X.copy()

        if self.feature_type == 'physical':
            X_transformed[:, 1] = np.sign(X[:, 1]) * np.log1p(np.abs(X[:, 1]))
            X_transformed[:, 2] = np.sign(X[:, 2]) * np.log1p(np.abs(X[:, 2]))
        elif self.feature_type == 'rgb':
            X_transformed = np.log1p(X)

        X_scaled = self.scaler.transform(X_transformed)
        return X_scaled


class SingleModalityVAE(nn.Module):
    """
    VAE for single modality (physical or RGB).
    Used for Stage 1a and 1b pre-training.
    """
    def __init__(self, input_dim=3, latent_dim=4, hidden_dims=[32, 16]):
        super(SingleModalityVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])

        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dims[1])
        self.bn3 = nn.BatchNorm1d(hidden_dims[1])
        self.fc4 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.bn4 = nn.BatchNorm1d(hidden_dims[0])
        self.fc5 = nn.Linear(hidden_dims[0], input_dim)

    def encode(self, x):
        h = F.relu(self.bn1(self.fc1(x)))
        h = F.relu(self.bn2(self.fc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.bn3(self.fc3(z)))
        h = F.relu(self.bn4(self.fc4(h)))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class DualEncoderVAE(nn.Module):
    """
    Dual encoder VAE with concatenated latent space.
    Used for Stage 2 fusion training.

    Architecture:
        Physical encoder: 3D → 4D latent
        RGB encoder: 3D → 4D latent
        Combined: 8D latent (4D + 4D)
        Physical decoder: 8D → 3D
        RGB decoder: 8D → 3D
    """
    def __init__(self, hidden_dims=[32, 16]):
        super(DualEncoderVAE, self).__init__()

        # Physical encoder (3D → 4D)
        self.phys_fc1 = nn.Linear(3, hidden_dims[0])
        self.phys_bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.phys_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.phys_bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.phys_fc_mu = nn.Linear(hidden_dims[1], 4)
        self.phys_fc_logvar = nn.Linear(hidden_dims[1], 4)

        # RGB encoder (3D → 4D)
        self.rgb_fc1 = nn.Linear(3, hidden_dims[0])
        self.rgb_bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.rgb_fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.rgb_bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.rgb_fc_mu = nn.Linear(hidden_dims[1], 4)
        self.rgb_fc_logvar = nn.Linear(hidden_dims[1], 4)

        # Physical decoder (8D → 3D)
        self.phys_dec_fc1 = nn.Linear(8, hidden_dims[1])
        self.phys_dec_bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.phys_dec_fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.phys_dec_bn2 = nn.BatchNorm1d(hidden_dims[0])
        self.phys_dec_fc3 = nn.Linear(hidden_dims[0], 3)

        # RGB decoder (8D → 3D)
        self.rgb_dec_fc1 = nn.Linear(8, hidden_dims[1])
        self.rgb_dec_bn1 = nn.BatchNorm1d(hidden_dims[1])
        self.rgb_dec_fc2 = nn.Linear(hidden_dims[1], hidden_dims[0])
        self.rgb_dec_bn2 = nn.BatchNorm1d(hidden_dims[0])
        self.rgb_dec_fc3 = nn.Linear(hidden_dims[0], 3)

    def encode_physical(self, x_phys):
        h = F.relu(self.phys_bn1(self.phys_fc1(x_phys)))
        h = F.relu(self.phys_bn2(self.phys_fc2(h)))
        return self.phys_fc_mu(h), self.phys_fc_logvar(h)

    def encode_rgb(self, x_rgb):
        h = F.relu(self.rgb_bn1(self.rgb_fc1(x_rgb)))
        h = F.relu(self.rgb_bn2(self.rgb_fc2(h)))
        return self.rgb_fc_mu(h), self.rgb_fc_logvar(h)

    def encode(self, x):
        """Encode full 6D input (3D physical + 3D RGB)."""
        x_phys = x[:, :3]  # GRA, MS, NGR
        x_rgb = x[:, 3:]   # R, G, B

        mu_phys, logvar_phys = self.encode_physical(x_phys)
        mu_rgb, logvar_rgb = self.encode_rgb(x_rgb)

        # Concatenate latent spaces
        mu = torch.cat([mu_phys, mu_rgb], dim=1)  # 8D
        logvar = torch.cat([logvar_phys, logvar_rgb], dim=1)  # 8D

        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_physical(self, z):
        h = F.relu(self.phys_dec_bn1(self.phys_dec_fc1(z)))
        h = F.relu(self.phys_dec_bn2(self.phys_dec_fc2(h)))
        return self.phys_dec_fc3(h)

    def decode_rgb(self, z):
        h = F.relu(self.rgb_dec_bn1(self.rgb_dec_fc1(z)))
        h = F.relu(self.rgb_dec_bn2(self.rgb_dec_fc2(h)))
        return self.rgb_dec_fc3(h)

    def decode(self, z):
        """Decode from 8D latent to 6D output."""
        x_phys = self.decode_physical(z)
        x_rgb = self.decode_rgb(z)
        return torch.cat([x_phys, x_rgb], dim=1)  # 6D output

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def load_pretrained_encoders(phys_checkpoint_path, rgb_checkpoint_path, device='cpu'):
    """
    Load pre-trained physical and RGB encoders into DualEncoderVAE.

    Args:
        phys_checkpoint_path: Path to Stage 1a physical VAE checkpoint
        rgb_checkpoint_path: Path to Stage 1b RGB VAE checkpoint
        device: Device to load model on

    Returns:
        model: DualEncoderVAE with pre-trained encoder weights
    """
    # Load Stage 1a and 1b checkpoints
    phys_checkpoint = torch.load(phys_checkpoint_path, map_location=device, weights_only=False)
    rgb_checkpoint = torch.load(rgb_checkpoint_path, map_location=device, weights_only=False)

    # Create Stage 1 models
    phys_vae = SingleModalityVAE(input_dim=3, latent_dim=4, hidden_dims=[32, 16])
    rgb_vae = SingleModalityVAE(input_dim=3, latent_dim=4, hidden_dims=[32, 16])

    phys_vae.load_state_dict(phys_checkpoint['model_state_dict'])
    rgb_vae.load_state_dict(rgb_checkpoint['model_state_dict'])

    # Create Stage 2 model
    dual_model = DualEncoderVAE(hidden_dims=[32, 16]).to(device)

    # Transfer encoder weights
    with torch.no_grad():
        # Physical encoder
        dual_model.phys_fc1.weight.copy_(phys_vae.fc1.weight)
        dual_model.phys_fc1.bias.copy_(phys_vae.fc1.bias)
        dual_model.phys_bn1.weight.copy_(phys_vae.bn1.weight)
        dual_model.phys_bn1.bias.copy_(phys_vae.bn1.bias)
        dual_model.phys_bn1.running_mean.copy_(phys_vae.bn1.running_mean)
        dual_model.phys_bn1.running_var.copy_(phys_vae.bn1.running_var)

        dual_model.phys_fc2.weight.copy_(phys_vae.fc2.weight)
        dual_model.phys_fc2.bias.copy_(phys_vae.fc2.bias)
        dual_model.phys_bn2.weight.copy_(phys_vae.bn2.weight)
        dual_model.phys_bn2.bias.copy_(phys_vae.bn2.bias)
        dual_model.phys_bn2.running_mean.copy_(phys_vae.bn2.running_mean)
        dual_model.phys_bn2.running_var.copy_(phys_vae.bn2.running_var)

        dual_model.phys_fc_mu.weight.copy_(phys_vae.fc_mu.weight)
        dual_model.phys_fc_mu.bias.copy_(phys_vae.fc_mu.bias)
        dual_model.phys_fc_logvar.weight.copy_(phys_vae.fc_logvar.weight)
        dual_model.phys_fc_logvar.bias.copy_(phys_vae.fc_logvar.bias)

        # RGB encoder
        dual_model.rgb_fc1.weight.copy_(rgb_vae.fc1.weight)
        dual_model.rgb_fc1.bias.copy_(rgb_vae.fc1.bias)
        dual_model.rgb_bn1.weight.copy_(rgb_vae.bn1.weight)
        dual_model.rgb_bn1.bias.copy_(rgb_vae.bn1.bias)
        dual_model.rgb_bn1.running_mean.copy_(rgb_vae.bn1.running_mean)
        dual_model.rgb_bn1.running_var.copy_(rgb_vae.bn1.running_var)

        dual_model.rgb_fc2.weight.copy_(rgb_vae.fc2.weight)
        dual_model.rgb_fc2.bias.copy_(rgb_vae.fc2.bias)
        dual_model.rgb_bn2.weight.copy_(rgb_vae.bn2.weight)
        dual_model.rgb_bn2.bias.copy_(rgb_vae.bn2.bias)
        dual_model.rgb_bn2.running_mean.copy_(rgb_vae.bn2.running_mean)
        dual_model.rgb_bn2.running_var.copy_(rgb_vae.bn2.running_var)

        dual_model.rgb_fc_mu.weight.copy_(rgb_vae.fc_mu.weight)
        dual_model.rgb_fc_mu.bias.copy_(rgb_vae.fc_mu.bias)
        dual_model.rgb_fc_logvar.weight.copy_(rgb_vae.fc_logvar.weight)
        dual_model.rgb_fc_logvar.bias.copy_(rgb_vae.fc_logvar.bias)

    print("Loaded pre-trained encoders:")
    print(f"  Physical encoder: {phys_checkpoint['epochs']} epochs, {phys_checkpoint['training_time']:.1f}s")
    print(f"  RGB encoder: {rgb_checkpoint['epochs']} epochs, {rgb_checkpoint['training_time']:.1f}s")
    print("  Decoders initialized randomly (will be trained in Stage 2)")

    return dual_model, phys_checkpoint['scaler'], rgb_checkpoint['scaler']


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with configurable β."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    loss = recon_loss + beta * kl_loss
    return loss, recon_loss, kl_loss


def train_vae_with_annealing(model, train_loader, val_loader, epochs=100,
                             learning_rate=1e-3, device='cpu',
                             beta_start=0.001, beta_end=0.5, anneal_epochs=50,
                             patience=20):
    """Train VAE with β annealing."""
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        'train_loss': [], 'train_recon': [], 'train_kl': [],
        'val_loss': [], 'val_recon': [], 'val_kl': [],
        'beta': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        # β annealing
        if epoch < anneal_epochs:
            progress = epoch / anneal_epochs
            current_beta = beta_start + (beta_end - beta_start) * progress
        else:
            current_beta = beta_end

        history['beta'].append(current_beta)

        # Training
        model.train()
        train_loss, train_recon, train_kl = 0, 0, 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=current_beta)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_kl /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_recon, val_kl = 0, 0, 0

        with torch.no_grad():
            for batch_idx, (data,) in enumerate(val_loader):
                data = data.to(device)
                recon, mu, logvar = model(data)
                loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta=current_beta)

                val_loss += loss.item()
                val_recon += recon_loss.item()
                val_kl += kl_loss.item()

        val_loss /= len(val_loader)
        val_recon /= len(val_loader)
        val_kl /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['val_loss'].append(val_loss)
        history['val_recon'].append(val_recon)
        history['val_kl'].append(val_kl)

        print(f'Epoch {epoch+1:3d}/{epochs} | β={current_beta:.4f} | '
              f'Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, KL: {train_kl:.4f}) | '
              f'Val Loss: {val_loss:.4f} (Recon: {val_recon:.4f}, KL: {val_kl:.4f})')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    return model, history
