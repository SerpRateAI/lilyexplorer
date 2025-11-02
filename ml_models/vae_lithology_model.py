"""
Variational Autoencoder (VAE) for Borehole Lithology Learning

This module implements a VAE that learns latent representations of lithological
properties from four continuous physical measurements:
1. Porosity (vol%)
2. Grain density (g/cm^3)
3. P-wave velocity (m/s)
4. Thermal conductivity (W/(m*K))

The learned latent space should capture lithological relationships that can be
validated against actual lithology labels from the LILY database.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
from typing import Tuple, Dict, List
import json
from datetime import datetime


class BoreholeDataset(Dataset):
    """PyTorch Dataset for borehole measurements with 4 physical properties."""

    def __init__(self, data: np.ndarray, labels: pd.DataFrame = None):
        """
        Args:
            data: numpy array of shape (n_samples, 4) with normalized measurements
            labels: DataFrame containing lithology labels and metadata (optional)
        """
        self.data = torch.FloatTensor(data)
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class VAE(nn.Module):
    """
    Variational Autoencoder for learning lithological relationships.

    Architecture:
    - Encoder: Maps 4D input to latent space (2D or 8D)
    - Decoder: Reconstructs 4D output from latent space
    - Uses reparameterization trick for sampling

    The latent space is designed to capture:
    - Lithology-dependent property relationships
    - Physical constraints (e.g., porosity-density correlations)
    - Multi-modal distributions reflecting different rock types
    """

    def __init__(self, input_dim: int = 4, latent_dim: int = 8, hidden_dims: List[int] = None):
        """
        Args:
            input_dim: Number of input features (4 for our properties)
            latent_dim: Dimensionality of latent space (2 for visualization, 8 for capacity)
            hidden_dims: List of hidden layer dimensions for encoder/decoder
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Default hidden dimensions if not specified
        if hidden_dims is None:
            hidden_dims = [32, 64, 128] if latent_dim > 2 else [16, 32]

        self.hidden_dims = hidden_dims

        # Encoder: Maps input to latent distribution parameters (mu, log_var)
        encoder_layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent distribution parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder: Maps latent space back to input space
        decoder_layers = []
        prev_dim = latent_dim

        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim

        # Final reconstruction layer (no activation for continuous outputs)
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            mu: Mean of latent distribution (batch_size, latent_dim)
            log_var: Log variance of latent distribution (batch_size, latent_dim)
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon

        Args:
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution

        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to reconstructed input.

        Args:
            z: Latent vector of shape (batch_size, latent_dim)

        Returns:
            x_recon: Reconstructed input (batch_size, input_dim)
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input tensor

        Returns:
            x_recon: Reconstructed input
            mu: Latent distribution mean
            log_var: Latent distribution log variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation (deterministic - using mean).

        Args:
            x: Input tensor

        Returns:
            z: Latent representation
        """
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu


def vae_loss(x_recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             log_var: torch.Tensor, beta: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE loss function: Reconstruction loss + KL divergence.

    Args:
        x_recon: Reconstructed input
        x: Original input
        mu: Latent distribution mean
        log_var: Latent distribution log variance
        beta: Weight for KL divergence term (β-VAE)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (MSE)
        kl_loss: KL divergence
    """
    # Reconstruction loss (MSE for continuous outputs)
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    kl_loss = kl_loss / x.size(0)  # Normalize by batch size

    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


def load_and_merge_data(borehole_list: List[str], data_dir: str = '/home/utig5/johna/bhai/datasets') -> pd.DataFrame:
    """
    Load and merge data from MAD, PWC, and TCON datasets.

    This function:
    1. Loads the three datasets
    2. Creates borehole IDs (Exp-Site-Hole)
    3. Filters to specified boreholes
    4. Merges on borehole ID and depth
    5. Extracts the 4 required measurements + lithology labels

    Args:
        borehole_list: List of borehole IDs (e.g., '318-U1356A')
        data_dir: Directory containing the CSV files

    Returns:
        merged_df: DataFrame with all measurements and labels
    """
    data_dir = Path(data_dir)

    logging.info("Loading MAD dataset (porosity, grain density)...")
    mad_df = pd.read_csv(data_dir / 'MAD_DataLITH.csv')
    logging.info(f"MAD dataset: {len(mad_df)} rows")

    logging.info("Loading PWC dataset (P-wave velocity)...")
    pwc_df = pd.read_csv(data_dir / 'PWC_DataLITH.csv')
    logging.info(f"PWC dataset: {len(pwc_df)} rows")

    logging.info("Loading TCON dataset (thermal conductivity)...")
    tcon_df = pd.read_csv(data_dir / 'TCON_DataLITH.csv')
    logging.info(f"TCON dataset: {len(tcon_df)} rows")

    # Create borehole IDs
    def create_borehole_id(df):
        return df['Exp'].astype(str) + '-' + df['Site'] + df['Hole']

    mad_df['Borehole_ID'] = create_borehole_id(mad_df)
    pwc_df['Borehole_ID'] = create_borehole_id(pwc_df)
    tcon_df['Borehole_ID'] = create_borehole_id(tcon_df)

    # Filter to specified boreholes
    logging.info(f"Filtering to {len(borehole_list)} specified boreholes...")
    mad_df = mad_df[mad_df['Borehole_ID'].isin(borehole_list)]
    pwc_df = pwc_df[pwc_df['Borehole_ID'].isin(borehole_list)]
    tcon_df = tcon_df[tcon_df['Borehole_ID'].isin(borehole_list)]

    logging.info(f"After filtering - MAD: {len(mad_df)}, PWC: {len(pwc_df)}, TCON: {len(tcon_df)}")

    # Select relevant columns and prepare for merge
    # Use Depth CSF-A (core depth below seafloor, method A) as merge key
    mad_cols = ['Borehole_ID', 'Depth CSF-A (m)', 'Porosity (vol%)', 'Grain density (g/cm^3)',
                'Principal', 'Prefix', 'Suffix', 'Full Lithology', 'Simplified Lithology',
                'Latitude (DD)', 'Longitude (DD)', 'Expanded Core Type']
    mad_subset = mad_df[mad_cols].copy()

    # P-wave velocity: use first available measurement (prefer unknown, then x, y, z)
    pwc_df['P_wave_velocity'] = pwc_df[['P-wave velocity unknown (m/s)',
                                         'P-wave velocity x (m/s)',
                                         'P-wave velocity y (m/s)',
                                         'P-wave velocity z (m/s)']].bfill(axis=1).iloc[:, 0]

    pwc_cols = ['Borehole_ID', 'Depth CSF-A (m)', 'P_wave_velocity']
    pwc_subset = pwc_df[pwc_cols].copy()

    tcon_cols = ['Borehole_ID', 'Depth CSF-A (m)', 'Thermal conductivity mean (W/(m*K))']
    tcon_subset = tcon_df[tcon_cols].copy()

    # For merging, create depth bins to group nearby measurements
    # Use 5cm resolution (0.05m) for depth binning
    depth_resolution = 0.05
    mad_subset['Depth_bin'] = (mad_subset['Depth CSF-A (m)'] / depth_resolution).round() * depth_resolution
    pwc_subset['Depth_bin'] = (pwc_subset['Depth CSF-A (m)'] / depth_resolution).round() * depth_resolution
    tcon_subset['Depth_bin'] = (tcon_subset['Depth CSF-A (m)'] / depth_resolution).round() * depth_resolution

    # Merge datasets on Borehole_ID and binned depth
    logging.info(f"Merging datasets on Borehole_ID and depth (resolution: {depth_resolution}m)...")

    # First merge MAD with PWC
    merged = pd.merge(
        mad_subset, pwc_subset,
        on=['Borehole_ID', 'Depth_bin'],
        how='inner',
        suffixes=('', '_pwc')
    )

    # Then merge with TCON
    merged = pd.merge(
        merged, tcon_subset,
        on=['Borehole_ID', 'Depth_bin'],
        how='inner',
        suffixes=('', '_tcon')
    )

    logging.info(f"Merged dataset: {len(merged)} rows")

    # Drop rows with missing values in the 4 key measurements
    key_cols = ['Porosity (vol%)', 'Grain density (g/cm^3)',
                'P_wave_velocity', 'Thermal conductivity mean (W/(m*K))']

    before_dropna = len(merged)
    merged = merged.dropna(subset=key_cols)
    logging.info(f"After removing missing values: {len(merged)} rows (dropped {before_dropna - len(merged)})")

    # Remove outliers (basic filtering for physically reasonable values)
    # Porosity: 0-100 vol%
    # Grain density: 1.5-5.0 g/cm^3 (covers most geological materials)
    # P-wave velocity: 1000-8000 m/s (seawater ~1500, basalt ~6000)
    # Thermal conductivity: 0.1-5.0 W/(m*K) (sediments ~1, basalt ~2-3)

    before_filter = len(merged)
    merged = merged[
        (merged['Porosity (vol%)'] >= 0) & (merged['Porosity (vol%)'] <= 100) &
        (merged['Grain density (g/cm^3)'] >= 1.5) & (merged['Grain density (g/cm^3)'] <= 5.0) &
        (merged['P_wave_velocity'] >= 1000) & (merged['P_wave_velocity'] <= 8000) &
        (merged['Thermal conductivity mean (W/(m*K))'] >= 0.1) &
        (merged['Thermal conductivity mean (W/(m*K))'] <= 5.0)
    ]
    logging.info(f"After outlier filtering: {len(merged)} rows (dropped {before_filter - len(merged)})")

    return merged


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
                 random_state: int = 42) -> Dict:
    """
    Prepare data for training: normalize, split by borehole, create datasets.

    Split strategy:
    - Split by borehole (not random samples) to avoid data leakage
    - Train/val/test splits respect borehole boundaries

    Args:
        df: Merged DataFrame with all measurements
        test_size: Fraction of boreholes for test set
        val_size: Fraction of training boreholes for validation
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing:
            - train_loader, val_loader, test_loader
            - scaler (fitted StandardScaler)
            - train_df, val_df, test_df (for labels)
            - feature_names
    """
    # Get unique boreholes
    unique_boreholes = df['Borehole_ID'].unique()
    n_boreholes = len(unique_boreholes)
    logging.info(f"Total unique boreholes: {n_boreholes}")

    # Split boreholes into train/test
    train_val_boreholes, test_boreholes = train_test_split(
        unique_boreholes, test_size=test_size, random_state=random_state
    )

    # Split train into train/val
    train_boreholes, val_boreholes = train_test_split(
        train_val_boreholes, test_size=val_size/(1-test_size), random_state=random_state
    )

    logging.info(f"Borehole split - Train: {len(train_boreholes)}, "
                 f"Val: {len(val_boreholes)}, Test: {len(test_boreholes)}")

    # Split data by boreholes
    train_df = df[df['Borehole_ID'].isin(train_boreholes)].copy()
    val_df = df[df['Borehole_ID'].isin(val_boreholes)].copy()
    test_df = df[df['Borehole_ID'].isin(test_boreholes)].copy()

    logging.info(f"Sample split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Extract features
    feature_names = ['Porosity (vol%)', 'Grain density (g/cm^3)',
                     'P_wave_velocity', 'Thermal conductivity mean (W/(m*K))']

    X_train = train_df[feature_names].values
    X_val = val_df[feature_names].values
    X_test = test_df[feature_names].values

    # Normalize features (fit on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logging.info("Feature statistics (before normalization):")
    for i, name in enumerate(feature_names):
        logging.info(f"  {name}: mean={X_train[:, i].mean():.3f}, std={X_train[:, i].std():.3f}")

    # Create PyTorch datasets
    train_dataset = BoreholeDataset(X_train_scaled, train_df)
    val_dataset = BoreholeDataset(X_val_scaled, val_df)
    test_dataset = BoreholeDataset(X_test_scaled, test_df)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'scaler': scaler,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'feature_names': feature_names,
        'train_boreholes': train_boreholes,
        'val_boreholes': val_boreholes,
        'test_boreholes': test_boreholes
    }


def train_vae(model: VAE, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int = 100, learning_rate: float = 1e-3, beta: float = 1.0,
              device: str = 'cpu', checkpoint_dir: str = '/home/utig5/johna/bhai/ml_models/checkpoints',
              log_dir: str = '/home/utig5/johna/bhai/ml_models/logs') -> Dict:
    """
    Train the VAE model.

    Args:
        model: VAE model
        train_loader: Training data loader
        val_loader: Validation data loader
        n_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        beta: Weight for KL divergence (β-VAE)
        device: Device for training ('cpu' or 'cuda')
        checkpoint_dir: Directory to save model checkpoints
        log_dir: Directory to save training logs

    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        'train_loss': [],
        'train_recon_loss': [],
        'train_kl_loss': [],
        'val_loss': [],
        'val_recon_loss': [],
        'val_kl_loss': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    logging.info(f"Starting training for {n_epochs} epochs...")
    logging.info(f"Device: {device}, Learning rate: {learning_rate}, Beta: {beta}")

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_recon_losses = []
        train_kl_losses = []

        for batch_idx, x in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_recon, mu, log_var = model(x)
            loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, log_var, beta)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_recon_losses.append(recon_loss.item())
            train_kl_losses.append(kl_loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        val_recon_losses = []
        val_kl_losses = []

        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                x_recon, mu, log_var = model(x)
                loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, log_var, beta)

                val_losses.append(loss.item())
                val_recon_losses.append(recon_loss.item())
                val_kl_losses.append(kl_loss.item())

        # Record epoch statistics
        epoch_train_loss = np.mean(train_losses)
        epoch_train_recon = np.mean(train_recon_losses)
        epoch_train_kl = np.mean(train_kl_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_val_recon = np.mean(val_recon_losses)
        epoch_val_kl = np.mean(val_kl_losses)

        history['train_loss'].append(epoch_train_loss)
        history['train_recon_loss'].append(epoch_train_recon)
        history['train_kl_loss'].append(epoch_train_kl)
        history['val_loss'].append(epoch_val_loss)
        history['val_recon_loss'].append(epoch_val_recon)
        history['val_kl_loss'].append(epoch_val_kl)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        scheduler.step(epoch_val_loss)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch+1}/{n_epochs} - "
                        f"Train Loss: {epoch_train_loss:.4f} "
                        f"(Recon: {epoch_train_recon:.4f}, KL: {epoch_train_kl:.4f}) - "
                        f"Val Loss: {epoch_val_loss:.4f} "
                        f"(Recon: {epoch_val_recon:.4f}, KL: {epoch_val_kl:.4f})")

        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'history': history
            }
            torch.save(checkpoint, checkpoint_dir / 'vae_lithology_model_best.pth')
            logging.info(f"  Saved best model (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': epoch_val_loss,
        'history': history
    }
    torch.save(checkpoint, checkpoint_dir / 'vae_lithology_model_final.pth')

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(Path(log_dir) / 'training_history.csv', index=False)

    logging.info("Training completed!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")

    return history


def main():
    """Main training script."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'/home/utig5/johna/bhai/ml_models/logs/vae_lithology_training_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("="*80)
    logging.info("VAE Lithology Model Training")
    logging.info("="*80)

    # Load borehole list from report.md
    borehole_list_str = """318-U1356A, 318-U1358B, 318-U1359A, 318-U1359B, 318-U1359D, 318-U1360A, 318-U1361A, 320-U1331A, 320-U1332A, 320-U1333A, 320-U1334A, 320-U1335A, 320-U1336A, 320-U1336B, 321-U1337A, 321-U1338A, 327-U1363C, 327-U1363D, 327-U1363F, 329-U1365A, 329-U1365B, 329-U1365C, 329-U1365D, 329-U1366B, 329-U1366D, 329-U1367B, 329-U1367D, 329-U1367E, 329-U1368B, 329-U1368C, 329-U1368D, 329-U1368E, 329-U1368F, 329-U1369B, 329-U1370B, 329-U1370D, 329-U1371D, 334-U1378B, 334-U1379C, 334-U1380A, 334-U1381A, 336-U1383D, 339-U1386A, 339-U1387A, 339-U1389A, 339-U1390A, 340-U1394A, 340-U1394B, 340-U1395A, 340-U1395B, 340-U1396A, 340-U1396C, 340-U1397A, 340-U1397B, 340-U1398A, 340-U1398B, 340-U1399A, 340-U1399B, 340-U1400B, 340-U1400C, 342-U1402B, 342-U1403A, 342-U1404A, 342-U1405A, 342-U1406A, 342-U1407A, 342-U1408A, 342-U1409A, 342-U1409C, 342-U1410A, 342-U1411B, 344-U1380C, 344-U1381C, 344-U1412A, 344-U1412B, 344-U1412C, 344-U1412D, 344-U1413A, 344-U1413C, 344-U1414A, 346-U1422C, 349-U1431D, 349-U1432C, 349-U1433A, 349-U1433B, 349-U1434A, 349-U1435A, 350-U1436A, 350-U1437B, 350-U1437D, 350-U1437E, 351-U1438B, 351-U1438D, 351-U1438E, 352-U1439A, 354-U1449A, 354-U1450A, 354-U1450B, 354-U1451A, 354-U1451B, 354-U1452B, 354-U1453A, 354-U1454A, 354-U1454B, 354-U1455C, 355-U1456A, 355-U1456C, 355-U1456D, 355-U1457A, 355-U1457C, 356-U1459A, 356-U1459B, 356-U1460A, 356-U1461A, 356-U1461B, 356-U1461C, 356-U1461D, 356-U1462A, 356-U1462C, 356-U1463B, 356-U1463C, 356-U1464B, 356-U1464C, 356-U1464D, 359-U1465B, 359-U1466A, 359-U1466B, 359-U1467A, 359-U1467B, 359-U1468A, 359-U1470A, 359-U1471A, 362-U1480E, 362-U1480F, 362-U1480G, 362-U1480H, 362-U1481A, 363-U1482A, 363-U1483A, 363-U1484A, 363-U1485A, 363-U1486B, 363-U1487A, 363-U1488A, 363-U1489B, 363-U1489C, 363-U1490A, 366-U1493B, 366-U1494A, 366-U1498B, 367-U1499A, 367-U1499B, 367-U1500A, 367-U1500B, 368-U1501A, 368-U1501B, 368-U1501C, 368-U1501D, 368-U1502A, 368-U1502B, 368-U1504A, 368-U1505C, 368X-U1503A, 369-U1512A, 369-U1513A, 369-U1513D, 369-U1514A, 369-U1514C, 369-U1515A, 369-U1516A, 369-U1516C, 371-U1506A, 371-U1507A, 371-U1507B, 371-U1508A, 371-U1508B, 371-U1508C, 371-U1509A, 371-U1510A, 371-U1510B, 371-U1511A, 371-U1511B, 372-U1517C, 374-U1521A, 374-U1522A, 374-U1523A, 374-U1523B, 374-U1523E, 374-U1524A, 374-U1525A, 375-U1518E, 375-U1518F, 375-U1519D, 375-U1519E, 375-U1520C, 375-U1520D, 376-U1527A, 376-U1527C, 376-U1528A, 376-U1528C, 376-U1528D, 376-U1530A, 376-U1531C, 379-U1532A, 379-U1532B, 379-U1532C, 379-U1532D, 379-U1532G, 379-U1533A, 379-U1533B, 379-U1533C, 379-U1533D"""

    borehole_list = [b.strip() for b in borehole_list_str.split(',')]
    logging.info(f"Target boreholes: {len(borehole_list)}")

    # Load and merge data
    logging.info("\n" + "="*80)
    logging.info("STEP 1: Loading and merging datasets")
    logging.info("="*80)
    merged_df = load_and_merge_data(borehole_list)

    # Data statistics
    logging.info("\n" + "="*80)
    logging.info("Data Statistics")
    logging.info("="*80)
    logging.info(f"Total samples: {len(merged_df)}")
    logging.info(f"Unique boreholes: {merged_df['Borehole_ID'].nunique()}")
    logging.info(f"Unique principal lithologies: {merged_df['Principal'].nunique()}")
    logging.info(f"\nTop 10 principal lithologies:")
    logging.info(merged_df['Principal'].value_counts().head(10))

    # Prepare data
    logging.info("\n" + "="*80)
    logging.info("STEP 2: Preparing data for training")
    logging.info("="*80)
    data_dict = prepare_data(merged_df)

    # Save preprocessed data info
    preprocess_info = {
        'n_samples_train': len(data_dict['train_df']),
        'n_samples_val': len(data_dict['val_df']),
        'n_samples_test': len(data_dict['test_df']),
        'n_boreholes_train': len(data_dict['train_boreholes']),
        'n_boreholes_val': len(data_dict['val_boreholes']),
        'n_boreholes_test': len(data_dict['test_boreholes']),
        'feature_names': data_dict['feature_names'],
        'scaler_mean': data_dict['scaler'].mean_.tolist(),
        'scaler_std': data_dict['scaler'].scale_.tolist()
    }

    with open('/home/utig5/johna/bhai/ml_models/checkpoints/preprocess_info.json', 'w') as f:
        json.dump(preprocess_info, f, indent=2)

    # Train models with different latent dimensions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"\nUsing device: {device}")

    for latent_dim in [2, 8]:
        logging.info("\n" + "="*80)
        logging.info(f"STEP 3: Training VAE with latent_dim={latent_dim}")
        logging.info("="*80)

        # Create model
        model = VAE(input_dim=4, latent_dim=latent_dim)
        logging.info(f"\nModel architecture:")
        logging.info(model)

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Total trainable parameters: {n_params:,}")

        # Train model
        history = train_vae(
            model=model,
            train_loader=data_dict['train_loader'],
            val_loader=data_dict['val_loader'],
            n_epochs=200,
            learning_rate=1e-3,
            beta=1.0,
            device=device
        )

        # Save model with latent dim in name
        final_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/vae_lithology_latent{latent_dim}.pth'
        best_path = f'/home/utig5/johna/bhai/ml_models/checkpoints/vae_lithology_latent{latent_dim}_best.pth'

        import shutil
        shutil.copy(
            '/home/utig5/johna/bhai/ml_models/checkpoints/vae_lithology_model_final.pth',
            final_path
        )
        shutil.copy(
            '/home/utig5/johna/bhai/ml_models/checkpoints/vae_lithology_model_best.pth',
            best_path
        )

        logging.info(f"Saved models: {final_path}, {best_path}")

    logging.info("\n" + "="*80)
    logging.info("Training completed successfully!")
    logging.info("="*80)
    logging.info(f"Log file: {log_file}")
    logging.info("Next steps: Run analysis script to generate UMAP projections and clustering")


if __name__ == '__main__':
    main()
