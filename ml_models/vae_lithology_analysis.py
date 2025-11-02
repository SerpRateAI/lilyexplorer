"""
VAE Lithology Analysis and Visualization

This script:
1. Loads trained VAE models
2. Generates latent representations for all data
3. Creates UMAP projections for visualization
4. Performs clustering analysis (KMeans and HDBSCAN)
5. Validates clusters against actual lithology labels
6. Generates comprehensive visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import json
import logging
from datetime import datetime
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import VAE model and utilities
from vae_lithology_model import VAE, load_and_merge_data, prepare_data

# Try importing UMAP (will install if needed)
try:
    import umap
    UMAP_AVAILABLE = True
except (ImportError, SystemError) as e:
    UMAP_AVAILABLE = False
    print(f"UMAP not available: {e}. Will use PCA for dimensionality reduction.")

# Try importing HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available. Install with: pip install hdbscan")


def load_trained_model(model_path: str, latent_dim: int = 8, device: str = 'cpu') -> VAE:
    """Load trained VAE model from checkpoint."""
    model = VAE(input_dim=4, latent_dim=latent_dim)
    # PyTorch 2.6+ requires weights_only=False for loading our checkpoints
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    logging.info(f"Loaded model from {model_path}")
    logging.info(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    return model


def get_latent_representations(model: VAE, data_loader, device: str = 'cpu') -> np.ndarray:
    """Extract latent representations for all samples."""
    latents = []
    model.eval()

    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            z = model.get_latent(x)
            latents.append(z.cpu().numpy())

    return np.vstack(latents)


def evaluate_reconstruction(model: VAE, data_loader, scaler, feature_names: list,
                            device: str = 'cpu') -> pd.DataFrame:
    """Evaluate reconstruction quality."""
    model.eval()
    all_inputs = []
    all_reconstructions = []

    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            x_recon, _, _ = model(x)
            all_inputs.append(x.cpu().numpy())
            all_reconstructions.append(x_recon.cpu().numpy())

    inputs = np.vstack(all_inputs)
    reconstructions = np.vstack(all_reconstructions)

    # Inverse transform to original scale
    inputs_orig = scaler.inverse_transform(inputs)
    recons_orig = scaler.inverse_transform(reconstructions)

    # Calculate metrics
    results = []
    for i, name in enumerate(feature_names):
        mse = np.mean((inputs_orig[:, i] - recons_orig[:, i])**2)
        mae = np.mean(np.abs(inputs_orig[:, i] - recons_orig[:, i]))
        mape = np.mean(np.abs((inputs_orig[:, i] - recons_orig[:, i]) / inputs_orig[:, i])) * 100

        results.append({
            'Feature': name,
            'MSE': mse,
            'MAE': mae,
            'MAPE (%)': mape,
            'Mean_Original': inputs_orig[:, i].mean(),
            'Std_Original': inputs_orig[:, i].std()
        })

    return pd.DataFrame(results)


def perform_clustering(latents: np.ndarray, n_clusters_range: range = range(5, 21)) -> dict:
    """Perform KMeans clustering with different numbers of clusters."""
    results = {}

    best_score = -1
    best_n = None

    for n_clusters in n_clusters_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(latents)
        score = silhouette_score(latents, labels)

        results[n_clusters] = {
            'labels': labels,
            'silhouette': score,
            'inertia': kmeans.inertia_
        }

        if score > best_score:
            best_score = score
            best_n = n_clusters

    logging.info(f"Best silhouette score: {best_score:.3f} with {best_n} clusters")

    return results, best_n


def analyze_cluster_lithology(labels: np.ndarray, lithology_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze lithology composition of each cluster."""
    cluster_stats = []

    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_lith = lithology_df[cluster_mask]['Principal']

        # Get top lithologies in this cluster
        lith_counts = cluster_lith.value_counts()
        total = len(cluster_lith)

        top_lithology = lith_counts.index[0] if len(lith_counts) > 0 else 'Unknown'
        top_percentage = (lith_counts.iloc[0] / total * 100) if len(lith_counts) > 0 else 0

        # Diversity metric (entropy)
        proportions = lith_counts.values / total
        entropy = -np.sum(proportions * np.log2(proportions + 1e-10))

        # Get top 3 lithologies
        top3 = list(lith_counts.head(3).items())
        top3_str = ', '.join([f"{lith} ({count})" for lith, count in top3])

        cluster_stats.append({
            'Cluster': cluster_id,
            'Size': total,
            'Size_Pct': total / len(labels) * 100,
            'Top_Lithology': top_lithology,
            'Top_Pct': top_percentage,
            'Entropy': entropy,
            'N_Lithologies': len(lith_counts),
            'Top_3': top3_str
        })

    return pd.DataFrame(cluster_stats).sort_values('Size', ascending=False)


def compute_contingency_matrix(cluster_labels: np.ndarray, lithology: pd.Series) -> pd.DataFrame:
    """Create contingency matrix between clusters and lithologies."""
    df = pd.DataFrame({
        'Cluster': cluster_labels,
        'Lithology': lithology
    })

    # Keep only top N lithologies for readability
    top_lithologies = lithology.value_counts().head(15).index
    df_filtered = df[df['Lithology'].isin(top_lithologies)]

    contingency = pd.crosstab(df_filtered['Cluster'], df_filtered['Lithology'])

    return contingency


def plot_training_history(history_path: str, output_dir: Path):
    """Plot training history."""
    history = pd.read_csv(history_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train', alpha=0.7)
    axes[0, 0].plot(history['val_loss'], label='Validation', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Reconstruction loss
    axes[0, 1].plot(history['train_recon_loss'], label='Train', alpha=0.7)
    axes[0, 1].plot(history['val_recon_loss'], label='Validation', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Reconstruction Loss (MSE)')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # KL divergence
    axes[1, 0].plot(history['train_kl_loss'], label='Train', alpha=0.7)
    axes[1, 0].plot(history['val_kl_loss'], label='Validation', alpha=0.7)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('KL Divergence')
    axes[1, 0].set_title('KL Divergence')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Learning rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved training history plot")


def plot_latent_space(latents: np.ndarray, labels: pd.Series, title: str,
                      output_path: Path, latent_dim: int = 2, projection_method: str = 'native'):
    """Plot 2D latent space visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get top lithologies for coloring
    top_lithologies = labels.value_counts().head(10).index
    color_map = {lith: i for i, lith in enumerate(top_lithologies)}
    colors = labels.map(lambda x: color_map.get(x, -1))

    if latent_dim == 2 and projection_method == 'native':
        # Direct 2D latent space
        x_plot, y_plot = latents[:, 0], latents[:, 1]
        plot_title = "2D Latent Space (Native)"
    elif projection_method == 'pca':
        # PCA projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(latents)
        x_plot, y_plot = coords[:, 0], coords[:, 1]
        plot_title = f"Latent Space (PCA, {latent_dim}D -> 2D)"
        explained = pca.explained_variance_ratio_
        plot_title += f"\nExplained variance: {explained[0]:.2%}, {explained[1]:.2%}"
    else:  # UMAP
        if not UMAP_AVAILABLE:
            logging.warning("UMAP not available, using PCA instead")
            pca = PCA(n_components=2)
            coords = pca.fit_transform(latents)
            x_plot, y_plot = coords[:, 0], coords[:, 1]
            plot_title = f"Latent Space (PCA, {latent_dim}D -> 2D)"
        else:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(latents)
            x_plot, y_plot = coords[:, 0], coords[:, 1]
            plot_title = f"Latent Space (UMAP, {latent_dim}D -> 2D)"

    # Plot 1: Colored by lithology (top 10)
    ax = axes[0]
    for lith in top_lithologies:
        mask = labels == lith
        ax.scatter(x_plot[mask], y_plot[mask], label=lith, alpha=0.5, s=10)

    # Plot remaining as gray
    other_mask = ~labels.isin(top_lithologies)
    if other_mask.sum() > 0:
        ax.scatter(x_plot[other_mask], y_plot[other_mask],
                  color='gray', alpha=0.2, s=5, label='Other')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{plot_title}\nColored by Top 10 Lithologies')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Density plot
    ax = axes[1]
    ax.hexbin(x_plot, y_plot, gridsize=50, cmap='viridis', mincnt=1)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'{plot_title}\nDensity Map')
    ax.grid(alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved latent space plot: {output_path}")


def plot_cluster_visualization(latents: np.ndarray, cluster_labels: np.ndarray,
                               lithology_labels: pd.Series, output_path: Path,
                               latent_dim: int = 8):
    """Plot clustering results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Get 2D projection for visualization
    if latent_dim == 2:
        x_plot, y_plot = latents[:, 0], latents[:, 1]
        method_label = "2D Latent Space"
    else:
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            coords = reducer.fit_transform(latents)
            x_plot, y_plot = coords[:, 0], coords[:, 1]
            method_label = f"UMAP Projection ({latent_dim}D -> 2D)"
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            coords = pca.fit_transform(latents)
            x_plot, y_plot = coords[:, 0], coords[:, 1]
            method_label = f"PCA Projection ({latent_dim}D -> 2D)"

    # Plot 1: Colored by cluster assignment
    ax = axes[0]
    scatter = ax.scatter(x_plot, y_plot, c=cluster_labels, cmap='tab20',
                         alpha=0.6, s=10)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'Cluster Assignments\n({method_label})')
    plt.colorbar(scatter, ax=ax, label='Cluster ID')
    ax.grid(alpha=0.3)

    # Plot 2: Colored by most common lithology
    ax = axes[1]
    top_lithologies = lithology_labels.value_counts().head(10).index
    color_map = {lith: i for i, lith in enumerate(top_lithologies)}
    colors = lithology_labels.map(lambda x: color_map.get(x, -1))

    for lith in top_lithologies:
        mask = lithology_labels == lith
        ax.scatter(x_plot[mask], y_plot[mask], label=lith, alpha=0.5, s=10)

    other_mask = ~lithology_labels.isin(top_lithologies)
    if other_mask.sum() > 0:
        ax.scatter(x_plot[other_mask], y_plot[other_mask],
                  color='gray', alpha=0.2, s=5, label='Other')

    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(f'True Lithology Labels\n({method_label})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved cluster visualization: {output_path}")


def plot_reconstruction_quality(recon_results: pd.DataFrame, output_path: Path):
    """Plot reconstruction quality metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    features = recon_results['Feature']

    # MSE
    axes[0].bar(range(len(features)), recon_results['MSE'])
    axes[0].set_xticks(range(len(features)))
    axes[0].set_xticklabels(features, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Reconstruction MSE by Feature')
    axes[0].grid(alpha=0.3, axis='y')

    # MAE
    axes[1].bar(range(len(features)), recon_results['MAE'])
    axes[1].set_xticks(range(len(features)))
    axes[1].set_xticklabels(features, rotation=45, ha='right')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Reconstruction MAE by Feature')
    axes[1].grid(alpha=0.3, axis='y')

    # MAPE
    axes[2].bar(range(len(features)), recon_results['MAPE (%)'])
    axes[2].set_xticks(range(len(features)))
    axes[2].set_xticklabels(features, rotation=45, ha='right')
    axes[2].set_ylabel('Mean Absolute Percentage Error (%)')
    axes[2].set_title('Reconstruction MAPE by Feature')
    axes[2].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved reconstruction quality plot: {output_path}")


def plot_contingency_heatmap(contingency: pd.DataFrame, output_path: Path):
    """Plot contingency matrix as heatmap."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Normalize by cluster (rows) to show percentage
    contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100

    sns.heatmap(contingency_pct, annot=False, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Percentage of Cluster (%)'}, ax=ax)

    ax.set_xlabel('Lithology (Principal)', fontweight='bold')
    ax.set_ylabel('Cluster ID', fontweight='bold')
    ax.set_title('Contingency Matrix: Cluster vs Lithology\n(Row-normalized percentages)',
                 fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Saved contingency heatmap: {output_path}")


def main():
    """Main analysis script."""
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'/home/utig5/johna/bhai/ml_models/logs/vae_lithology_analysis_{timestamp}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info("="*80)
    logging.info("VAE Lithology Analysis")
    logging.info("="*80)

    # Setup paths
    output_dir = Path('/home/utig5/johna/bhai/vae_outputs')
    output_dir.mkdir(exist_ok=True)
    checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # Load preprocessed data
    logging.info("\nLoading preprocessed data...")
    with open(checkpoint_dir / 'preprocess_info.json', 'r') as f:
        preprocess_info = json.load(f)

    # Load borehole list and data
    borehole_list_str = """318-U1356A, 318-U1358B, 318-U1359A, 318-U1359B, 318-U1359D, 318-U1360A, 318-U1361A, 320-U1331A, 320-U1332A, 320-U1333A, 320-U1334A, 320-U1335A, 320-U1336A, 320-U1336B, 321-U1337A, 321-U1338A, 327-U1363C, 327-U1363D, 327-U1363F, 329-U1365A, 329-U1365B, 329-U1365C, 329-U1365D, 329-U1366B, 329-U1366D, 329-U1367B, 329-U1367D, 329-U1367E, 329-U1368B, 329-U1368C, 329-U1368D, 329-U1368E, 329-U1368F, 329-U1369B, 329-U1370B, 329-U1370D, 329-U1371D, 334-U1378B, 334-U1379C, 334-U1380A, 334-U1381A, 336-U1383D, 339-U1386A, 339-U1387A, 339-U1389A, 339-U1390A, 340-U1394A, 340-U1394B, 340-U1395A, 340-U1395B, 340-U1396A, 340-U1396C, 340-U1397A, 340-U1397B, 340-U1398A, 340-U1398B, 340-U1399A, 340-U1399B, 340-U1400B, 340-U1400C, 342-U1402B, 342-U1403A, 342-U1404A, 342-U1405A, 342-U1406A, 342-U1407A, 342-U1408A, 342-U1409A, 342-U1409C, 342-U1410A, 342-U1411B, 344-U1380C, 344-U1381C, 344-U1412A, 344-U1412B, 344-U1412C, 344-U1412D, 344-U1413A, 344-U1413C, 344-U1414A, 346-U1422C, 349-U1431D, 349-U1432C, 349-U1433A, 349-U1433B, 349-U1434A, 349-U1435A, 350-U1436A, 350-U1437B, 350-U1437D, 350-U1437E, 351-U1438B, 351-U1438D, 351-U1438E, 352-U1439A, 354-U1449A, 354-U1450A, 354-U1450B, 354-U1451A, 354-U1451B, 354-U1452B, 354-U1453A, 354-U1454A, 354-U1454B, 354-U1455C, 355-U1456A, 355-U1456C, 355-U1456D, 355-U1457A, 355-U1457C, 356-U1459A, 356-U1459B, 356-U1460A, 356-U1461A, 356-U1461B, 356-U1461C, 356-U1461D, 356-U1462A, 356-U1462C, 356-U1463B, 356-U1463C, 356-U1464B, 356-U1464C, 356-U1464D, 359-U1465B, 359-U1466A, 359-U1466B, 359-U1467A, 359-U1467B, 359-U1468A, 359-U1470A, 359-U1471A, 362-U1480E, 362-U1480F, 362-U1480G, 362-U1480H, 362-U1481A, 363-U1482A, 363-U1483A, 363-U1484A, 363-U1485A, 363-U1486B, 363-U1487A, 363-U1488A, 363-U1489B, 363-U1489C, 363-U1490A, 366-U1493B, 366-U1494A, 366-U1498B, 367-U1499A, 367-U1499B, 367-U1500A, 367-U1500B, 368-U1501A, 368-U1501B, 368-U1501C, 368-U1501D, 368-U1502A, 368-U1502B, 368-U1504A, 368-U1505C, 368X-U1503A, 369-U1512A, 369-U1513A, 369-U1513D, 369-U1514A, 369-U1514C, 369-U1515A, 369-U1516A, 369-U1516C, 371-U1506A, 371-U1507A, 371-U1507B, 371-U1508A, 371-U1508B, 371-U1508C, 371-U1509A, 371-U1510A, 371-U1510B, 371-U1511A, 371-U1511B, 372-U1517C, 374-U1521A, 374-U1522A, 374-U1523A, 374-U1523B, 374-U1523E, 374-U1524A, 374-U1525A, 375-U1518E, 375-U1518F, 375-U1519D, 375-U1519E, 375-U1520C, 375-U1520D, 376-U1527A, 376-U1527C, 376-U1528A, 376-U1528C, 376-U1528D, 376-U1530A, 376-U1531C, 379-U1532A, 379-U1532B, 379-U1532C, 379-U1532D, 379-U1532G, 379-U1533A, 379-U1533B, 379-U1533C, 379-U1533D"""

    borehole_list = [b.strip() for b in borehole_list_str.split(',')]

    merged_df = load_and_merge_data(borehole_list)
    data_dict = prepare_data(merged_df)

    # Plot training history
    logging.info("\nPlotting training history...")
    plot_training_history(
        '/home/utig5/johna/bhai/ml_models/logs/training_history.csv',
        output_dir
    )

    # Analyze both latent dimensions
    for latent_dim in [2, 8]:
        logging.info("\n" + "="*80)
        logging.info(f"Analyzing VAE with latent_dim={latent_dim}")
        logging.info("="*80)

        # Load model
        model_path = checkpoint_dir / f'vae_lithology_latent{latent_dim}_best.pth'
        if not model_path.exists():
            logging.warning(f"Model not found: {model_path}")
            continue

        model = load_trained_model(str(model_path), latent_dim, device)

        # Get latent representations for all splits
        logging.info("\nExtracting latent representations...")
        train_latents = get_latent_representations(model, data_dict['train_loader'], device)
        val_latents = get_latent_representations(model, data_dict['val_loader'], device)
        test_latents = get_latent_representations(model, data_dict['test_loader'], device)

        # Combine for visualization
        all_latents = np.vstack([train_latents, val_latents, test_latents])
        all_df = pd.concat([data_dict['train_df'], data_dict['val_df'], data_dict['test_df']])

        logging.info(f"Latent representations shape: {all_latents.shape}")

        # Evaluate reconstruction quality
        logging.info("\nEvaluating reconstruction quality...")
        recon_results = evaluate_reconstruction(
            model, data_dict['test_loader'], data_dict['scaler'],
            data_dict['feature_names'], device
        )
        logging.info("\nReconstruction quality (test set):")
        logging.info(recon_results.to_string(index=False))

        recon_results.to_csv(output_dir / f'reconstruction_quality_latent{latent_dim}.csv', index=False)
        plot_reconstruction_quality(recon_results, output_dir / f'reconstruction_quality_latent{latent_dim}.png')

        # Latent space visualization
        logging.info("\nGenerating latent space visualizations...")

        if latent_dim == 2:
            plot_latent_space(
                all_latents, all_df['Principal'],
                f'VAE Latent Space (2D)',
                output_dir / f'latent_space_2d.png',
                latent_dim=2, projection_method='native'
            )
        else:
            # PCA projection
            plot_latent_space(
                all_latents, all_df['Principal'],
                f'VAE Latent Space (8D -> 2D via PCA)',
                output_dir / f'latent_space_8d_pca.png',
                latent_dim=8, projection_method='pca'
            )

            # UMAP projection
            if UMAP_AVAILABLE:
                plot_latent_space(
                    all_latents, all_df['Principal'],
                    f'VAE Latent Space (8D -> 2D via UMAP)',
                    output_dir / f'latent_space_8d_umap.png',
                    latent_dim=8, projection_method='umap'
                )

        # Clustering analysis
        logging.info("\nPerforming clustering analysis...")
        clustering_results, best_n = perform_clustering(all_latents)

        # Use best n_clusters
        best_labels = clustering_results[best_n]['labels']

        # Analyze cluster composition
        cluster_stats = analyze_cluster_lithology(best_labels, all_df)
        logging.info(f"\nCluster statistics (n_clusters={best_n}):")
        logging.info(cluster_stats.to_string(index=False))

        cluster_stats.to_csv(output_dir / f'cluster_statistics_latent{latent_dim}.csv', index=False)

        # Compute clustering metrics
        ari = adjusted_rand_score(all_df['Principal'].astype('category').cat.codes, best_labels)
        nmi = normalized_mutual_info_score(all_df['Principal'].astype('category').cat.codes, best_labels)

        logging.info(f"\nClustering performance metrics:")
        logging.info(f"  Adjusted Rand Index: {ari:.3f}")
        logging.info(f"  Normalized Mutual Information: {nmi:.3f}")
        logging.info(f"  Silhouette Score: {clustering_results[best_n]['silhouette']:.3f}")

        # Plot cluster visualization
        plot_cluster_visualization(
            all_latents, best_labels, all_df['Principal'],
            output_dir / f'cluster_visualization_latent{latent_dim}.png',
            latent_dim
        )

        # Contingency matrix
        contingency = compute_contingency_matrix(best_labels, all_df['Principal'])
        contingency.to_csv(output_dir / f'contingency_matrix_latent{latent_dim}.csv')

        plot_contingency_heatmap(contingency, output_dir / f'contingency_heatmap_latent{latent_dim}.png')

        # Save summary
        summary = {
            'latent_dim': latent_dim,
            'n_samples': len(all_latents),
            'n_clusters': best_n,
            'silhouette_score': float(clustering_results[best_n]['silhouette']),
            'adjusted_rand_index': float(ari),
            'normalized_mutual_info': float(nmi),
            'n_lithologies': all_df['Principal'].nunique(),
            'top_10_lithologies': all_df['Principal'].value_counts().head(10).to_dict(),
            'reconstruction_metrics': recon_results.to_dict('records')
        }

        with open(output_dir / f'summary_latent{latent_dim}.json', 'w') as f:
            json.dump(summary, f, indent=2)

    logging.info("\n" + "="*80)
    logging.info("Analysis completed successfully!")
    logging.info("="*80)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Log file: {log_file}")


if __name__ == '__main__':
    main()
