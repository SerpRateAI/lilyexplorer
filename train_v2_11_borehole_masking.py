"""
VAE v2.11 - Borehole-level masking (10% random features per borehole)

More realistic: some boreholes missing certain measurements entirely
e.g., Borehole A: no NGR (mask feature 2 for all samples)
     Borehole B: no RGB (mask features 3,4,5 for all samples)
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

from vae_lithology_gra_v2_11_masked import *

def apply_borehole_masking(X, borehole_ids, mask_prob=0.1, mask_value=0.0):
    """
    Apply masking at borehole level: each borehole randomly masks ~10% of features.

    Args:
        X: Input tensor [batch_size, 6]
        borehole_ids: Borehole ID for each sample [batch_size]
        mask_prob: Probability of masking each feature PER BOREHOLE
        mask_value: Value to use for masked features

    Returns:
        X_masked, mask
    """
    device = X.device
    batch_size = X.shape[0]
    n_features = X.shape[1]

    # Get unique boreholes in this batch
    unique_boreholes = torch.unique(borehole_ids)

    # Initialize mask (True = keep, False = masked)
    mask = torch.ones_like(X, dtype=torch.bool)

    # For each borehole, randomly mask features
    for borehole in unique_boreholes:
        # Find samples from this borehole
        borehole_mask = (borehole_ids == borehole)

        # Randomly select features to mask (10% probability per feature)
        feature_mask = torch.rand(n_features, device=device) > mask_prob

        # Apply feature mask to all samples from this borehole
        for feat_idx in range(n_features):
            if not feature_mask[feat_idx]:
                mask[borehole_mask, feat_idx] = False

    # Apply masking
    X_masked = X * mask.float() + mask_value * (~mask).float()

    return X_masked, mask


class LithologyDatasetWithBorehole(torch.utils.data.Dataset):
    """Dataset that returns borehole IDs"""
    def __init__(self, X, borehole_ids):
        self.X = torch.FloatTensor(X)
        self.borehole_ids = torch.LongTensor(borehole_ids)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.borehole_ids[idx]


def train_borehole_masked_vae(model, train_loader, val_loader, n_epochs=100,
                               lr=0.001, device='cpu', mask_prob=0.1):
    """Train VAE with borehole-level masking"""
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
        # β annealing schedule
        if epoch < 50:
            beta = 0.001 + (0.5 - 0.001) * (epoch / 50)
        else:
            beta = 0.5

        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, borehole_ids) in enumerate(train_loader):
            data = data.to(device)
            borehole_ids = borehole_ids.to(device)

            # Apply borehole-level masking
            data_masked, mask = apply_borehole_masking(data, borehole_ids,
                                                        mask_prob=mask_prob)

            optimizer.zero_grad()

            # Forward pass
            recon, mu, logvar = model(data_masked)

            # Loss: reconstruct all features (learn dependencies)
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


if __name__ == "__main__":
    print("="*80)
    print("VAE GRA v2.11 - Borehole-Level Masking (10%)")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv('vae_training_data_v2_20cm.csv')

    feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                    'NGR total counts (cps)', 'R', 'G', 'B']

    X = df[feature_cols].values
    lithology = df['Principal'].values
    borehole_ids = df['Borehole_ID'].values

    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    lithology = lithology[valid_mask]
    borehole_ids = borehole_ids[valid_mask]

    print(f"Dataset: {len(X):,} samples")

    # Encode borehole IDs as integers
    unique_boreholes = np.unique(borehole_ids)
    borehole_to_id = {bh: i for i, bh in enumerate(unique_boreholes)}
    borehole_ids_encoded = np.array([borehole_to_id[bh] for bh in borehole_ids])

    # Borehole-level split
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

    borehole_ids_train = borehole_ids_encoded[train_mask]
    borehole_ids_val = borehole_ids_encoded[val_mask]

    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Scale
    scaler = DistributionAwareScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # DataLoaders with borehole IDs
    train_dataset = LithologyDatasetWithBorehole(X_train_scaled, borehole_ids_train)
    val_dataset = LithologyDatasetWithBorehole(X_val_scaled, borehole_ids_val)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512)

    # Train model
    print("\n" + "="*80)
    print("Training with borehole-level masking (10% features per borehole)")
    print("="*80)
    print("Each borehole randomly masks ~10% of features for ALL its samples")
    print("Simulates realistic missing data: e.g., some boreholes have no NGR")
    print()

    device = 'cpu'
    print(f"Device: {device}")

    model = MaskedVAE(input_dim=6, latent_dim=8, hidden_dims=[32, 16]).to(device)

    start_time = time.time()
    model, history = train_borehole_masked_vae(
        model, train_loader, val_loader,
        n_epochs=100, lr=0.001, device=device,
        mask_prob=0.1  # 10% per borehole
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
    save_path = Path('ml_models/checkpoints/vae_gra_v2_11_masked_borehole_10pct.pth')
    save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'results': results,
        'scaler': scaler,
        'mask_strategy': 'borehole_10pct',
        'train_time': train_time
    }, save_path)

    print(f"\nModel saved: {save_path}")

    # Compare to all variants
    print("\n" + "="*80)
    print("Comparison: All Masking Strategies")
    print("="*80)
    print(f"v2.6 (no masking):         ARI (k=12) = 0.258")
    print(f"v2.11 (block NGR+RGB 30%): ARI (k=12) = 0.236 (-8.5%)")
    print(f"v2.11 (random 15%):        ARI (k=12) = 0.248 (-3.8%)")
    print(f"v2.11 (borehole 10%):      ARI (k=12) = {results[12]['ari']:.3f}", end="")

    improvement = (results[12]['ari'] - 0.258) / 0.258 * 100
    print(f" ({improvement:+.1f}%)")

    if results[12]['ari'] > 0.258:
        print("\n✓ BOREHOLE MASKING BEATS v2.6!")
    else:
        print(f"\n✗ Still {abs(improvement):.1f}% worse than v2.6")
