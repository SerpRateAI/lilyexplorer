"""
Train VAE v2.10 with VampPrior
"""

import sys
sys.path.insert(0, '/home/utig5/johna/bhai/ml_models')

from vae_lithology_gra_v2_10_vampprior import *

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

    total_params = sum(p.numel() for p in model.parameters())
    model_params = sum(p.numel() for n, p in model.named_parameters() if 'pseudo' not in n)
    pseudo_params = model.pseudo_inputs.numel()

    print(f"Parameters: {total_params:,}")
    print(f"  Model (encoder/decoder): {model_params:,}")
    print(f"  Pseudo-inputs (K=50): {pseudo_params:,} (50 × 6 features)")

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
