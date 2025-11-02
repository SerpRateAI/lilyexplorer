"""Quick VAE v2.9 training with porosity feature."""
import sys
sys.path.append('/home/utig5/johna/bhai/ml_models')

from vae_lithology_gra_v2_5_model import *

# Override data loading to use v2.9 dataset
def load_and_prepare_data_v2_9(data_path):
    print("Loading v2.9 data with porosity...")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

    # Extract features (8D: GRA, Grain density, Porosity, MS, NGR, R, G, B)
    feature_cols = [
        'Bulk density (GRA)',
        'Grain density (g/cm^3)',
        'Porosity',
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
    print("  Grain density:            Gaussian      → StandardScaler")
    print("  Porosity:                 Beta-like     → StandardScaler")
    print("  Magnetic susceptibility:  Poisson       → sign(x)*log(|x|+1) + StandardScaler")
    print("  NGR:                      Bimodal       → sign(x)*log(|x|+1) + StandardScaler")
    print("  R, G, B:                  Log-normal    → log(x+1) + StandardScaler")

    # Custom scaler for 8D
    class DistributionAwareScaler8D:
        def __init__(self):
            self.scaler = StandardScaler()
            self.signed_log_indices = [3, 4]  # MS, NGR
            self.log_indices = [5, 6, 7]  # R, G, B

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

    scaler = DistributionAwareScaler8D()
    X_scaled = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    lithology_encoded = label_encoder.fit_transform(lithology)

    print(f"Found {len(label_encoder.classes_)} unique lithologies")

    return X_scaled, lithology_encoded, lithology, borehole_ids, scaler, label_encoder

# Run training
print("="*80)
print("VAE GRA v2.9 - WITH POROSITY FEATURE")
print("="*80)

data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_9_20cm_porosity.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# Load data
X, y, lithology, borehole_ids, scaler, label_encoder = load_and_prepare_data_v2_9(data_path)

# Split data
(X_train, y_train, lith_train), (X_val, y_val, lith_val), (X_test, y_test, lith_test) = \
    split_by_borehole(X, y, lithology, borehole_ids)

# Create data loaders (smaller batch size for small dataset)
train_dataset = LithologyDataset(X_train, y_train)
val_dataset = LithologyDataset(X_val, y_val)
test_dataset = LithologyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Train 8D model
latent_dim = 8
print(f"\nTraining VAE with 8D latent space, 8D input (with porosity)")
model = VAE(input_dim=8, latent_dim=latent_dim, hidden_dims=[32, 16]).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# β annealing (same as v2.6)
start_time = time.time()
model, history = train_vae_with_annealing(
    model, train_loader, val_loader,
    epochs=100, device=device,
    beta_start=0.001, beta_end=0.5, anneal_epochs=50
)

# Save
checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')
model_path = checkpoint_dir / 'vae_gra_v2_9_latent8_porosity.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': label_encoder,
    'history': history,
    'latent_dim': latent_dim,
    'input_dim': 8,
    'version': 'v2.9'
}, model_path)
print(f"\nModel saved to: {model_path}")

# Evaluate
latent_test, labels_test = get_latent_representations(model, test_loader, device)
results = cluster_analysis(latent_test, labels_test, lith_test, n_clusters_list=[10, 12, 15, 20])

print("\nv2.9 Best Results:")
for r in results:
    if r['n_clusters'] == 12:
        print(f"  k=12: ARI={r['ari']:.3f}, Silhouette={r['silhouette']:.3f}")
