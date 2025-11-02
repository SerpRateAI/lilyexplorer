"""Train VAE v2.9 with 12D engineered features."""
import sys
sys.path.append('/home/utig5/johna/bhai')
from train_beta_annealing import *

# Custom scaler for 12D features
class DistributionAwareScaler12D:
    def __init__(self):
        self.scaler = StandardScaler()
        # Indices for signed log (MS, NGR, density_contrast, depth_normalized_gra, mag_gamma_interaction)
        self.signed_log_indices = [1, 2, 9, 11, 10]
        # Indices for regular log (RGB, brightness, chroma)
        self.log_indices = [3, 4, 5, 7, 8]
        # Others: GRA, red_green_ratio - just StandardScaler

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

# Load v2.9 data
print("="*80)
print("VAE GRA v2.9 - 12D ENGINEERED FEATURES")
print("="*80)

data_path = Path('/home/utig5/johna/bhai/vae_training_data_v2_9_engineered.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

print("Loading data...")
df = pd.read_csv(data_path)
print(f"Loaded {len(df):,} samples from {df['Borehole_ID'].nunique()} boreholes")

# Extract 12D features
feature_cols = [
    'Bulk density (GRA)',
    'Magnetic susceptibility (instr. units)',
    'NGR total counts (cps)',
    'R', 'G', 'B',
    'red_green_ratio',
    'brightness',
    'chroma',
    'density_contrast',
    'mag_gamma_interaction',
    'depth_normalized_gra'
]

X = df[feature_cols].values
lithology = df['Principal'].values
borehole_ids = df['Borehole_ID'].values

# Remove NaN
valid_mask = ~np.isnan(X).any(axis=1)
X = X[valid_mask]
lithology = lithology[valid_mask]
borehole_ids = borehole_ids[valid_mask]
print(f"After removing NaN: {len(X):,} samples")

# Scale
print("\nApplying distribution-aware scaling for 12D features...")
scaler = DistributionAwareScaler12D()
X_scaled = scaler.fit_transform(X)

# Encode lithology
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(lithology)
print(f"Found {len(label_encoder.classes_)} unique lithologies")

# Split by borehole
unique_boreholes = np.unique(borehole_ids)
train_boreholes, test_boreholes = train_test_split(unique_boreholes, train_size=0.85, random_state=42)
train_boreholes, val_boreholes = train_test_split(train_boreholes, train_size=0.82, random_state=42)

train_mask = np.isin(borehole_ids, train_boreholes)
val_mask = np.isin(borehole_ids, val_boreholes)
test_mask = np.isin(borehole_ids, test_boreholes)

X_train, y_train, lith_train = X_scaled[train_mask], y[train_mask], lithology[train_mask]
X_val, y_val, lith_val = X_scaled[val_mask], y[val_mask], lithology[val_mask]
X_test, y_test, lith_test = X_scaled[test_mask], y[test_mask], lithology[test_mask]

print(f"\nData split:")
print(f"  Train: {len(train_boreholes)} boreholes, {len(X_train):,} samples")
print(f"  Val:   {len(val_boreholes)} boreholes, {len(X_val):,} samples")
print(f"  Test:  {len(test_boreholes)} boreholes, {len(X_test):,} samples")

# Create loaders
train_dataset = LithologyDataset(X_train, y_train)
val_dataset = LithologyDataset(X_val, y_val)
test_dataset = LithologyDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# Train 8D latent model with 12D input
latent_dim = 8
print(f"\n{'='*80}")
print(f"TRAINING VAE: 12D input → {latent_dim}D latent")
print(f"{'='*80}")

model = VAE(input_dim=12, latent_dim=latent_dim, hidden_dims=[64, 32]).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Train with β annealing
start_time = time.time()
model, history = train_vae_with_annealing(
    model, train_loader, val_loader,
    epochs=100, device=device,
    beta_start=0.001, beta_end=0.5, anneal_epochs=50
)
train_time = time.time() - start_time
print(f"Training time: {train_time:.1f}s")

# Save
checkpoint_dir = Path('/home/utig5/johna/bhai/ml_models/checkpoints')
model_path = checkpoint_dir / 'vae_gra_v2_9_latent8_engineered.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'scaler': scaler,
    'label_encoder': label_encoder,
    'history': history,
    'latent_dim': latent_dim,
    'input_dim': 12,
    'feature_cols': feature_cols,
    'version': 'v2.9'
}, model_path)
print(f"Model saved to: {model_path}")

# Evaluate
latent_test, labels_test = get_latent_representations(model, test_loader, device)
results = cluster_analysis(latent_test, labels_test, lith_test, n_clusters_list=[10, 12, 15, 20])

print("\n" + "="*80)
print("v2.9 RESULTS (12D Engineered Features)")
print("="*80)
for r in results:
    print(f"k={r['n_clusters']:2d}: ARI={r['ari']:.3f}, Silhouette={r['silhouette']:.3f}")
