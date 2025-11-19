#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train Depth Predictor from Semi-Supervised VAE v2.14 Embeddings

Uses 10D latent embeddings as features to predict depth (Depth_Bin).
Tests whether VAE embeddings contain stratigraphic information.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Set matplotlib font to monospace
plt.rcParams['font.family'] = 'monospace'

print("="*80)
print("DEPTH PREDICTION FROM SEMI-SUPERVISED VAE v2.14 EMBEDDINGS")
print("="*80)

# Define model architecture
class DistributionAwareScaler:
    """Distribution-aware scaler for physical properties"""
    def __init__(self):
        self.median = None
        self.iqr = None
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

        self.median = np.median(X_transformed, axis=0)
        q75 = np.percentile(X_transformed, 75, axis=0)
        q25 = np.percentile(X_transformed, 25, axis=0)
        self.iqr = q75 - q25
        self.iqr[self.iqr == 0] = 1.0

        X_scaled = (X_transformed - self.median) / self.iqr
        return X_scaled

    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.signed_log_indices:
            X_transformed[:, idx] = self.signed_log_transform(X[:, idx])
        for idx in self.log_indices:
            X_transformed[:, idx] = np.log1p(X[:, idx])
        X_scaled = (X_transformed - self.median) / self.iqr
        return X_scaled


class SemiSupervisedVAE(nn.Module):
    """Semi-supervised VAE with classification head"""
    def __init__(self, input_dim=6, latent_dim=10, n_classes=139,
                 encoder_dims=[32, 16], classifier_hidden=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes

        # Encoder: 6D → [32, 16] → 10D latent
        layers = []
        prev_dim = input_dim
        for h_dim in encoder_dims:
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(encoder_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(encoder_dims[-1], latent_dim)

        # Decoder: 10D → [16, 32] → 6D (symmetric)
        layers = []
        prev_dim = latent_dim
        for h_dim in reversed(encoder_dims):
            layers.extend([nn.Linear(prev_dim, h_dim), nn.ReLU()])
            prev_dim = h_dim
        self.decoder = nn.Sequential(*layers)
        self.fc_out = nn.Linear(encoder_dims[0], input_dim)

        # Classification head: 10D → [32, ReLU, Dropout] → 139 classes
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(classifier_hidden, n_classes)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return self.fc_out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        logits = self.classifier(z)
        return recon, mu, logvar, logits


# Load dataset
print("\nLoading dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')
print(f"Total samples: {len(df):,}")
print(f"Depth range: {df['Depth_Bin'].min():.1f} - {df['Depth_Bin'].max():.1f} m")

# Prepare features
feature_cols = ['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
                'NGR total counts (cps)', 'R', 'G', 'B']
X = df[feature_cols].values
depths = df['Depth_Bin'].values
boreholes = df['Borehole_ID'].values

# Scale features
print("\nScaling features...")
scaler = DistributionAwareScaler()
X_scaled = scaler.fit_transform(X)

# Load v2.14 model
print("\nLoading Semi-Supervised VAE v2.14...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

lithologies = sorted(df['Principal'].unique())
n_classes = len(lithologies)

model = SemiSupervisedVAE(
    input_dim=6,
    latent_dim=10,
    n_classes=n_classes,
    encoder_dims=[32, 16],
    classifier_hidden=32
)

checkpoint_path = 'ml_models/checkpoints/semisup_vae_alpha0.1.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()
print(f"✓ Model loaded (epoch {checkpoint['epoch']})")

# Generate latent embeddings
print("\nGenerating 10D latent embeddings...")
X_tensor = torch.FloatTensor(X_scaled).to(device)
with torch.no_grad():
    mu, _ = model.encode(X_tensor)
    embeddings = mu.cpu().numpy()

print(f"Embeddings shape: {embeddings.shape}")
print(f"Embedding statistics:")
for i in range(10):
    print(f"  z{i}: mean={np.mean(embeddings[:, i]):7.4f}, std={np.std(embeddings[:, i]):7.4f}")

# Create borehole-level train/test split (80/20)
print("\nCreating borehole-level train/test split...")
unique_boreholes = np.unique(boreholes)
np.random.seed(42)
np.random.shuffle(unique_boreholes)
split_idx = int(0.8 * len(unique_boreholes))
train_boreholes = set(unique_boreholes[:split_idx])

train_mask = np.array([b in train_boreholes for b in boreholes])
test_mask = ~train_mask

X_train = embeddings[train_mask]
y_train = depths[train_mask]
X_test = embeddings[test_mask]
y_test = depths[test_mask]

print(f"Training samples: {len(X_train):,} ({len(train_boreholes)} boreholes)")
print(f"Test samples: {len(X_test):,} ({len(unique_boreholes) - len(train_boreholes)} boreholes)")
print(f"Train depth range: {y_train.min():.1f} - {y_train.max():.1f} m")
print(f"Test depth range: {y_test.min():.1f} - {y_test.max():.1f} m")

# Train CatBoost regressor
print("\nTraining CatBoost depth predictor...")
model_depth = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    eval_metric='R2',
    random_seed=42,
    verbose=100
)

model_depth.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50,
    use_best_model=True
)

# Save model
model_depth.save_model('ml_models/depth_predictor_v2_14.cbm')
print("\n✓ Model saved: ml_models/depth_predictor_v2_14.cbm")

# Evaluate
print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)

y_pred_train = model_depth.predict(X_train)
y_pred_test = model_depth.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
train_mae = mean_absolute_error(y_train, y_pred_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"\nTraining Set:")
print(f"  R² Score:  {train_r2:.4f}")
print(f"  MAE:       {train_mae:.2f} m")
print(f"  RMSE:      {train_rmse:.2f} m")

print(f"\nTest Set:")
print(f"  R² Score:  {test_r2:.4f}")
print(f"  MAE:       {test_mae:.2f} m")
print(f"  RMSE:      {test_rmse:.2f} m")

# Feature importance
print("\nFeature Importance (Latent Dimensions):")
feature_importance = model_depth.get_feature_importance()
for i, imp in enumerate(feature_importance):
    print(f"  z{i}:  {imp:7.2f}")

# Save feature importance
importance_df = pd.DataFrame({
    'Latent_Dimension': [f'z{i}' for i in range(10)],
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
importance_df.to_csv('depth_predictor_feature_importance.csv', index=False)
print("\n✓ Saved: depth_predictor_feature_importance.csv")

# Visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Test set scatter plot
axes[0].scatter(y_test, y_pred_test, s=1, alpha=0.5, c='steelblue', rasterized=True)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('True Depth (m)', fontsize=12, weight='bold')
axes[0].set_ylabel('Predicted Depth (m)', fontsize=12, weight='bold')
axes[0].set_title(f'Test Set Predictions\n(R²={test_r2:.4f}, MAE={test_mae:.2f}m)',
                 fontsize=14, weight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Residual plot
residuals = y_test - y_pred_test
axes[1].scatter(y_pred_test, residuals, s=1, alpha=0.5, c='steelblue', rasterized=True)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Depth (m)', fontsize=12, weight='bold')
axes[1].set_ylabel('Residual (m)', fontsize=12, weight='bold')
axes[1].set_title(f'Residual Plot\n(RMSE={test_rmse:.2f}m)',
                 fontsize=14, weight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('depth_predictor_v2_14_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: depth_predictor_v2_14_results.png")
plt.close()

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
