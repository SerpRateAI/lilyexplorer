"""
Visualize predicted vs true RGB to assess prediction quality.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

print("Loading models...")
model_r = CatBoostRegressor()
model_r.load_model('ml_models/rgb_predictor_r.cbm')

model_g = CatBoostRegressor()
model_g.load_model('ml_models/rgb_predictor_g.cbm')

model_b = CatBoostRegressor()
model_b.load_model('ml_models/rgb_predictor_b.cbm')

print("Loading VAE v2.6.7 dataset...")
df = pd.read_csv('vae_training_data_v2_20cm.csv')

# Features and targets
X = df[['Bulk density (GRA)', 'Magnetic susceptibility (instr. units)',
        'NGR total counts (cps)']].values
y_r = df['R'].values
y_g = df['G'].values
y_b = df['B'].values

# Get test set (same split as training)
X_train, X_test, y_r_train, y_r_test = train_test_split(
    X, y_r, test_size=0.2, random_state=42
)
_, _, y_g_train, y_g_test = train_test_split(
    X, y_g, test_size=0.2, random_state=42
)
_, _, y_b_train, y_b_test = train_test_split(
    X, y_b, test_size=0.2, random_state=42
)

print(f"Making predictions on {len(X_test):,} test samples...")
y_r_pred = model_r.predict(X_test)
y_g_pred = model_g.predict(X_test)
y_b_pred = model_b.predict(X_test)

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Top row: Predicted vs True scatter plots
for i, (y_true, y_pred, channel, color) in enumerate([
    (y_r_test, y_r_pred, 'R', 'red'),
    (y_g_test, y_g_pred, 'G', 'green'),
    (y_b_test, y_b_pred, 'B', 'blue')
]):
    ax = axes[0, i]

    # Scatter plot with transparency
    ax.scatter(y_true, y_pred, alpha=0.1, s=1, c=color)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect prediction')

    # Calculate metrics
    from sklearn.metrics import r2_score, mean_squared_error
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    ax.set_xlabel(f'True {channel}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Predicted {channel}', fontsize=14, fontweight='bold')
    ax.set_title(f'{channel} Channel: R²={r2:.3f}, RMSE={rmse:.1f}',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_aspect('equal')

    # Set same limits for both axes
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

# Bottom row: Residual plots
for i, (y_true, y_pred, channel, color) in enumerate([
    (y_r_test, y_r_pred, 'R', 'red'),
    (y_g_test, y_g_pred, 'G', 'green'),
    (y_b_test, y_b_pred, 'B', 'blue')
]):
    ax = axes[1, i]

    residuals = y_pred - y_true

    # Residual scatter
    ax.scatter(y_true, residuals, alpha=0.1, s=1, c=color)

    # Zero line
    ax.axhline(y=0, color='k', linestyle='--', linewidth=2)

    # Calculate residual statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)

    ax.set_xlabel(f'True {channel}', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Residual (Pred - True)', fontsize=14, fontweight='bold')
    ax.set_title(f'{channel} Residuals: μ={mean_residual:.2f}, σ={std_residual:.1f}',
                 fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('rgb_prediction_quality.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved rgb_prediction_quality.png")

# Also create example predictions visualization
print("\nCreating example predictions visualization...")

# Sample 100 random test points
np.random.seed(42)
sample_indices = np.random.choice(len(X_test), size=min(100, len(X_test)), replace=False)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# True colors
true_colors = np.stack([y_r_test[sample_indices],
                        y_g_test[sample_indices],
                        y_b_test[sample_indices]], axis=1) / 255.0
true_colors = np.clip(true_colors, 0, 1)

# Predicted colors
pred_colors = np.stack([y_r_pred[sample_indices],
                        y_g_pred[sample_indices],
                        y_b_pred[sample_indices]], axis=1) / 255.0
pred_colors = np.clip(pred_colors, 0, 1)

# Display as color swatches
n_cols = 10
n_rows = len(sample_indices) // n_cols

# True colors
for idx, i in enumerate(sample_indices[:n_rows*n_cols]):
    row = idx // n_cols
    col = idx % n_cols
    rect = plt.Rectangle((col, n_rows-row-1), 1, 1,
                         facecolor=true_colors[idx])
    ax[0].add_patch(rect)

ax[0].set_xlim(0, n_cols)
ax[0].set_ylim(0, n_rows)
ax[0].set_aspect('equal')
ax[0].set_title('True RGB Colors (100 random test samples)',
                fontsize=16, fontweight='bold', pad=20)
ax[0].axis('off')

# Predicted colors
for idx, i in enumerate(sample_indices[:n_rows*n_cols]):
    row = idx // n_cols
    col = idx % n_cols
    rect = plt.Rectangle((col, n_rows-row-1), 1, 1,
                         facecolor=pred_colors[idx])
    ax[1].add_patch(rect)

ax[1].set_xlim(0, n_cols)
ax[1].set_ylim(0, n_rows)
ax[1].set_aspect('equal')
ax[1].set_title('Predicted RGB Colors (from GRA+MS+NGR)',
                fontsize=16, fontweight='bold', pad=20)
ax[1].axis('off')

plt.tight_layout()
plt.savefig('rgb_prediction_examples.png', dpi=150, bbox_inches='tight')
print("✓ Saved rgb_prediction_examples.png")

# Print statistics
print("\n" + "="*80)
print("PREDICTION QUALITY STATISTICS")
print("="*80)

for channel, y_true, y_pred in [('R', y_r_test, y_r_pred),
                                 ('G', y_g_test, y_g_pred),
                                 ('B', y_b_test, y_b_pred)]:
    residuals = y_pred - y_true

    print(f"\n{channel} channel:")
    print(f"  R² = {r2_score(y_true, y_pred):.4f}")
    print(f"  RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.2f}")
    print(f"  Mean residual = {np.mean(residuals):+.2f} (bias)")
    print(f"  Std residual = {np.std(residuals):.2f}")
    print(f"  Max absolute error = {np.max(np.abs(residuals)):.1f}")
    print(f"  95th percentile error = {np.percentile(np.abs(residuals), 95):.1f}")

print("\n✓ Visualization complete!")
