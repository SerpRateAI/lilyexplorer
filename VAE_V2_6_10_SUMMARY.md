# VAE v2.6.10: Predicted RGB Experiment - FAILURE

**Date:** 2025-11-04
**Status:** ❌ Failed (-53% vs v2.6.7)

## Motivation

Expand training dataset by predicting RGB from GRA+MS+NGR using supervised learning, then training VAE on mixed real+predicted RGB.

**Problem**: RGB camera coverage bottleneck (only 296/534 boreholes have RGB imaging)

**Hypothesis**: CatBoost RGB prediction (R²=0.72) + 77% more boreholes → better or similar clustering performance

## Approach

### Step 1: RGB Prediction Models

Train CatBoost regressors to predict RGB from physical properties:

```python
X = [GRA, MS, NGR]
y = [R, G, B]  # 3 separate models
```

**RGB Predictor Performance:**
- R channel: R²=0.731, RMSE=24.1
- G channel: R²=0.730, RMSE=23.8
- B channel: R²=0.708, RMSE=22.5
- Average: R²=0.723

**Feature importance:**
- NGR: 40% (high NGR → clay → darker)
- MS: 41% (magnetic minerals affect color)
- GRA: 18% (density less predictive of color)

**95th percentile error:** ~50 RGB units per channel

### Step 2: Dataset Creation

**Real RGB boreholes (296):** Use actual RGB camera measurements
**Predicted RGB boreholes (228):** Apply CatBoost models to predict RGB from GRA+MS+NGR

**Final dataset:**
- **395,682 samples** (+66% vs v2.6.7's 238,506)
- **523 boreholes** (+77% vs v2.6.7's 296)
- **60.3% real RGB** (238,506 samples)
- **39.7% predicted RGB** (157,176 samples)

### Step 3: VAE Training

**Architecture:** Same as v2.6.7
- 10D latent space
- β annealing: 1e-10 → 0.75 over 50 epochs
- Distribution-aware scaling
- [32, 16] hidden layers

**Training:** 100 epochs in 16.2 minutes on GPU

**Latent space:** 3 effective dimensions (7/10 collapsed)

## Results

### Clustering Performance (GMM)

| k | ARI | vs v2.6.7 | Silhouette |
|---|-----|-----------|------------|
| 10 | 0.112 | -43% | 0.284 |
| 12 | 0.106 | -46% | 0.290 |
| 15 | 0.079 | -60% | 0.272 |
| 20 | 0.073 | -63% | 0.250 |
| **Avg** | **0.093** | **-53%** | **0.274** |

**v2.6.7 baseline:** ARI = 0.196 ± 0.037

### Verdict: ✗ FAILURE

Despite +77% more boreholes and +66% more samples, v2.6.10 performs **-53% worse** than v2.6.7.

## Why It Failed

### 1. RGB Prediction Noise Corrupts Cross-Modal Correlations

**28% unexplained variance** in RGB predictions (1 - R²) introduces systematic errors:
- Predicted RGB has ±50 RGB unit errors (95th percentile)
- These errors break critical cross-modal patterns:
  - "Dark (RGB~30) + Dense (GRA~2.8) = Basalt" becomes ambiguous when RGB has ±50 unit noise
  - "Light (RGB~200) + Low density (GRA~1.4) = Carbonate ooze" loses discriminative power

### 2. More Data Doesn't Compensate for Quality Loss

**Feature quality > dataset size:**
- v2.6.7: 239K samples, 100% real RGB → ARI = 0.196
- v2.6.10: 396K samples, 60% real RGB → ARI = 0.093

The 157K predicted RGB samples dilute the 239K real RGB samples instead of augmenting them.

### 3. Supervised Imputation R²=0.72 Insufficient

**Comparison:**
- VAE v2.11 masked encoding: R² < 0 for imputation (complete failure)
- v2.6.10 CatBoost prediction: R²=0.72 (better than VAE, still insufficient)
- Threshold for clustering: Appears to require R² > ~0.9

**Why R²=0.72 fails:**
- Feature correlations are weak (NGR↔RGB r=-0.42, MS↔RGB r=-0.19)
- CatBoost captures bulk trends but misses fine-grained variations
- Geological heterogeneity at cm-scale exceeds prediction capability

### 4. Similar Pattern to Other Failures

v2.6.10 joins unsuccessful data expansion attempts:
- **v2.6.1 (RSC+MSP):** +44% data, -54% performance (wrong color space)
- **v2.6.8 (fuzzy matching):** +5% data, -55% performance (misalignment)
- **v2.6.10 (predicted RGB):** +66% data, -53% performance (prediction noise)

All share the same lesson: **Cannot overcome poor feature quality with more data**

## Scientific Insights

### RGB Prediction Quality Analysis

**Qualitative assessment** (from `rgb_prediction_examples.png`):
- ✓ Captures dark vs light reasonably well
- ✓ General color trends preserved
- ✗ Exact hues less accurate
- ✗ Subtle color variations lost (critical for lithology discrimination)

**Quantitative metrics:**
```
True RGB:      Well-separated lithology-specific colors
Predicted RGB: Blurred, less distinct color clusters
```

### CatBoost vs VAE for Imputation

| Method | Approach | R² | Clustering Impact |
|--------|----------|-----|-------------------|
| VAE v2.11 (masked) | Reconstruct from latent | <0 | -4% to -8% |
| v2.6.10 (CatBoost) | Supervised regression | 0.72 | -53% |
| **Requirement** | **Unknown prior** | **~0.9?** | **No degradation** |

CatBoost outperforms VAE for imputation but still falls short of clustering requirements.

### Cross-Modal Correlation Importance

VAE learns **emergent patterns** from feature combinations:
- "Dark clay" vs "dark basalt" distinguished by density+magnetism, not RGB alone
- Predicted RGB breaks these correlations because errors aren't consistent across features

When RGB is wrong by ±50 units, the multimodal signal "dark + dense + magnetic" degrades to "uncertain color + dense + magnetic".

## Files

**RGB Predictor Training:**
- `train_rgb_predictor.py` - CatBoost training script
- `ml_models/rgb_predictor_r.cbm` - R channel model
- `ml_models/rgb_predictor_g.cbm` - G channel model
- `ml_models/rgb_predictor_b.cbm` - B channel model
- `ml_models/rgb_predictor_summary.json` - Performance metrics

**RGB Prediction Visualization:**
- `visualize_rgb_predictions.py` - Scatter plots, residuals, color swatches
- `rgb_prediction_quality.png` - Scatter plots showing R²~0.73
- `rgb_prediction_examples.png` - 100 true vs predicted color swatches

**Dataset Creation:**
- `create_vae_v2_6_10_dataset.py` - Merge real+predicted RGB
- `vae_training_data_v2_6_10.csv` - 396K samples (47MB)
- `v2_6_10_dataset_creation.log` - Creation output

**VAE Training:**
- `train_vae_v2_6_10.py` - Training script
- `ml_models/checkpoints/vae_gra_v2_6_10.pth` - Model checkpoint
- `vae_v2_6_10_training.log` - Training output
- `vae_v2_6_10_clustering_results.csv` - GMM clustering results

## Conclusion

v2.6.10 demonstrates that:

1. **Supervised RGB imputation (R²=0.72) is insufficient** for multimodal clustering
2. **28% unexplained variance corrupts cross-modal correlations** essential for lithology discrimination
3. **More data with noisy features degrades performance** - quality trumps quantity
4. **RGB camera coverage bottleneck is real** - cannot be engineered around with prediction models

**Recommendation:** Continue using v2.6.7 with 239K samples and 100% real RGB. The 228 boreholes without RGB camera data are not usable for current multimodal approach.

**Alternative paths** (not pursued):
- Collect RGB camera data for remaining 228 boreholes (not feasible retrospectively)
- Use physical-only models (v1: 403K samples, ARI~0.06) - much worse performance
- Develop better RGB predictors (would require R² > 0.9, likely impossible given weak feature correlations)

**Final lesson:** Feature quality dominates dataset size. RGB camera imaging is essential and irreplaceable for lithology clustering.
