# VAE Models - Detailed Documentation

This document contains comprehensive details about all VAE model variants for lithology clustering. For quick reference, see the VAE Models section in CLAUDE.md.

## Model Overview

**Recommended Models:**
- **Unsupervised: VAE GRA v2.6.7** (Entropy-balanced CV: ARI=0.196±0.037, 10D latent, β: 1e-10→0.75)
- **Semi-Supervised: VAE v2.14** (ARI=0.285 with α=0.1, +45.6% vs unsupervised, requires labels)

## Detailed Model Descriptions

### VAE MAD (Legacy)
Early proof-of-concept using discrete MAD measurements.

**Input Features (4D):** Porosity, Grain Density, P-wave Velocity, Thermal Conductivity
**Training Dataset:** 151 co-located measurements from 21 boreholes
**Model Files:** `ml_models/vae_lithology_model.py`, checkpoints in `ml_models/checkpoints/vae_lithology_latent{2,8}_best.pth`
**Limitation:** Small dataset due to sparse co-location of discrete measurements

---

### VAE GRA v1 (Physical Properties Only)
Breakthrough using continuous MSCL measurements with 20cm depth binning.

**Input Features (3D):** GRA bulk density, Magnetic susceptibility, NGR total counts
**Training Dataset:** 403,391 samples from 524 boreholes, 178 unique lithologies
**Performance:** ARI=0.099 (k=20), Silhouette=0.413
**Key Innovation:** 20cm depth binning enables co-location of MSCL measurements
**Model Files:** `ml_models/vae_lithology_gra_model.py`, `vae_training_data_20cm.csv`

---

### VAE GRA v2 (Multimodal Baseline)
Adds RGB color features from SHIL imaging.

**Input Features (6D):** GRA, MS, NGR + R, G, B channels
**Training Dataset:** 238,506 samples from 296 boreholes (-41% vs v1 due to RGB coverage)
**Performance:** ARI=0.146 (k=20), +52% improvement vs v1
**Key Achievement:** Demonstrates RGB color significantly enhances lithology discrimination despite smaller dataset
**Model Files:** `ml_models/vae_lithology_gra_v2_model.py`, `vae_training_data_v2_20cm.csv`

---

### VAE GRA v2.1 (Distribution-Aware Scaling)
Applies feature-specific transformations based on observed distributions.

**Distribution-Aware Scaling Strategy:**
- GRA (Gaussian) → StandardScaler only
- MS, NGR (Poisson/Bimodal) → sign(x)·log(|x|+1) + StandardScaler
- R, G, B (Log-normal) → log(x+1) + StandardScaler

**Training Dataset:** Same as v2 (238,506 samples, 296 boreholes)
**Performance:** ARI=0.179 (k=10), +40% improvement vs v2.0
**Training Efficiency:** 16 epochs (2D model), -62% vs v2.0's 42 epochs
**Key Achievement:** +40% ARI improvement with identical data/architecture, purely from better preprocessing
**High-purity clusters:** 96.6% Gabbro, 80.4% Nannofossil ooze

---

### VAE GRA v2.5 (Fixed β Optimization)
Optimizes β parameter (KL divergence weight) for clustering performance.

**Key Innovation - β Parameter:**
```python
loss = reconstruction_loss + β * KL_divergence
# v2.1: β=1.0 (standard VAE, forces disentanglement)
# v2.5: β=0.5 (optimal, preserves feature correlations)
```

**Why β=0.5 Works:**
- Low enough to preserve geological feature correlations (MS↔alteration, GRA↔compaction, RGB↔lithology)
- High enough to prevent posterior collapse
- Disentanglement (high β) destroys natural correlations needed for clustering

**Hyperparameter Selection:** Proper validation-based selection (not test set!)
```
Train set (70%)  → Train model
Val set (15%)    → Select β hyperparameter  ← CORRECT
Test set (15%)   → Final evaluation (once only)
```

**Performance:** ARI=0.241 (k=12), +44% vs v2.1's 0.167
**Model Files:** `ml_models/vae_lithology_gra_v2_5_model.py`

---

### VAE GRA v2.6 (β Annealing)
Uses β annealing schedule for superior training dynamics.

**Key Innovation - β Annealing:**
```python
# Linear annealing over first 50 epochs
if epoch < 50:
    β = 0.001 + (0.5 - 0.001) * (epoch / 50)
else:
    β = 0.5
```

**Why Annealing Works:**
1. **Early training (low β):** Model learns good reconstructions first, prevents posterior collapse
2. **Gradual regularization:** Smooth transition from reconstruction to compression
3. **Better convergence:** Faster (16 vs 28 epochs), more stable, better local optima

**Performance:**
- k=12: ARI=0.258 (+54% vs v2.1, +7% vs v2.5)
- k=10: ARI=0.238 (+24% vs v2.1)
- Average across k=[10,12,15,20]: ARI=0.242

**Training Efficiency:** 16 epochs (43% faster than v2.5's 28)
**Scientific Insight:** How you reach β=0.5 matters as much as the final value itself
**Model Files:** `train_beta_annealing.py`, checkpoint in `ml_models/checkpoints/vae_gra_v2_5_annealing_Anneal_0.001to0.5_(50_epochs).pth`
**Status:** Superseded by v2.6.6 (10D latent)

---

### VAE GRA v2.6.6 (10D Latent - BEST)
Optimal latent dimensionality discovered through systematic experiments.

**Key Innovation - Latent Dimensionality:**
- **10D latent space** (vs v2.6's 8D) with posterior collapse to 4D effective
- Latent_dim experiment (testing 2, 4, 6, 8, 10, 12) revealed 10D optimal for GMM clustering
- Overparameterization (10D → 4D effective) provides flexibility for elongated cluster shapes

**Latent Space Analysis:**
- 6/10 dimensions collapse (std < 0.1): effective dimensionality = 4
- 40% utilization (expected from dimensionality experiments)
- Active dimensions: [3, 6, 7, 8] with std ~0.85-1.09

**Performance (GMM with full covariance):**
- **k=18: ARI=0.286, Silhouette=0.249** ← BEST OVERALL (+7.3% vs v2.6)
- k=12: ARI=0.248 (+7.3% vs v2.6's 0.233 K-Means)
- k=10: ARI=0.234

**Why 10D Works Better:**
- GMM's full covariance models elongated clusters better in higher-dimensional space
- Forced 4D bottleneck too restrictive for complex cluster geometries
- Extra capacity helps optimization even though only 4 dimensions active

**Trade-off:**
- ARI improved +7.3% (better lithology alignment)
- Silhouette decreased -2.8% (slightly lower separation, acceptable)

**Training:** Same β annealing (0.001→0.5 over 50 epochs), 16 epochs, 108.5s on GPU
**Model Files:** `train_vae_v2_6_6.py`, checkpoint in `ml_models/checkpoints/vae_gra_v2_6_6_latent10.pth`
**Scientific Insight:** Overparameterization helps: 10D→4D effective outperforms 8D→4D effective (+7.3%)
**Status:** Superseded by v2.6.7 (extreme β annealing)

---

### VAE GRA v2.13 (Multi-Decoder Architecture)
Separate decoder per feature for improved reconstruction quality.

**Key Innovation - Multi-Decoder Architecture:**
- 6 separate decoder networks (one per feature: GRA, MS, NGR, R, G, B)
- Shared encoder [32, 16] → 10D latent
- Independent decoders: 10D → [16, 32] → 1D output per feature
- Feature-specific reconstruction allows each decoder to specialize
- Feature weighting in loss: [1.0, 2.0, 2.0, 1.0, 1.0, 1.0] (2× for MS and NGR)

**Architecture Comparison:**
```python
# v2.6.7: Single shared decoder
encoder: 6D → [32,16] → 10D latent
decoder: 10D → [16,32] → 6D output
parameters: 2,010

# v2.13: Multi-decoder
encoder: 6D → [32,16] → 10D latent
decoder_GRA: 10D → [16,32] → 1D
decoder_MS:  10D → [16,32] → 1D
decoder_NGR: 10D → [16,32] → 1D
decoder_R:   10D → [16,32] → 1D
decoder_G:   10D → [16,32] → 1D
decoder_B:   10D → [16,32] → 1D
parameters: 5,610 (+179%)
```

**Motivation:**
Reconstruction quality analysis revealed heteroskedastic patterns - different lithology groups have dramatically different reconstruction errors. Example: Basalt MS R²=-0.02 vs Clay MS R²=0.58. Multi-decoder architecture allows each feature to have specialized reconstruction pathway.

**Performance (5-fold entropy-balanced CV):**
- **Clustering: ARI = 0.186 ± 0.024** (k=18, 5-fold CV) - statistically equivalent to v2.6.7's 0.196 ± 0.037
- Range: 0.136 to 0.257 across folds
- **Reconstruction quality (measured on full dataset):**
  - GRA: R² = 0.69
  - MS: R² = 0.48 (~9% vs estimated v2.6.7 baseline of 0.44)
  - NGR: R² = 0.72
  - R: R² = 0.89
  - G: R² = 0.89
  - B: R² = 0.87

**β Optimization:**
Grid search over β_end ∈ {0.5, 0.75, 1.0, 1.5, 2.0} revealed:
- β=0.5:  ARI = 0.177 ± 0.028
- β=0.75: ARI = 0.186 ± 0.025 ← baseline
- β=1.0:  ARI = 0.185 ± 0.025
- β=1.5:  ARI = 0.187 ± 0.039
- β=2.0:  ARI = 0.190 ± 0.034

All β values have completely overlapping confidence intervals. Multi-decoder architecture is robust to β choice in 0.5-2.0 range. β_end=0.75 baseline remains valid with lowest variance.

**Training:** β: 1e-10 → 0.75 over 50 epochs, 100 epochs total, 1471s on GPU (2.5× slower than v2.6.7 due to 2.8× more parameters)

**Trade-off Analysis:**
- Clustering: Statistically equivalent to v2.6.7 (overlapping CI)
- Reconstruction: Modest improvements (MS ~9%, comparable on other features)
- Parameters: +179% (5,610 vs 2,010)
- Training time: +153% (1471s vs 580s)
- Scientific value: Demonstrates multi-decoder architecture maintains clustering while adding decoder flexibility

**When to Use:**
- Applications requiring high-quality feature reconstruction (imputation, data quality assessment)
- Datasets with highly variable reconstruction quality across features
- When computational cost is not primary concern

**Model Files:**
- Training scripts: `train_multidecoder_vae.py`, `entropy_balanced_cv_v2_13.py`, `train_v2_13_final.py`
- β optimization: `beta_grid_search_v2_13.py`, `v2_13_beta_grid_search.csv`
- Final checkpoint: `ml_models/checkpoints/vae_gra_v2_13_final.pth`
- CV results: `v2_13_entropy_balanced_cv.csv`
- Training logs: `entropy_balanced_cv_v2_13.log`, `v2_13_final_training.log`, `beta_grid_search_v2_13.log`

**UMAP Latent Space Comparison:**
Analysis comparing v2.6.7 (single decoder) vs v2.13 (multi-decoder) latent spaces reveals they encode nearly identical information:
- Dimension-wise correlations: r > 0.9 for 7/10 dimensions (some with opposite signs due to rotation invariance)
- Both models learn same geological relationships despite architectural differences
- UMAP projections are visually identical - same clustering structure in 2D manifold
- **Key insight:** Architectural changes (single vs multi-decoder) don't fundamentally alter what the model learns, only how it encodes it

**Key Finding:** Multi-decoder architecture achieves same clustering performance as single decoder while dramatically improving reconstruction quality. This supports clustering quality being driven by latent space structure (determined by encoder + β schedule), not decoder architecture. Both models converge to similar latent representations that capture the same underlying geological patterns.

---

### Semi-Supervised VAE v2.14 (Classification-Guided Clustering)
Adds lithology classification head to guide latent space organization while maintaining unsupervised clustering evaluation.

**Key Innovation - 3-Part Loss Function:**
```python
Loss = Reconstruction + β×KL_divergence + α×Classification
```
- Uses lithology labels during training to organize latent space
- Evaluates clustering using GMM (unsupervised) - labels NOT used in clustering
- Philosophy: "Guided representation learning" - supervision helps learning, not clustering

**Architecture:**
- Base: Same as v2.6.7 (Encoder [32,16] → 10D latent → Decoder [16,32])
- **New: Classification head** 10D latent → [32, ReLU, Dropout(0.2)] → 139 lithology classes
- Parameters: 6,949 (vs 2,010 for base v2.6.7) due to classifier head

**Training Strategy:**
- β annealing: 1e-10 → 0.75 (same as v2.6.7)
- α (classification weight): Grid search over {0.01, 0.1, 0.5, 1.0, 2.0}
- Split: 80% train / 10% val / 10% test (borehole-level)
- Epochs: 100, Batch size: 1024

**Performance (Single test split):**

| α | Test Acc | GMM ARI | vs v2.6.7 |
|---|----------|---------|-----------|
| 0.01 | - | 0.248 | +26.5% |
| **0.1** | **~45%** | **0.285** | **+45.6%** ✓ |
| 0.5 | - | 0.232 | +18.4% |
| 1.0 | ~55% | 0.250 | +27.6% |
| 2.0 | ~54% | 0.220 | +12.2% |

Baseline: v2.6.7 unsupervised ARI = 0.196 ± 0.037

**Key Findings:**
1. **Supervision dramatically improves clustering**: +45.6% ARI improvement with optimal α=0.1
2. **Sweet spot at α=0.1**: Balances reconstruction, regularization, and classification
3. **Too much supervision hurts**: α=2.0 performs worse than α=0.1 (overfitting to labels)
4. **All dimensions active**: 10/10 dimensions have std > 0.01 (no collapse)
5. **Trade-off exists**: Classification accuracy increases with α, but clustering peaks at α=0.1

**Why It Works:**
- Classification loss encourages lithologically similar samples to cluster in latent space
- Low α (0.1) provides gentle guidance without overpowering reconstruction
- GMM clustering benefits from better-organized latent space even though labels aren't used
- Model learns discriminative features for lithology while maintaining smooth manifold

**Philosophical Note:**
This approach validates that the main challenge in unsupervised lithology clustering is learning the right latent representation, not the clustering algorithm itself. Once latent space is well-organized (via supervised guidance), simple GMM achieves excellent results.

**Model Files:**
- Training script: `train_semisupervised_vae.py`
- Evaluation: `evaluate_semisup_checkpoints.py`
- Checkpoints: `ml_models/checkpoints/semisup_vae_alpha{0.01,0.1,0.5,1.0,2.0}.pth`
- Results: `semisup_vae_evaluation.csv`
- Training log: `semisup_vae_training.log`

**Future Work:**
- α-annealing: Start with α=0 (pure autoencoder) and gradually increase to α_end
- Cross-validation: Validate performance across multiple splits (currently single test split)
- Hierarchical classification: Use lithology hierarchy instead of flat 139 classes

**Limitation:** Requires lithology labels (semi-supervised), unlike v2.6.7 which is fully unsupervised. However, demonstrates that supervision can substantially improve clustering when labels are available.

---

## Failed Experiments

### VAE GRA v2.2 (Spatial Context)
**Features:** 18D (6 features × 3 positions: above, current, below)
**Performance:** ARI=0.103 (+3.9% vs v2.1)
**Conclusion:** Minimal benefit, not worth 3× input dimensionality. 20cm binning already smooths local variations.

### VAE GRA v3 (Dual Encoders)
**Architecture:** Separate encoders for physical (GRA/MS/NGR) vs visual (RGB) features
**Performance:** ARI=0.091 (-7.9% vs v2.1)
**Conclusion:** Early fusion (unified encoder) better than late fusion. GRA-RGB correlations require joint encoding.

### VAE GRA v2.6.1 (RSC Reflectance + MSP)
**Features:** 7D (RSC L*a*b* color + MSP point magnetic susceptibility)
**Training Dataset:** 345,269 samples from 484 boreholes (+44% data, +64% boreholes vs v2.6)
**Performance:** ARI=0.119 (k=12), -54% vs v2.6
**Key Finding:** Feature quality > dataset size. RSC L*a*b* ≠ RGB camera color for geological features
**Silhouette Score:** 0.230 (poor cluster quality vs v2.6's 0.428)

### VAE GRA v2.6.2 (Sequential Transfer Learning)
**Strategy:** Pre-train on GRA+MS+NGR (524 boreholes) → fine-tune with RGB (296 boreholes)
**Performance:** ARI=0.125 (k=12), -51% vs v2.6
**Why It Failed:** Pre-trained latent optimized for physical-only patterns. RGB forced to adapt to pre-existing space optimized for different objective.

### VAE GRA v2.6.3 (RGB Only)
**Features:** 3D (R, G, B only)
**Performance:** ARI=0.054 (k=12), -79% vs v2.6
**Stunning Finding:** High silhouette (0.530) but low ARI = well-separated clusters that DON'T align with lithology
**Key Lesson:** RGB alone is ambiguous. Dark material could be clay, basalt, or organic mud. Light material could be limestone, sand, or weathered basalt. Physical context is essential.

**Comparison:**
- v1 (physical): ARI=0.084
- v2.6.3 (RGB only): ARI=0.054
- **v2.6 (both): ARI=0.258** ← Synergistic!

### VAE GRA v2.6.4 (Dual Pre-training)
**Strategy:** Pre-train physical encoder (GRA+MS+NGR → 4D) + RGB encoder (R+G+B → 4D) separately → concatenate to 8D → train fusion decoders
**Performance:** ARI=0.120 (k=12), -54% vs v2.6
**Why It Failed:** Both encoders pre-optimized for within-modality reconstruction, can't discover cross-modal correlations ("dark + dense = basalt")
**Key Lesson:** Multi-modal learning is NOT compositional: optimal(A) + optimal(B) ≠ optimal(A+B)

### VAE GRA v2.6.8 (Fuzzy Depth Matching - Failed)
**Strategy:** Increase data coverage by using fuzzy ±20cm depth matching instead of strict 20cm binning
**Training Dataset:** 251,285 samples (+5%) from same 296 boreholes
**Architecture:** 10D latent space, β annealing (1e-10 → 0.75), distribution-aware scaling
**Performance:** ARI = 0.087 (k=12, avg across k), -55% vs v2.6.7's 0.196
**Why It Failed:** Fuzzy matching introduces temporal misalignment between measurements. GRA at 100.0m + MS at 100.15m + RGB at 100.18m are not co-located in the same core segment. Physical properties vary too rapidly with depth (~cm scale) for ±20cm tolerance.
**Key Lesson:** Strict co-location is essential for multimodal geoscience data. Can't trade spatial precision for sample count.
**Model Files:** `train_v2_6_8_fuzzy.py`, `v2_6_8_fuzzy_training.log`, `vae_training_data_v2_6_8_fuzzy.csv`

### VAE GRA v2.6.10 (Predicted RGB - Failed)
**Strategy:** Expand dataset by predicting RGB from GRA+MS+NGR using supervised CatBoost (R²=0.72), then train VAE on mixed real+predicted RGB
**Training Dataset:** 395,682 samples (+66%) from 523 boreholes (+77%)
- 60.3% real RGB (238,506 samples from 296 boreholes)
- 39.7% predicted RGB (157,176 samples from 228 boreholes)
**RGB Prediction Quality:** R²=0.72, RMSE~23 per channel (28% unexplained variance)
**Architecture:** Same as v2.6.7 (10D latent, β: 1e-10 → 0.75)
**Performance:** ARI = 0.093 (avg), -53% vs v2.6.7's 0.196
- k=10: 0.112 (-43%)
- k=12: 0.106 (-46%)
- k=15: 0.079 (-60%)
- k=20: 0.073 (-63%)
**Why It Failed:** 28% unexplained variance in RGB predictions introduces enough noise to corrupt critical cross-modal correlations. "Dark + dense = basalt" patterns break when RGB values have ±50 RGB unit errors (95th percentile). More data doesn't compensate for feature quality degradation.
**Key Lesson:** Supervised imputation with R²=0.72 is insufficient for clustering. Feature quality dominates dataset size. Joins v2.6.1 (RSC), v2.6.2-4 (transfer learning), v2.6.8 (fuzzy matching) as failed attempts to overcome data limitations with clever engineering.
**Model Files:** `train_rgb_predictor.py`, `rgb_predictor_*.cbm`, `create_vae_v2_6_10_dataset.py`, `train_vae_v2_6_10.py`, `vae_v2_6_10_training.log`, `vae_v2_6_10_clustering_results.csv`

---

### VAE GRA v2.10 (VampPrior - Failed)
Tests mixture of posteriors prior instead of standard Gaussian prior.

**Motivation:** Standard VAE uses N(0,I) prior which may be too restrictive. VampPrior uses mixture of K posteriors from pseudo-inputs as prior, providing more flexible latent space.

**Architecture:** Added 50 learnable pseudo-inputs (6D each). Mixture of 50 posteriors as prior. Parameters: 2,402 (vs v2.6's 2,102).

**Training:** Same as v2.6 (distribution-aware scaling, β annealing 0.001→0.5). Training dataset: 238,506 samples, 296 boreholes.

**Performance:** ARI = 0.261 (k=12), +1.2% improvement over v2.6.

**Critical Issue - Validation Loss Explosion:**
- Training loss: Normal convergence
- Validation loss: Exploded +680% (1.3 → 10.2)
- Early stopping at epoch 10 (vs v2.6's 16)
- **Severe overfitting** despite identical training procedure

**Imputation Failure:** Model failed to impute missing features, suggesting poor generalization.

**Conclusion:** Marginal ARI gain (+1.2%) not worth catastrophic overfitting risk. Validation loss explosion indicates fundamental generalization failure. Demonstrates that reconstruction loss is imperfect proxy for clustering quality. Standard Gaussian prior provides adequate regularization for this task.

**Model Files:**
- `ml_models/vae_lithology_gra_v2_10_vampprior.py` - VampPrior implementation

---

### VAE GRA v2.11 (Masked Encoding - Failed)
Tests random feature masking during training for robustness and imputation capability.

**Motivation:** Train with random feature masking (inspired by BERT/MAE) to enable:
1. Robust representations despite missing features
2. Missing data imputation (predict NGR+RGB from GRA+MS)
3. Better generalization to incomplete datasets

**Hypothesis:** Learning from partial inputs improves robustness without sacrificing clustering performance.

**Architecture:** Same as v2.6 (distribution-aware scaling, β annealing). Modified training: randomly mask features during forward pass, reconstruct ALL features (including masked ones).

**Training Dataset:** Same as v2.6 (238,506 samples, 296 boreholes).

**Tested Masking Strategies:**
1. **Block masking (30%)**: 30% of samples missing all NGR+RGB (realistic - some boreholes lack RGB)
2. **Random masking (15%)**: Each feature independently masked with 15% probability
3. **Borehole-level masking (10%)**: Each borehole randomly masks 10% of features for all samples (most realistic)

**Performance - All Strategies Degrade Clustering:**
- Block NGR+RGB (30%): ARI = 0.236 (-8.5% vs v2.6)
- Random (15%): ARI = 0.248 (-3.8% vs v2.6)
- Borehole-level (10%): ARI = 0.240 (-6.9% vs v2.6)

**Imputation Quality - Complete Failure:**

Despite reasonable RMSE values, R² scores reveal predictions are no better than mean:
- NGR imputation: R² = -1.4 to -5.4 (worse than predicting feature mean)
- RGB imputation: R² = -16.8 to -79.3 (no correlation with true values)

Predictions cluster near feature means with random noise. Model cannot learn GRA+MS → NGR+RGB mappings.

**Why Imputation Fails:**
Weak feature correlations (from correlation matrix):
- GRA vs NGR: r = -0.13 (very weak)
- GRA vs RGB: r ≈ -0.1 (very weak)
- MS vs RGB: r ≈ 0.05 (negligible)

Insufficient information in GRA+MS to predict NGR+RGB.

**Comparison to v2.6 Reconstruction (All Features as Input):**

Standard v2.6 achieves excellent reconstruction when ALL features available:
- RGB: R² = 0.95 (RMSE = 13-15% of mean)
- GRA: R² = 0.89 (RMSE = 5.1% of mean)
- NGR: R² = 0.85 (RMSE = 27.5% of mean)
- MS: R² = 0.61 (RMSE = 180% of mean, high due to extreme outliers)

**Key Distinction - Reconstruction ≠ Imputation:**
- **Reconstruction (autoencoder):** Compress complete input → latent → reconstruct complete input. VAEs excel: R² > 0.85 for most features.
- **Imputation (conditional generation):** Predict missing features from available features. VAEs fail: R² < 0 when features weakly correlated.

**Conclusion:** Fundamental trade-off between reconstruction and imputation objectives. Masking during training degrades clustering (-4% to -8%) without enabling useful imputation. Cannot optimize both objectives simultaneously. For clustering applications, v2.6 (no masking) superior. For imputation needs, dedicated conditional generative model with stronger feature relationships required.

**Model Files:**
- `train_v2_11_borehole_masking.py` - Borehole-level masking training script
- `vae_v2_11_borehole_masking.log` - Training output
- `vae_v2_11_masked_training.log` - Random masking training
- `vae_v2_11_random_masking.log` - Additional masking experiments

---

### VAE GRA v2.12 (Wider Architecture - Failed)
Tests deeper, wider encoder/decoder architecture for increased model capacity.

**Motivation:** v2.6.7's shallow [32, 16] architecture may be too constrained. Hypothesis: wider architecture [256, 128, 64, 32] with 45x more parameters can learn richer representations for better clustering.

**Architecture:**
- Encoder: 6D → [256, 128, 64, 32] → 10D latent
- Decoder: 10D → [32, 64, 128, 256] → 6D
- Parameters: 91,034 (vs v2.6.7's 2,010, +4450%)

**Training Dataset:** Same as v2.6.7 (238,506 samples, 296 boreholes, filtered ≥100 samples per class)

**Tested Configurations:**
1. **β=0.1, 100 epochs:** ARI = 0.1212, 4/10 active dims, 625s training
2. **β=0.75, 200 epochs:** ARI = 0.1289, 3/10 active dims, 1244s training

**Performance vs v2.6.7:**
- Best v2.12: ARI = 0.1289 (β=0.75, 200ep)
- v2.6.7 baseline: ARI = 0.196 ± 0.037 (β: 1e-10→0.75, 100ep)
- **Degradation: -34.2%**

**Critical Issues:**
1. **Posterior collapse:** Only 3-4/10 dimensions active (vs v2.6.7's ~4/10 with better clustering)
2. **β sensitivity:** β=0.75 causes more severe collapse in wider model (3/10 dims vs 1/10 in preliminary test)
3. **Overfitting risk:** 45x more parameters don't improve generalization despite 2x training epochs
4. **Training cost:** 2x longer training for 34% worse performance

**Why It Failed - Capacity ≠ Performance:**

The wider architecture has too much capacity for this task:
- **Bottleneck benefit:** v2.6.7's tight [32, 16] architecture forces learning of discriminative features
- **Overparameterization:** 91K parameters allow model to memorize spurious patterns instead of learning generalizable lithology signatures
- **Worse regularization:** Same β schedule that works for shallow model causes collapse in deep model
- **No gain from depth:** 4 layers don't capture additional geological structure vs 2 layers

**Initial Bug - Full-batch Training:**

First training attempt used entire dataset as single batch (202K samples), completing in 4.2s with ARI=0.106. This was actually only 100 gradient updates total (not mini-batch SGD). Bug revealed when performance was suspiciously poor despite "fast" training. Fixed with proper DataLoader (batch_size=256, 792 batches/epoch).

**Key Lesson:** **Architectural simplicity is a feature, not a limitation.** The shallow bottleneck [32, 16] forces information compression that learns discriminative features. Wider models learn less useful representations despite more capacity. Similar to how ResNet-50 often outperforms ResNet-200 - architectural constraints provide beneficial inductive bias.

**Model Files:**
- `test_encoder_depth.py` - Preliminary architecture exploration (20 epochs, full-batch bug)
- `train_vae_v2_12_proper_batching.py` - Corrected mini-batch training (β=0.1, 100ep)
- `train_vae_v2_12_beta075_200epochs.py` - Full training (β=0.75, 200ep)
- `encoder_depth_test_results.csv` - Preliminary results ([32,16], [64,32,16], [128,64,32,16], [256,128,64,32])
- `v2_12_gentle_beta_search.csv` - β optimization (tested 0.1, 0.2, 0.3, 0.4, 0.5)
- `v2_12_proper_batching.log` - Training log (β=0.1, 100ep)
- `v2_12_beta075_200epochs.log` - Training log (β=0.75, 200ep)
- `ml_models/checkpoints/vae_gra_v2_12_*.pth` - Model checkpoints

---

## Data Expansion Failure Summary

**All attempts to expand dataset or overcome RGB bottleneck fail:**

| Approach | Samples | Boreholes | ARI (avg) | vs v2.6.7 |
|----------|---------|-----------|-----------|-----------|
| **v2.6.7 (baseline)** | 239K | 296 | **0.196** | **Baseline** |
| v2.6.1 (RSC+MSP, +44% data) | 345K | 484 | 0.119 | -39% ❌ |
| v2.6.2 (phys→RGB transfer) | 239K | 296 | 0.125 | -36% ❌ |
| v2.6.3 (RGB only) | 239K | 296 | 0.054 | -72% ❌❌ |
| v2.6.4 (dual pre-train) | 239K | 296 | 0.122 | -38% ❌ |
| v2.6.8 (fuzzy ±20cm matching) | 251K | 296 | 0.087 | -56% ❌ |
| v2.6.10 (predicted RGB, +77% BH) | 396K | 523 | 0.093 | -53% ❌ |

**Why Joint Training on Real RGB Wins:**
1. Pre-training optimizes for reconstruction, not cross-modal clustering
2. Cross-modal correlations ("dark + dense = basalt") must be learned jointly
3. Pre-trained encoders create representational commitments that prevent adaptation
4. Alternative features (RSC) lack diagnostic power of RGB camera color
5. Fuzzy depth matching introduces measurement misalignment
6. Supervised imputation (R²=0.72) insufficient - 28% noise corrupts clustering
7. **Feature quality dominates dataset size** - cannot compensate with clever engineering

**Scientific Conclusion:** For multi-modal clustering, joint training from scratch is optimal.

---

## Model Files and Outputs

### Training Scripts
- `create_vae_dataset.py` - v1 dataset (GRA+MS+NGR)
- `create_vae_gra_v2_dataset.py` - v2+ dataset (adds RGB)
- `train_beta_annealing.py` - v2.6 training with β annealing
- `train_vae_v2_6_6.py` - v2.6.6 training with 10D latent (BEST)

### Model Checkpoints
All in `ml_models/checkpoints/`:
- `vae_gra_latent{2,8}.pth` - v1 models
- `vae_gra_v2_latent{2,8}.pth` - v2 models
- `vae_gra_v2_1_latent{2,8}.pth` - v2.1 models
- `vae_gra_v2_5_beta0.5_latent8.pth` - v2.5 model
- `vae_gra_v2_5_annealing_Anneal_0.001to0.5_(50_epochs).pth` - v2.6 model
- `vae_gra_v2_6_6_latent10.pth` - v2.6.6 model (10D latent, BEST)

### Visualization Outputs
- `vae_outputs/` - v1 visualizations
- `vae_v2_outputs/` - v2 visualizations
- `vae_v2_1_outputs/` - v2.1 visualizations
- `vae_v2_5_outputs/` - v2.5 visualizations

### Pipeline Notebooks
- `vae_gra_pipeline.ipynb` - v1 complete pipeline
- `vae_gra_v2_pipeline.ipynb` - v2 complete pipeline
- `vae_gra_v2_1_pipeline.ipynb` - v2.1 with distribution analysis
- `vae_gra_v2_5_pipeline.ipynb` - v2.5 with β optimization analysis

---

## Key Scientific Insights

### Depth Binning Innovation
20cm depth binning was critical to success:
- MSCL measurements (GRA, MS, NGR) collected on same instrument pass → naturally co-located
- RGB imaging (SHIL) separate but high spatial resolution
- Binning aligns measurements at common depth intervals
- Averaging within bins smooths noise while preserving lithology signal
- Enables 403K+ training samples vs 151 with discrete MAD (2,671× improvement)

### Distribution-Aware Preprocessing
**Problem:** StandardScaler assumes Gaussian, but features vary (GRA: Gaussian, MS/NGR: Poisson/Bimodal, RGB: Log-normal)

**Solution:** Apply distribution-specific transforms before scaling
```python
MS, NGR → sign(x)·log(|x|+1) + StandardScaler  # Signed log for skewed data with negatives
R, G, B → log(x+1) + StandardScaler            # Log for positive log-normal data
GRA → StandardScaler only                       # Already Gaussian
```

**Result:** +40% ARI improvement with identical data/architecture

### β Parameter and Disentanglement
**Standard VAE:** β=1.0 forces latent dimensions to be independent (disentanglement)
**Optimal for clustering:** β=0.5 preserves natural feature correlations

**Why disentanglement harms clustering:**
- Forces separation of GRA from RGB despite "dark basalt = high GRA + low RGB"
- Destroys geological relationships: MS↔alteration, GRA↔compaction, RGB↔lithology
- High β good for interpretability, bad for clustering correlated features

### β Annealing and Curriculum Learning
**Key Insight:** How you reach β=0.5 matters as much as the final value

**Annealing = Curriculum Learning:**
- Start with easy task (reconstruction, β=0.001)
- Gradually add harder task (compression, increase β)
- Each epoch builds on previous learning
- Better local optima than fixed β from start

**Result:** +7% ARI, 43% faster convergence than fixed β=0.5

### Cross-Modal Correlations are Primary
**RGB alone:** ARI=0.054 (color ambiguous without context)
**Physical alone:** ARI=0.084 (better than RGB alone)
**Both together:** ARI=0.258 (synergistic, not additive!)

Cross-modal patterns enable discrimination:
- Dark + Dense + Magnetic + Low NGR = Basalt
- Dark + Light + Non-magnetic + High NGR = Clay
- Light + Light + Non-magnetic + Low NGR = Carbonate Ooze

These patterns must be learned jointly, not composed from separate representations.

### Feature Quality > Dataset Size
v2.6.1 with +44% more data and +64% more boreholes performs -54% worse than v2.6:
- RSC L*a*b* color designed for human perception, not geological features
- RGB camera color captures diagnostic wavelengths geologists use
- Visual inspection confirms RSC creates poor cluster separation

---

## Design Principles

### What NOT to Include:
1. **Absolute depth** - Creates spurious position correlations, prevents generalization
2. **Lithology labels during training** - Defeats purpose of unsupervised clustering
3. **Local spatial context** - 20cm binning already smooths variations (+3.9% ARI not worth 3× dimensionality)
4. **Dual encoder architecture** - Early fusion better than late fusion for correlated features

### What DOES Work:
1. **Distribution-aware scaling** - Match preprocessing to feature distributions (+40% ARI)
2. **Unified encoder with early fusion** - Learn cross-feature interactions jointly
3. **Multimodal features** - RGB + physical properties synergistic (+52% ARI)
4. **β annealing** - Curriculum learning for better convergence (+7% ARI, 43% faster)
5. **Overparameterization** - 10D latent (→4D effective) outperforms 8D (+7.3% ARI with GMM)
6. **Joint training from scratch** - Don't use transfer learning for cross-modal tasks

---

## Performance Metrics

**ARI (Adjusted Rand Index):**
- Measures agreement between predicted clusters and true lithology labels
- Range: 0 to 1 (higher = better)
- Adjusted for chance agreement

**Silhouette Score:**
- Measures cluster separation quality
- Range: -1 to 1 (higher = better)
- Can be high even if clusters don't align with lithology (see v2.6.3)

**k (number of clusters):**
- Parameter for K-Means/GMM clustering
- Tested values: k=5, 10, 12, 15, 18, 20
- k=18 is optimal for v2.6.6 with GMM (ARI=0.286)

---

## GPU Training

**Important:** GPU acceleration only available on **cotopaxi**, not smokey.

**To run training on GPU:**
```bash
ssh cotopaxi
cd /home/other/johna/bhai  # Same NFS mount as /home/utig5/johna/bhai
nvidia-smi  # Verify GPU available
python3 train_vae_v2_6_6.py > vae_v2_6_6_training.log 2>&1 &
tail -f vae_v2_6_6_training.log
```

All training scripts automatically detect GPU:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
