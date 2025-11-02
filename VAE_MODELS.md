# VAE Models - Detailed Documentation

This document contains comprehensive details about all VAE model variants for lithology clustering. For quick reference, see the VAE Models section in CLAUDE.md.

## Model Overview

**Recommended for production: VAE GRA v2.6.6** (GMM ARI=0.286, 10D latent, β annealing)

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

## Transfer Learning Failure Summary

**All transfer learning approaches fail (-50% to -79% vs v2.6):**

| Approach | ARI (k=12) | vs v2.6 |
|----------|------------|---------|
| **v2.6 (baseline)** | **0.258** | **Baseline** |
| v2.6.1 (RSC+MSP, +44% data) | 0.119 | -54% ❌ |
| v2.6.2 (phys→RGB transfer) | 0.125 | -51% ❌ |
| v2.6.3 (RGB only) | 0.054 | -79% ❌❌ |
| v2.6.4 (dual pre-train) | 0.122 | -54% ❌ |

**Why Joint Training Wins:**
1. Pre-training optimizes for reconstruction, not cross-modal clustering
2. Cross-modal correlations ("dark + dense = basalt") must be learned jointly
3. Pre-trained encoders create representational commitments that prevent adaptation
4. 228 extra physical-only boreholes don't help because patterns don't transfer

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
