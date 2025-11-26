# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Supplemental Documentation

For detailed information on specific topics, see:
- **[DEVELOPMENT_SETUP.md](DEVELOPMENT_SETUP.md)** - Python setup, GPU configuration (cotopaxi), UMAP issues
- **[ANALYSIS_NOTEBOOKS.md](ANALYSIS_NOTEBOOKS.md)** - Measurement-specific notebooks, VAE analysis pipelines
- **[SCIENTIFIC_INSIGHTS.md](SCIENTIFIC_INSIGHTS.md)** - Core innovations, key lessons from VAE development
- **[GIT_WORKFLOW.md](GIT_WORKFLOW.md)** - Commit workflow and best practices (commit after each major task)
- **[VAE_CLASSIFIER_INVESTIGATION_SUMMARY.md](VAE_CLASSIFIER_INVESTIGATION_SUMMARY.md)** - Complete lithology classifier investigation (2025-11-01)
- **[SEMISUPERVISED_VAE_CLASSIFIER_SUMMARY.md](SEMISUPERVISED_VAE_CLASSIFIER_SUMMARY.md)** - Semi-supervised classifier experiment (2025-11-11, failed)

## Project Overview

This repository contains the LILY Database (LIMS with Lithology) - a comprehensive dataset linking lithological information to physical, chemical, and magnetic properties data from IODP (International Ocean Discovery Program) expeditions 2009-2019. The dataset contains over 34 million measurements from 89 km of core recovered across 42 expeditions.

## Key Reference Paper

The paper at `/papers/lilypaper.pdf` (Childress et al., 2024, Geochemistry, Geophysics, Geosystems) describes the complete dataset construction, methodology, and boundary conditions. **Read this paper first** to understand the data structure and scientific context.

## Data Structure

### Dataset Location
All datasets are located in `/datasets/` (note: CLAUDE.md mentions `/data_set` but actual directory is `/datasets/`)

### Available Datasets
The repository contains multiple CSV files with different measurement types, all following the naming pattern `{MEASUREMENT_TYPE}_DataLITH.csv`:

- **GRA_DataLITH.csv** - Gamma ray attenuation bulk density (largest file, >1GB, 3.7M+ measurements)
- **MAD_DataLITH.csv** - Moisture and density measurements (grain density, bulk density, porosity)
- **RGB_DataLITH.csv** - Digital image color data (largest dataset by count, >10M measurements)
- **MS_DataLITH.csv** - Magnetic susceptibility
- **NGR_DataLITH.csv** - Natural gamma radiation
- **RSC_DataLITH.csv** - Reflectance spectroscopy (color reflectance data)
- **SRM_DataLITH.csv** - Natural remanent magnetization
- **IW_DataLITH.csv** - Interstitial water chemistry
- **CARB_DataLITH.csv** - Carbonate content (CaCO3 measurements)
- **ICP_DataLITH.csv** - Inductively coupled plasma measurements
- **AVS_DataLITH.csv** - Automated Vane Shear strength measurements
- **PWC_DataLITH.csv** - P-wave velocity measurements
- **THCN_DataLITH.csv** - Thermal conductivity measurements
- Plus additional measurement types (see paper Table S1 for complete list)

### Common Data Fields
All datasets share core identification fields:
- Expedition, Site, Hole, Core, Type, Section (Sect), Archive/Working half (A/W)
- Depth measurements: `Depth CSF-A (m)` and `Depth CSF-B (m)`
- Lithology: `Prefix`, `Principal`, `Suffix`, `Full Lithology`, `Simplified Lithology`
- Location metadata: `Latitude (DD)`, `Longitude (DD)`, `Water Depth (mbsl)`
- `Expanded Core Type`: Coring system (APC, HLAPC, XCB, RCB, etc.)

### Lithology Classification
Lithologies follow modified Mazzullo et al. (1988) and Dean et al. (1985) schemes:
- **Principal lithology**: Dominant composition (>50-60%)
- **Prefix (major modifier)**: Components 25-50% of composition
- **Suffix (minor modifier)**: Components 10-25% of composition

Common principal lithologies include: nannofossil ooze, clay, silty clay, diatom ooze, basalt, mud, chalk, sand, etc.

## Primary Task: Figure Reproduction

The main objective is to recreate all plots from the paper in `/papers/lilypaper.pdf`.

### Workflow
1. **Output directory**: Generate Python scripts in `/paper_plots_code/`
2. **Script naming**: `figure_{number}.py` for each figure in the paper
3. **Image output**: Save plots as PNG in `/paper_plots/`
4. **Execution script**: Create `generate_paper_plots.sh` in repo root
5. **Validation**: Compare generated plots with originals, iterate to improve accuracy
6. **Documentation**: Create `data_comparison.pdf` with side-by-side comparisons

### Critical Rules
- **Never fabricate data** - all data must come from the datasets in `/datasets/`
- Use Python for all code unless explicitly specified otherwise
- Match original plots as closely as possible (colors, scales, labels, styles)
- Handle large files efficiently (GRA and RGB datasets are multi-GB)

## Data Analysis Insights from Paper

### Key Statistics
- 209 unique principal lithologies
- Most common: nannofossil ooze (>20% of descriptions)
- 431 unique prefix values
- 185 unique suffix values
- Unconsolidated sediments: ~71% of cores
- Biogenic oozes: 41% of unconsolidated sediments

### Coring Systems
- **APC** (Advanced Piston Corer): Soft sediments, high recovery (~103%)
- **HLAPC** (Half-Length APC): Intermediate sediments
- **XCB** (Extended Core Barrel): Semi-lithified materials
- **RCB** (Rotary Core Barrel): Hard rock, variable recovery

### Important Data Relationships
- Grain density depends on lithology (e.g., calcite-rich ~2.71 g/cm³, basalt ~2.89 g/cm³)
- Porosity = (ρ_fluid - ρ_grain)/(ρ_bulk - ρ_grain) for seawater (ρ_fluid = 1.024 g/cm³)
- GRA bulk densities have systematic biases for RCB cores (need correction factors from paper Table S8)
- MAD measurements are discrete samples; GRA provides continuous high-resolution data

## Development Environment

**Python**: 3.11+ with dependencies in `pyproject.toml` (pandas, numpy, matplotlib, torch, scikit-learn, catboost)

**GPU Training**: Available on **cotopaxi** only (PyTorch 2.8.0 + CUDA 12.8)
- Smokey: CPU-only (no NVIDIA driver)
- Shared NFS filesystem between systems

**UMAP Issue**: Import fails on smokey due to numba/numpy conflicts. Run full analysis notebooks on **cotopaxi** for UMAP visualizations.

**For detailed setup instructions**: See [`DEVELOPMENT_SETUP.md`](DEVELOPMENT_SETUP.md)

## Working with Large Files

Several datasets exceed 1GB. Use efficient loading strategies:
```python
# Read in chunks for large files
import pandas as pd
df = pd.read_csv('datasets/GRA_DataLITH.csv', chunksize=100000)

# Or read specific columns only
df = pd.read_csv('datasets/GRA_DataLITH.csv', usecols=['Exp', 'Depth CSF-A (m)', 'Bulk density (GRA)', 'Principal'])
```

## Figure Types in Paper

Based on the paper structure, expect to reproduce:
- Histograms (density distributions, recovery percentages)
- Scatter plots (MAD vs GRA comparisons, density vs porosity)
- Geographic maps (expedition locations, Figure 2)
- Bar charts (lithology statistics)
- Box plots (coring system performance)
- Multi-panel comparison figures

## Additional Resources

- Paper supporting information contains extensive tables (S1-S8) with:
  - Complete data type lists
  - Data quantity by expedition
  - Grain density lookup tables by lithology
  - GRA correction factors
  - Lithology dictionaries

## Analysis Notebooks

**Measurement-specific**: `lily_analysis.ipynb`, `gra_analysis.ipynb`, `LILY-AVS.ipynb`, `LILY-RSC.ipynb`, `explore_CaCO3.ipynb`, `borehole-age.ipynb`

**VAE pipelines**: `vae_gra_pipeline.ipynb` (v1: 3D features), `vae_gra_v2_pipeline.ipynb` (v2: 6D multimodal), `vae_gra_v2_1_pipeline.ipynb` (distribution-aware scaling), `vae_gra_v2_5_pipeline.ipynb` (β optimization)

**Production model**: **`vae_v2_6_7_analysis.ipynb`** - Gold standard v2.6.7 analysis with Q-Q plots, UMAP, GMM clustering (ARI=0.196±0.037)

**For complete notebook descriptions**: See [`ANALYSIS_NOTEBOOKS.md`](ANALYSIS_NOTEBOOKS.md)

## Machine Learning Models

### GRA Bulk Density Prediction
CatBoost regression models for predicting bulk density from GRA measurements:

**Model Files:**
- `bulk_density_gra_model.cbm` - Standard model
- `bulk_density_gra_model_borehole_split.cbm` - Model with borehole-based train/test split

**Training Scripts:**
- `bulk_density_gra_model.py` - Model training and evaluation code

**Training Logs:**
- `bulk_density_gra_training.log` - Standard model training output
- `bulk_density_gra_borehole_split.log` - Borehole-split model training output

**Feature Analysis:**
- `feature_importance_gra.csv` - Feature importance for standard model
- `feature_importance_gra_borehole_split.csv` - Feature importance for borehole-split model

**Model Artifacts:**
- `catboost_info/` - CatBoost training metadata and intermediate files

### Model Usage
Models predict MAD bulk density from GRA measurements and other features, accounting for systematic biases mentioned in the paper. The borehole-split version ensures no data leakage between training and testing by splitting at the borehole level rather than randomly.

### RGB Color Prediction (Failed Experiment)

CatBoost regression models for predicting RGB color from physical properties (used in failed v2.6.10 experiment):

**Model Files:**
- `ml_models/rgb_predictor_r.cbm` - Red channel predictor
- `ml_models/rgb_predictor_g.cbm` - Green channel predictor
- `ml_models/rgb_predictor_b.cbm` - Blue channel predictor
- `ml_models/rgb_predictor_summary.json` - Performance metrics

**Training Scripts:**
- `train_rgb_predictor.py` - CatBoost training for RGB prediction

**Visualization:**
- `visualize_rgb_predictions.py` - Prediction quality assessment
- `rgb_prediction_quality.png` - Scatter plots and residuals
- `rgb_prediction_examples.png` - True vs predicted color swatches

**Performance:**
- R²=0.72 (avg), RMSE~23 per channel
- Feature importance: NGR 40%, MS 41%, GRA 18%
- 95th percentile error: ~50 RGB units

**Usage:** These models were used to expand the VAE v2.6.10 dataset by predicting RGB for 228 boreholes lacking camera data. Experiment failed (-53% ARI) because 28% unexplained variance corrupted cross-modal correlations. **Not recommended for production use.** See `VAE_V2_6_10_SUMMARY.md` for details.

### Lithology Prediction from Physical Properties (VAE Models)

**Recommendation:**
- **Unsupervised clustering**: Use VAE GRA v2.6.7 (ARI = 0.196 ± 0.037)
- **Supervised classification + clustering**: Use Semi-Supervised VAE v2.14 (ARI = 0.285, Pooled AUC = 0.917)

**Quick Reference:**

| Model | Features | Samples | GMM ARI / Classification | Status |
|-------|----------|---------|--------------------------|--------|
| **Semi-Sup VAE v2.14** | **10D latent + 139-class head** (α=0.1, β: 1e-10→0.75) | 239K | **ARI=0.285**, Pooled AUC=0.917 | **✓ Best overall** |
| **VAE GRA v2.6.7** | **10D latent, [32,16] arch** (GRA+MS+NGR+RGB, β: 1e-10→0.75) | 239K | **0.196 ± 0.037** | **✓ Best unsupervised** |
| VAE GRA v2.13 | 10D latent, multi-decoder (6 decoders, β: 1e-10→0.75) | 239K | 0.187 ± 0.045 | Equivalent clustering, architectural study |
| VAE GRA v2.12 | 10D latent, [256,128,64,32] arch (β: 1e-10→0.75, 200ep) | 239K | 0.129 | ❌ Failed (-34%, wider arch) |
| VAE GRA v2.6.10 | 10D latent (60% real + 40% predicted RGB) | 396K | 0.093 | ❌ Failed (-53%) |
| VAE GRA v2.6.8 | 10D latent (fuzzy ±20cm depth matching) | 251K | 0.087 | ❌ Failed (-55%) |
| VAE GRA v2.6.6 | 10D latent (GRA+MS+NGR+RGB, β: 0.001→0.5) | 239K | 0.19 ± 0.05 | Superseded by v2.6.7 |
| VAE GRA v2.6 | 8D latent (GRA+MS+NGR+RGB, β anneal) | 239K | ~0.19 (est.) | Superseded by v2.6.7 |
| VAE GRA v2.5 | 6D (fixed β=0.5) | 239K | ~0.18 (est.) | Fixed β baseline |
| VAE GRA v2.1 | 6D (dist-aware, β=1.0) | 239K | ~0.13 (est.) | Distribution baseline |
| VAE GRA v2 | 6D (standard scaling) | 239K | ~0.10 (est.) | Multimodal baseline |
| VAE GRA v1 | 3D (physical only) | 403K | ~0.06 (est.) | Max coverage |
| Failed experiments (10 variants) | Various | — | 0.04-0.19 (est.) | ❌ Don't use |

**Key Model Files:**

**v2.6.7 (Production - Recommended):**
- **Production Model**: `ml_models/checkpoints/vae_gra_v2_6_7_final.pth` (trained on all 238,506 samples)
- Training CV: `entropy_balanced_cv_v2_6_7.py` (5-fold CV with β: 1e-10→0.75)
- Final Training: `train_v2_6_7_final.py` (production model on 100% data)
- CV Results: `v2_6_7_entropy_balanced_cv.csv`
- Training Log: `v2_6_7_final_training.log`
- Dataset: `vae_training_data_v2_20cm.csv` (239K samples, 24MB)
- Implementation: `ml_models/vae_lithology_gra_v2_5_model.py` (same architecture as v2.6.6, different β schedule)
- Analysis Notebook: `vae_v2_6_7_analysis.ipynb` (Q-Q plots, UMAP, statistics)

**v2.13 (Multi-Decoder - Architectural Study):**
- Model: `ml_models/checkpoints/vae_gra_v2_13_final.pth` (6 separate decoders, 5,610 parameters)
- Training CV: `entropy_balanced_cv_v2_13.py` (5-fold CV with β: 1e-10→0.75)
- Final Training: `train_v2_13_final.py` (production model on 100% data)
- β Grid Search: `beta_grid_search_v2_13.py` (tested β ∈ {0.5, 0.75, 1.0, 1.5, 2.0})
- CV Results: `v2_13_entropy_balanced_cv.csv`, `beta_grid_search_v2_13.csv`
- Training Logs: `v2_13_final_training.log`, `entropy_balanced_cv_v2_13.log`, `beta_grid_search_v2_13.log`
- Implementation: `ml_models/vae_lithology_gra_v2_13_model.py` (multi-decoder architecture)
- Visualization: `visualize_vae_v2_13_multidecoder_architecture.py` (architecture diagram)
- Reconstruction: `visualize_v2_13_reconstruction.py` (predicted vs true scatter plots)
- Investigation: `V2_13_RECONSTRUCTION_INVESTIGATION.md` (documents early experiments vs final model)

**v2.14 (Semi-Supervised VAE - Best Overall Performance):**
- Model: `ml_models/checkpoints/semisup_vae_alpha0.1.pth` (classification head: 10D→[32,ReLU,Dropout]→139 classes)
- Training: `train_semisupervised_vae.py` (α grid search: {0.01, 0.1, 0.5, 1.0, 2.0}, β: 1e-10→0.75)
- α Grid Search Results: `semisup_alpha_grid_search.csv` (best: α=0.1, ARI=0.285)
- Training Log: `semisup_vae_training.log` (5 epochs, 188s)
- Implementation: `ml_models/semisup_vae_model.py` (semi-supervised architecture)
- Visualization:
  - Architecture: `visualize_vae_v2_14_architecture.py` (horizontal flow diagram)
  - Reconstruction: `visualize_v2_14_reconstruction.py` (dark red scatter plots)
  - UMAP: `plot_v2_14_umap.py` (cluster + lithology projections with gabbro label)
  - ROC Curves: `plot_v2_14_roc_curves.py` (pooled AUC=0.917, top 20 classes, AUC distribution)
- Results:
  - `v2_14_umap_projection.png`, `v2_14_umap_by_lithology.png`
  - `v2_14_reconstruction.png`, `v2_14_residuals.png`
  - `v2_14_roc_curves_all.png`, `v2_14_roc_curves_top20.png`, `v2_14_auc_distribution.png`
  - `v2_14_roc_auc_results.csv` (per-class AUC scores)
- Performance: ARI=0.285 (+45.6% vs v2.6.7), Pooled AUC=0.917, 81/139 classes with test data
- Reconstruction: R²=0.863 avg (GRA=0.788, MS=0.788, NGR=0.835, R=0.925, G=0.929, B=0.916)
- Documentation: `SEMISUP_VAE_V2_14_SUMMARY.md`

**v2.14.1 (Adaptive Weighting - Failed Experiment):**
- Model: `ml_models/checkpoints/adaptive_vae_v2_14_1_final.pth` (sample-specific α weights based on class frequency)
- Training: `train_adaptive_vae_v2_14_1.py` (adaptive α ∝ 1/√class_frequency)
- Implementation: `ml_models/adaptive_semisup_vae_model.py`
- Training Log: `adaptive_vae_v2_14_1_training.log`
- Performance: ARI=0.075 (-73.7% vs v2.14, failed)
- Issue: Adaptive weighting degraded performance, removed class hierarchy benefits

**v2.14.2 (Random Masking - Missing Data Regularization SUCCESS):**
- Model: `ml_models/checkpoints/vae_v2_14_2_best.pth` (random 30% feature masking during training)
- Training: `train_vae_v2_14_2.py` (mask_prob=0.3, forces robust feature learning)
- Implementation: Same architecture as v2.14, added `apply_mask()` method
- Training Log: `vae_v2_14_2_training.log`
- Performance:
  - **Clustering: ARI=0.129** (+159% vs v2.14 when v2.14 trained with same settings)
  - Reconstruction: R²=0.403 avg (worse than v2.14, expected tradeoff)
  - Per-feature R²: GRA=-0.014, MS=0.326, NGR=0.057, R=0.700, G=0.699, B=0.660
  - Imputation quality (on masked data): R=0.893, G=0.897, B=0.882
- Evaluation: `evaluate_v2_14_2_reconstruction.py`
- **Key insight**: Random masking acts as powerful regularizer, improving clustering at cost of reconstruction

**v2.15 (Real Missing Data - Failed):**
- Architecture: Same as v2.14.2 but trained on actual incomplete data (real NaN values)
- Performance: ARI=0.075 (-54% vs v2.14.2)
- Issue: Real missing data patterns correlate with lithology, creating confounds

**Masking Hyperparameter Sweep (Complete):**
- Sweep: 51 models (0% to 50% masking in 1% increments), 4 GPUs parallel
- Training: `train_vae_masking_sweep.py` (50 epochs per model)
- Orchestration: `run_masking_sweep.sh`, `monitor_sweep.sh`
- Results: `masking_sweep_results.csv`, `masking_sweep_results.png`
- Plotting: `plot_masking_sweep_results.py` (2×3 grid: R² vs masking % for each feature)
- Visualization: `/home/utig5/johna/bhai/masking_sweep_results.png`
- **Key finding**: Reconstruction-clustering tradeoff
  - 0% masking: Best reconstruction (R²=0.86 avg), worst clustering
  - 30% masking: Optimal clustering (ARI=0.129), moderate reconstruction (R²=0.40 avg)
  - 50% masking: Poor reconstruction (R²=0.25 avg), untested clustering
- **Conclusion**: Masking percentage controls regularization strength; optimal depends on task

### Depth Prediction from VAE Embeddings

CatBoost regression model for predicting stratigraphic depth (CSF-A) from Semi-Supervised VAE v2.14 latent embeddings:

**Model Files:**
- `ml_models/depth_predictor_v2_14.cbm` - CatBoost depth predictor (336 iterations, early stopped)

**Training Scripts:**
- `train_depth_predictor_v2_14.py` - Training and evaluation code (loads v2.14 embeddings, trains CatBoost)

**Training Logs:**
- `depth_predictor_training.log` - Training output and performance metrics

**Results:**
- `depth_predictor_v2_14_results.png` - Scatter plot (predicted vs true) and residual plot
- `depth_predictor_feature_importance.csv` - Latent dimension importance scores

**Performance:**
- Test R² = 0.407 (explains 40.7% of depth variance)
- Test MAE = 198.46 m (mean absolute error)
- Test RMSE = 293.53 m (root mean squared error)
- Borehole-level 80/20 split (236 train / 60 test boreholes)
- Training: 336 iterations, ~4 seconds

**Most Important Latent Dimensions:**
1. z5: 18.69% (highest importance for depth)
2. z3: 16.00%
3. z6: 14.16%
4. z9: 9.82%
5. z4: 9.64%

**Key Insight:** VAE embeddings contain stratigraphic information beyond lithology classification. Moderate R² (0.407) is reasonable given that boreholes sample vastly different geological settings (oceanic crust, continental margins, deep-sea sediments) where depth alone doesn't determine properties. Demonstrates latent space captures vertical structure in addition to lithological composition.

**Critical Findings:**
1. **Extreme β annealing optimal**: Starting from pure autoencoder (β=1e-10) and annealing to β=0.75 achieves best performance (ARI=0.196 ± 0.037, +3% vs β: 0.001→0.5, +18% lower variance). Sweet spot at β_end=0.75 balances regularization without destroying feature correlations.
2. **Cross-validation essential**: Original single-split results (ARI=0.286) were 33.5% inflated due to lucky test set with lower lithology diversity (entropy 2.95 vs 3.12). Entropy-balanced 5-fold CV reveals true performance.
3. **Variance reflects geology**: High performance variance (±0.04-0.05) is inherent to geological heterogeneity across boreholes, not a flaw in methodology
4. **Overparameterization helps (latent)**: 10D latent (→4D effective) outperforms 8D (+7.3% relative improvement)
5. **Architectural simplicity wins**: Shallow [32,16] bottleneck (2K params) outperforms wider [256,128,64,32] (91K params) by +52%. Tight bottleneck forces discriminative feature learning. v2.12 wider architecture failed (-34%).
6. **Feature quality > dataset size**: More data with inferior features performs -54% worse
7. **Cross-modal correlations are primary**: "Dark + dense = basalt" must be learned jointly
8. **Joint training > transfer learning**: All pre-training approaches fail (-50% to -79%)
9. **VampPrior overfits**: v2.10 validation loss explodes +680% despite +1.2% ARI gain
10. **Masking degrades clustering**: v2.11 feature masking reduces ARI -4% to -8%, fails at imputation (R²<0)
11. **Predicted RGB fails**: v2.6.10 using supervised RGB prediction (R²=0.72, +77% boreholes) degrades ARI -53%, demonstrating 28% unexplained variance corrupts cross-modal correlations
12. **Clustering determined by latent space, not decoder**: v2.13 multi-decoder (6 separate decoders, 5.6K params) achieves identical clustering to v2.6.7 single decoder (2K params): ARI=0.187±0.045 vs 0.196±0.037 (statistically equivalent). Demonstrates decoder architecture doesn't affect clustering performance - latent space structure is what matters. Multi-decoder robust across β ∈ {0.5, 0.75, 1.0, 1.5, 2.0} with overlapping confidence intervals.
13. **Dimension collapse is intentional and desirable**: Both v2.6.7 and v2.13 collapse from 10D nominal to ~3-4D effective latent space. This is β-VAE working correctly - only the most discriminative dimensions survive the KL penalty (β=0.75). The 3 active dimensions (z5, z6, z7 for v2.13) capture 100% of variance and represent the most informative geological patterns. More dimensions wouldn't improve clustering - β-VAE automatically performs feature selection.
14. **Semi-supervised learning dramatically improves clustering**: v2.14 adds classification head (10D→[32,ReLU,Dropout]→139 classes) with α=0.1 weight, achieving ARI=0.285 (+45.6% vs v2.6.7 unsupervised). Gentle classification guidance organizes latent space more effectively than pure reconstruction loss. Also provides excellent classification (Pooled AUC=0.917). Sweet spot at α=0.1 - higher values degrade both clustering and classification.
15. **VAE vastly outperforms linear baselines for physical properties**: Baseline Lasso regression (predict each feature from other 5) achieves R² of only 0.18 (GRA), 0.25 (MS), 0.41 (NGR) - demonstrating weak linear correlations. VAE reconstruction improves these by +282% (GRA), +94% (MS), +77% (NGR), proving the VAE learns meaningful non-linear geological relationships. RGB channels show opposite pattern (R²=0.99+ for linear baseline) because color channels are nearly perfectly linearly correlated; VAE's latent bottleneck sacrifices RGB perfection to capture physical property structure.
16. **Feature weighting (v2.13)**: Multi-decoder uses weights [1.0, 2.0, 2.0, 1.0, 1.0, 1.0] for GRA/MS/NGR/R/G/B. The 2× weight on MS and NGR reflects (1) their importance for lithology discrimination (MS distinguishes basalt, NGR distinguishes clay) and (2) their lower baseline reconstruction quality (MS R²=0.44, NGR R²=0.74 vs RGB R²=0.94-0.96). This encourages the model to focus on geologically informative but challenging features.
17. **Latent space captures stratigraphic information**: CatBoost depth predictor trained on v2.14 10D embeddings achieves R²=0.407, demonstrating the latent space encodes vertical stratigraphic structure beyond lithology labels. Most important dimensions for depth (z5, z3, z6) overlap with those important for lithology clustering, suggesting the VAE learns a unified geological representation capturing both composition and stratigraphic position. Moderate R² reflects inherent geological heterogeneity - boreholes sample vastly different settings (oceanic crust, continental margins, deep-sea sediments) where depth alone doesn't determine properties.

**Performance Context**: ARI=0.196±0.037 is **strong performance** for unsupervised lithology clustering from physical properties. In geoscience literature, ARI>0.15 is "good agreement" and ARI>0.20 is "strong clustering". The problem is inherently difficult: 209 lithologies with overlapping physical property distributions, many-to-many mapping (same lithology → different measurements, different lithologies → similar measurements), and subjective ground-truth labels. Random clustering yields ARI≈0.02; v2.6.7/v2.13 achieve ~10× better, demonstrating real geological structure learning.

**For detailed documentation:** See `VAE_MODELS.md` for complete model descriptions, all 10 failed experiments (v2.2, v3, v2.6.1-4, v2.6.8, v2.6.10-11, v2.12), and scientific insights.

### VAE Pipeline Notebooks

Complete end-to-end pipelines demonstrating raw data → trained models → UMAP visualizations:

- **`vae_gra_pipeline.ipynb`** - v1 pipeline (physical properties only: GRA+MS+NGR)
- **`vae_gra_v2_pipeline.ipynb`** - v2 pipeline (multimodal: GRA+MS+NGR+RGB)
- **`vae_gra_v2_1_pipeline.ipynb`** - v2.1 pipeline (distribution-aware scaling)
- **`vae_gra_v2_5_pipeline.ipynb`** - v2.5 pipeline (β optimization)
- **`vae_v2_6_6_analysis.ipynb`** - v2.6.6 analysis (10D latent): Q-Q plots, normality tests, distribution histograms, correlation matrix, UMAP with labeled centroids
- **`vae_v2_6_7_analysis.ipynb`** - **v2.6.7 production model analysis** (10D latent, β: 1e-10→0.75): Complete analysis with Q-Q plots, normality tests, distribution histograms, correlation matrix, UMAP projections with labeled centroids. **USE THIS for current gold standard model.**

### VAE Outputs

- **Model checkpoints:** `ml_models/checkpoints/vae_gra_*.pth`
  - **`vae_gra_v2_6_7_final.pth`** - **Production model** (trained on all 238,506 samples)
- **Visualizations:** `vae_outputs/`, `vae_v2_outputs/`, `vae_v2_1_outputs/`, `vae_v2_5_outputs/`
- **Training logs:** `vae_gra_training.log`, `vae_gra_v2_training.log`, `v2_6_7_final_training.log`, etc.
- **Reports:** `vae_lithology_gra_summary.md`, `vae_gra_v2_summary.md`
- **Cross-validation results:**
  - **`v2_6_7_entropy_balanced_cv.csv`** - **v2.6.7 5-fold entropy-balanced CV** (ARI = 0.196 ± 0.037)
  - **`entropy_balanced_cv_v2_6_7.log`** - v2.6.7 CV training output
  - `v2_6_6_entropy_balanced_cv.csv` - v2.6.6 5-fold entropy-balanced CV (stratified by dominant lithology)
  - `entropy_balanced_cv.log` - v2.6.6 CV training output showing entropy balance per fold
  - `v2_6_6_stratified_cv.csv` - Geographic stratified CV (5 regions)
  - `v2_6_6_cross_validation.csv` - Random 5-fold CV (baseline)
  - `check_original_split.py` - Analysis of why original v2.6.6 test split was easier to cluster

## Lithology Classification from VAE Embeddings

**Investigation Status**: Completed 2025-11-01

### Key Finding

Direct classifiers on **raw 6D features** outperform VAE-based classifiers on **10D embeddings** by **42.3%**:

| Approach | Balanced Acc | Status |
|----------|--------------|--------|
| **Direct Classifier (raw features)** | **42.32%** | ✓ Recommended |
| VAE Classifier v1.1 (hierarchical) | 29.73% | Hierarchical (139→14 groups) |
| VAE Classifier v1.0 (class-balanced) | 7.51% | ✗ Failed (extreme weights) |

**Why**: VAE optimizes for reconstruction (R²=0.904), not class separation. Only 4/10 latent dimensions active (6 collapsed). Physically similar lithologies cluster together.

### Files

**Training scripts**: `train_vae_classifier_v1_0.py`, `train_vae_classifier_v1_1.py`, `train_direct_classifier_baseline.py`
**Analysis**: `analyze_lithology_distribution.py`, `validate_hierarchy_with_embeddings.py`, `validate_vae_reconstruction.py`
**Hierarchy**: `create_lithology_hierarchy.py`, `lithology_hierarchy_mapping.csv`

**For complete investigation details**: See [`VAE_CLASSIFIER_INVESTIGATION_SUMMARY.md`](VAE_CLASSIFIER_INVESTIGATION_SUMMARY.md)

## Utility Scripts

### Data Analysis
- **`count_measurements_per_borehole.py`** - Analyzes measurement coverage per borehole
- **`extract_borehole_coordinates.py`** - Extract lat/lon coordinates for all boreholes to CSV
- **`plot_gra_vs_mad.py`** - GRA vs MAD density comparison plots
- **`plot_gra_vs_mad_scatter.py`** - Scatter plots comparing GRA and MAD
- **`pca_gra_analysis.py`** - PCA analysis on VAE GRA v1 features
- **`plot_training_distributions.py`** - Generate 2×3 distribution plots for all 6 training features
- **`plot_latent_distributions.py`** - VAE latent space distribution analysis (shows 3D effective dimensionality from 10D nominal)
- **`lasso_latent_to_features.py`** - Lasso regression from latent space to features (shows which latent dims predict which features)
- **`lasso_baseline_cross_prediction.py`** - Baseline Lasso cross-prediction (predict each feature from other 5, comparison to VAE)
- **`visualize_lasso_baseline_predictions.py`** - Predicted vs true scatter plots for baseline Lasso model

### Visualization Scripts
- **`visualize_vae_v2_6_7_detailed_architecture.py`** - v2.6.7 detailed architecture diagram with scaling
- **`visualize_vae_v2_13_multidecoder_architecture.py`** - v2.13 multi-decoder architecture diagram with trapezoid shapes
- **`create_network_diagram_horizontal.py`** - Horizontal flow network diagram
- **`visualize_vae_torchviz.py`** - PyTorch computational graph visualization
- **`create_network_diagram.py`** - Vertical network diagram (superseded by horizontal)
- **`plot_vae_reconstructions.py`** - VAE v2.6.7 reconstruction quality 2×3 hexbin plots (true vs predicted for all 6 features)
- **`plot_vae_reconstructions_scatter.py`** - VAE v2.6.7 reconstruction quality 2×3 scatter plots (alternative to hexbin)
- **`visualize_v2_13_reconstruction.py`** - VAE v2.13 reconstruction quality 2×3 scatter plots (black points, no grid)

### VAE Dataset Creation
- **`create_vae_dataset.py`** - VAE GRA v1 dataset (403K samples, 3D features)
- **`create_vae_gra_v2_dataset.py`** - VAE GRA v2+ dataset (239K samples, 6D features)

### VAE Experiments
- **`test_latent_dimensionality.py`** - Systematic test of latent_dim ∈ {2,4,6,8,10,12}, discovered 10D optimal for GMM clustering
- **`test_beta_end_grid_search.py`** - Fine grid search confirming β_end=0.75 is optimal
- **`test_encoder_depth.py`** - Architecture exploration (v2.12 preliminary test, [32,16] vs [256,128,64,32])
- **`test_vae_v2_6_7_notebook.py`** - Validation script for vae_v2_6_7_analysis.ipynb (tests PyTorch loading)
- **`test_vae_v2_6_6_notebook.py`** - Validation script for vae_v2_6_6_analysis.ipynb
- **`entropy_balanced_cv_v2_6_7.py`** - **v2.6.7 cross-validation** (5-fold entropy-balanced, β: 1e-10→0.75)
- **`train_v2_6_7_final.py`** - **v2.6.7 production model training** (all 238,506 samples)
- **`entropy_balanced_cv_v2_13.py`** - **v2.13 cross-validation** (5-fold entropy-balanced, multi-decoder)
- **`train_v2_13_final.py`** - **v2.13 production model training** (multi-decoder, all samples)
- **`beta_grid_search_v2_13.py`** - **v2.13 β optimization** (tested β ∈ {0.5, 0.75, 1.0, 1.5, 2.0})
- **`train_semisupervised_vae.py`** - **v2.14 semi-supervised training** (α grid search, classification head)
- **`train_depth_predictor_v2_14.py`** - **Depth predictor from v2.14 embeddings** (CatBoost regression, R²=0.407)
- **`train_vae_v2_6_6.py`** - Training script for v2.6.6 (10D latent, β annealing)
- **`train_vae_v2_12_proper_batching.py`** - v2.12 wider architecture training (β=0.1, 100ep, failed -38%)
- **`train_vae_v2_12_beta075_200epochs.py`** - v2.12 full training (β=0.75, 200ep, failed -34%)
- **`train_vae_v2_12_gentle_beta.py`** - v2.12 β grid search (0.1, 0.2, 0.3, 0.4, 0.5)
- **`latent_dimensionality_comparison.csv`** - Results showing 10D→4D effective outperforms 8D by +7.3%
- **`beta_end_grid_search.csv`** - Grid search results confirming β_end=0.75 optimal (v2.6.7)
- **`v2_13_entropy_balanced_cv.csv`** - v2.13 5-fold CV results (ARI = 0.187 ± 0.045)
- **`beta_grid_search_v2_13.csv`** - v2.13 β optimization results (all β values overlap)
- **`encoder_depth_test_results.csv`** - Architecture comparison results
- **`v2_12_gentle_beta_search.csv`** - v2.12 β optimization results
- **`v2_12_proper_batching.log`** - v2.12 training log (β=0.1)
- **`v2_12_beta075_200epochs.log`** - v2.12 training log (β=0.75, 200ep)
- **`v2_13_final_training.log`** - v2.13 production model training log (100 epochs, 1471s)
- **`entropy_balanced_cv_v2_13.log`** - v2.13 5-fold CV training output
- **`beta_grid_search_v2_13.log`** - v2.13 β grid search output

## Reports and Outputs

### Analysis Reports
- **`report.md`** - Comprehensive borehole measurement coverage report (437 boreholes, 212 with complete measurements)
- **`borehole_coordinates.csv`** - Lat/lon coordinates for all 534 boreholes (Borehole_ID, Latitude, Longitude, Water_Depth_mbsl)

### Visualization Outputs

**Paper Figures:**
- `paper_plots/figure_9.png`, `paper_plots/figure_11.png` - Reproduced paper figures

**Network Architecture Diagrams:**
- **`network_structure.md`** - ASCII art network diagram with complete architecture details
- `vae_network_diagram_horizontal.png` - Horizontal flow network diagram (left→right)
- `vae_v2_6_7_detailed_architecture_diagram.png` - v2.6.7 detailed architecture with scaling transformations
- **`vae_v2_13_multidecoder_architecture_diagram.png`** - v2.13 multi-decoder architecture with trapezoid shapes
- `vae_torchviz_diagram.dot` - PyTorch computational graph (DOT format)
- `training_data_distributions.png` - 2×3 distribution plots for all 6 input features
- `training_data_distributions_logscale.png` - Same with log scale for MS/NGR

**VAE Reconstruction Quality:**

*v2.6.7 (Single Decoder):*
- **`vae_reconstruction_quality.png`** - 2×3 hexbin plots showing true vs predicted values for all 6 features (GRA, MS, NGR, RGB)
- **`vae_reconstruction_quality_scatter.png`** - 2×3 scatter plots (alternative visualization)
- Reconstruction R² scores: GRA=0.83, MS=0.44, NGR=0.74, R=0.96, G=0.96, B=0.94

*v2.13 (Multi-Decoder):*
- **`v2_13_reconstruction_scatter.png`** - 2×3 scatter plots with black points, no grid
- Reconstruction R² scores: GRA=0.69, MS=0.48, NGR=0.72, R=0.89, G=0.89, B=0.87

**Latent Space Analysis:**
- **`v2_13_latent_distributions.png`** - 2×5 histogram grid showing distribution of all 10 latent dimensions
- Analysis reveals 3D effective dimensionality: z5 (33.2%), z6 (39.2%), z7 (27.6%) capture 100% of variance
- Dimensions z0, z1, z2, z3, z4, z8, z9 collapsed (Var ≈ 0.0001 each)

**Baseline Comparison (Lasso Regression):**
- **`lasso_baseline_prediction_scatter.png`** - 2×3 predicted vs true for baseline Lasso (each feature from other 5)
- **`vae_vs_baseline_comparison.png`** - Bar chart comparing VAE vs baseline R² scores
- **`lasso_baseline_coefficient_heatmap.png`** - Coefficient matrix showing feature cross-correlations
- **`lasso_baseline_performance.png`** - Performance metrics and sparsity analysis
- **`lasso_coefficient_heatmap.png`** - Lasso coefficients mapping latent dims to features
- **`lasso_performance.png`** - Latent→feature prediction performance
- Baseline results: Physical properties (GRA R²=0.18, MS R²=0.25, NGR R²=0.41) vs RGB (R²=0.99+)
- VAE improvements: +282% (GRA), +94% (MS), +77% (NGR) - demonstrates non-linear learning

**VAE Outputs:**
- `vae_outputs/`, `vae_v2_outputs/`, `vae_v2_1_outputs/`, `vae_v2_5_outputs/` - Model visualizations
- `vae_lithology_gra_summary.md`, `vae_gra_v2_summary.md` - Comprehensive reports

*v2.6.7 Production Model:*
- **`v2_6_7_entropy_balanced_cv.csv`** - **v2.6.7 cross-validation results (ARI = 0.196 ± 0.037)**
- **`v2_6_7_final_training.log`** - **v2.6.7 production model training** (100 epochs, 580s, all data)
- **`entropy_balanced_cv_v2_6_7.log`** - v2.6.7 5-fold CV output
- **`beta_end_grid_search.log`** - Grid search log confirming β_end=0.75 optimal

*v2.13 Multi-Decoder Model:*
- **`v2_13_entropy_balanced_cv.csv`** - v2.13 cross-validation results (ARI = 0.187 ± 0.045)
- **`v2_13_final_training.log`** - v2.13 production model training (100 epochs, 1471s, 6 decoders)
- **`entropy_balanced_cv_v2_13.log`** - v2.13 5-fold CV output
- **`beta_grid_search_v2_13.csv`** - v2.13 β optimization results (all β ∈ {0.5-2.0} overlap)
- **`beta_grid_search_v2_13.log`** - v2.13 β grid search output
- **`V2_13_RECONSTRUCTION_INVESTIGATION.md`** - Documents early experiments vs final model, corrects false "+91%" claim

*Other Models:*
- `vae_v2_6_6_clustering_results.csv` - v2.6.6 performance (GMM k=18: ARI=0.286, lucky split)
- `vae_v2_6_6_training.log` - v2.6.6 training output (16 epochs, 108.5s)
- `latent_dim_test.log` - Latent dimensionality experiment results
- `v2_12_clustering_results.csv` - v2.12 performance (failed architecture test)

## Reference Papers

### Primary Reference
- **`papers/lilypaper.pdf`** - Childress et al., 2024 (LILY Database main paper)

### Supporting Literature
- **`papers/Geochem Geophys Geosyst - 2016 - Olson`** - Sediment thickness and crustal age relationships

### IODP Expedition Proceedings
- Collection of expedition reports (301, 309/312, 327, 335, 345, 360, 390/393, 395, 402)
- `papers/oceanic_crust_iodp/` - Specialized papers for oceanic crust expeditions

## Key Scientific Insights

**Core innovations**: 20cm depth binning (403K samples), distribution-aware scaling (+40% ARI), β annealing (+7% ARI), cross-modal joint training

**Key lessons**: Feature quality > dataset size, joint training > transfer learning, disentanglement harms clustering, multimodal synergy is non-compositional

**For detailed scientific insights**: See [`SCIENTIFIC_INSIGHTS.md`](SCIENTIFIC_INSIGHTS.md)

## Working with the Project

### Key Workflows

#### 1. Exploratory Analysis
- Load datasets from `/datasets/` directory
- Use efficient loading for large files (GRA, RGB)
- Follow analysis patterns from existing notebooks

#### 2. Machine Learning Development
- Use **ml-model-builder agent** for designing/implementing/optimizing models
- Key considerations: Borehole-level splits, feature engineering, CatBoost for tabular data
- PyTorch VAEs for learning latent representations

#### 3. Figure Reproduction
Follow workflow in "Primary Task" section for recreating paper figures

### Best Practices
- **Data integrity**: Always verify data from actual datasets, never fabricate
- **Performance**: Use chunked reading for files >1GB
- **Reproducibility**: Save models, training logs, feature importance
- **Validation**: Use borehole-level splits to avoid data leakage
- dont tell me how brilliant and smart i am, none of this sycophantic behavior
- whenever you make a plot always look at the plot and think to yourself, does this plot match the request? should i change it first?
- you really need to start talking like you are data from star trek