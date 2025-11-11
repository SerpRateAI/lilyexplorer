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

**Recommendation: Use VAE GRA v2.6.7** for all lithology clustering (entropy-balanced CV: ARI = 0.196 ± 0.037, best performance).

**Quick Reference:**

| Model | Features | Samples | GMM ARI (5-fold CV) | Status |
|-------|----------|---------|---------------------|--------|
| **VAE GRA v2.6.7** | **10D latent** (GRA+MS+NGR+RGB, β: 1e-10→0.75) | 239K | **0.196 ± 0.037** | **✓ USE THIS** |
| VAE GRA v2.6.10 | 10D latent (60% real + 40% predicted RGB) | 396K | 0.093 | ❌ Failed (-53%) |
| VAE GRA v2.6.8 | 10D latent (fuzzy ±20cm depth matching) | 251K | 0.087 | ❌ Failed (-55%) |
| VAE GRA v2.6.6 | 10D latent (GRA+MS+NGR+RGB, β: 0.001→0.5) | 239K | 0.19 ± 0.05 | Superseded by v2.6.7 |
| VAE GRA v2.6 | 8D latent (GRA+MS+NGR+RGB, β anneal) | 239K | ~0.19 (est.) | Superseded by v2.6.7 |
| VAE GRA v2.5 | 6D (fixed β=0.5) | 239K | ~0.18 (est.) | Fixed β baseline |
| VAE GRA v2.1 | 6D (dist-aware, β=1.0) | 239K | ~0.13 (est.) | Distribution baseline |
| VAE GRA v2 | 6D (standard scaling) | 239K | ~0.10 (est.) | Multimodal baseline |
| VAE GRA v1 | 3D (physical only) | 403K | ~0.06 (est.) | Max coverage |
| Failed experiments (9 variants) | Various | — | 0.04-0.19 (est.) | ❌ Don't use |

**Key Model Files:**
- **Production Model**: `ml_models/checkpoints/vae_gra_v2_6_7_final.pth` (trained on all 238,506 samples)
- Training CV: `entropy_balanced_cv_v2_6_7.py` (5-fold CV with β: 1e-10→0.75)
- Final Training: `train_v2_6_7_final.py` (production model on 100% data)
- CV Results: `v2_6_7_entropy_balanced_cv.csv`
- Training Log: `v2_6_7_final_training.log`
- Dataset: `vae_training_data_v2_20cm.csv` (239K samples, 24MB)
- Implementation: `ml_models/vae_lithology_gra_v2_5_model.py` (same architecture as v2.6.6, different β schedule)
- Analysis Notebook: `vae_v2_6_7_analysis.ipynb` (Q-Q plots, UMAP, statistics)

**Critical Findings:**
1. **Extreme β annealing optimal**: Starting from pure autoencoder (β=1e-10) and annealing to β=0.75 achieves best performance (ARI=0.196 ± 0.037, +3% vs β: 0.001→0.5, +18% lower variance). Sweet spot at β_end=0.75 balances regularization without destroying feature correlations.
2. **Cross-validation essential**: Original single-split results (ARI=0.286) were 33.5% inflated due to lucky test set with lower lithology diversity (entropy 2.95 vs 3.12). Entropy-balanced 5-fold CV reveals true performance.
3. **Variance reflects geology**: High performance variance (±0.04-0.05) is inherent to geological heterogeneity across boreholes, not a flaw in methodology
4. **Overparameterization helps**: 10D latent (→4D effective) outperforms 8D (+7.3% relative improvement)
5. **Feature quality > dataset size**: More data with inferior features performs -54% worse
6. **Cross-modal correlations are primary**: "Dark + dense = basalt" must be learned jointly
7. **Joint training > transfer learning**: All pre-training approaches fail (-50% to -79%)
8. **VampPrior overfits**: v2.10 validation loss explodes +680% despite +1.2% ARI gain
9. **Masking degrades clustering**: v2.11 feature masking reduces ARI -4% to -8%, fails at imputation (R²<0)
10. **Predicted RGB fails**: v2.6.10 using supervised RGB prediction (R²=0.72, +77% boreholes) degrades ARI -53%, demonstrating 28% unexplained variance corrupts cross-modal correlations

**For detailed documentation:** See `VAE_MODELS.md` for complete model descriptions, all 9 failed experiments (v2.2, v3, v2.6.1-4, v2.6.8, v2.6.10-11), and scientific insights.

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
- **`plot_gra_vs_mad.py`** - GRA vs MAD density comparison plots
- **`plot_gra_vs_mad_scatter.py`** - Scatter plots comparing GRA and MAD
- **`pca_gra_analysis.py`** - PCA analysis on VAE GRA v1 features

### VAE Dataset Creation
- **`create_vae_dataset.py`** - VAE GRA v1 dataset (403K samples, 3D features)
- **`create_vae_gra_v2_dataset.py`** - VAE GRA v2+ dataset (239K samples, 6D features)

### VAE Experiments
- **`test_latent_dimensionality.py`** - Systematic test of latent_dim ∈ {2,4,6,8,10,12}, discovered 10D optimal for GMM clustering
- **`test_beta_end_grid_search.py`** - Fine grid search confirming β_end=0.75 is optimal
- **`test_vae_v2_6_7_notebook.py`** - Validation script for vae_v2_6_7_analysis.ipynb (tests PyTorch loading)
- **`test_vae_v2_6_6_notebook.py`** - Validation script for vae_v2_6_6_analysis.ipynb
- **`entropy_balanced_cv_v2_6_7.py`** - **v2.6.7 cross-validation** (5-fold entropy-balanced, β: 1e-10→0.75)
- **`train_v2_6_7_final.py`** - **v2.6.7 production model training** (all 238,506 samples)
- **`train_vae_v2_6_6.py`** - Training script for v2.6.6 (10D latent, β annealing)
- **`latent_dimensionality_comparison.csv`** - Results showing 10D→4D effective outperforms 8D by +7.3%
- **`beta_end_grid_search.csv`** - Grid search results confirming β_end=0.75 optimal

## Reports and Outputs

### Analysis Reports
- **`report.md`** - Comprehensive borehole measurement coverage report (437 boreholes, 212 with complete measurements)

### Visualization Outputs

**Paper Figures:**
- `paper_plots/figure_9.png`, `paper_plots/figure_11.png` - Reproduced paper figures

**VAE Outputs:**
- `vae_outputs/`, `vae_v2_outputs/`, `vae_v2_1_outputs/`, `vae_v2_5_outputs/` - Model visualizations
- `vae_lithology_gra_summary.md`, `vae_gra_v2_summary.md` - Comprehensive reports
- **`v2_6_7_entropy_balanced_cv.csv`** - **v2.6.7 cross-validation results (ARI = 0.196 ± 0.037)**
- **`v2_6_7_final_training.log`** - **v2.6.7 production model training** (100 epochs, 580s, all data)
- **`entropy_balanced_cv_v2_6_7.log`** - v2.6.7 5-fold CV output
- **`beta_end_grid_search.log`** - Grid search log confirming β_end=0.75 optimal
- `vae_v2_6_6_clustering_results.csv` - v2.6.6 performance (GMM k=18: ARI=0.286, lucky split)
- `vae_v2_6_6_training.log` - v2.6.6 training output (16 epochs, 108.5s)
- `latent_dim_test.log` - Latent dimensionality experiment results

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
