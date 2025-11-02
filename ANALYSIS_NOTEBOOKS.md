# Analysis Notebooks

The repository contains Jupyter notebooks for exploratory data analysis and specific measurement type investigations.

## Measurement-Specific Analysis

### General Database Exploration
- **`lily_analysis.ipynb`** - General LILY database exploration and analysis

### Physical Properties
- **`gra_analysis.ipynb`** - Gamma ray attenuation (GRA) bulk density analysis
- **`LILY-AVS.ipynb`** - Automated Vane Shear (AVS) strength measurements
- **`PWC analysis`** - P-wave velocity measurements (if exists)
- **`THCN analysis`** - Thermal conductivity measurements (if exists)

### Optical/Color Analysis
- **`LILY-RSC.ipynb`** - Reflectance Spectroscopy (RSC) data exploration
- **`RGB analysis`** - Digital image color data (if exists)

### Geochemistry
- **`explore_CaCO3.ipynb`** - Carbonate content (CARB) analysis
- **`ICP analysis`** - Inductively coupled plasma measurements (if exists)
- **`IW analysis`** - Interstitial water chemistry (if exists)

### Age and Chronology
- **`explore_accum.ipynb`** - Accumulation rate investigations
- **`borehole-age.ipynb`** - Borehole age and chronology analysis

### Data Format Exploration
- **`explore_netcdf.ipynb`** - NetCDF format exploration (for gridded/multi-dimensional data)

## VAE Model Analysis Pipelines

### Legacy VAE Models (v1, v2)

**`vae_gra_pipeline.ipynb`** - VAE GRA v1 complete pipeline
- **Dataset**: 403K samples from 524 boreholes
- **Features**: 3D (GRA + MS + NGR)
- **Latent**: 2D and 8D latent space models
- **Analysis**: K-Means clustering, reconstruction quality, UMAP visualizations

**`vae_gra_v2_pipeline.ipynb`** - VAE GRA v2 complete pipeline
- **Dataset**: 239K samples from 296 boreholes
- **Features**: 6D (GRA + MS + NGR + RGB)
- **Analysis**: RGB color analysis by lithology, UMAP projections, clustering, v1 vs v2 comparison

**`vae_gra_v2_1_pipeline.ipynb`** - VAE GRA v2.1 complete pipeline
- **Innovation**: Distribution-aware scaling
- **Analysis**: Before/after scaling visualizations, distribution normality tests
- **Comparison**: v2.0 vs v2.1 performance (+40% ARI improvement)

**`vae_gra_v2_5_pipeline.ipynb`** - VAE GRA v2.5 complete pipeline
- **Innovation**: β parameter optimization (β=0.5)
- **Analysis**: v2.1 (β=1.0) vs v2.5 (β=0.5) comparison, feature-latent correlations
- **Performance**: +32% ARI improvement from β tuning

### Production VAE Models (v2.6.6, v2.6.7)

**`vae_v2_6_6_analysis.ipynb`** - VAE v2.6.6 analysis
- **Model**: 10D latent, β annealing (1e-10 → 0.75)
- **Dataset**: Test set split (70%/15%/15%)
- **Analysis**:
  - Q-Q plots and normality tests for all 10 latent dimensions
  - Distribution histograms
  - Latent-feature correlation matrix
  - UMAP projection with labeled lithology centroids
  - GMM clustering performance (k=18: ARI=0.286)
- **Note**: Test split was easier to cluster than expected (see `check_original_split.py`)

**`vae_v2_6_7_analysis.ipynb`** - **VAE v2.6.7 production model analysis** ⭐
- **Model**: 10D latent, β annealing (1e-10 → 0.75)
- **Dataset**: All 238,506 samples (no split)
- **Analysis**:
  - Q-Q plots and normality tests for all 10 latent dimensions
  - Distribution histograms
  - Latent-feature correlation matrix
  - UMAP projection with labeled lithology centroids
  - GMM clustering performance across k values
- **Cross-validation**: 5-fold entropy-balanced CV (ARI = 0.196 ± 0.037)
- **Status**: **Current gold standard model - USE THIS**

## Notebook Usage Notes

### Running on Smokey vs Cotopaxi

**UMAP Visualization Issue on Smokey:**
- UMAP import fails due to numba/numpy version conflicts
- Notebooks automatically fall back to PCA if UMAP unavailable
- For full UMAP visualizations, run on **cotopaxi**

**Recommendations:**
- **Development/editing**: Either system
- **Full analysis with UMAP**: Run on **cotopaxi**
- **Quick tests**: Run on **smokey** (PCA fallback sufficient)

### Notebook Structure

Most VAE pipeline notebooks follow this structure:
1. **Data loading**: Load training dataset (e.g., `vae_training_data_v2_20cm.csv`)
2. **Model loading**: Load trained VAE checkpoint
3. **Latent space extraction**: Encode all samples to latent space
4. **Distribution analysis**: Q-Q plots, histograms, normality tests
5. **Correlation analysis**: Feature-latent correlation matrices
6. **Dimensionality reduction**: UMAP/PCA projections
7. **Clustering**: GMM/K-Means clustering on latent space
8. **Visualization**: Lithology-colored UMAP plots with centroids
9. **Performance metrics**: ARI, silhouette scores across k values

## Related Documentation

- **VAE model details**: See `VAE_MODELS.md` (or main CLAUDE.md Machine Learning Models section)
- **Dataset creation**: Scripts like `create_vae_gra_v2_dataset.py`
- **Model training**: Scripts like `train_v2_6_7_final.py`
- **Development setup**: See `DEVELOPMENT_SETUP.md` for GPU/UMAP configuration
