# VAE for Lithology Clustering: Scientific Methodology

**Target Journal:** JGR: Machine Learning and Computation
**Date:** October 2025

---

## Executive Summary

This document outlines the rigorous scientific methodology needed to prepare VAE lithology clustering results for peer-reviewed publication. Moving beyond "production deployment" thinking, we focus on:

1. **Statistical significance** of model improvements
2. **Reproducibility** through multiple random seeds
3. **Proper experimental design** to avoid data leakage
4. **Ablation studies** to isolate contribution of each component
5. **Comparison to baselines** and existing literature
6. **Scientific interpretation** of why methods work/fail

---

## Current Status: What We Have

### Models Developed
| Model | Features | ARI (k=12) | Key Innovation |
|-------|----------|------------|----------------|
| v2.1 | 6D, β=1.0 | 0.167 | Distribution-aware scaling baseline |
| v2.5 | 6D, β=0.5 | 0.241 | Fixed β optimization |
| v2.6 | 6D, β anneal | 0.258 | Curriculum learning (β annealing) |
| v2.10 | 6D, VampPrior | 0.261 | Flexible prior (K=50 components) |

### Dataset
- **238,506 samples** from **296 boreholes**
- **6 features**: GRA, MS, NGR, R, G, B
- **139 unique lithologies**
- **20cm depth binning** for co-location
- Borehole-level train/val/test split (70/15/15)

### Key Findings (Preliminary)
1. **Distribution-aware scaling**: +40% ARI vs standard scaling
2. **β optimization**: β=0.5 > β=1.0 for clustering (+44% ARI)
3. **β annealing**: +7% ARI vs fixed β=0.5
4. **VampPrior**: +1.2% ARI but +680% validation loss (overfitting)

---

## What's Missing for Publication

### 1. Statistical Significance Testing ⚠️

**Current problem:** Single runs with random seed=42
- No confidence intervals
- Can't determine if improvements are significant or noise
- Reviewers will reject without statistical validation

**What we need:**
```python
# Run each model 10 times with different seeds
seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

results = []
for seed in seeds:
    # Train model with this seed
    model, history = train_vae(..., random_state=seed)

    # Evaluate clustering
    ari = evaluate_clustering(model, X_test, y_test)
    results.append(ari)

# Report: mean ± std
print(f"ARI = {np.mean(results):.3f} ± {np.std(results):.3f}")

# Statistical test (e.g., paired t-test)
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(v2_5_results, v2_6_results)
print(f"v2.6 vs v2.5: p={p_value:.4f}")
```

**Expected outcome:**
- v2.1: ARI = 0.167 ± 0.012 (example)
- v2.5: ARI = 0.241 ± 0.008
- v2.6: ARI = 0.258 ± 0.010
- v2.10: ARI = 0.261 ± 0.015

**Statistical tests:**
- Paired t-test: v2.6 vs v2.5 (is β annealing significantly better?)
- Paired t-test: v2.10 vs v2.6 (is VampPrior significantly better?)
- Bonferroni correction for multiple comparisons

---

### 2. Proper Hyperparameter Selection ⚠️

**Current problem:** Some experiments used test set for hyperparameter tuning
- v2.5 initially selected β by testing on test set (wrong!)
- This biases results optimistically
- Reviewers will reject methodology

**What we did right (later):**
```
Train set (70%)  → Train models
Val set (15%)    → Select hyperparameters (β, K, etc.)
Test set (15%)   → Report final results (ONCE ONLY)
```

**What we need to re-verify:**
1. All β selection done on validation set ✓
2. VampPrior K selection (K=50) - was this validated?
3. Annealing schedule selection - was this validated?
4. Number of clusters k - this is okay to sweep on test since it's not a training hyperparameter

**Action items:**
- Document that β=0.5 was selected using validation set
- If VampPrior K=50 was arbitrary, need validation sweep over K=[20, 50, 100, 200]
- If annealing schedule was arbitrary, need validation comparison of schedules

---

### 3. Ablation Studies 📊

**Purpose:** Isolate contribution of each component

**Ablation 1: Feature Scaling**
- Model A: StandardScaler only (no distribution-aware)
- Model B: Distribution-aware scaling (v2.1)
- **Result:** Quantify impact of signed log transforms

**Ablation 2: β Parameter**
- β=0.1, 0.5, 1.0, 2.0
- **Result:** Plot ARI vs β, find optimal

**Ablation 3: Annealing Schedule**
- Fixed β=0.5
- Anneal 0.01→0.5 (50 epochs)
- Anneal 0.001→0.5 (50 epochs)
- Anneal 0.001→0.5 (25 epochs)
- **Result:** Quantify benefit of annealing vs schedule choice

**Ablation 4: Input Features**
- GRA + MS + NGR only (no RGB)
- RGB only
- GRA + MS + NGR + RGB (full)
- **Result:** Multimodal feature value

**Ablation 5: Latent Dimensionality**
- 2D, 4D, 8D, 16D latent space
- **Result:** Optimal compression level

**Ablation 6: VampPrior Components**
- K=10, 20, 50, 100, 200
- **Result:** Optimal mixture complexity

---

### 4. Baseline Comparisons 📈

**Current baselines:**
- VAE v1 (no RGB): ARI = 0.084
- VAE v2 (standard scaling): ARI = 0.128
- VAE v2.1 (β=1.0): ARI = 0.167

**Additional baselines needed:**

**PCA + K-Means**
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=8)
z_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=12)
labels_pca = kmeans.fit_predict(z_pca)
ari_pca = adjusted_rand_score(y_test, labels_pca)
```

**Raw K-Means (no dimensionality reduction)**
```python
kmeans_raw = KMeans(n_clusters=12)
labels_raw = kmeans_raw.fit_predict(X_scaled)
ari_raw = adjusted_rand_score(y_test, labels_raw)
```

**UMAP + K-Means**
```python
import umap
reducer = umap.UMAP(n_components=8, random_state=42)
z_umap = reducer.fit_transform(X_scaled)
kmeans_umap = KMeans(n_clusters=12)
labels_umap = kmeans_umap.fit_predict(z_umap)
ari_umap = adjusted_rand_score(y_test, labels_umap)
```

**t-SNE + K-Means**
```python
from sklearn.manifold import TSNE
z_tsne = TSNE(n_components=8, random_state=42).fit_transform(X_scaled)
kmeans_tsne = KMeans(n_clusters=12)
labels_tsne = kmeans_tsne.fit_predict(z_tsne)
ari_tsne = adjusted_rand_score(y_test, labels_tsne)
```

**Autoencoder (no probabilistic)**
- Standard autoencoder with same architecture
- Deterministic latent code (no reparameterization)
- Compares to VAE probabilistic formulation

**Expected comparison table:**
| Method | ARI (k=12) | Silhouette | Inference Time |
|--------|------------|------------|----------------|
| Raw K-Means | 0.05 | 0.30 | 0.1s |
| PCA + K-Means | 0.12 | 0.35 | 0.5s |
| UMAP + K-Means | 0.18 | 0.40 | 10s |
| t-SNE + K-Means | 0.15 | 0.38 | 30s |
| Autoencoder + K-Means | 0.22 | 0.42 | 2s |
| VAE v2.6 + K-Means | **0.258** | **0.420** | 2s |
| VampPrior v2.10 | 0.261 | 0.415 | 3s |

---

### 5. Cluster Quality Analysis 🔬

**Beyond ARI:** Cluster purity, geological interpretability

**High-purity cluster analysis:**
```python
# For each cluster, find dominant lithology
for cluster_id in range(12):
    cluster_mask = (labels == cluster_id)
    cluster_lithologies = y_test[cluster_mask]

    # Count lithologies
    counts = pd.Series(cluster_lithologies).value_counts()
    dominant_litho = counts.index[0]
    purity = counts.iloc[0] / len(cluster_lithologies)

    print(f"Cluster {cluster_id}: {purity:.1%} {dominant_litho} (n={len(cluster_lithologies)})")
```

**Confusion matrix:**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Map clusters to lithologies (Hungarian algorithm)
from scipy.optimize import linear_sum_assignment
cm = confusion_matrix(y_test, labels)
row_ind, col_ind = linear_sum_assignment(-cm)

# Plot confusion matrix
sns.heatmap(cm[row_ind][:, col_ind], cmap='Blues')
```

**Geological interpretation:**
- Which lithologies cluster well? (gabbro, nannofossil ooze)
- Which lithologies are mixed? (clays, muds)
- Why? Physical property overlaps

---

### 6. Validation Loss vs ARI Discrepancy 🔍

**Critical finding:** v2.10 VampPrior has higher ARI but 680% higher validation loss

**Scientific questions:**
1. Why does validation loss increase while ARI improves?
2. Is this overfitting or a mismatch between loss and clustering objective?
3. Should we use a different metric during training?

**Hypotheses to test:**

**H1: Validation loss measures reconstruction, not clustering**
- VAE optimizes: `L = reconstruction + β·KL`
- Clustering needs: separable latent representations
- **Test:** Plot validation loss vs ARI across epochs
- **Prediction:** Decorrelation after some point

**H2: VampPrior overfits to training lithology distribution**
- K=50 components may memorize training clusters
- Generalization to test set degrades
- **Test:** Compare train vs test ARI for VampPrior vs standard
- **Prediction:** Larger train-test gap for VampPrior

**H3: Posterior collapse in different dimensions**
- VampPrior may use more latent dimensions effectively
- Standard VAE has 4 collapsed dimensions (std~0.01)
- **Test:** Measure effective dimensionality (participation ratio)
- **Prediction:** VampPrior has higher effective dimensionality

**Analysis to include in paper:**
```python
# Plot loss vs ARI during training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Validation loss over epochs
ax1.plot(epochs, val_loss_v26, label='v2.6')
ax1.plot(epochs, val_loss_v210, label='v2.10 VampPrior')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Loss')
ax1.legend()

# ARI over epochs
ax2.plot(epochs, ari_v26, label='v2.6')
ax2.plot(epochs, ari_v210, label='v2.10 VampPrior')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('ARI (k=12)')
ax2.legend()

plt.suptitle('Validation Loss vs Clustering Performance')
```

**Discussion points for paper:**
- Reconstruction loss ≠ clustering quality
- Task-specific metrics needed during training
- VampPrior provides flexible prior but at cost of generalization
- For clustering applications, ARI early stopping > loss early stopping

---

### 7. Reproducibility Requirements ✅

**Code release:**
- Clean, documented code for all models
- Training scripts with all hyperparameters
- Data preprocessing pipeline
- Evaluation scripts
- Environment specification (pyproject.toml)

**Data release:**
- Preprocessed `vae_training_data_v2_20cm.csv` (24MB)
- Train/val/test split specification (borehole IDs)
- Feature normalization parameters (scaler checkpoints)

**Model checkpoints:**
- Best models for v2.1, v2.5, v2.6, v2.10
- Include full training history
- Random seeds documented

**Documentation:**
- Architecture diagrams
- Loss function equations
- Hyperparameter choices with justification
- Computational requirements (GPU/CPU, time)

---

### 8. Computational Efficiency Analysis ⚙️

**Training time:**
- v2.6: 16 epochs, 165 seconds
- v2.10: 10 epochs, 88 seconds (early stop)
- Compare to baselines (PCA instant, UMAP slow)

**Inference time:**
```python
import time

# Encode 10,000 samples
start = time.time()
with torch.no_grad():
    z = model.encode(X_test_tensor[:10000])
encode_time = time.time() - start

print(f"Encoding: {encode_time:.3f}s for 10K samples ({10000/encode_time:.0f} samples/s)")
```

**Scalability:**
- How does training time scale with dataset size?
- Batch size impact on performance
- GPU vs CPU training time

---

### 9. Comparison to Literature 📚

**Relevant papers to cite and compare:**

1. **Geological ML baselines:**
   - Decision trees for lithology classification
   - Random forests for facies prediction
   - SVMs for well log interpretation

2. **VAE applications in geoscience:**
   - VAE for seismic data (if exists)
   - Autoencoders for well log analysis
   - Deep learning for lithology prediction

3. **Clustering methods:**
   - K-means benchmarks
   - Hierarchical clustering
   - DBSCAN for spatial clustering

4. **Representation learning:**
   - VAE vs β-VAE vs VampPrior comparisons in other domains
   - UMAP/t-SNE for visualization

**Gap in literature:**
- First application of VAEs to IODP lithology clustering?
- First to combine physical + visual features?
- First to optimize β for geological clustering?

---

### 10. Limitations and Future Work 🔮

**Limitations to acknowledge:**

1. **Imbalanced lithologies:**
   - 139 lithologies, highly imbalanced (nannofossil ooze >> rare lithologies)
   - ARI may be dominated by common lithologies
   - Could weight rare lithologies higher

2. **Borehole coverage:**
   - Only 5.8% of LILY data has all 6 features co-located
   - NGR is bottleneck (822K vs 4.1M GRA measurements)
   - RGB from separate instrument (SHIL) reduces overlap

3. **Depth binning:**
   - 20cm bins smooth local variations
   - May lose information at sharp lithological boundaries
   - Trade-off between co-location and resolution

4. **Supervised evaluation:**
   - Using human-labeled lithologies as ground truth
   - Human labels may be inconsistent or simplified
   - Clustering may discover valid sub-types

5. **VampPrior overfitting:**
   - K=50 may be too flexible for test set
   - +680% validation loss indicates poor generalization
   - Marginal ARI gain not statistically significant?

**Future work:**

1. **Semi-supervised learning:**
   - Use small labeled set to guide clustering
   - Constrained clustering with known lithology pairs

2. **Hierarchical clustering:**
   - Multi-scale lithology organization
   - Coarse (sediment vs igneous) → fine (clay types)

3. **Temporal/spatial context:**
   - Sequence models (LSTM) for stratigraphic transitions
   - Graph neural networks for borehole networks

4. **Generative applications:**
   - Synthetic lithology generation (if VampPrior is improved)
   - Missing data imputation (needs different training)

5. **Transfer learning:**
   - Pre-train on LILY, fine-tune on other drilling programs
   - Cross-expedition generalization

6. **Explainability:**
   - Feature importance for each cluster
   - Latent dimension interpretation
   - SHAP values for individual predictions

---

## Recommended Publication Structure

### Abstract
- 200 words
- Problem: Lithology clustering from multi-modal measurements
- Method: VAE with distribution-aware scaling + β annealing
- Result: ARI = 0.258 ± X.XXX, 96.6% purity for gabbro
- Significance: First deep learning approach for IODP lithology clustering

### 1. Introduction
- IODP drilling program, LILY database
- Challenge: Manual lithology description subjective, time-consuming
- Opportunity: Physical properties (GRA, MS, NGR) + visual (RGB) co-located
- Goal: Unsupervised clustering to discover natural lithology groups
- Contribution: VAE framework, preprocessing innovations, β optimization

### 2. Data
- LILY database overview (Childress et al. 2024)
- 238,506 samples, 296 boreholes, 139 lithologies
- 6 features: GRA, MS, NGR, RGB
- 20cm depth binning strategy
- Borehole-level train/val/test split

### 3. Methods
- VAE architecture ([32, 16] encoder/decoder)
- Distribution-aware scaling (signed log transforms)
- β parameter optimization (0.5 optimal)
- β annealing curriculum learning (0.001→0.5)
- K-means clustering in latent space
- Evaluation metrics (ARI, Silhouette, purity)

### 4. Experiments
- Ablation studies (scaling, β, annealing, features, latent dim)
- Baseline comparisons (PCA, UMAP, t-SNE, autoencoder)
- Statistical significance testing (10 seeds, paired t-tests)
- Cluster quality analysis
- VampPrior investigation (loss vs ARI discrepancy)

### 5. Results
- v2.6 best performance: ARI = 0.258 ± X.XXX
- High-purity clusters: 96.6% gabbro, 80.4% nannofossil ooze
- Ablation results (quantify each component)
- Baseline comparisons (VAE > UMAP > PCA)
- VampPrior: +1.2% ARI not statistically significant (p=X.XX)

### 6. Discussion
- Why distribution-aware scaling works (+40% ARI)
- Why β=0.5 > β=1.0 for clustering (correlation preservation)
- Why β annealing works (+7% ARI) (curriculum learning)
- Why VampPrior fails (overfitting vs marginal improvement)
- Validation loss ≠ clustering quality
- Limitations and geological interpretation

### 7. Conclusions
- VAE effective for multimodal lithology clustering
- Preprocessing as important as architecture
- Loss function design critical (β optimization)
- Training dynamics matter (annealing)
- Future: Semi-supervised, hierarchical, transfer learning

### 8. Data/Code Availability
- LILY database (Childress et al. 2024)
- Code: GitHub repository
- Preprocessed data: Zenodo DOI
- Model checkpoints: Zenodo DOI

---

## Action Items for Publication

### High Priority (Required)
- [ ] Run each model 10 times with different seeds → confidence intervals
- [ ] Paired t-tests for statistical significance
- [ ] Complete ablation studies (scaling, β, annealing, features, latent dim)
- [ ] Baseline comparisons (PCA, UMAP, t-SNE, autoencoder, raw k-means)
- [ ] Document hyperparameter selection (validation set)
- [ ] Cluster quality analysis (purity, confusion matrix)
- [ ] Loss vs ARI analysis during training
- [ ] Write Methods section with full mathematical detail

### Medium Priority (Recommended)
- [ ] VampPrior K sweep (K=10, 20, 50, 100, 200)
- [ ] Train-test ARI gap analysis (overfitting check)
- [ ] Effective dimensionality analysis (participation ratio)
- [ ] Computational efficiency benchmarks
- [ ] Literature review and comparison
- [ ] Geological interpretation with domain expert
- [ ] Create publication-quality figures

### Low Priority (Nice to Have)
- [ ] Cross-validation (5-fold borehole-level)
- [ ] Sensitivity analysis (batch size, learning rate)
- [ ] Visualization gallery (cluster examples with images)
- [ ] Interactive web demo
- [ ] Extended experiments (semi-supervised, hierarchical)

---

## Timeline Estimate

### Month 1: Core Experiments
- Week 1-2: Multiple seed runs + statistical tests
- Week 3-4: Ablation studies + baselines

### Month 2: Analysis and Writing
- Week 1: VampPrior investigation + cluster analysis
- Week 2-3: Draft Methods + Results sections
- Week 4: Create figures + tables

### Month 3: Refinement and Submission
- Week 1-2: Draft Introduction + Discussion
- Week 3: Domain expert review + revisions
- Week 4: Final polish + submission to JGR:ML&C

---

## Key Differences: Production vs Publication

| Aspect | Production | Publication |
|--------|-----------|-------------|
| **Goal** | Best single model | Understand why methods work |
| **Evaluation** | Single run, pick best | Multiple seeds, confidence intervals |
| **Comparisons** | Internal (v2.5 vs v2.6) | Baselines (PCA, UMAP, etc.) |
| **Decision** | "Lock in v2.6" | "v2.6 significantly better (p<0.01)" |
| **Validation loss** | Use for early stopping | Investigate loss vs ARI discrepancy |
| **VampPrior** | Reject (+680% val loss) | Investigate and discuss why it fails |
| **Documentation** | Summary markdown | Full Methods section with equations |
| **Code** | Working scripts | Clean, documented, reproducible |
| **Figures** | Quick plots | Publication-quality with captions |

---

**Bottom Line:**

For JGR: Machine Learning and Computation, we need to shift from "which model should we deploy?" to "what can we learn about VAE-based lithology clustering, and can we demonstrate it rigorously?"

The science is in:
1. **Understanding** why things work (ablations, analysis)
2. **Quantifying** uncertainty (multiple seeds, statistics)
3. **Comparing** to alternatives (baselines, literature)
4. **Discussing** limitations and future directions

The +54% improvement is exciting, but reviewers will want to see:
- Is it statistically significant?
- Does it hold across random seeds?
- How does it compare to UMAP/PCA/t-SNE?
- What drives the improvement? (ablations)
- Why did VampPrior fail? (scientific investigation)
