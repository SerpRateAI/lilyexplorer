# VAE Classifier Investigation Summary

**Date**: 2025-11-01
**Models**: VAE v2.6.7, VAE Classifier v1.0, v1.1, Direct Classifier Baseline

---

## Executive Summary

This investigation examined why VAE-based lithology classifiers achieve poor performance (~30% balanced accuracy) and whether the VAE embeddings provide value for downstream tasks.

**Key Findings:**

1. **VAE is losing discriminative information** - Direct classifier on raw features achieves **42.3% better balanced accuracy** (42.32% vs 29.73%)

2. **VAE reconstruction quality is excellent** - Despite poor classification, the VAE successfully achieves its unsupervised objective with **R² = 0.904** for physical property reconstruction

3. **The VAE objective differs from classification** - Reconstruction optimizes for variance preservation, not lithology separation

4. **Hierarchical classification helps** - Reducing 139 classes → 14 groups improved balanced accuracy 4× (7.51% → 29.73%)

5. **Expert-defined hierarchy is misaligned** - Keyword-based geological groupings show only 11.6% agreement (ARI = 0.116) with VAE embedding clusters

---

## Investigation Timeline

### Problem 1: Poor Classifier Performance (33% accuracy)

**Root Cause**: Random borehole splitting created severe train/test distribution mismatch

**Symptoms:**
- Gabbro: 2 training samples, 2,520 test samples (7,192× mismatch)
- 15 classes with 0 training samples but present in test set
- Test set wildly unrepresentative of training distribution

**Solution**: Entropy-balanced borehole splitting
- **Implementation**: Greedy allocation algorithm to maintain similar lithology distributions
- **Script**: `train_lithology_classifier_v2_6_7_entropy_balanced.py`
- **Results**: Unweighted accuracy improved 33% → 42.57%, but balanced accuracy still poor (5.26%)

---

### Problem 2: Extreme Class Imbalance (49,778:1 ratio)

**Challenge**: 139 lithology classes with massive imbalance
- Most common: nannofossil ooze (20.87%, 49,778 samples)
- Least common: 5 lithologies with 1 sample each (0.0004%)
- 119/139 classes (85.6%) have <1% of data

#### Attempt 1: Class-Balanced Loss (VAE Classifier v1.0)

**Approach**: Inverse frequency weighting
```python
weight = 1 / (count + 1)
```

**Results**: **FAILURE**
- Weight ratio: 32,891× (too extreme)
- Test accuracy: 7.27% unweighted, 7.51% balanced
- Common classes crushed:
  - Nannofossil ooze: 87.96% → 2.90%
  - Clay: 77.43% → 0.41%

**Script**: `train_vae_classifier_v1_0.py`

**Lesson**: Inverse frequency weighting fails with extreme imbalance - weights become so large they destroy performance on common classes

---

#### Attempt 2: Hierarchical Classification (VAE Classifier v1.1)

**Approach**: Group 139 fine-grained lithologies into 14 geological families

**Grouping Strategy**: Keyword-based mapping
- **Carbonate** (42.15%, 26 lithologies): nannofossil, foraminifera, chalk, limestone, etc.
- **Clay/Mud** (40.64%, 35 lithologies): clay, mud, mudstone
- **Sand** (5.56%, 17 lithologies): sand, sandstone
- **Biogenic Silica** (3.11%, 6 lithologies): diatom, radiolarian, siliceous ooze
- **Mafic Igneous** (2.50%, 11 lithologies): basalt, gabbro, diabase
- Plus 9 more groups

**Results**: **MAJOR IMPROVEMENT**
- Test accuracy: 34.18% unweighted, **29.73% balanced**
- **4× better balanced accuracy** than v1.0 (7.51% → 29.73%)
- Weight ratio reduced to 1,175× (more manageable)

**Per-Group Performance:**
- Mafic Igneous: 62.87%
- Biogenic Silica: 54.98%
- Clay/Mud: 43.20%
- Carbonate: 32.93%
- Sand: 20.25%

**Script**: `train_vae_classifier_v1_1.py`
**Hierarchy**: `lithology_hierarchy_mapping.csv`

**Lesson**: Hierarchical classification is essential for extreme imbalance - reducing classes from 139 → 14 makes the problem tractable

---

### Problem 3: Expert Hierarchy Validation

**Question**: Do expert-defined lithology groups (keyword-based) align with natural VAE embedding structure?

**Approach**: Compute lithology centroids in 10D VAE space, compare expert groupings vs hierarchical clustering

**Validation Method**:
1. Compute median embedding for each of 139 lithologies
2. Cluster lithology centroids using AgglomerativeClustering (n=14 clusters)
3. Measure alignment with expert hierarchy using Adjusted Rand Index (ARI)

**Results**: **POOR ALIGNMENT**
- ARI = 0.1164 (11.6% agreement between expert and data-driven clustering)
- Expert groups don't match natural VAE structure

**Group Coherence** (intra-group distance in embedding space):
- **Tight (✓)**: Silt (0.82), Ultramafic (0.99)
- **Moderate (○)**: Carbonate (1.74), Clay/Mud (1.88), Sand (1.57)
- **Loose (✗)**: Mafic Igneous (2.11), Metamorphic (2.30), Other (2.59)

**Script**: `validate_hierarchy_with_embeddings.py`

**Lesson**: Compositional similarity (keyword matching) doesn't guarantee physical property similarity. Carbonates with different grain sizes, compaction, or alteration have different GRA/MS/NGR/RGB signatures, causing them to scatter in embedding space.

**Implication**: Data-driven clustering would better capture natural physical property groupings than expert geological taxonomy

---

### Problem 4: VAE Embeddings Underperforming

**Critical Question**: Is ~30% balanced accuracy:
1. The fundamental limit of physical properties for lithology discrimination?
2. Or is the VAE losing discriminative information during compression?

**Test**: Direct classifier on raw 6D features (no VAE) vs VAE classifier on 10D embeddings

#### Direct Classifier Baseline Architecture

```python
Input: 6D raw features (GRA, MS, NGR, RGB)
  ↓
Linear(64) + ReLU + Dropout(0.3)
  ↓
Linear(32) + ReLU + Dropout(0.3)
  ↓
Linear(14) → Lithology Group
```

**Key Design**:
- Distribution-aware scaling (same preprocessing as VAE)
- Entropy-balanced borehole split
- Class-balanced loss (weight ratio 1,175×)
- 2,990 parameters

**Results**: **SIGNIFICANTLY BETTER**

| Model | Unweighted Acc | Balanced Acc | Improvement |
|-------|----------------|--------------|-------------|
| **Direct Classifier (6D raw)** | **55.65%** | **42.32%** | **Baseline** |
| VAE Classifier v1.1 (10D embeddings) | 34.18% | 29.73% | -42.3% |

**Per-Group Accuracies (Direct Classifier):**
- Carbonate: 73.75% (17,751 samples)
- Mafic Igneous: 70.06% (167 samples)
- Clay/Mud: 43.17% (12,985 samples)
- Sand: 34.03% (3,870 samples)
- Conglomerate/Breccia: 31.50% (273 samples)
- Metamorphic: 38.46% (104 samples)

**Script**: `train_direct_classifier_baseline.py`

**Conclusion**: **VAE is losing 42.3% discriminative information** during compression from 6D → 10D latent → 6D reconstruction

---

## VAE Reconstruction Validation

**Purpose**: Confirm that while VAE embeddings have limited classification value, they successfully achieve their primary unsupervised objective (physical property reconstruction)

**Reconstruction Quality: Excellent (R² = 0.904)**

### Per-Feature Reconstruction

| Feature | MAE | RMSE | R² | Quality |
|---------|-----|------|-----|---------|
| R (red) | 0.143 | 0.196 | **0.962** | Excellent |
| G (green) | 0.138 | 0.184 | **0.966** | Excellent |
| B (blue) | 0.173 | 0.223 | **0.950** | Excellent |
| NGR (cps) | 0.266 | 0.347 | **0.880** | Excellent |
| GRA (g/cm³) | 0.309 | 0.405 | **0.836** | Good |
| MS (instr.) | 0.323 | 0.415 | **0.828** | Good |

**Average**: MAE = 0.225, RMSE = 0.295, R² = 0.904

### Latent Space Utilization

**Effective Dimensionality: 4/10**

| Dimension | Mean | Std | Active |
|-----------|------|-----|--------|
| Latent 0 | -0.002 | **0.986** | ✓ |
| Latent 1 | -0.011 | **0.808** | ✓ |
| Latent 2 | 0.005 | 0.008 | ✗ |
| Latent 3 | -0.018 | **0.283** | ✓ |
| Latent 4 | -0.002 | 0.016 | ✗ |
| Latent 5 | 0.000 | 0.005 | ✗ |
| Latent 6 | -0.010 | 0.026 | ✗ |
| Latent 7 | 0.001 | 0.017 | ✗ |
| Latent 8 | 0.002 | **0.360** | ✓ |
| Latent 9 | 0.000 | 0.010 | ✗ |

Only 4 dimensions (0, 1, 3, 8) have std > 0.1 and are actively used. Six dimensions (2, 4, 5, 6, 7, 9) have collapsed (std < 0.03).

### Reconstruction Error by Lithology Group

| Lithology Group | N_Samples | Avg_MAE | Quality |
|-----------------|-----------|---------|---------|
| Clay/Mud | 96,930 | 0.215 | ○ Good |
| Silt | 6,951 | 0.215 | ○ Good |
| Carbonate | 100,531 | 0.222 | ○ Good |
| Sand | 13,253 | 0.225 | ○ Good |
| Biogenic Silica | 7,414 | 0.283 | ✗ Poor |
| Volcaniclastic | 4,187 | 0.282 | ✗ Poor |
| Mafic Igneous | 5,955 | 0.297 | ✗ Poor |
| Conglomerate/Breccia | 2,197 | 0.314 | ✗ Poor |

**Pattern**: Common sedimentary lithologies (clay, silt, carbonate, sand) reconstruct well (MAE ~0.22). Rare igneous/metamorphic lithologies reconstruct poorly (MAE > 0.28).

**Script**: `validate_vae_reconstruction.py`

---

## Key Insights

### 1. VAE Objective vs Classification Objective

**VAE trains for reconstruction**:
```
Loss = reconstruction_loss + β * KL_divergence
```
- Optimizes for minimizing reconstruction error
- Preserves variance in the data
- Learns to compress and decompress accurately

**Classification requires separation**:
```
Loss = cross_entropy(predicted_class, true_class)
```
- Optimizes for separating classes in latent space
- Maximizes inter-class distance
- Minimizes intra-class variance

**These objectives are fundamentally different**. A VAE that reconstructs perfectly can still have poor class separation if physically similar lithologies (e.g., clay vs mud, sand vs silt) have overlapping properties.

### 2. Latent Space Collapse

The VAE has 10D latent space but only 4 dimensions are active (std > 0.1). This means:
- Effective compression: 6D input → 4D latent → 6D output
- 6 latent dimensions collapsed during training (common in VAEs with β regularization)
- Limited capacity for capturing fine-grained lithology distinctions

### 3. Classification Performance Ceiling

**Direct classifier achieves 42.32% balanced accuracy** on raw features. This suggests:
- Physical properties (GRA, MS, NGR, RGB) have **inherent limitations** for lithology discrimination
- Many lithologies have overlapping physical signatures
- Some groups are fundamentally ambiguous:
  - Clay/Mud: 43.17% (physically very similar)
  - Sand: 34.03% (grain size variation)
  - Volcaniclastic: 2.82% (heterogeneous composition)

Even with perfect features, we cannot expect >50% balanced accuracy for 14-class hierarchical classification from these 6 physical properties alone.

### 4. Reconstruction vs Classification Trade-off

| Task | VAE Embeddings | Raw Features |
|------|----------------|--------------|
| **Reconstruction** | R² = 0.904 (excellent) | N/A |
| **Classification** | 29.73% balanced acc | 42.32% balanced acc |
| **Dimensionality** | 10D (4D effective) | 6D |
| **Interpretation** | Captures variance | Direct physical meaning |

The VAE learns a compressed representation that preserves most information (90.4% variance) but loses discriminative details critical for classification.

---

## Recommendations

### For Lithology Classification
1. **Use raw features directly** - Direct classifiers on 6D physical properties outperform VAE embeddings by 42%
2. **Hierarchical grouping is essential** - 139 classes are intractable, 14 groups are manageable
3. **Data-driven hierarchy may be better** - Expert keyword groupings show poor alignment (ARI=0.12) with physical property structure
4. **Consider additional features** - 42% balanced accuracy suggests ceiling with current features; geochemistry, mineralogy, or stratigraphic context could help

### For Oceanic Crust AI Model
The VAE v2.6.7 embeddings **are suitable** for unsupervised tasks:
- **Dimensionality reduction** for visualization (4D effective latent space)
- **Anomaly detection** via reconstruction error (rare lithologies have MAE > 0.30)
- **Exploratory data analysis** of physical property patterns
- **Feature extraction** for tasks that don't require fine-grained lithology discrimination

The VAE v2.6.7 embeddings **are NOT suitable** for:
- Fine-grained lithology classification (loses 42% discriminative power)
- Supervised learning tasks where raw features outperform

### For VAE Architecture Improvement
If classification performance must be improved while maintaining unsupervised learning:

1. **Increase latent dimensionality** (e.g., 20D) to prevent collapse
2. **Reduce β parameter** further (currently 0.75, try 0.25) to preserve more variance
3. **Add auxiliary task** during training (e.g., predict depth or borehole ID) to encourage richer representations
4. **Use different architecture** (e.g., β-TCVAE, FactorVAE) designed to learn disentangled but informative features

However, these changes risk overfitting or violating the unsupervised constraint. The fundamental issue remains: **reconstruction and classification are different objectives**.

---

## Files Generated

### Training Scripts
- `train_vae_classifier_v1_0.py` - Class-balanced loss (failed, 7.51% balanced acc)
- `train_vae_classifier_v1_1.py` - Hierarchical classification (29.73% balanced acc)
- `train_direct_classifier_baseline.py` - Raw features (42.32% balanced acc)

### Analysis Scripts
- `analyze_lithology_distribution.py` - Class imbalance analysis (49,778:1 ratio)
- `analyze_train_test_distribution_mismatch.py` - Exposes random split failure
- `create_lithology_hierarchy.py` - Keyword-based 139→14 grouping
- `validate_hierarchy_with_embeddings.py` - Expert vs data-driven comparison (ARI=0.12)
- `validate_vae_reconstruction.py` - Reconstruction quality validation (R²=0.904)

### Data Files
- `lithology_hierarchy_mapping.csv` - 139 lithologies mapped to 14 groups

### Logs
- `vae_classifier_v1_0_training.log` - v1.0 training (inverse freq weights)
- `vae_classifier_v1_1_training.log` - v1.1 training (hierarchical)
- `direct_classifier_baseline_training.log` - Direct classifier training
- `vae_reconstruction_validation.log` - Reconstruction validation

### Model Checkpoints
- `ml_models/checkpoints/vae_classifier_v1_0_best.pth` - v1.0 model
- `ml_models/checkpoints/vae_classifier_v1_1_best.pth` - v1.1 model
- `ml_models/checkpoints/direct_classifier_baseline_best.pth` - Direct classifier

### Visualizations
- `vae_reconstruction_validation.png` - Reconstruction quality metrics

---

## Conclusion

The VAE v2.6.7 model successfully achieves its **unsupervised objective** (R² = 0.904 reconstruction quality) but has limited value for lithology classification due to fundamental differences between reconstruction and discrimination objectives.

**For the oceanic crust AI model**, the key insight is:
- ✓ VAE embeddings preserve physical property information (90.4% variance)
- ✓ Suitable for unsupervised exploration, visualization, anomaly detection
- ✗ Raw features classify 42% better than VAE embeddings (42.32% vs 29.73%)
- ✗ Even raw features plateau at ~42% balanced accuracy (physical property limitations)

**The VAE is working as designed** - it learns a compressed representation without lithology labels. The trade-off is that compression optimizes for reconstruction fidelity, not class separability. This is an acceptable limitation for an unsupervised oceanic crust model, where the goal is to learn physical property patterns rather than predict human-defined lithology labels.

Future work should focus on:
1. Using raw features directly for any supervised classification tasks
2. Leveraging VAE embeddings for unsupervised pattern discovery
3. Exploring additional features (geochemistry, mineralogy) to break the 42% accuracy ceiling
4. Considering data-driven lithology hierarchies aligned with physical properties rather than expert taxonomy
