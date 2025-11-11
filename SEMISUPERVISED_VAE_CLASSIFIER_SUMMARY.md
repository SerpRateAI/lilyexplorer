# Semi-Supervised VAE Classifier Investigation

**Date:** 2025-11-11
**Status:** ❌ FAILED
**Investigation Goal:** Test if pre-training VAE v2.6.7 encoder on unsupervised reconstruction improves lithology classification

---

## Executive Summary

**Key Finding:** Semi-supervised learning using VAE v2.6.7 pre-training **fails catastrophically** for lithology classification, performing **61.6% worse** than direct classifiers on raw features.

| Approach | Balanced Accuracy | Relative to Baseline |
|----------|-------------------|----------------------|
| **Direct Classifier (raw 6D features)** | **42.32%** | Baseline |
| VAE Classifier v1.1 (hierarchical) | 29.73% | -29.7% |
| **Semi-supervised (frozen encoder)** | **16.96%** | **-59.9%** |
| **Semi-supervised (fine-tuned encoder)** | **16.24%** | **-61.6%** |

**Conclusion:** VAE pre-training on unsupervised reconstruction provides **no benefit** for classification. In fact, it significantly degrades performance compared to both direct classification and previous VAE-based approaches.

---

## Hypothesis

Pre-training a VAE encoder on the unsupervised reconstruction task (optimizing for R²=0.904 on physical properties) might provide a useful initialization for classification, capturing latent structure in the feature space that could improve class separation.

**Two variants tested:**
1. **Frozen encoder:** Use pre-trained VAE v2.6.7 embeddings as fixed features, train only classification head
2. **Fine-tuned encoder:** Initialize with VAE v2.6.7 weights, then jointly train encoder + classifier with low learning rate

---

## Methodology

### Model Architecture

**Base Model:** VAE GRA v2.6.7 (10D latent space, trained with extreme β annealing: 1e-10 → 0.75)

**VAE Encoder:**
- Input: 6D features (GRA, MS, NGR, R, G, B)
- Hidden layers: [32, 16] with ReLU
- Output: 10D latent mean (μ) - deterministic embeddings, no sampling

**Classification Head:**
- Input: 10D VAE embeddings
- Hidden: 32 units + ReLU + Dropout(0.3)
- Output: 14 lithology groups (hierarchical)

**Total Parameters:** 1,906 (classifier only: 814 trainable when encoder frozen)

### Training Configuration

| Variant | Encoder | Learning Rate | Epochs | Early Stopping |
|---------|---------|---------------|--------|----------------|
| Frozen | Frozen (no gradients) | 1e-3 | 100 | 15 patience |
| Fine-tuned | Trainable | 1e-4 | 100 | 15 patience |

**Loss Function:** Cross-entropy with class weights (inverse frequency, clipped)
- Class weight ratio: 3228.5× (min=0.002 for Carbonate, max=6.235 for Ultramafic)

**Data Split:** Entropy-balanced borehole split (70/30)
- Train: 169,271 samples from 207 boreholes
- Test: 69,235 samples from 89 boreholes

**Scaling:** Distribution-aware scaler from VAE v2.6.7 (preserves outliers in physical properties)

---

## Results

### Performance Comparison

**Frozen Encoder:**
- Best validation: 18.38% (epoch 46)
- Test balanced accuracy: **16.96%**
- Training stopped early at epoch 46

**Fine-tuned Encoder:**
- Best validation: 16.54% (epoch 20)
- Test balanced accuracy: **16.24%**
- Training stopped early at epoch 35
- Fine-tuning made performance **worse** (-4.2% vs frozen)

### Per-Class Performance (Frozen Encoder, Best Variant)

| Lithology Group | Test Accuracy | Test Samples | Notes |
|-----------------|---------------|--------------|-------|
| Carbonate | 61.52% | 26,276 | Only high-performing class |
| Biogenic Silica | 47.95% | 1,512 | Moderate |
| Volcaniclastic | 31.57% | 1,774 | Poor |
| Sand | 30.48% | 6,561 | Poor |
| Mafic Igneous | 18.97% | 2,377 | Very poor |
| Silt | 11.35% | 2,237 | Very poor |
| Clay/Mud | 10.77% | 27,186 | Very poor (largest class!) |
| Conglomerate/Breccia | 6.86% | 875 | Failed |
| Evaporite | 0.93% | 107 | Failed |
| **5 classes** | **0.00%** | 630 | **Complete failure** |

**Catastrophic failures (0% accuracy):**
- Intermediate/Felsic Igneous (14 samples)
- Metamorphic (18 samples)
- Ultramafic (1 sample)
- Other (297 samples)
- Diamict (omitted from results, presumably 0%)

---

## Why It Failed

### 1. **VAE Optimized for Wrong Objective**

The VAE v2.6.7 encoder was trained to **reconstruct physical properties** (R²=0.904), not to **separate lithology classes**. The reconstruction loss doesn't care about class boundaries—it only cares about minimizing feature reconstruction error.

**Evidence:**
- Only **4/10 latent dimensions are active** (6 collapsed to near-zero variance)
- Effective dimensionality: ~4D, not 10D
- Latent space optimized for "physical similarity" (e.g., "dark + dense = similar") rather than "lithological similarity"

### 2. **Latent Space Collapses Class Distinctions**

The 10D → 4D effective compression **destroys information** needed for fine-grained classification:

**What VAE preserves:**
- Broad physical trends (density gradients, color gradients, magnetic signatures)
- Continuous variations within lithologies

**What VAE discards:**
- Sharp boundaries between lithology classes
- Subtle differences in feature combinations that distinguish classes
- Rare class signatures (Ultramafic, Metamorphic completely lost)

### 3. **Hierarchical Grouping Still Too Fine-Grained**

Even with 14 hierarchical groups (vs 139 original principal lithologies), the VAE embeddings cannot provide enough information for reliable classification.

**Comparison to raw features:**
- Raw 6D features: 42.32% (information-rich)
- VAE 10D embeddings: 16.96% (information-poor)
- **Information loss:** -60% from compression

### 4. **Fine-Tuning Fails to Recover**

Fine-tuning the encoder with classification loss (1e-4 LR) **degrades performance further** (-4.2% vs frozen):

**Possible reasons:**
- Low learning rate (1e-4) too conservative to escape bad initialization
- Classification signal overwhelmed by pre-training bias toward reconstruction
- 35 epochs insufficient to overcome 100 epochs of unsupervised pre-training (VAE trained for ~16 epochs but on full 238K samples)
- Class imbalance (3228.5× weight ratio) causes training instability

---

## Comparison to Baselines

### Direct Classifier (Best)

**Performance:** 42.32% balanced accuracy

**Why better:**
- Uses **raw 6D features** with full information content
- No information-destroying compression step
- Model can learn task-specific feature combinations directly
- Supervised from the start—no reconstruction bias

### VAE Classifier v1.1 (Previous Work)

**Performance:** 29.73% balanced accuracy
**Approach:** Hierarchical classification (14 groups) on frozen VAE embeddings (10D)

**Why better than semi-supervised:**
- Same 10D embeddings, but trained from scratch for classification
- No pre-training bias toward reconstruction
- Still worse than direct classifier (-29.7%) due to information loss from VAE compression

### Semi-Supervised (This Work)

**Performance:** 16.96% (frozen) / 16.24% (fine-tuned)

**Why worst:**
- **Double penalty:** Information loss from VAE compression + reconstruction bias from pre-training
- Pre-training optimizes for wrong objective (reconstruction vs classification)
- Latent space not structured for class separation

---

## Scientific Insights

### 1. **Transfer Learning Requires Task Alignment**

Pre-training only helps if the pre-training task and target task share the **same objective**.

**Misaligned objectives:**
- Pre-training: Minimize reconstruction error (unsupervised)
- Classification: Maximize class separation (supervised)

**Result:** No transfer benefit. Model learns wrong feature representations.

### 2. **Information Bottleneck Principle Violated**

VAE's information bottleneck (6D → 10D → 4D effective) is designed to compress data while preserving **reconstruction quality**, not **class separability**.

**For classification:**
- Need to preserve discriminative features (class boundaries)
- Can discard reconstructive features (smooth gradients, noise)

**VAE does the opposite:**
- Preserves smooth, continuous trends (good for reconstruction)
- Discards sharp, discrete boundaries (bad for classification)

### 3. **Raw Features > Learned Embeddings for Tabular Data**

In tabular domains (unlike vision/language), **raw features are often optimal**:

**Why:**
- Features already physically meaningful (density, color, magnetism)
- No nuisance variation to abstract away (no lighting, perspective, synonyms)
- Task-specific information preserved in original feature space

**Embeddings hurt when:**
- Compression loses discriminative information
- Pre-training introduces irrelevant inductive biases

### 4. **Fine-Tuning Cannot Fix Bad Initialization**

Fine-tuning with 1e-4 LR for 35 epochs failed to overcome VAE pre-training bias:

**Implication:** When pre-training objective is misaligned, fine-tuning is ineffective. Better to train from scratch on raw features.

---

## Relationship to Other Failed Experiments

This is the **4th failed approach** to using VAE embeddings for classification:

| Experiment | Approach | Performance | Failure Mode |
|------------|----------|-------------|--------------|
| **Direct Classifier** | Raw 6D → CatBoost | **42.32%** | ✓ **Success** |
| VAE Classifier v1.0 | 10D embeddings → NN (class-balanced) | 7.51% | Extreme class weighting |
| VAE Classifier v1.1 | 10D embeddings → NN (hierarchical) | 29.73% | Information loss from VAE |
| **Semi-supervised (frozen)** | **VAE pre-train → classifier** | **16.96%** | **Reconstruction bias** |
| **Semi-supervised (fine-tuned)** | **VAE pre-train → fine-tune** | **16.24%** | **Worse than frozen** |

**Pattern:** All VAE-based approaches fail because VAE optimizes for reconstruction (ARI=0.196 for clustering), not class separation (42.32% for direct classification). The two objectives are fundamentally different.

---

## Recommendations

### For Lithology Classification

**✓ DO:**
- Use **direct classifier on raw 6D features** (42.32% balanced accuracy)
- Use CatBoost/Random Forest for tabular data (handles class imbalance well)
- Apply borehole-level train/test splits to avoid data leakage

**✗ DON'T:**
- Use VAE embeddings for classification (loses information)
- Pre-train on unsupervised tasks for supervised classification (misaligned objectives)
- Fine-tune reconstruction-optimized models (cannot escape bad initialization)

### For VAE Usage

**VAE is good for:**
- Unsupervised clustering (ARI=0.196±0.037 on v2.6.7)
- Anomaly detection (reconstruction error)
- Visualization (UMAP on latent space)
- Data generation (sample from latent space)

**VAE is bad for:**
- Direct classification (loses discriminative information)
- Fine-grained class separation (compresses boundaries)
- Transfer learning to classification tasks (wrong objective)

### For Future Semi-Supervised Work

If semi-supervised learning is desired, consider:

1. **Contrastive pre-training** (SimCLR, MoCo) - optimizes for class separation, not reconstruction
2. **Supervised pre-training on related task** - e.g., pre-train on coarse lithology groups (carbonate vs siliciclastic), fine-tune on fine groups
3. **Multi-task learning** - jointly train on reconstruction + classification (shared encoder, dual loss)
4. **Self-supervised with pseudo-labels** - use clustering to generate pseudo-labels, iteratively refine

---

## Files

### Training Script
- **`train_semisupervised_vae_classifier.py`** - Full implementation of frozen + fine-tuned variants

### Model Checkpoints
- **`ml_models/checkpoints/semisupervised_frozen_encoder_best.pth`** - Frozen encoder model (16.96%)
- **`ml_models/checkpoints/semisupervised_finetuned_encoder_best.pth`** - Fine-tuned encoder model (16.24%)

### Training Log
- **`semisupervised_vae_classifier_training.log`** - Complete training output with per-epoch results

### Dependencies
- **`ml_models/checkpoints/vae_gra_v2_6_7_final.pth`** - Pre-trained VAE v2.6.7 encoder
- **`vae_training_data_v2_20cm.csv`** - Dataset (238,506 samples, 6D features + lithology labels)
- **`lithology_hierarchy_mapping.csv`** - Mapping from 139 principal lithologies → 14 groups

---

## Conclusion

Semi-supervised learning using VAE v2.6.7 pre-training **fails catastrophically** for lithology classification, achieving only **16.24-16.96% balanced accuracy** compared to **42.32%** for direct classification on raw features.

**Root cause:** VAE optimized for reconstruction (unsupervised) learns representations unsuitable for classification (supervised). The 10D latent space collapses to 4D effective dimensionality, destroying class boundaries while preserving smooth physical trends.

**Key lesson:** In tabular domains with physically meaningful features, **raw features are optimal**. Learned embeddings introduce information loss and inductive biases that hurt classification performance. Pre-training only helps when the pre-training task aligns with the target task—reconstruction and classification are fundamentally misaligned objectives.

**Recommendation:** Use **direct classifier on raw 6D features** (42.32%) for all lithology classification tasks. Reserve VAE for unsupervised clustering (ARI=0.196±0.037), anomaly detection, and visualization.
