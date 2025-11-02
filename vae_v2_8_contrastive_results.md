# VAE GRA v2.8 - Contrastive Learning Results

## Summary

Tested contrastive learning with pseudo-label based positive/negative pairs to improve latent space discrimination.

**Result**: Contrastive loss **significantly degraded** performance (-43.8% at k=12 vs v2.6).

## Model Configuration

### VAE GRA v2.8 (Contrastive Loss with Pseudo-Labels)
- **Architecture**: Standard VAE with 6D input → 8D latent
- **Loss**: `L = Reconstruction + β * KL + γ * Contrastive`
  - Reconstruction: MSE loss
  - KL: Standard VAE divergence with β annealing (0.001 → 0.5)
  - Contrastive: InfoNCE loss with pseudo-label pairs
- **Pseudo-labeling**: K-means clustering (K=12) on latent codes, updated every 10 epochs
- **Positive pairs**: Samples with same pseudo-label
- **Negative pairs**: Samples with different pseudo-labels
- **Hyperparameters**:
  - β annealing: 0.001 → 0.5 over 50 epochs
  - γ (contrastive weight): 0.1
  - τ (temperature): 0.5
  - Update interval: 10 epochs
- **Training**: Early stopping at epoch 30 (227.5s)

## Performance Results

### VAE GRA v2.6 (Baseline - Best Model)

| k  | Silhouette | ARI   |
|----|------------|-------|
| 10 | 0.428      | 0.238 |
| 12 | -          | **0.258** |
| 15 | -          | 0.237 |
| 20 | -          | 0.237 |

### VAE GRA v2.8 (Contrastive Loss)

| k  | Silhouette | ARI   | vs v2.6 |
|----|------------|-------|---------|
| 10 | 0.366      | 0.124 | -47.9% |
| 12 | 0.376      | 0.145 | **-43.8%** |
| 15 | 0.381      | 0.147 | -38.0% |
| 20 | 0.359      | 0.132 | -44.3% |

**Best v2.8 result**: ARI = 0.147 at k=15 (still 38% worse than v2.6)

## Direct Comparison

| Model | Method | k=12 ARI | Improvement vs v2.1 |
|-------|--------|----------|---------------------|
| v2.1 (baseline) | K-Means | 0.167 | - |
| v2.6 (β anneal) | K-Means | **0.258** | **+54.5%** ✓ |
| v2.7 (VaDE) | K-Means | 0.248 | +48.5% |
| v2.8 (contrastive) | K-Means | 0.145 | **-13.2%** ✗ |

v2.8 performs **worse than the v2.1 baseline** that it builds upon!

## Why Contrastive Loss Failed

### 1. Circular Dependency Problem
- Pseudo-labels derived from k-means on latent codes
- Contrastive loss then optimizes to cluster by these same pseudo-labels
- This creates a **self-fulfilling loop** that may lock in poor initial clusters
- Model can't escape early random k-means initialization

### 2. Conflicting Objectives
The three loss terms may be pulling in different directions:
- **Reconstruction**: "Preserve input information"
- **KL divergence**: "Compress to simple Gaussian"
- **Contrastive**: "Separate based on pseudo-labels"

If pseudo-labels don't align well with true lithology, contrastive loss actively works against discrimination.

### 3. Pseudo-Label Quality
Initial k-means (epoch 0) runs on random latent representations:
- Random encoder initialization → poor initial latent space
- K-means on poor latent space → poor pseudo-labels
- Training to optimize these poor labels → locked into bad solution
- Updates every 10 epochs may be too infrequent to escape

### 4. InfoNCE Loss Mechanics
InfoNCE pulls positive pairs together and pushes negatives apart:
- If pseudo-labels are wrong (e.g., different lithologies get same label)
- Contrastive loss forces model to merge truly different samples
- This destroys natural lithology structure
- γ=0.1 weight still strong enough to corrupt latent space

### 5. Overfitting to Pseudo-Labels
Model optimizes to match pseudo-labels, not true lithology:
- True task: cluster by lithology (unsupervised)
- Actual task: cluster by pseudo-labels (self-supervised)
- These diverge when pseudo-labels are poor
- No ground truth feedback to correct pseudo-label errors

### 6. Training Dynamics
Early stopping at epoch 30 vs v2.6's 16 epochs:
- Longer training but worse results
- Suggests model struggling with conflicting objectives
- May be converging to poor local optimum

## Cluster Quality Analysis

### v2.6 High-Purity Clusters (k=12)
- 96.6% Gabbro (from v2.1 experiments)
- 64.7% Nannofossil ooze
- Clear lithology separation

### v2.8 High-Purity Clusters (k=12)
- 96.1% Gabbro (cluster 4, n=1276)
- 71.2% Nannofossil ooze (cluster 1, n=1599)
- 48.8% Nannofossil ooze (cluster 11, n=1819)

v2.8 identifies Gabbro well but fails on other lithologies compared to v2.6.

## Loss Component Analysis

From training log at epoch 30:
- Reconstruction: 5.6382 (comparable to v2.6)
- KL divergence: 0.7696 (comparable to v2.6)
- **Contrastive: 5.2203** (large contributor)

The contrastive term is dominating the loss, potentially overwhelming the VAE objectives.

## Alternative Approaches That Failed

This is the **4th experimental approach** that underperformed v2.6:

1. **v2.2 (Spatial context)**: +3.9% vs v2.1, but not worth complexity
2. **v3 (Dual encoders)**: -7.9% vs v2.1 - architectural elegance failed
3. **v2.7 (VaDE loss)**: -3.9% vs v2.6 - cluster-aware prior too restrictive
4. **v2.8 (Contrastive)**: **-43.8% vs v2.6** - worst performing variant

## Key Insights

### What We Learned

1. **Pseudo-labeling is brittle**: Self-supervised signals need high-quality initial representations
2. **Conflicting objectives hurt**: Three loss terms with different goals degrade performance
3. **Circular dependencies fail**: Optimizing based on model's own predictions creates self-fulfilling loops
4. **Simple is better**: v2.6's two-term loss (recon + KL) outperforms complex multi-objective losses
5. **Domain mismatch**: Contrastive learning excels in vision (where augmentation creates true positive pairs), but geological data doesn't have obvious augmentations

### When Contrastive Learning Works

Contrastive learning succeeds in scenarios v2.8 lacks:
- **Strong augmentations**: Vision (crop, color, rotate), NLP (backtranslation)
- **True positive pairs**: Multiple views of same underlying entity
- **Large batch sizes**: 1000s of negatives for good InfoNCE gradients
- **Pre-training then fine-tuning**: Learn general features, then adapt
- **Labeled supervision**: SimCLR → supervised fine-tuning

Our setting has none of these:
- Batch size: 512 (not 1000s)
- No natural augmentations for geological features
- Pseudo-labels instead of true positives
- Direct clustering (no fine-tuning stage)

### Why β Annealing Alone Works Best

v2.6 succeeds because:
- **Two aligned objectives**: Reconstruction and regularization both serve representation learning
- **No conflicting signals**: Model freely learns natural data structure
- **Curriculum learning**: β annealing provides training dynamics benefit
- **Simplicity**: Fewer hyperparameters to tune (just β schedule)
- **Robustness**: Works across different k values (10, 12, 15, 20)

## Recommendations

1. **Use VAE GRA v2.6** - Simple β annealing remains best approach (ARI=0.258)
2. **Avoid contrastive loss** for unsupervised geological clustering
3. **Document v2.8** as cautionary example of complexity hurting performance
4. **Key lesson**: Self-supervised signals must be high-quality; circular dependencies fail

## Scientific Contribution

This negative result is valuable:
- Demonstrates contrastive learning isn't universally applicable
- Shows pseudo-labeling can create harmful circular dependencies
- Reinforces principle: **Follow the data, not the hype**

Contrastive learning is powerful in vision/NLP but **doesn't transfer** to geological unsupervised clustering.

## Files Generated

- `ml_models/vae_lithology_gra_v2_8_model.py` - Contrastive VAE implementation
- `ml_models/checkpoints/vae_gra_v2_8_latent8_contrastive.pth` - Trained model
- `vae_v2_8_outputs/training_summary_v2_8_latent8.png` - Training curves
- `vae_gra_v2_8_training.log` - Full training log
- `vae_v2_8_contrastive_results.md` - This analysis

## Conclusion

Contrastive learning with pseudo-labels **significantly degraded** lithology clustering performance (-43.8% vs v2.6).

The circular dependency between pseudo-label generation and contrastive optimization creates a harmful feedback loop that locks the model into poor solutions.

**VAE GRA v2.6 (β annealing alone) remains the best model** for unsupervised lithology clustering.

## Future Directions

Given 4 failed experimental approaches (v2.2, v3, v2.7, v2.8), recommendations:

1. **Focus on v2.6 optimization**: Different latent dims, annealing schedules
2. **Semi-supervised approaches**: If labels available, use them directly (not pseudo-labels)
3. **Different architectures**: Transformers, graph neural networks?
4. **Feature engineering**: Better preprocessing, different feature combinations
5. **Ensemble methods**: Combine multiple v2.6 models

But most importantly: **Recognize when good enough is good enough**.

v2.6 achieves 54.5% improvement over v2.1 baseline - this may be close to the ceiling for unsupervised clustering on this data.
