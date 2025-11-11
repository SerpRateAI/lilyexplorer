# Retrain VAE v2.6.7 on Filtered Data - Execution Plan

**Date:** 2025-11-11
**Goal:** Retrain VAE v2.6.7 on filtered dataset (≥100 samples per class) for consistency with downstream classification experiments
**Machine:** cotopaxi (GPU required)

---

## Motivation

Current inconsistency:
- **VAE v2.6.7 pre-training**: 238,506 samples (14 classes including Ultramafic n=81, Diamict n=66)
- **Classification downstream**: 238,359 samples (12 classes, filtered ≥100)

**Problem:** Semi-supervised classifier uses encoder pre-trained on different data than it's evaluated on.

**Solution:** Retrain VAE v2.6.7 on exact same filtered dataset (238,359 samples, 12 classes).

---

## Expected Outcomes

1. **Perfect consistency**: Pre-training ↔ downstream task use identical data
2. **Possibly better ARI**: Removing 1-sample classes (noise) might improve clustering
3. **Cleaner embeddings**: Encoder never learns from impossible classes
4. **Fair comparison**: Semi-supervised starts from truly aligned pre-training

**Hypothesis:** ARI will stay similar or improve (0.196±0.037 → 0.20±0.04?)

---

## Execution Steps

### Step 1: Create Filtered Dataset for VAE (on smokey, already done)

Dataset already exists at:
- **File:** `/home/utig5/johna/bhai/vae_training_data_v2_20cm.csv`
- **Filter script:** Already applied in `train_clean_lithology_classifiers.py`

We need to create a permanent filtered version:

**Script:** `create_filtered_vae_dataset.py` (see below)

**Output:** `vae_training_data_v2_20cm_filtered_100.csv` (238,359 samples, 12 classes)

### Step 2: Switch to cotopaxi

```bash
# On smokey
cd /home/utig5/johna/bhai
git pull  # Make sure latest code

# Switch to cotopaxi
ssh cotopaxi
cd /home/utig5/johna/bhai  # NFS shared filesystem
git status  # Verify same repo

# Check GPU
nvidia-smi
```

### Step 3: Run 5-Fold Entropy-Balanced CV (on cotopaxi)

**Script:** `train_vae_v2_6_7_filtered_cv.py` (see below)

**Estimated time:** 1-2 hours on GPU

**Command:**
```bash
python3 train_vae_v2_6_7_filtered_cv.py 2>&1 | tee vae_v2_6_7_filtered_cv.log
```

**Output:**
- `vae_v2_6_7_filtered_cv.csv` - Cross-validation results (ARI per fold)
- `vae_v2_6_7_filtered_cv.log` - Training log
- 5 fold checkpoints (for inspection)

### Step 4: Train Final Model on All Data (on cotopaxi)

**Script:** `train_vae_v2_6_7_filtered_final.py` (see below)

**Estimated time:** 10 minutes on GPU

**Command:**
```bash
python3 train_vae_v2_6_7_filtered_final.py 2>&1 | tee vae_v2_6_7_filtered_final.log
```

**Output:**
- `ml_models/checkpoints/vae_gra_v2_6_7_filtered_final.pth` - Production model
- `vae_v2_6_7_filtered_final.log` - Training log

### Step 5: Re-run Semi-Supervised Classifiers (can do on smokey, CPU)

**Script:** `train_clean_semisupervised_with_filtered_vae.py` (see below)

**Estimated time:** 15 minutes on CPU

**Command:**
```bash
# Can switch back to smokey if desired (CPU sufficient)
python3 train_clean_semisupervised_with_filtered_vae.py 2>&1 | tee clean_semisupervised_filtered_vae.log
```

**Output:**
- Updated frozen/fine-tuned models using filtered VAE encoder
- Performance comparison: old encoder vs filtered encoder

### Step 6: Regenerate Figures and Documentation

**Script:** `compare_vae_filtered_vs_original.py` (see below)

**Output:**
- Comparison figures (ARI, per-class, embeddings)
- Updated paper section

### Step 7: Commit Everything

Following git workflow - commit after each major step:
1. After CV completes → commit CV results
2. After final training → commit production model
3. After semi-supervised re-run → commit updated classifiers
4. After analysis → commit figures and documentation

---

## Scripts to Create

All scripts are provided below. Create them on **smokey** (they're on shared NFS), then run VAE training on **cotopaxi**.

### 1. create_filtered_vae_dataset.py

Creates permanent filtered dataset for VAE training.

### 2. train_vae_v2_6_7_filtered_cv.py

5-fold entropy-balanced CV on filtered data.

### 3. train_vae_v2_6_7_filtered_final.py

Final model trained on all filtered data.

### 4. train_clean_semisupervised_with_filtered_vae.py

Re-run semi-supervised classifiers using filtered VAE encoder.

### 5. compare_vae_filtered_vs_original.py

Compare original vs filtered VAE performance.

---

## Expected Results Comparison

### VAE Clustering Performance

| Version | Data | ARI (5-fold CV) | Status |
|---------|------|-----------------|--------|
| v2.6.7 original | 238,506 (14 classes) | 0.196 ± 0.037 | ✅ Done |
| v2.6.7 filtered | 238,359 (12 classes) | ? ± ? | ⏳ To run |

**Hypothesis:** Similar or slightly better (removing noise)

### Semi-Supervised Classification

| Encoder | Frozen | Fine-tuned |
|---------|--------|------------|
| Original v2.6.7 (14 classes) | 22.05% | 17.60% |
| Filtered v2.6.7 (12 classes) | ? | ? |

**Hypothesis:** Similar or slightly better (consistent pre-training)

---

## Success Criteria

1. ✅ **CV completes successfully** - 5 folds, no crashes
2. ✅ **ARI within expected range** - 0.15-0.25 (similar to original)
3. ✅ **Final model trains** - 100 epochs, R²>0.90
4. ✅ **Semi-supervised re-runs** - Results comparable or better
5. ✅ **Documentation updated** - Paper reflects filtered results

---

## Contingency Plans

### If ARI drops significantly (<0.15):

**Possible causes:**
- Fewer classes easier to cluster → should improve ARI, not decrease
- Dataset too homogeneous after filtering

**Action:** Investigate which classes benefit/hurt from filtering

### If training crashes on cotopaxi:

**Possible causes:**
- CUDA out of memory
- Dataset loading issues

**Actions:**
1. Check GPU memory: `nvidia-smi`
2. Reduce batch size (256 → 128)
3. Check dataset file exists and is readable

### If semi-supervised gets worse with filtered encoder:

**Possible cause:** Encoder overfits to 12 classes, less generalizable

**Action:** Document as finding - more consistent pre-training may not always help

---

## Timeline

**Assuming starting on cotopaxi:**

- **Step 1 (smokey):** Create filtered dataset - 5 minutes
- **Step 2:** Switch to cotopaxi - 2 minutes
- **Step 3 (cotopaxi):** 5-fold CV - 1-2 hours
- **Step 4 (cotopaxi):** Final training - 10 minutes
- **Step 5 (either):** Semi-supervised re-run - 15 minutes
- **Step 6 (either):** Analysis and figures - 10 minutes
- **Step 7 (either):** Commit - 5 minutes

**Total: ~2-3 hours** (mostly waiting for CV)

---

## Execution Instructions for cotopaxi

### Quick Start

```bash
# Step 1: On smokey (DONE)
# - Filtered dataset created: vae_training_data_v2_20cm_filtered_100.csv (238,359 samples)
# - Scripts ready: train_vae_v2_6_7_filtered_cv.py, train_vae_v2_6_7_filtered_final.py

# Step 2: Switch to cotopaxi
ssh cotopaxi
cd /home/utig5/johna/bhai  # NFS shared filesystem
git pull  # Get latest code
nvidia-smi  # Verify GPU available

# Step 3: Run 5-fold CV (~1-2 hours)
python3 train_vae_v2_6_7_filtered_cv.py 2>&1 | tee vae_v2_6_7_filtered_cv.log

# Check results
cat vae_v2_6_7_filtered_cv.csv  # ARI per fold
tail -50 vae_v2_6_7_filtered_cv.log  # Summary stats

# Step 4: Train final model (~10 minutes)
python3 train_vae_v2_6_7_filtered_final.py 2>&1 | tee vae_v2_6_7_filtered_final.log

# Verify checkpoint created
ls -lh ml_models/checkpoints/vae_gra_v2_6_7_filtered_final.pth

# Step 5: Optional - switch back to smokey for semi-supervised re-run (CPU sufficient)
# Or stay on cotopaxi and run there

# Step 6: Commit results (following git workflow)
git add vae_training_data_v2_20cm_filtered_100.csv \
        vae_v2_6_7_filtered_cv.py \
        vae_v2_6_7_filtered_cv.csv \
        vae_v2_6_7_filtered_cv.log \
        ml_models/checkpoints/vae_gra_v2_6_7_filtered_final.pth \
        vae_v2_6_7_filtered_final.log

git commit -m "Train VAE v2.6.7 on filtered dataset: ARI=X.XXX±0.XXX

5-fold entropy-balanced CV on filtered dataset (≥100 samples per class):
- Dataset: 238,359 samples, 12 classes (removed Ultramafic n=81, Diamict n=66)
- Architecture: 10D latent, β: 1e-10→0.75
- Results: ARI = X.XXX ± 0.XXX (vs original 0.196 ± 0.037)

Production model trained on all filtered data for consistency with
downstream classification experiments.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### Files Ready to Run

✅ **Created (on smokey):**
1. `vae_training_data_v2_20cm_filtered_100.csv` - Filtered dataset (238,359 samples, 12 classes)
2. `create_filtered_vae_dataset.py` - Dataset creation script
3. `train_vae_v2_6_7_filtered_cv.py` - 5-fold CV script
4. `train_vae_v2_6_7_filtered_final.py` - Final training script

⏳ **To create after CV completes:**
5. `train_clean_semisupervised_with_filtered_vae.py` - Re-run semi-supervised classifiers
6. `compare_vae_filtered_vs_original.py` - Analysis and comparison figures

---

## What Happens Next

After you run the CV and final training on cotopaxi, I'll help you:

1. **Analyze CV results** - Compare ARI to original v2.6.7
2. **Re-run semi-supervised classifiers** - Using new filtered encoder
3. **Generate comparison figures** - Show impact of filtering on:
   - VAE clustering performance
   - Semi-supervised classification
   - Embeddings visualization
4. **Update paper** - Revise with consistent filtered results
5. **Commit everything** - Following our git workflow

---

## Expected Timeline

| Step | Duration | Machine | Status |
|------|----------|---------|--------|
| 1. Create filtered dataset | 5 min | smokey | ✅ DONE |
| 2. Switch to cotopaxi | 2 min | --- | ⏳ TODO |
| 3. Run 5-fold CV | 1-2 hours | cotopaxi (GPU) | ⏳ TODO |
| 4. Train final model | 10 min | cotopaxi (GPU) | ⏳ TODO |
| 5. Re-run semi-supervised | 15 min | either (CPU ok) | ⏳ TODO |
| 6. Analysis & figures | 10 min | smokey | ⏳ TODO |
| 7. Commit all results | 5 min | either | ⏳ TODO |

**Total: ~2-3 hours** (mostly automated, waiting for CV)

---

## Ready to Go!

All scripts are prepared and ready to run. Filtered dataset created successfully:
- **File:** `/home/utig5/johna/bhai/vae_training_data_v2_20cm_filtered_100.csv`
- **Size:** 238,359 samples (99.94% of original)
- **Classes:** 12 (removed 2: Ultramafic, Diamict)

Just switch to **cotopaxi** and run the commands above!
