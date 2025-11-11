# Fuzzy Matching Tolerance Experiment

**Date**: 2025-11-02
**Status**: Running
**Goal**: Extract more data from existing 296 boreholes using relaxed depth alignment

## Motivation

Current VAE v2.6.7 dataset uses **exact 20cm binning**:
- 238,506 samples from 296 boreholes
- Requires all 4 measurements (GRA+MS+NGR+RGB) at same 20cm bin
- This is likely too strict - measurements don't align perfectly

**Discovery from missing data analysis**:
- We're using ALL 296/304 boreholes with RGB camera (97% coverage)
- RGB camera is the bottleneck (only 304/534 IODP boreholes have RGB)
- **Cannot get more boreholes** without losing RGB
- **CAN potentially get more samples from existing 296 boreholes**

## Hypothesis

By allowing **fuzzy depth matching** (measurements within ±tolerance of bin center), we can extract more samples without adding new boreholes.

**Trade-off**:
- Larger tolerance → More samples (good for training)
- Larger tolerance → Measurements farther apart (may span lithological boundaries)

**Question**: What tolerance maximizes data while maintaining clustering performance (ARI)?

## Experiment Design

### Tolerances Tested

| Tolerance | Description | Geological Rationale |
|-----------|-------------|---------------------|
| ± 5 cm | Tight | Measurement precision only |
| ± 10 cm | Conservative | Measurement precision + minor core disturbance |
| ± 20 cm | Baseline | Current bin size (exact matching) |
| ± 30 cm | Moderate | Within typical sampling resolution |
| ± 50 cm | Relaxed | Usually within same lithological unit |
| ± 1 m | Loose | May span thin beds |
| ± 2 m | Very loose | Likely spans multiple units |
| ± 3 m | Extreme | Definitely spans boundaries |
| ± 5 m | Maximum | Far beyond single lithological unit |

### Method

For each 20cm depth bin center:
1. Find measurements within ±tolerance
2. Take the **closest measurement** within tolerance for each feature
3. Require ALL 4 measurements (GRA, MS, NGR, RGB) within tolerance
4. Use lithology from closest GRA measurement

### Implementation

**Script**: `test_fuzzy_matching_tolerances.py`

**Steps**:
1. Quick test on 20 sample boreholes (estimate potential gain)
2. Create full datasets for promising tolerances (± 10cm, 20cm, 50cm, 1m, 2m)
3. Save as `vae_training_data_fuzzy_{tolerance}cm.csv`
4. Later: Train VAE models and compare ARI

## Expected Results

### Sample Count Predictions

Based on measurement spacing (GRA ~2-5cm, RGB ~0.5cm):

| Tolerance | Expected Samples | vs Baseline | Notes |
|-----------|------------------|-------------|-------|
| ± 5 cm | ~240K | +1% | Minimal gain |
| ± 10 cm | ~260K | +9% | Modest gain |
| ± 20 cm | 238K | Baseline | Current exact matching |
| ± 50 cm | ~320K | +34% | Significant gain |
| ± 1 m | ~420K | +76% | Large gain |
| ± 2 m | ~550K | +131% | Very large gain |
| ± 5 m | ~700K+ | +194% | Maximum gain |

**Note**: These are rough estimates. Actual numbers depend on measurement density per borehole.

### Performance Predictions

**Optimistic scenario**: ARI improves with more data
- More samples → Better statistics
- Averaging within tolerance reduces noise
- Model learns better representations

**Pessimistic scenario**: ARI degrades with large tolerance
- Measurements 5m apart may be different lithologies
- Adds label noise (mixing units)
- Model learns spurious correlations

**Most likely**: Sweet spot around ±50cm to ±1m
- Enough samples to improve training
- Small enough to stay within units
- Balance sample size vs data quality

## Geological Considerations

### What tolerance is "reasonable"?

**Core disturbance**: ±5-10cm
- Drilling can compress/expand cores
- Section breaks may have gaps
- Measurement precision ~1cm

**Lithological unit thickness**:
- Thin beds: 10-50 cm
- Typical layers: 0.5-2 m
- Thick units: >2 m

**MSCL measurement spacing**:
- GRA: 2-5 cm (continuous)
- MS: 2-5 cm (continuous)
- NGR: 10-20 cm (less frequent)
- RGB: 0.5 cm (very dense)

### Geologist concerns

**"You can't connect rocks 5m apart!"**
- Valid concern: 5m definitely spans multiple lithological units
- But: Empirical test is needed - does ARI actually degrade?
- Maybe: Extra data outweighs label noise at model level

**"Measurements should be co-located"**
- Ideal: Yes, exact co-location is best
- Reality: MSCL sensors have different spacing
- Compromise: Fuzzy matching balances precision vs sample size

## Next Steps

1. **Wait for dataset creation** (script running)
2. **Examine sample counts** for each tolerance
3. **Train VAE models** on fuzzy-matched datasets
   - Use same architecture as v2.6.7
   - Same β annealing (1e-10 → 0.75)
   - Same training protocol
4. **Compare ARI performance**
   - Plot ARI vs tolerance
   - Find optimal tolerance
5. **Select best tolerance** based on ARI/sample trade-off

## Success Criteria

**Experiment succeeds if**:
- We find tolerance with +20% samples AND equal/better ARI
- Example: ± 50cm with 320K samples and ARI ≥ 0.196

**Experiment fails if**:
- All tolerances > 20cm degrade ARI
- Extra samples don't compensate for label noise

**Acceptable outcome**:
- Confirm ± 20cm (current) is optimal
- Document that fuzzy matching doesn't help
- Provides empirical justification for current approach

## Files Generated

**Datasets** (will be created):
- `vae_training_data_fuzzy_10cm.csv` - ± 10cm tolerance
- `vae_training_data_fuzzy_20cm.csv` - ± 20cm (baseline)
- `vae_training_data_fuzzy_50cm.csv` - ± 50cm tolerance
- `vae_training_data_fuzzy_100cm.csv` - ± 1m tolerance
- `vae_training_data_fuzzy_200cm.csv` - ± 2m tolerance

**Logs**:
- `fuzzy_matching_experiment.log` - Dataset creation output

**Training scripts** (to be created):
- `train_vae_fuzzy_comparison.py` - Train models on all tolerances

**Results** (to be generated):
- `fuzzy_tolerance_comparison.csv` - ARI vs tolerance results
- `fuzzy_tolerance_analysis.png` - Performance visualization

## Timeline

- **Dataset creation**: ~30-60 minutes (running now)
- **VAE training**: ~2 hours per tolerance × 5 tolerances = 10 hours total
  - Can run in parallel on 4 A100 GPUs (cotopaxi)
  - Actual time: ~3 hours with parallelization
- **Analysis**: ~30 minutes

**Total**: ~4-5 hours for complete experiment

## Related Work

**Previous experiments that failed**:
- v2.6.1 (RSC+MSP): +44% data but -54% ARI (feature quality > quantity)
- v2.6.2 (pre-training): +228 boreholes but -51% ARI (transfer learning fails)

**Why this might work**:
- Same features (GRA+MS+NGR+RGB), just relaxed alignment
- Same boreholes, just more depth bins
- Not changing measurement types or transfer learning

**Key difference**: We're adding MORE SAMPLES from SAME BOREHOLES with SAME FEATURES, just relaxing depth precision. This is fundamentally different from adding different features or different boreholes.
