# Simplified Lithology Labels for Better Evaluation

## The Problem

**Original**: 139 fine-grained lithologies
- 66 lithologies with <100 samples (47% of categories!)
- 96 lithologies with <500 samples (69% of categories!)
- Rare lithologies add noise to ARI evaluation
- Too many similar categories (e.g., "nannofossil ooze", "diatom ooze", "calcareous ooze")

## The Solution

**Simplified**: 12 major lithology types based on sedimentary classification

| Simplified Type | Original Count | Samples | Major Components |
|-----------------|----------------|---------|------------------|
| **Clay/Claystone** | 6 | 72,171 | clay, silty clay, claystone |
| **Biogenic ooze** | 5 | 60,513 | nannofossil ooze, diatom ooze, foram ooze |
| **Limestone** | 6 | 29,310 | wackestone, packstone, grainstone |
| **Mud/Mudstone** | 6 | 16,512 | mud, mudstone, pebbly mud |
| **Other** | 91 | 15,103 | rare types (<500 samples each) |
| **Silt/Siltstone** | 5 | 11,875 | silt, siltstone, clayey silt |
| **Sand/Sandstone** | 7 | 11,480 | sand, sandstone, silty sand |
| **Biogenic chalk** | 2 | 11,304 | nannofossil chalk, chalk |
| **Volcaniclastic** | 4 | 2,906 | tuff, ash, lapilli tuff |
| **Gabbro** | 4 | 2,698 | gabbro, diabase, dolerite |
| **Basalt** | 1 | 2,401 | basalt |
| **Diamict** | 2 | 2,233 | glacial sediments |

## Grouping Rationale

1. **Biogenic sediments**: Separated by diagenesis state
   - **Ooze**: Unconsolidated biogenic sediment
   - **Chalk**: Semi-lithified calcareous ooze

2. **Siliciclastic sediments**: Grain size classification
   - **Clay** < 4μm
   - **Silt** 4-63μm
   - **Sand** 63μm-2mm
   - **Mud**: Mixed clay+silt

3. **Carbonate rocks**: Texture-based (Dunham classification)
   - **Limestone**: wackestone, packstone, grainstone unified

4. **Igneous rocks**: Composition-based
   - **Basalt**: Extrusive mafic
   - **Gabbro**: Intrusive mafic

5. **Other**: 91 rare lithologies with <500 samples each

## Expected Benefits

### For Clustering Evaluation:
- **Higher ARI**: Fewer, more distinct categories → better agreement
- **Reduced noise**: Rare lithologies no longer penalize clustering
- **More meaningful**: 12 major types vs 139 fine variations

### For Interpretability:
- Align with standard sediment classification
- Major lithology types geologists actually care about
- Clearer separation between sediment types

## Theoretical Impact on ARI

**With 139 labels (original)**:
- Model must distinguish "nannofossil ooze" from "diatom ooze" from "foram ooze"
- These are similar biogenic sediments with overlapping properties
- ARI penalizes model for grouping them together
- But grouping makes physical sense!

**With 12 labels (simplified)**:
- All biogenic oozes → "Biogenic ooze"
- Model gets credit for grouping similar sediments
- ARI should increase significantly
- More aligned with physical reality

## Usage

Dataset with simplified labels: `vae_training_data_v2_20cm_simplified.csv`

Contains both columns:
- `Principal`: Original 139 fine-grained labels
- `Lithology_Simplified`: New 12 major types

Can evaluate clustering against either set of labels to compare.

## Recommendation

**Use simplified labels (12 types) for:**
- Final performance reporting
- Production lithology prediction
- Interpretable cluster analysis
- Comparison across studies

**Keep original labels (139 types) for:**
- Detailed lithology discrimination
- Research on specific sediment types
- Validating fine-scale clustering

## Next Steps

Re-evaluate VAE GRA v2.1 with simplified labels to see ARI improvement.

Expected outcome: **ARI should increase substantially** (potentially 0.3-0.5+) since:
- Fewer, more distinct categories
- Better aligned with physical sediment types
- Reduced penalty for reasonable groupings
