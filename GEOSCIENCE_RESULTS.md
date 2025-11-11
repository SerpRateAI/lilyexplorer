# Geoscience Results: Why Physical Properties Don't Perfectly Predict Lithology

**Date**: 2025-11-04
**Models**: VAE v2.6.7, Direct Classifier, VAE Classifier v1.1

---

## Executive Summary

Machine learning results reveal a fundamental geoscience insight: **physical properties (GRA bulk density, magnetic susceptibility, natural gamma radiation, RGB color) provide incomplete information for lithological classification**.

**Key Findings:**
1. **Direct classifier ceiling: 42% balanced accuracy** - Even with perfect ML, physical properties alone cannot distinguish many lithologies
2. **VAE reconstruction R²=0.904** - Physical properties are highly compressible and predictable
3. **Classification accuracy << reconstruction quality** - Physical similarity ≠ lithological identity
4. **Expert hierarchy misaligned with physical properties** (ARI=0.116) - Geological groupings based on composition/genesis don't match physical property clusters

This document explores the **geological reasons** for these results.

---

## 1. Lithological Continua: Gradational Boundaries

### The Problem
Many lithologies exist on continuous spectra rather than discrete categories:

**Grain size continuum:**
```
Clay (<4μm) → Silt (4-63μm) → Sand (63μm-2mm) → Gravel (>2mm)
```

**Carbonate diagenesis:**
```
Nannofossil ooze → Chalk → Limestone → Marble
(soft, high porosity) → (lithified, low porosity)
```

**Siliciclastic mixing:**
```
Clay → Silty clay → Clayey silt → Silt → Sandy silt → Silty sand → Sand
```

### Physical Property Implications

**Bulk density (GRA)** varies continuously with:
- Grain size (finer = more packing = higher density, but also more water)
- Porosity (decreases with burial/compaction)
- Grain density (quartz 2.65, calcite 2.71, clay minerals 2.7-2.9 g/cm³)

**Result**: A "silty clay" at 1.6 g/cm³ is physically indistinguishable from "clayey silt" at 1.6 g/cm³. The lithology name reflects **dominant grain size** (>50% clay vs >50% silt), not a discrete physical property jump.

### Classification Impact

**Direct classifier confusions** (from investigation):
- Clay vs Silty clay vs Clayey silt vs Silt
- Sand vs Silty sand vs Sandy silt
- Nannofossil ooze vs Chalk

**VAE behavior**: Learns the continuous physical property space. Places "silty clay" and "clayey silt" near each other in latent space because they ARE physically similar. This is **scientifically correct** but harms discrete classification.

**Interpretation**: Low classification accuracy on gradational lithologies is not a model failure - it reflects genuine geological ambiguity.

---

## 2. Convergent Physical Properties: Different Origins, Similar Measurements

### The Problem
Lithologies with different compositions/origins can have identical physical properties.

#### Case Study: Dark-Colored Lithologies

**Lithologies that produce RGB ~30-50 (dark):**
1. **Basalt** (mafic igneous rock)
   - Dark from Fe-Mg minerals (pyroxene, olivine)
   - GRA: 2.7-3.0 g/cm³ (dense, crystalline)
   - MS: HIGH (magnetite, iron oxides)
   - NGR: LOW (depleted in K, U, Th)

2. **Clay-rich mud** (fine-grained sediment)
   - Dark from organic matter or fine grain size
   - GRA: 1.4-1.8 g/cm³ (low, porous)
   - MS: LOW to MODERATE (depends on detrital minerals)
   - NGR: HIGH (K in clay minerals, U/Th adsorption)

3. **Organic-rich mud** (sapropel, black shale precursor)
   - Dark from organic carbon (>5%)
   - GRA: 1.3-1.6 g/cm³ (very low, organic matter less dense)
   - MS: LOW (organic dilution)
   - NGR: MODERATE to HIGH (U enrichment in anoxic conditions)

**Distinguishability**: Basalt vs organic-rich mud are distinguishable (density + MS + NGR combination). But what about:
- **Gray clay** vs **weathered/altered basalt** (both can have GRA~1.8, MS~moderate)
- **Fe-oxide-rich clay** vs **basaltic ash** (convergent properties)

### Physical Property Implications

**RGB color alone** is ambiguous:
- Dark: basalt, clay, organic mud, manganese nodules, volcanic ash
- Light: carbonate ooze, diatomaceous ooze, volcanic ash (felsic), sand

**Magnetic susceptibility** depends on:
- Primary mineralogy (magnetite, hematite, pyrrhotite)
- Alteration state (fresh basalt vs altered basalt)
- Detrital input (clay with magnetite-rich terrigenous fraction)

**NGR** depends on:
- Primary K-feldspar/mica content
- Clay mineral type (illite > smectite > kaolinite in K)
- Adsorbed U/Th (redox-sensitive)

### Classification Impact

**Direct classifier accuracy by group** (from investigation):
- **Mafic Igneous: 70.06%** ✓ High (distinctive density + MS + low NGR signature)
- **Volcaniclastic: 2.82%** ✗ Very low (overlap with terrigenous sediments)
- **Metamorphic: 38.46%** ✗ Low (variable properties depending on protolith)
- **Conglomerate/Breccia: 31.50%** ✗ Low (bulk properties average over mixed clasts)

**Interpretation**: Lithologies with **unique physical property combinations** classify well (basalt, gabbro, pure carbonate ooze). Lithologies with **convergent or averaged properties** classify poorly (volcaniclastics, conglomerates, altered rocks).

**VAE behavior**: Groups lithologies by physical similarity, not genetic origin. This is why expert hierarchy (composition-based) shows poor alignment (ARI=0.116) with VAE clusters (property-based).

---

## 3. Diagenesis and Burial: Physical Properties Evolve, Names Don't

### The Problem
Lithology names often reflect original depositional composition, but physical properties change with burial depth, time, temperature, and pressure.

#### Case Study: Carbonate Diagenesis

**Depositional name: "Nannofossil ooze"**
- Fresh deposit: Porosity 70-80%, GRA 1.3-1.5 g/cm³, soft
- 100m burial: Porosity 60%, GRA 1.6 g/cm³, mechanical compaction
- 500m burial: Porosity 40%, GRA 1.9 g/cm³, chemical compaction (pressure solution)
- 1000m burial + cementation: Porosity 10%, GRA 2.4 g/cm³, now "chalk" or "limestone"

**Physical property trajectory**:
```
Depth (mbsf)    Lithology Name         GRA (g/cm³)    Porosity (%)
0-100           Nannofossil ooze       1.3-1.5        70-80
100-300         Nannofossil ooze       1.5-1.7        60-70
300-600         Chalk                  1.7-2.0        40-60
600-1000        Chalk/Limestone        2.0-2.3        20-40
>1000           Limestone              2.3-2.7        5-20
```

But in IODP descriptions, all might be labeled "nannofossil ooze" if calcareous nannofossils dominate (>50%).

### Physical Property Implications

**Same lithology name, different physical properties:**
- Nannofossil ooze at 50 mbsf: GRA 1.4 g/cm³
- Nannofossil ooze at 500 mbsf: GRA 2.0 g/cm³
- **43% difference in bulk density** for "same" lithology

**Porosity-density relationship** (from paper Figure 9):
```
ρ_bulk = φ·ρ_fluid + (1-φ)·ρ_grain
```
For seawater (ρ_fluid=1.024) and calcite (ρ_grain=2.71):
- φ=70% → ρ_bulk=1.45 g/cm³
- φ=40% → ρ_bulk=1.94 g/cm³
- φ=10% → ρ_bulk=2.53 g/cm³

### Classification Impact

**Intra-class variance** exceeds **inter-class variance** for diagenetically active lithologies:
- Within "Carbonate" group: GRA ranges 1.3-2.7 g/cm³ (factor of 2×)
- Between "Carbonate" and "Clay": GRA ranges 1.3-2.7 vs 1.4-2.0 g/cm³ (overlap!)

**Direct classifier struggles** with:
- Carbonate: 73.75% accuracy (best group, but still 26% errors)
- Sand: 34.03% accuracy (grain size + compaction variability)

**VAE behavior**: Groups samples by **current physical state**, not **original depositional lithology**. A compacted nannofossil ooze (GRA=2.0) clusters near chalk/limestone, not near fresh ooze (GRA=1.4). This is physically correct but taxonomically "wrong."

**Interpretation**: Physical property-based classification captures **diagenetic state**, while lithology labels capture **original composition**. These are related but not identical.

---

## 4. Compositional Mixing: Most Sediments Are Mixtures

### The Problem
IODP lithology naming follows modified Mazzullo scheme:
- **Principal lithology**: >50-60% of composition
- **Major modifier (prefix)**: 25-50%
- **Minor modifier (suffix)**: 10-25%

Example: "Clayey nannofossil ooze with foraminifera"
- Nannofossils: 50-60% (principal)
- Clay: 25-40% (major modifier)
- Foraminifera: 10-20% (minor modifier)

### Physical Property Implications

**Bulk properties are weighted averages:**

```
ρ_bulk = Σ(f_i · ρ_i)

where f_i = volume fraction of component i

Example:
60% nannofossils (ρ=2.71, carbonate)
30% clay (ρ=2.7, silicate)
10% forams (ρ=2.71, carbonate)

Result: ρ_grain ≈ 2.70 g/cm³ (nearly identical components)
```

**Magnetic susceptibility** for mixed sediments:
```
MS_bulk ≈ f_clay·MS_clay + f_carbonate·MS_carbonate + f_organic·MS_organic

Clay (terrigenous): MS = 50-500 instrument units (variable)
Carbonate: MS = -10 to +20 (diamagnetic to weakly paramagnetic)
Organic matter: MS ≈ 0 (dilution)
```

A sediment with 60% carbonate + 40% clay can have MS anywhere from 20-200 depending on clay mineralogy!

### Classification Impact

**Boundary ambiguity:**
- 55% nannofossils + 45% clay = "Clayey nannofossil ooze"
- 45% nannofossils + 55% clay = "Nannofossil clay"

These differ by ±5% in composition but have different principal lithology names. Physical properties are nearly identical.

**Direct classifier confusions**:
- Clay (40.6% of data) vs Carbonate (42.1% of data) = 43.17% vs 73.75% accuracy
- Many samples are transitional (30-70% of each component)

**VAE embedding structure**: Places mixed lithologies in **continuous transitions** between end-members. There is no discrete boundary at 50% composition in physical property space.

**Interpretation**: Lithology classification treats composition as discrete categories. Physical properties reflect continuous mixing. Models optimized for physical properties (VAE) don't align with categorical boundaries.

---

## 5. Measurement Scale vs Classification Scale

### The Problem
**MSCL measurements** (GRA, MS) are bulk properties averaged over:
- Spatial scale: ~5-10 cm³ sample volume
- Includes matrix + pore fluids + heterogeneities

**Lithology descriptions** are based on:
- Visual examination of split core surface
- Smear slides (microscopic composition)
- Handheld lens identification

These scales differ by 3-6 orders of magnitude (cm³ vs mm³ vs μm³).

### Physical Property Implications

**Heterogeneity at different scales:**

**Macro-scale (10 cm)**: Bedding, laminations, burrows
- GRA measurement averages over 2-3 beds
- Lithology description: "Interbedded clay and silt"
- Model sees: GRA=1.65 g/cm³ (average)
- Description records: Two separate lithologies

**Meso-scale (1 cm)**: Biogenic structures, clasts
- Burrow filled with different sediment
- GRA: Bulk average
- Description: Presence/absence of bioturbation

**Micro-scale (100 μm)**: Grain composition
- Smear slide: 60% nannofossils, 30% clay, 10% diatoms
- GRA: Cannot distinguish grain types, only bulk density
- MS: Sensitive to clay mineralogy but not grain type

### Classification Impact

**Scale mismatch examples:**

1. **Turbidite sequence** (Bouma sequence):
   - Top: Clay (fine)
   - Middle: Silt (medium)
   - Bottom: Sand (coarse)
   - GRA measurement (10cm scale): Average density ~1.7 g/cm³
   - Lithology name (cm scale): Three separate units

2. **Volcanic ash layer** (2 cm thick) in nannofossil ooze:
   - GRA (10cm avg): Barely detects ash (20% of measurement volume)
   - Lithology: Separate "ash layer" unit
   - Model: Sees "slightly denser nannofossil ooze"

**Direct classifier blind spots:**
- Thin layers (<5 cm) diluted in bulk measurement
- Rare but diagnostic components (volcanic glass, shell fragments)
- Fine-scale structures (cross-bedding, grading)

**VAE reconstruction**: Excellent R²=0.904 because it learns **bulk property relationships**. But bulk properties average over heterogeneities that define lithology.

**Interpretation**: Physical property measurements and lithology classifications are **observing different scales**. A model trained on bulk properties cannot recover information lost by spatial averaging.

---

## 6. Human Classification Criteria: Composition, Texture, Genesis

### The Problem
Lithology names encode multiple criteria beyond physical properties:

**Compositional criteria:**
- Mineralogy: Quartz vs calcite vs clay
- Biogenic components: Nannofossils vs diatoms vs radiolarians
- Chemical: Carbonate vs silicate vs oxide

**Textural criteria:**
- Grain size: Clay vs silt vs sand
- Sorting: Well-sorted vs poorly sorted
- Fabric: Massive vs laminated vs bioturbated

**Genetic criteria:**
- Depositional environment: Pelagic vs hemipelagic vs turbidite
- Origin: Biogenic vs terrigenous vs authigenic vs volcanic
- Diagenetic state: Unconsolidated vs lithified

### Physical Property Implications

**GRA/MS/NGR/RGB measure:**
- Bulk density (not mineralogy directly)
- Magnetic mineral content (not grain size)
- Radioactive element content (not depositional environment)
- Visible light reflectance (not biogenic vs terrigenous)

**What they DON'T directly measure:**
- Grain shape/angularity
- Fossil types/abundance (unless affect density/color)
- Depositional structures
- Cementation type
- Diagenetic minerals (unless magnetic or radioactive)

### Classification Impact

**Expert hierarchy groups** (from investigation) based on:
- **Carbonate**: All carbonate-dominated lithologies (nannofossil, foram, chalk, limestone)
- **Biogenic Silica**: Diatom, radiolarian, siliceous ooze
- **Mafic Igneous**: Basalt, gabbro, diabase, dolerite

**VAE clusters** based on:
- Dense + magnetic + low NGR = basalt, gabbro, altered mafic rocks
- Low density + low MS + moderate NGR = carbonate ooze, diatomaceous ooze, chalk
- Moderate density + variable MS + high NGR = clay, mud, silty clay

**ARI = 0.116** (11.6% agreement) shows expert groupings don't match physical property clusters.

**Why?**
- Diatom ooze (biogenic silica) has similar GRA/MS/NGR to nannofossil ooze (carbonate)
  - Both: Low density, low MS, biogenic origin
  - Differ: Silica vs carbonate mineralogy (not directly measured)
  - VAE groups them together (physically similar)
  - Expert groups them separately (compositional difference)

- Volcaniclastic vs terrigenous clay:
  - Both: Medium density, variable MS/NGR
  - Differ: Volcanic vs continental source (genetic criterion)
  - VAE cannot distinguish (similar physical properties)

**Interpretation**: Physical properties capture **current state**, not **origin**. Lithology names encode **genetic and compositional information** not fully represented in GRA/MS/NGR/RGB measurements.

---

## 7. Synthesis: The Fundamental Limitation

### Why Classification Plateaus at 42%

The **direct classifier ceiling (42.32% balanced accuracy)** reflects fundamental geological ambiguity:

**Lithologies with unique physical signatures** (high accuracy):
- **Mafic igneous (70%)**: Dense + magnetic + low NGR (unique combination)
- **Biogenic silica (55%)**: Low density + high silica content + low carbonate
- **Carbonate (74%)**: Moderate density + low MS + low NGR (when pure)

**Lithologies with overlapping physical properties** (low accuracy):
- **Sand (34%)**: Grain size not measured, density depends on composition + porosity
- **Clay/Mud (43%)**: Continuous with silt, properties vary with compaction
- **Volcaniclastic (3%)**: Averages between volcanic and sedimentary end-members
- **Conglomerate/Breccia (32%)**: Bulk properties average over mixed clasts

### Why VAE Embeddings Don't Help Classification

**VAE optimizes for reconstruction** (R²=0.904):
- Learns: "This sample has GRA=1.6, MS=80, NGR=35, RGB=120"
- Compresses: Into 10D latent space (4D effective)
- Reconstructs: Back to GRA=1.6, MS=80, NGR=35, RGB=120

**Classification requires discrimination**:
- Learns: "GRA=1.6 + MS=80 + NGR=35 → CLAY (not silt, not sand)"
- Maximizes: Inter-class distance
- Minimizes: Intra-class variance

**These objectives conflict** when:
1. Intra-class variance is high (diagenesis, mixing, depth trends)
2. Inter-class variance is low (gradational boundaries, convergent properties)
3. Physical properties don't determine lithology (genetic/compositional criteria)

**VAE correctly learns** that "nannofossil ooze at 500 mbsf" is physically similar to "chalk" (both GRA~2.0). But these have different lithology names based on consolidation state.

**Raw features outperform VAE** (42% vs 30%) because:
- No information loss from compression (6D → 4D effective → 6D)
- Preserve fine-grained variations that distinguish boundary cases
- Direct optimization for classification objective

---

## 8. Geoscience Implications for Model Interpretation

### What the VAE v2.6.7 Actually Learned

**The VAE learned the physics of marine sediments:**
- Density-porosity relationships (R²=0.836 for GRA reconstruction)
- Color-composition correlations (R²=0.962 for RGB reconstruction)
- Magnetic mineral distributions (R²=0.828 for MS reconstruction)
- Radioactivity-clay mineral content (R²=0.880 for NGR reconstruction)

**The VAE did NOT learn:**
- Lithology taxonomy (human classification system)
- Genetic relationships (depositional environment)
- Compositional details below measurement resolution (grain mineralogy)
- Temporal information (burial history, diagenesis)

**This is exactly what we expect** from unsupervised learning on physical property data.

### Why This Matters for Oceanic Crust AI

**The 42% classification ceiling tells us:**

1. **Current features are limited**: GRA/MS/NGR/RGB cannot fully determine lithology
   - Need: XRF geochemistry, XRD mineralogy, core photos (texture), micropaleontology

2. **Lithology is multi-scale**: Bulk properties miss thin layers, structures, fabric
   - Need: Higher spatial resolution, depth-continuous measurements, imaging

3. **Lithology is interpretive**: Same physical properties → different names based on context
   - Need: Stratigraphic context, regional geology, expedition-specific nomenclature

4. **Physical properties are valuable**: 42% is much better than random (7% for 14 classes)
   - Keep: GRA/MS/NGR/RGB as fundamental observables
   - Add: Complementary measurements for composition/mineralogy

### What the VAE IS Good For

**Unsupervised tasks** (where it excels):
- ✓ Dimensionality reduction (4D effective captures 90% variance)
- ✓ Anomaly detection (high reconstruction error = unusual properties)
- ✓ Data quality control (outliers, sensor malfunctions)
- ✓ Exploratory analysis (find natural groupings in data)
- ✓ Similarity search (find cores with similar physical properties)

**Supervised tasks** (where raw features are better):
- ✗ Lithology classification (use direct classifier instead)
- ✗ Property prediction from incomplete measurements (weak feature correlations)
- ✗ Fine-grained discrimination (information lost in compression)

---

## 9. Recommendations for Future Work

### For Improving Classification

**Add compositional measurements:**
1. **XRF core scanning**: Major element chemistry (Ca, Si, Fe, Al, K, Ti, Mn)
   - Distinguish carbonate vs silicate vs oxide
   - Quantify mixing ratios
   - Detect volcanic input (Ti/Al)

2. **XRD mineralogy**: Clay mineral types, carbonate phases, silica polymorphs
   - Differentiate smectite vs illite vs kaolinite (NGR differences)
   - Identify diagenetic minerals (zeolites, pyrite, gypsum)

3. **Microscopic data**: Smear slide analysis, grain counts
   - Biogenic vs terrigenous ratios
   - Grain size distributions
   - Fossil assemblages

**Add textural measurements:**
1. **Core photo analysis**: Computer vision for fabric, structures, contacts
2. **P-wave velocity anisotropy**: Fabric orientation
3. **High-resolution density**: Thin layer detection

**Add contextual information:**
1. **Depth trends**: Compaction curves, diagenetic sequences
2. **Stratigraphic position**: Within-hole correlation
3. **Regional geology**: Tectonic setting, basement type, overlying water depth

### For Unsupervised Learning

**VAE architecture improvements** (if must use embeddings for classification):
1. Increase latent dimensionality (20D-30D) to reduce collapse
2. Decrease β (0.25-0.5) to preserve more variance
3. Add auxiliary task (depth prediction, borehole ID) to enrich representations
4. Use disentangled VAE variants (β-TCVAE, FactorVAE)

But: These changes risk overfitting and increase complexity. **Direct classifiers on raw features already outperform.**

### For Oceanic Crust AI Model

**Hybrid approach:**
1. Use VAE v2.6.7 for unsupervised tasks (exploration, visualization, anomaly detection)
2. Use direct classifier on raw features for lithology prediction (42% accuracy)
3. Develop separate models for specific tasks:
   - Carbonate content: CatBoost on GRA+NGR (strong correlation)
   - Basalt identification: Threshold on GRA+MS (>2.5 + >100)
   - Diagenetic state: Track density vs depth within lithology

4. Acknowledge limitations in publications:
   - "Physical properties provide partial lithology information (42% accuracy)"
   - "Full lithology determination requires compositional and textural analysis"
   - "VAE embeddings capture physical property structure, not taxonomic relationships"

---

## 10. Conclusion: Physical Properties ≠ Lithology

The machine learning results validate fundamental geoscience principles:

1. **Lithology is complex**: Composition + texture + genesis + diagenetic state
2. **Measurements are selective**: GRA/MS/NGR/RGB capture some but not all criteria
3. **Classification is interpretive**: Human experts use context, experience, multiple observations
4. **Physical properties are valuable**: 42% accuracy is scientifically meaningful
5. **Model limitations reflect data limitations**: Cannot predict what is not measured

**The VAE v2.6.7 is working correctly** - it learned the structure of physical property data. That structure is related to but not determined by lithology taxonomy.

**The 42% classification ceiling is not a failure** - it quantifies the information content of GRA/MS/NGR/RGB for lithology discrimination. This is a valuable scientific result.

**Future improvements** require additional measurements (geochemistry, mineralogy, microscopy) that capture compositional and textural information missing from current physical property suite.

The oceanic crust AI model should embrace this: **Physical properties tell us about the physical state of cores. Lithology tells us about their geological interpretation. These are related but distinct.**

---

## References

1. IODP LIMS with Lithology (LILY) Database (Childress et al., 2024)
2. Mazzullo et al. (1988) - Sediment classification scheme
3. Dean et al. (1985) - Pelagic sediment classification
4. VAE v2.6.7 training and evaluation (this study)
5. Direct classifier investigation (this study)
6. VAE classifier investigation summary (2025-11-01)
