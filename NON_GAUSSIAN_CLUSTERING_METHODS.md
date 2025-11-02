# Clustering Methods for Non-Gaussian Data

## Overview

Most classical clustering algorithms assume Gaussian (normal) distributions, but real-world data often violates this assumption. When your data has:
- Heavy tails (extreme values)
- Skewed distributions
- Multimodal clusters
- Non-elliptical shapes
- Arbitrary cluster geometries

...you need clustering methods designed for **non-Gaussian data**.

---

## 1. Density-Based Methods

### **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

**Key Idea:** Clusters are dense regions separated by low-density regions.

**How it works:**
- Define two parameters: `eps` (neighborhood radius) and `min_samples`
- **Core points**: Points with ≥ `min_samples` neighbors within `eps`
- **Border points**: Non-core points within `eps` of a core point
- **Noise**: Points that are neither core nor border

**Advantages:**
- ✓ Finds arbitrary-shaped clusters
- ✓ Identifies outliers/noise (label -1)
- ✓ No need to specify number of clusters
- ✓ Works with non-Gaussian distributions

**Disadvantages:**
- ✗ Struggles with varying densities (one `eps` for all clusters)
- ✗ High-dimensional data (curse of dimensionality affects density)
- ✗ Sensitive to parameter choice

**When to use:**
- Clusters have varying shapes (not spherical/elliptical)
- Need outlier detection
- Clusters have similar density

**Python:**
```python
from sklearn.cluster import DBSCAN

clusterer = DBSCAN(eps=0.5, min_samples=10)
labels = clusterer.fit_predict(X)
```

---

### **HDBSCAN (Hierarchical DBSCAN)**

**Key Idea:** Build a hierarchy of clusterings at different densities, then extract optimal clusters.

**How it works:**
1. Build minimum spanning tree based on mutual reachability distance
2. Create hierarchy of clusters at all density levels
3. Extract "persistent" clusters using stability measure

**Advantages:**
- ✓ All DBSCAN advantages
- ✓ **Handles varying densities** (major improvement)
- ✓ Automatic cluster extraction
- ✓ More robust to parameter choice

**Disadvantages:**
- ✗ Slower than DBSCAN (hierarchical construction)
- ✗ Still struggles in very high dimensions
- ✗ May label many points as noise if clusters are not well-separated

**When to use:**
- Clusters have different densities
- Unsure about optimal `eps` parameter
- Need robust outlier detection

**Why it failed for VAE latent space:**
- Latent clusters have **uniform density** (no gradients to exploit)
- VAE already "clustered" data → compact, similar-density regions
- 26% noise classification loses signal (those points still have lithology!)

**Python:**
```python
import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=10)
labels = clusterer.fit_predict(X)
```

---

## 2. Distribution-Based Methods

### **Gaussian Mixture Models (GMM)**

**Key Idea:** Data generated from mixture of K Gaussian distributions with different parameters.

**How it works:**
- Assume each cluster is a Gaussian with mean μ_k and covariance Σ_k
- EM algorithm finds optimal parameters
- Soft assignments: P(cluster k | point x)

**Covariance Types:**
- **`full`**: Each cluster has its own general covariance (elliptical, any orientation) ← **Most flexible**
- **`tied`**: All clusters share same covariance (parallel ellipses)
- **`diag`**: Diagonal covariance (axes-aligned ellipses)
- **`spherical`**: Single variance parameter (spherical, like K-Means but probabilistic)

**Advantages:**
- ✓ **Elliptical clusters** (not just spherical)
- ✓ **Soft assignments** (probabilistic)
- ✓ Different cluster sizes/variances
- ✓ Model selection via BIC/AIC

**Disadvantages:**
- ✗ **Still assumes Gaussian components** (limitation for heavy-tailed data)
- ✗ Sensitive to initialization
- ✗ Can overfit with too many components

**When to use:**
- Clusters are roughly elliptical
- Need probabilistic assignments
- Data is approximately Gaussian (even if clusters aren't)

**Why it worked better than K-Means for VAE:**
- VAE latent clusters are **elliptical** (different variances in different directions)
- `full` covariance captures elongated cluster shapes
- +13% improvement shows clusters aren't perfectly spherical

**Python:**
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=12, covariance_type='full')
labels = gmm.fit_predict(X)
```

---

### **t-Distributed Mixture Models**

**Key Idea:** Like GMM, but use **Student's t-distribution** instead of Gaussian.

**Why Student's t?**
- **Heavy tails** → robust to outliers
- **Degrees of freedom (ν)** controls tail heaviness
  - ν→∞: Approaches Gaussian
  - ν=1: Cauchy distribution (very heavy tails)
  - ν=3-5: Common choice for robust clustering

**Advantages:**
- ✓ **Robust to outliers** (heavy tails absorb extreme points)
- ✓ Elliptical clusters (like GMM)
- ✓ Better for non-Gaussian data with outliers

**Disadvantages:**
- ✗ More parameters to estimate (ν for each component)
- ✗ Slower than GMM
- ✗ Harder to implement (not in sklearn)

**When to use:**
- Data has outliers/heavy tails
- GMM overfits to extreme points
- Robust clustering needed

**Python (requires external package):**
```python
# Option 1: pomegranate
from pomegranate import GeneralMixtureModel, MultivariateGaussianDistribution

# Option 2: scikit-learn-extra
from sklearn_extra.cluster import KMedoids  # Not t-mixture, but robust alternative
```

---

### **Skew-Normal Mixture Models**

**Key Idea:** Allow **asymmetric (skewed) distributions** in each cluster.

**Why skew-normal?**
- Gaussian assumes symmetry (mean = median = mode)
- Real data often has skewness (tail on one side)
- Skew-normal has **skewness parameter α**:
  - α=0: Standard Gaussian
  - α>0: Right-skewed
  - α<0: Left-skewed

**Advantages:**
- ✓ Handles skewed clusters
- ✓ More flexible than GMM
- ✓ Reduces misclassification when clusters overlap asymmetrically

**Disadvantages:**
- ✗ More complex (additional skewness parameters)
- ✗ Not standard in sklearn
- ✗ Harder to interpret

**When to use:**
- Clusters have visible skewness
- GMM forces symmetric clusters poorly
- Asymmetric overlaps between clusters

---

## 3. Non-Parametric Methods

### **Mean Shift**

**Key Idea:** Find modes (peaks) in the density function by iteratively shifting points toward higher density.

**How it works:**
1. For each point, compute mean of neighbors within bandwidth `h`
2. Shift point to that mean
3. Repeat until convergence
4. Points converging to same mode = same cluster

**Advantages:**
- ✓ **No assumption about cluster shape**
- ✓ Automatic number of clusters (finds all modes)
- ✓ Works with arbitrary distributions
- ✓ No random initialization

**Disadvantages:**
- ✗ **Very slow** (O(n²) for n points)
- ✗ Sensitive to bandwidth `h`
- ✗ Doesn't scale to high dimensions

**When to use:**
- Small-medium datasets
- Arbitrary cluster shapes
- Don't know number of clusters

**Python:**
```python
from sklearn.cluster import MeanShift

clusterer = MeanShift(bandwidth=0.5)
labels = clusterer.fit_predict(X)
```

---

### **Spectral Clustering**

**Key Idea:** Transform data into graph, then cluster in **spectral space** (eigenvectors of graph Laplacian).

**How it works:**
1. Build similarity graph (k-nearest neighbors or ε-neighborhoods)
2. Compute graph Laplacian matrix
3. Find k smallest eigenvectors
4. Run K-Means in eigenvector space

**Advantages:**
- ✓ **Non-convex cluster shapes** (works with "moons", "circles", etc.)
- ✓ Only needs similarity measure (not distances)
- ✓ Effective for non-Gaussian manifold data

**Disadvantages:**
- ✗ Need to specify k (number of clusters)
- ✗ Sensitive to similarity graph construction
- ✗ Computationally expensive (eigendecomposition)
- ✗ Doesn't scale well (O(n³))

**When to use:**
- Clusters are connected manifolds (not necessarily convex)
- Have good similarity/kernel function
- Moderate dataset size

**Python:**
```python
from sklearn.cluster import SpectralClustering

clusterer = SpectralClustering(n_clusters=12, affinity='nearest_neighbors')
labels = clusterer.fit_predict(X)
```

---

## 4. Subspace/Projected Clustering

### **Subspace Clustering**

**Key Idea:** Different clusters may exist in **different subspaces** of the feature space.

**Examples:**
- **CLIQUE**: Grid-based density in axis-parallel subspaces
- **SUBCLU**: DBSCAN in subspaces
- **PROCLUS**: Projected clustering with local dimensionality

**Why important:**
- High-dimensional data: clusters may only be meaningful in some dimensions
- Different features matter for different clusters

**Advantages:**
- ✓ Handles high-dimensional data
- ✓ Finds clusters in local subspaces
- ✓ Avoids curse of dimensionality

**Disadvantages:**
- ✗ Computationally expensive
- ✗ Many parameters
- ✗ Hard to interpret (which subspace?)

---

## 5. Model-Free Robust Methods

### **K-Medoids (PAM - Partitioning Around Medoids)**

**Key Idea:** Like K-Means, but use **medoids** (actual data points) as cluster centers instead of means.

**How it works:**
- Medoid = most central point in cluster (minimizes sum of dissimilarities)
- Robust to outliers (medoid is actual point, not affected by extremes)
- Can use any distance metric (not just Euclidean)

**Advantages:**
- ✓ **Robust to outliers** (medoids not affected by extreme values)
- ✓ Works with any distance metric
- ✓ Interpretable centers (actual data points)

**Disadvantages:**
- ✗ Still assumes spherical-ish clusters
- ✗ Slower than K-Means (O(n²) vs O(n))
- ✗ Doesn't handle arbitrary shapes

**When to use:**
- Outliers present
- Need interpretable centers (real samples)
- Non-Euclidean distance (e.g., Manhattan, cosine)

**Python:**
```python
from sklearn_extra.cluster import KMedoids

clusterer = KMedoids(n_clusters=12, metric='euclidean')
labels = clusterer.fit_predict(X)
```

---

### **Affinity Propagation**

**Key Idea:** Clusters emerge from **message passing** between points about their suitability as exemplars.

**How it works:**
- Each point sends "responsibility" messages (how well-suited to be exemplar)
- Each point sends "availability" messages (how appropriate to choose as exemplar)
- Iterate until convergence
- Exemplars = cluster centers

**Advantages:**
- ✓ **Automatic number of clusters**
- ✓ No initialization
- ✓ Works with any similarity measure

**Disadvantages:**
- ✗ Very slow (O(n²) iterations)
- ✗ Memory intensive (similarity matrix)
- ✗ Can be unstable (oscillations)

**When to use:**
- Don't know number of clusters
- Have good similarity measure
- Small-medium datasets

**Python:**
```python
from sklearn.cluster import AffinityPropagation

clusterer = AffinityPropagation(damping=0.9)
labels = clusterer.fit_predict(X)
```

---

## 6. Deep Learning-Based Methods

### **Deep Embedded Clustering (DEC)**

**Key Idea:** Learn representations and cluster assignments **jointly**.

**How it works:**
1. Pre-train autoencoder
2. Initialize cluster centers (e.g., K-Means on latent space)
3. Iteratively:
   - Compute soft assignments using Student's t-distribution
   - Update encoder to improve cluster purity

**Advantages:**
- ✓ End-to-end learning
- ✓ Nonlinear manifold learning
- ✓ Can handle complex, non-Gaussian data

**Disadvantages:**
- ✗ Requires neural network training
- ✗ Hyperparameter tuning
- ✗ Initialization sensitive

---

### **JULE (Joint Unsupervised Learning)**

**Key Idea:** Merge clustering and representation learning into single optimization.

**Why useful:**
- Standard VAE: Learn representation → cluster (two-stage)
- JULE: Jointly optimize both

---

## 7. Hierarchical Methods

### **Agglomerative Clustering**

**Key Idea:** Bottom-up merging of clusters based on linkage criterion.

**Linkage types:**
- **Single**: Minimum distance between clusters → can create elongated clusters
- **Complete**: Maximum distance → compact spherical clusters
- **Average**: Mean distance → balanced
- **Ward**: Minimize within-cluster variance → similar to K-Means

**Advantages:**
- ✓ **Single linkage handles arbitrary shapes** (chain-like clusters)
- ✓ Dendrogram visualization
- ✓ No need to specify k upfront (cut tree at any level)

**Disadvantages:**
- ✗ O(n²) or O(n³) complexity
- ✗ Single linkage sensitive to noise (chaining effect)

**When to use:**
- Want cluster hierarchy
- Elongated/arbitrary shapes (single linkage)
- Moderate dataset size

**Python:**
```python
from sklearn.cluster import AgglomerativeClustering

clusterer = AgglomerativeClustering(n_clusters=12, linkage='single')
labels = clusterer.fit_predict(X)
```

---

## Summary Table: Method Selection Guide

| Method | Best For | Cluster Shape | Outlier Robust | Auto k | Speed |
|--------|----------|---------------|----------------|--------|-------|
| **K-Means** | Spherical, balanced | Spherical | No | No | Very Fast |
| **GMM (full)** | Elliptical, probabilistic | Elliptical | No | BIC/AIC | Fast |
| **t-Mixture** | Heavy tails, outliers | Elliptical | **Yes** | BIC/AIC | Medium |
| **DBSCAN** | Arbitrary shapes, similar density | Arbitrary | **Yes** | **Yes** | Fast |
| **HDBSCAN** | Arbitrary shapes, varying density | Arbitrary | **Yes** | **Yes** | Medium |
| **Mean Shift** | Arbitrary shapes, small data | Arbitrary | Moderate | **Yes** | Slow |
| **Spectral** | Manifolds, non-convex | Manifolds | No | No | Slow |
| **K-Medoids** | Outliers, need actual centers | Spherical-ish | **Yes** | No | Medium |
| **Agglom (single)** | Chain-like, hierarchy | Arbitrary | No | Cut tree | Slow |
| **Agglom (ward)** | Spherical, hierarchy | Spherical | No | Cut tree | Slow |

---

## Recommendations for VAE Latent Space

Based on your VAE analysis:

### ✅ **Use GMM (full covariance)**
**Why:**
- Latent clusters are **compact** (HDBSCAN failed → no density gradients)
- Latent clusters are **elliptical** (GMM > K-Means by 13%)
- Latent space is **non-Gaussian but not arbitrary-shaped**
- Full covariance captures different variances per dimension

### ❌ **Avoid:**
- **HDBSCAN**: Latent space has uniform density (no gradients to exploit)
- **Mean Shift**: Too slow, no advantage over GMM for elliptical clusters
- **t-Mixture**: VAE already handles outliers via reconstruction loss
- **Spectral**: Latent space already low-dimensional and well-structured

### 🤔 **Worth Testing:**

**1. Hierarchical Ward Clustering**
- May reveal cluster hierarchy (lithology sub-types)
- Dendrogram shows relationships

**2. Subspace Clustering**
- Some latent dimensions collapsed → effective dimensionality < 8
- Different lithologies may use different subspaces

**3. Deep Embedded Clustering (DEC)**
- Joint optimization of VAE + clustering
- Could improve over two-stage approach

---

## Key Insight: Non-Gaussian ≠ Arbitrary Shape

**Your latent space has:**
- ✗ Non-Gaussian marginal distributions (Q-Q plots show deviations)
- ✗ Correlated dimensions (violates N(0,I) prior)
- ✗ Posterior collapse (some dims have std ≪ 1)

**BUT:**
- ✓ Clusters are still **compact** (HDBSCAN failed)
- ✓ Clusters are **elliptical** (GMM > K-Means)
- ✓ Uniform density within clusters (no gradients)

**Conclusion:**
Non-Gaussian distributions can still produce elliptical, compact clusters. The VAE learned meaningful structure despite violating its own prior assumptions. This is actually **good** for clustering - the β parameter optimization (β=0.5) **preserved feature correlations** which helps distinguish lithologies.

---

## Further Reading

- **GMM**: Bishop, "Pattern Recognition and Machine Learning" (2006), Chapter 9
- **HDBSCAN**: Campello et al., "Density-Based Clustering Based on Hierarchical Density Estimates" (2013)
- **t-Mixture**: Peel & McLachlan, "Robust mixture modelling using the t distribution" (2000)
- **Spectral**: Von Luxburg, "A tutorial on spectral clustering" (2007)
- **DEC**: Xie et al., "Unsupervised Deep Embedding for Clustering Analysis" (2016)
