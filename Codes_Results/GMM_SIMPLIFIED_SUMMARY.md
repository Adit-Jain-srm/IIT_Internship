# GMM Temperature Clustering - Simplified Fundamentals

## Overview
This implementation focuses on **pure Gaussian Mixture Model (GMM) clustering** with a clear emphasis on fundamentals:
- ✓ No feature engineering
- ✓ No predefined temperature assumptions
- ✓ Raw sensor data only
- ✓ Unsupervised learning approach

---

## Key Changes from Previous Version

### Removed
1. **Hardcoded Temperature Categories**: No more assumptions about 20-30°C = Cold, etc.
2. **Feature Engineering**: All engineered features (ratios, differences, CV) eliminated
3. **Temperature-guided Initialization**: Removed reliance on ground truth labels
4. **Complexity**: Simplified pipeline for clarity

### Added
1. **Pure GMM Clustering**: Standard GMM with default random initialization
2. **Raw Sensor Data**: Uses only 4 original sensor readings (sensor_1 through sensor_4)
3. **Unsupervised Characterization**: Clusters defined by data patterns alone
4. **Clear Documentation**: Emphasis on methodology over assumptions

---

## Methodology

### Step 1: Data Loading
- Load balanced dataset: **98,820 samples**
- 6 temperature ranges (20-30, 30-40, 40-50, 50-60, 60-70, 70-85°C)
- **4 sensor features** (no assumptions about what they mean)

### Step 2: Data Standardization
```python
StandardScaler: zero mean, unit variance
Input: 98,820 × 4 (raw sensor values)
Output: 98,820 × 4 (standardized)
```

### Step 3: GMM Fitting
```python
GaussianMixture(
    n_components=3,
    covariance_type='full',
    random_state=42,
    n_init=20
)
```

**Results:**
- **Cluster 0**: 49,406 samples (50.00%)
- **Cluster 1**: 33,082 samples (33.48%)
- **Cluster 2**: 16,332 samples (16.53%)
- **Convergence**: 7 iterations ✓

### Step 4: Cluster Characterization
Each cluster has distinct sensor value patterns:

| Metric | Cluster 0 | Cluster 1 | Cluster 2 |
|--------|-----------|-----------|-----------|
| **Size** | 49,406 | 33,082 | 16,332 |
| **sensor_1** mean | 169.25 | 92.32 | 93.23 |
| **sensor_2** mean | 442.27 | 366.25 | 441.97 |
| **sensor_3** mean | 500.66 | 818.93 | 914.67 |
| **sensor_4** mean | 449.39 | 179.42 | 244.75 |

---

## Model Evaluation

### Quality Metrics
| Metric | Value | Interpretation |
|--------|-------|-----------------|
| **Silhouette Score** | 0.5489 | Good clustering structure |
| **Davies-Bouldin Index** | 0.8953 | Good separation (lower is better) |
| **Calinski-Harabasz Index** | 184,282.55 | Excellent cluster definition |

### Information Criteria
- **BIC**: 146,738.08 (lower is better)
- **AIC**: 146,320.03 (balance between fit and complexity)

---

## Visualizations

### 1. 2D PCA Projection
- **File**: `gmm_clusters_2d_pca.png`
- **Variance Explained**: 94.74% (PC1: 76.59%, PC2: 18.15%)
- **Shows**: Clear separation of 3 clusters in 2D space

### 2. 3D PCA Projection
- **File**: `gmm_clusters_3d_pca.png`
- **Variance Explained**: 99.20% (PC1: 76.59%, PC2: 18.15%, PC3: 4.46%)
- **Shows**: Complete cluster structure with minimal information loss

### 3. Sensor Distributions
- **File**: `gmm_sensor_distributions.png`
- **Content**: 4 histograms showing sensor value ranges per cluster
- **Shows**: Distinct feature distributions between clusters

### 4. Cluster Probability Distribution
- **File**: `gmm_cluster_probabilities.png`
- **Content**: Heatmap of membership probabilities
- **Shows**: Model confidence levels for each sample

---

## Output Files

### Data Files
| File | Size | Content |
|------|------|---------|
| `gmm_clustering_results.csv` | 5.99 MB | Cluster assignments + probabilities for all 98,820 samples |
| `gmm_model.pkl` | ~0.01 MB | Trained GMM model + StandardScaler (ready for predictions) |
| `gmm_model_statistics.csv` | <0.01 MB | All model metrics and convergence info |

### Visualizations
| File | Size | Description |
|------|------|-------------|
| `gmm_clusters_2d_pca.png` | 2.82 MB | 2D PCA visualization |
| `gmm_clusters_3d_pca.png` | 1.97 MB | 3D PCA visualization |
| `gmm_sensor_distributions.png` | 0.27 MB | Sensor histograms by cluster |
| `gmm_cluster_probabilities.png` | 0.55 MB | Probability heatmap |

---

## Key Findings

### Cluster Characteristics

**Cluster 0 (50% of data)**
- Highest sensor_1 values (mean: 169.25)
- Intermediate sensor_3 values (mean: 500.66)
- Balanced sensor_2 and sensor_4
- Interpretation: Distinct pattern A

**Cluster 1 (33.48% of data)**
- Lowest sensor_1 values (mean: 92.32)
- Highest sensor_3 values (mean: 818.93)
- Lowest sensor_4 values (mean: 179.42)
- Interpretation: Distinct pattern B

**Cluster 2 (16.53% of data)**
- Low sensor_1 values (similar to Cluster 1)
- Highest sensor_3 values (mean: 914.67, highest of all)
- Intermediate sensor_2 and sensor_4
- Interpretation: Distinct pattern C

### What These Clusters Represent

**Important Note**: These are empirically discovered clusters based on sensor data patterns. 
- Without external ground truth labels, we cannot definitively say which cluster corresponds to "Hot", "Cold", or "Normal"
- The clusters reflect the natural structure of the 4-sensor feature space
- Interpretation depends on domain knowledge of what the sensors measure

---

## Usage

### Load the Trained Model
```python
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Load saved model
with open('gmm_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

gmm = model_data['gmm_model']
scaler = model_data['scaler']
sensor_columns = model_data['sensor_columns']
```

### Make Predictions on New Data
```python
# New data: shape (n_samples, 4) with columns [sensor_1, sensor_2, sensor_3, sensor_4]
X_new = ...  # your new sensor data

# Preprocess
X_new_scaled = scaler.transform(X_new)

# Predict cluster
cluster_assignments = gmm.predict(X_new_scaled)

# Get probabilities
cluster_probabilities = gmm.predict_proba(X_new_scaled)
```

### Interpret Results
```python
# cluster_assignments: array of 0, 1, or 2 for each sample
# cluster_probabilities: shape (n_samples, 3) with membership probabilities
```

---

## Advantages of This Approach

✓ **No Assumptions**: Clusters discovered purely from data patterns  
✓ **Transparent**: Simple pipeline, easy to understand and reproduce  
✓ **Scalable**: Uses fundamental ML (GMM with standardization)  
✓ **Flexible**: Clusters can be interpreted with domain knowledge  
✓ **Reproducible**: Fixed random_state=42 ensures consistent results  

---

## Considerations

⚠ **Interpretation**: Cluster-to-"temperature-label" mapping requires external validation  
⚠ **Imbalance**: Natural cluster sizes are unequal (50%, 33%, 17%)  
⚠ **Feature Selection**: Clustering depends heavily on these 4 sensors  
⚠ **Generalization**: Model trained on balanced dataset from one gesture type  

---

## Next Steps

1. **Validate**: Compare discovered clusters with external ground truth if available
2. **Interpret**: Determine what physical phenomena each cluster represents
3. **Deploy**: Use `gmm_model.pkl` for real-time classification of new sensor streams
4. **Improve**: If needed, collect more data, add sensors, or try different algorithms
5. **Monitor**: Track cluster distribution in production to detect anomalies

---

**Generated**: December 22, 2025  
**Notebook**: `GMM_Temperature_3Clusters.ipynb`  
**Framework**: Scikit-learn, pandas, NumPy, matplotlib
