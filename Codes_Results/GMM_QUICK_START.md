# GMM Clustering - Quick Start Guide

## What Changed?
✅ **Removed**: Feature engineering (ratios, differences, statistics)  
✅ **Removed**: Hardcoded temperature assumptions (20-30°C = Cold, etc.)  
✅ **Kept**: Pure GMM algorithm, clear methodology  

---

## 3 Clusters Discovered
```
Cluster 0 (50%)  ← Large cluster, distinct sensor pattern
Cluster 1 (33%)  ← Medium cluster, different sensor pattern  
Cluster 2 (17%)  ← Small cluster, unique sensor pattern
```

**Note**: Cluster names are empirical. What they represent (Hot/Cold/Normal) depends on your domain knowledge.

---

## Quick Use

### Load Model
```python
import pickle
with open('gmm_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['gmm_model']
scaler = data['scaler']
```

### Predict
```python
X_new = [[s1, s2, s3, s4], ...]  # your sensor data
X_scaled = scaler.transform(X_new)
clusters = model.predict(X_scaled)
probs = model.predict_proba(X_scaled)
```

---

## Files

| File | Purpose |
|------|---------|
| `gmm_clustering_results.csv` | Cluster assignments for all 98,820 samples |
| `gmm_model.pkl` | Ready-to-use trained model |
| `gmm_clusters_2d_pca.png` | Visual: 2D cluster projection |
| `gmm_clusters_3d_pca.png` | Visual: 3D cluster projection |
| `gmm_sensor_distributions.png` | Visual: Feature distributions |
| `gmm_cluster_probabilities.png` | Visual: Model confidence |

---

## Cluster Profiles

**Cluster 0** (50% of data)
- sensor_1: ~169 (high)
- sensor_2: ~442 (medium-high)
- sensor_3: ~501 (low)
- sensor_4: ~449 (high)

**Cluster 1** (33% of data)
- sensor_1: ~92 (low)
- sensor_2: ~366 (low-medium)
- sensor_3: ~819 (high)
- sensor_4: ~179 (low)

**Cluster 2** (17% of data)
- sensor_1: ~93 (low)
- sensor_2: ~442 (medium-high)
- sensor_3: ~915 (very high)
- sensor_4: ~245 (medium)

---

## Quality Metrics

| Metric | Value | Meaning |
|--------|-------|---------|
| Silhouette | 0.549 | Good separation |
| Davies-Bouldin | 0.895 | Compact clusters |
| Calinski-Harabasz | 184,283 | Well-defined clusters |

✓ All metrics indicate good clustering quality

---

## Key Points

1. **No Assumptions**: Pure data-driven clustering
2. **Reproducible**: Same results every time (random_state=42)
3. **Validated**: 98,820 samples analyzed, converged in 7 iterations
4. **Production-Ready**: Model saved and ready for predictions

---

## Interpretation

The three clusters represent distinct patterns in your 4-sensor data. To map them to "Hot", "Cold", "Normal":

1. Compare cluster sensor profiles with known temperature ranges
2. Validate with labeled test data
3. Assign semantic meaning based on domain knowledge

**Example**: If you know Cluster 0's samples come from high-temperature recordings, label it as "Hot".

---

Generated: December 22, 2025
