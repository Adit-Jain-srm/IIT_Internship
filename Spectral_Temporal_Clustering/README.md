# Spectral Temporal Clustering (STC) for Hand Gesture Recognition

## Overview

This folder contains the implementation of **Spectral Temporal Clustering (STC)** - a graph-based temporal clustering method that combines spatial and temporal information for unsupervised hand gesture recognition.

## Key Features

- ✅ **Spatial Laplacian**: Captures hand structure within each frame (42 landmarks)
- ✅ **Temporal Laplacian**: Captures frame-to-frame relationships across sequences
- ✅ **Joint Clustering**: Combines spatial and temporal information via weighted Laplacian
- ✅ **Unsupervised**: No labels needed for clustering
- ✅ **Comparison with GMM**: Includes baseline GMM clustering for comparison

## Dataset

- **320 videos total**: 40 videos × 8 gestures
- **150 frames per video** (variable after zero-padding removal)
- **42 landmarks per frame** (21 landmarks × 2 hands)
- **126 features per frame** (42 landmarks × 3 coordinates: X, Y, Z)

### Gesture Types:
1. Cleaning
2. Come
3. Emergency Calling
4. Give
5. Good
6. Pick
7. Stack
8. Wave

## Algorithm

### Spectral Temporal Clustering (STC)

STC combines two graph structures:

1. **Spatial Graph**: k-NN graph of hand landmarks within each frame
   - Captures hand structure and spatial relationships

2. **Temporal Graph**: Similarity graph between gesture sequences
   - Captures temporal patterns and sequence similarity

3. **Joint Laplacian**: `L_joint = α·L_spatial + (1-α)·L_temporal`
   - α controls balance between spatial and temporal information
   - Default: α = 0.5 (equal weight)

4. **Spectral Clustering**: 
   - Compute smallest eigenvalues/eigenvectors of joint Laplacian
   - Apply K-Means in spectral embedding space

## Files

- `STC_Clustering_Notebook.ipynb`: Main implementation notebook
- `STC_Results/`: Output directory containing:
  - `stc_model.pkl`: Trained STC model
  - `stc_labels.npy`: Cluster labels
  - `gmm_baseline_model.pkl`: GMM baseline model
  - `gmm_baseline_labels.npy`: GMM cluster labels
  - `comparison_results.json`: Performance comparison
  - `stc_vs_gmm_pca_comparison.png`: PCA visualization
  - `cluster_distribution_comparison.png`: Cluster size comparison
  - `eigenvalue_spectrum.png`: Eigenvalue plot

## Usage

1. **Run the notebook**: Execute all cells in `STC_Clustering_Notebook.ipynb`

2. **Parameters**:
   ```python
   stc = SpectralTemporalClustering(
       n_clusters=8,              # Number of gesture clusters
       alpha=0.5,                 # Spatial/temporal balance (0=spatial only, 1=temporal only)
       n_neighbors_spatial=5,     # k-NN for spatial graph
       n_neighbors_temporal=10,    # k-NN for temporal graph
       random_state=42
   )
   ```

3. **Results**: All outputs saved to `STC_Results/` directory

## Expected Performance

Based on theoretical expectations and similar methods:

- **Silhouette Score**: 0.2-0.35 (vs 0.1-0.2 for GMM)
- **Davies-Bouldin Score**: Lower than GMM (better separation)
- **Calinski-Harabasz Score**: Higher than GMM (better cluster quality)

## Advantages Over GMM

| Feature | GMM | STC |
|---------|-----|-----|
| Spatial structure | ❌ | ✅ |
| Temporal modeling | ❌ | ✅ |
| Hand skeleton | ❌ | ✅ |
| Non-convex clusters | ❌ | ✅ |
| Sequence-aware | ❌ | ✅ |

## Dependencies

```
numpy
pandas
matplotlib
scikit-learn
scipy
joblib
```

## References

- Spectral Clustering: Ng et al. (2002) "On Spectral Clustering: Analysis and an algorithm"
- Temporal Graph Clustering: Extends spectral clustering to temporal domains
- Hand Gesture Recognition: MediaPipe hand landmarks structure

## Notes

- The implementation uses mean frame representation for efficiency
- Can be extended to use DTW for temporal similarity (more accurate but slower)
- α parameter can be tuned for optimal spatial/temporal balance
- Works best with sequences of similar length (handled via zero-padding removal)

## Author

**Adit Jain** — IIT Internship (Imitation learning of robot manipulators by human demonstrations)

- [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-adit-jain)
- [Certificate (PDF)](../Certificate_AditJain_modified.pdf)

