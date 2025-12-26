# DTW-GC: Dynamic Time Warping Graph Clustering for Hand Gesture Recognition

## Overview

This folder contains the implementation of **Dynamic Time Warping Graph Clustering (DTW-GC)** for unsupervised clustering of 8 hand gestures from the `combined.csv` dataset.

## Key Features

- ✅ **Temporal alignment**: Handles gestures performed at different speeds using DTW
- ✅ **Sequence-aware**: Captures gesture dynamics, not just static poses
- ✅ **Robust to timing variations**: DTW aligns sequences optimally
- ✅ **Better for dynamic gestures**: Distinguishes "Wave", "Come", "Emergency Calling"

## Gesture Types

1. Cleaning
2. Come
3. Emergency Calling
4. Give
5. Good
6. Pick
7. Stack
8. Wave

## Algorithm Overview

```
1. Segment Data: Split into gesture sequences (using zero-padding as boundaries)
2. Compute DTW Matrix: Calculate DTW distance between all sequence pairs
3. Build Graph: Create k-NN similarity graph with DTW distances as edge weights
4. Graph Clustering: Apply spectral clustering on similarity matrix
```

## Installation

### Required Packages

```bash
pip install numpy pandas matplotlib scikit-learn networkx joblib

# For DTW computation (recommended)
pip install dtaidistance

# Alternative: If dtaidistance is not available, the code falls back to Euclidean distance
```

## Usage

1. **Open the notebook**: `DTW_GC_Clustering_Notebook.ipynb`

2. **Run all cells** in order:
   - Cell 1: Import libraries
   - Cell 2: Load and preprocess data
   - Cell 3: Define DTWGraphClustering class
   - Cell 4: Train-test split
   - Cell 5: Apply DTW-GC clustering
   - Cell 6: Visualization
   - Cell 7: DTW distance matrix visualization
   - Cell 8: Save results summary

## Output Files

All outputs are saved in the `DTW_GC_Results/` directory:

- `dtw_gc_model.pkl`: Trained DTW-GC model
- `dtw_gc_train_labels.npy`: Cluster labels for training sequences
- `scaler.pkl`: StandardScaler used for preprocessing
- `dtw_gc_clustering_visualization.png`: 3D PCA, 2D PCA, and cluster distribution plots
- `dtw_distance_matrix.png`: DTW distance matrix heatmap and distribution
- `dtw_gc_results_summary.json`: Complete results summary with metrics

## Parameters

The DTW-GC implementation uses the following parameters:

- `n_clusters=8`: Number of gesture clusters
- `dtw_window=10`: DTW warping window (Sakoe-Chiba band)
- `graph_method='spectral'`: Clustering method ('spectral' or 'modularity')
- `k_neighbors=10`: Number of nearest neighbors for graph construction
- `random_state=42`: Random seed for reproducibility

## Evaluation Metrics

The implementation computes three standard clustering metrics:

1. **Silhouette Score**: Measures how similar a point is to its own cluster vs. other clusters (range: -1 to +1, higher is better)
2. **Davies-Bouldin Index**: Ratio of average intra-cluster distance to minimum inter-cluster distance (lower is better)
3. **Calinski-Harabasz Score**: Ratio of between-cluster dispersion to within-cluster dispersion (higher is better)

## Advantages Over GMM

| Feature | GMM | DTW-GC |
|---------|-----|--------|
| Temporal alignment | ❌ No | ✅ Yes (DTW) |
| Sequence awareness | ❌ Frame-by-frame | ✅ Sequence-level |
| Speed variations | ❌ Fails | ✅ Handles well |
| Dynamic gestures | ❌ Poor | ✅ Better |

## Performance Notes

- **DTW computation**: O(n²) complexity - may take time for large datasets
- **Memory usage**: Stores full DTW distance matrix (n_sequences × n_sequences)
- **Optimization**: Uses sampling for visualization if dataset is too large

## Reference

This implementation follows the structure and best practices from:
- `GMM Hand Gesture/Copy of Raw_feature_double_hand_gesture.ipynb`
- `GMM Hand Gesture/NOTEBOOK_EXPLANATION.md`
- `GMM Hand Gesture/GRAPH_BASED_CLUSTERING_METHODS.md`

## Troubleshooting

### Issue: "dtaidistance not found"
- **Solution**: Install with `pip install dtaidistance`
- **Alternative**: Code will fall back to Euclidean distance approximation

### Issue: "No sequences found"
- **Solution**: Check zero-padding detection threshold in `_segment_sequences()`
- **Adjust**: Modify `zero_threshold` parameter if needed

### Issue: Memory error during DTW computation
- **Solution**: Reduce dataset size or use a subset for initial testing
- **Optimization**: Consider using a smaller `dtw_window` value

## Contact

For questions or issues, refer to the main project documentation.

