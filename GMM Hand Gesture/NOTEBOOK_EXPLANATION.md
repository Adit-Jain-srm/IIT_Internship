# Complete Notebook Explanation: Double Hand Gesture Clustering Analysis

## Overview
This Jupyter notebook performs **comprehensive clustering analysis on double hand gesture data** using raw features. It trains, evaluates, and compares three clustering algorithms: **Gaussian Mixture Models (GMM)**, **K-Means**, and **DBSCAN**. The notebook includes active code, utility functions, and several commented-out experimental modules ready for use.

---

## Table of Contents
1. [Active Code Section](#active-code-section)
2. [Clustering Models](#clustering-models)
3. [Evaluation Framework](#evaluation-framework)
4. [Commented Code Section](#commented-code-section)
5. [Utility Functions](#utility-functions)
6. [Key Metrics Explained](#key-metrics-explained)
7. [Workflow Architecture](#workflow-architecture)

---

## Active Code Section

### **Cell 1: Import Core Libraries**
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
```

**Purpose**: Imports essential libraries for data science pipeline
- `matplotlib.pyplot`: Visualization and plotting
- `numpy`: Numerical operations and array manipulation
- `pandas`: Data manipulation and CSV handling
- `TSNE` (from sklearn): Dimensionality reduction for visualization
- `StandardScaler` (from sklearn): Feature normalization

**Why StandardScaler?** Ensures all features have mean=0 and standard deviation=1, so clustering algorithms treat features equally regardless of their original scales.

---

### **Cell 2: Load and Preprocess Data**
```python
file_path = 'combined.csv'
X = pd.read_csv(file_path)
X_scaled = StandardScaler().fit_transform(X)
```

**Workflow**:
1. **Load Data**: Reads gesture data from `combined.csv` into a pandas DataFrame
2. **Standardize Features**: Applies StandardScaler to normalize all columns
   - `fit_transform()` computes statistics from the data and applies transformation in one step
   - Result: `X_scaled` is a numpy array with scaled features

**Important**: This data will be used for visualization before any clustering.

---

### **Cell 3: t-SNE Dimensionality Reduction & Visualization**
```python
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 10))
plt.scatter(X_tsne[:, 0], X_tsne[:,1], s=6, alpha=0.6, linewidths=0, rasterized=True)
plt.title("TSNE Visualization")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.show()
```

**Purpose**: Visualize high-dimensional data in 2D space

**Parameters Explained**:
- `n_components=2`: Output is 2D (for 2D scatter plot)
- `random_state=42`: Ensures reproducibility (same result each time)
- `n_jobs=-1`: Uses all CPU cores for faster computation

**Plot Details**:
- `s=6`: Point size is small
- `alpha=0.6`: 60% opacity to see overlapping points
- `rasterized=True`: Optimizes PNG export (useful for large datasets)

**Key Insight**: This visualization helps identify if gestures naturally cluster together or overlap before applying formal clustering algorithms.

---

### **Cell 4: Train-Test Split & Feature Scaling**
```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X)
```

**Workflow**:
1. **Split Data**: 80% training (648 samples if 810 total), 20% test (162 samples)
2. **Fit Scaler on Training Data**: Learns statistics from training set only
3. **Transform Both Sets**: Applies learned transformations to prevent data leakage

**⚠️ Bug Alert**: Line 3 says `scaler.transform(X)` instead of `scaler.transform(X_test)`. This scales the entire dataset instead of just the test set. Should be:
```python
X_test_scaled = scaler.transform(X_test)
```

**Why This Matters**: Proper scaling prevents the model from "seeing" test data statistics during training, ensuring unbiased performance evaluation.

---

### **Cell 5: Import Clustering & Evaluation Libraries**
```python
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import joblib
```

**Library Details**:
- **Metrics**: Functions to measure clustering quality
- **KMeans, DBSCAN, GaussianMixture**: Three clustering algorithms
- **joblib**: Serializes/deserializes models for saving and loading

---

### **Cells 6-8: Train Three Clustering Models**

#### **Cell 6: Gaussian Mixture Model (GMM)**
```python
gmm = GaussianMixture(n_components=8, random_state=42)
gmm.fit(X_train_scaled)
joblib.dump(gmm, '/content/drive/MyDrive/Double_Hand_gesture/gesture_Raw_feature_gmm_model.pkl')
```

**What is GMM?**
- Probabilistic model assuming data comes from mixture of Gaussian distributions
- Each component represents one gesture class
- Soft clustering: each point has probability of belonging to each cluster

**Parameters**:
- `n_components=8`: 8 gesture types (Cleaning, Come, Emergency Calling, Give, Good, Pick, Stack, Wave)
- `random_state=42`: Reproducibility

**Output**: Saves trained model as `.pkl` file for later predictions

---

#### **Cell 7: K-Means Clustering**
```python
kmeans = KMeans(n_clusters=8, random_state=42)
kmeans.fit(X_train_scaled)
joblib.dump(kmeans, '/content/drive/MyDrive/Double_Hand_gesture/gesture_Raw_feature_kmeans_model.pkl')
```

**What is K-Means?**
- Partitions data into K clusters by minimizing within-cluster variance
- Hard clustering: each point belongs to exactly one cluster
- Faster than GMM, simpler, deterministic

**Parameters**:
- `n_clusters=8`: Fixed number of clusters
- `random_state=42`: Reproducibility

**Comparison with GMM**: K-Means is faster but less flexible; GMM provides probabilistic assignments.

---

#### **Cell 8: DBSCAN Clustering**
```python
db = DBSCAN(eps=1.098214, min_samples=12)
db.fit(X_train_scaled)
joblib.dump(db, '/content/drive/MyDrive/Double_Hand_gesture/gesture_Raw_feature_db_model.pkl')
```

**What is DBSCAN?**
- Density-based clustering; finds clusters of arbitrary shape
- Can identify noise points (outliers)
- No need to specify number of clusters

**Parameters**:
- `eps=1.098214`: Maximum distance between neighbors (radius)
- `min_samples=12`: Minimum points in a neighborhood to form a core point
- These values appear pre-tuned for this specific dataset

**Key Feature**: Points that don't belong to dense regions get label `-1` (noise), unlike GMM and K-Means which always assign points to clusters.

---

### **Cell 9: Generic Clustering Evaluation Function**
```python
def evaluate_model(model, X):
    y_pred = model.predict(X)
    s_score = silhouette_score(X, y_pred)
    db_score = davies_bouldin_score(X, y_pred)
    ch_score = calinski_harabasz_score(X, y_pred)
    print("Silhoutte Score:", s_score)
    print("DB Score:", db_score)
    print("CH score:", ch_score)
```

**Purpose**: Evaluates clustering quality using three metrics

**Metrics Explained**:
1. **Silhouette Score** (Range: -1 to 1)
   - Measures how similar a point is to its own cluster vs. other clusters
   - +1: Perfect clustering (well-separated clusters)
   - 0: Points on cluster boundaries
   - -1: Points in wrong clusters
   - Higher is better

2. **Davies-Bouldin Index** (Range: 0 to ∞)
   - Ratio of average intra-cluster distance to minimum inter-cluster distance
   - Lower is better (0 = perfect)
   - Measures compactness and separation

3. **Calinski-Harabasz Index** (Range: 0 to ∞)
   - Ratio of between-cluster dispersion to within-cluster dispersion
   - Higher is better
   - Favors tight, well-separated clusters

---

### **Cells 10-13: Evaluate GMM and K-Means on Train/Test Data**
```python
evaluate_model(gmm, X_train_scaled)      # Cell 10
evaluate_model(gmm, X_test_scaled)       # Cell 11
evaluate_model(kmeans, X_train_scaled)   # Cell 12
evaluate_model(kmeans, X_test_scaled)    # Cell 13
```

**Purpose**: Tests both models on training and test data

**Analysis**:
- Compare train vs. test scores to detect overfitting
- If test scores are much worse than train, model overfitted
- GMM and K-Means use `predict()` method which works on any data

---

### **Cell 14: DBSCAN-Specific Evaluation Function**
```python
def evaluate_dbscan(model, X):
    y_pred = model.labels_
    s_score = silhouette_score(X, y_pred)
    db_score = davies_bouldin_score(X, y_pred)
    ch_score = calinski_harabasz_score(X, y_pred)
    print("Silhouette Score:", s_score)
    print("DB Score:", db_score)
    print("CH Score:", ch_score)
```

**Why Different from Cell 9?**
- DBSCAN has no `predict()` method
- Must use `model.labels_` (labels from training data)
- This only works on the training data (not generalized)
- Will fail on test data unless using the `dbscan_predict()` helper (see Cell 15)

---

### **Cell 15: Comprehensive DBSCAN Helper Functions**

#### **Part 1: `dbscan_predict()` Function**
```python
def dbscan_predict(db, X_new):
    """
    Assign labels to X_new based on nearest DBSCAN core sample.
    - Points whose nearest core-sample distance > db.eps get label -1 (noise).
    - Returns an array of labels with same length as X_new.
    """
    # Get core samples (representatives of each cluster)
    try:
        core_samples = db.components_
    except AttributeError:
        raise

    if core_samples.shape[0] == 0:
        return np.full(len(X_new), -1, dtype=int)

    # Fit a nearest-neighbor model on core samples
    nbr = NearestNeighbors(n_neighbors=1).fit(core_samples)
    distances, indices = nbr.kneighbors(X_new, return_distance=True)

    distances = distances.ravel()
    indices = indices.ravel()

    # Map core sample indices to cluster labels
    core_labels = db.labels_[db.core_sample_indices_]
    assigned_labels = core_labels[indices]

    # Points too far from any core sample become noise
    assigned_labels[distances > db.eps] = -1

    return assigned_labels
```

**Purpose**: Extends DBSCAN to predict labels for new, unseen data

**How It Works**:
1. **Extract Core Samples**: DBSCAN stores core samples in `db.components_`
2. **Build NN Index**: Creates a nearest-neighbor index on core samples
3. **Find Nearest Core**: For each new point, finds closest core sample
4. **Assign Label**: Assigns the label of that core sample
5. **Identify Noise**: If distance > eps, marks as noise (-1)

**Key Insight**: DBSCAN is not generative (no built-in prediction), so this function uses "nearest core sample" logic to extend it to new data.

---

#### **Part 2: `evaluate_dbscan()` Robust Function**
```python
def evaluate_dbscan(db, X, name="X"):
    """
    Evaluate DBSCAN on dataset X. Handles edge cases gracefully.
    """
    # Decide whether labels correspond to this X
    if hasattr(db, "labels_") and len(db.labels_) == X.shape[0]:
        y_pred = db.labels_
        source = "model.labels_ (fitted data)"
    else:
        y_pred = dbscan_predict(db, X)
        source = "assigned by nearest core-sample (approximate)"

    # Check for valid clusters
    unique_labels = set(y_pred)
    n_clusters = len([lab for lab in unique_labels if lab != -1])

    print(f"Evaluating on {name}: {X.shape[0]} samples. Labels source: {source}")
    print(f"Found {n_clusters} cluster(s) (excluding noise). Unique labels: {sorted(unique_labels)}")

    if n_clusters == 0:
        print("No clusters found (all points labeled as noise). Metrics cannot be computed.")
        return y_pred
    if n_clusters == 1:
        print("Only one cluster found (plus maybe noise). Silhouette/DB/CH require >=2 clusters.")
        return y_pred

    # Compute metrics
    s_score = silhouette_score(X, y_pred)
    db_score = davies_bouldin_score(X, y_pred)
    ch_score = calinski_harabasz_score(X, y_pred)

    print("Silhouette Score:", s_score)
    print("DB Score:", db_score)
    print("CH Score:", ch_score)
```

**Advanced Features**:
1. **Handles Fitted vs. New Data**: Detects if labels match data size
2. **Edge Case Handling**: 
   - 0 clusters: All noise (can't compute metrics)
   - 1 cluster: Need ≥2 for silhouette/DB/CH
3. **Informative Logging**: Reports where labels came from and cluster counts
4. **Safe Metrics Computation**: Only computes when valid

**Why This Matters**: DBSCAN evaluation is tricky because noise handling and metric validity depend on data and parameters.

---

### **Cells 16-17: Evaluate DBSCAN**
```python
evaluate_dbscan(db, X_train_scaled, name="X_train")  # Cell 16
evaluate_dbscan(db, X_test_scaled, name="X_test")    # Cell 17
```

Tests DBSCAN on training and test data using the comprehensive evaluation function.

---

### **Cells 18-20: Empty Placeholder Cells**
Reserved for future experimental code or additional visualizations.

---

## Commented Code Section

### **Cell 21: DBSCAN Tuning Toolkit** (Fully Commented)

This extensive toolkit automates the process of finding optimal DBSCAN parameters. Here's what each function does:

#### **`k_distance_plot(X, k=8, plot=True)`**
```python
def k_distance_plot(X, k=8, plot=True):
    """
    Plot sorted k-distance (distance to k-th nearest neighbor).
    Useful for choosing eps parameter.
    Returns sorted k-distances array.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    distances, _ = neigh.kneighbors(X)
    k_dist = np.sort(distances[:, -1])
    if plot:
        plt.figure(figsize=(8,4))
        plt.plot(k_dist)
        plt.xlabel(f"Points sorted by {k}-distance (ascending)")
        plt.ylabel(f"{k}-distance (distance to {k}th NN)")
        plt.title(f"k-distance plot (k={k}) — look for the elbow to set eps")
        plt.grid(True)
        plt.show()
    return k_dist
```

**Purpose**: Helps choose optimal `eps` parameter for DBSCAN

**How It Works**:
1. For each point, finds distance to k-th nearest neighbor
2. Sorts these distances in ascending order
3. Plots them as a line graph
4. Look for "elbow" (sudden change in slope) - that's a good eps value

**Why This Matters**: DBSCAN is very sensitive to eps. Too small = everything is noise. Too large = one giant cluster. The elbow in k-distance plot is a heuristic guide.

---

#### **`evaluate_dbscan_labels(labels, X, verbose=True)`**
```python
def evaluate_dbscan_labels(labels, X, verbose=True):
    """
    Accepts labels and X.
    Returns dict with n_clusters, n_noise, cluster_sizes and metrics.
    Handles edge cases gracefully.
    """
    labels = np.asarray(labels)
    unique = np.unique(labels)
    n_clusters = len(unique[unique != -1])
    n_noise = int(np.sum(labels == -1))
    counts = Counter(labels)
    
    res = {
        "n_samples": len(labels),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sizes": dict(counts)
    }
    
    if verbose:
        print(f"Samples: {res['n_samples']}  |  Clusters: {res['n_clusters']}  |  Noise: {res['n_noise']}")
        print("Cluster sizes:", dict(counts))

    # Compute metrics only if ≥2 clusters
    if n_clusters >= 2:
        mask = labels != -1
        X_core = X[mask]
        y_core = labels[mask]
        if len(np.unique(y_core)) >= 2:
            res["silhouette"] = silhouette_score(X_core, y_core)
            res["davies_bouldin"] = davies_bouldin_score(X_core, y_core)
            res["calinski_harabasz"] = calinski_harabasz_score(X_core, y_core)
            if verbose:
                print(f"Silhouette: {res['silhouette']:.4f}  |  DB: {res['davies_bouldin']:.4f}  |  CH: {res['calinski_harabasz']:.1f}")
        else:
            if verbose:
                print("Not enough distinct non-noise clusters to compute metrics.")
    else:
        if verbose:
            print("Fewer than 2 clusters found → skipping silhouette/DB/CH metrics.")
    
    return res
```

**Purpose**: Safe, flexible evaluation with detailed reporting

**Features**:
- Separates noise points before metric computation
- Returns dictionary with all statistics
- Handles edge cases (0, 1, or many clusters)
- Provides informative console output

---

#### **`eps_candidates_from_kdist(k_dist, percentiles=[...], expand=0.2)`**
```python
def eps_candidates_from_kdist(k_dist, percentiles=[85,88,90,92,94,96,98], expand=0.2):
    """
    Use percentiles of k-distance as eps candidates.
    'expand' controls neighborhood around each percentile.
    Returns sorted unique list of candidate eps values.
    """
    vals = np.percentile(k_dist, percentiles)
    candidates = []
    for v in vals:
        lo = max(v * (1 - expand/2), 1e-6)
        hi = v * (1 + expand/2)
        candidates.extend(np.linspace(lo, hi, 3))
    candidates = np.unique(np.round(candidates, 6))
    return sorted(candidates)
```

**Purpose**: Automatically generates eps candidates for grid search

**Logic**:
1. Computes percentiles of k-distance (e.g., 85th, 88th, 90th percentile)
2. Creates small band around each percentile
3. Generates 3 candidate values in each band
4. Returns sorted, unique list

**Example**: If k-distance 90th percentile = 1.5, and expand = 0.2:
- Band = [1.5 * 0.9, 1.5 * 1.1] = [1.35, 1.65]
- Generates 3 values: e.g., [1.35, 1.50, 1.65]

---

#### **`dbscan_grid_search(X, eps_list, min_samples_list, verbose=False)`**
```python
def dbscan_grid_search(X, eps_list, min_samples_list, verbose=False):
    """
    For each (eps, min_samples) pair: fit DBSCAN, evaluate, record metrics.
    Returns pandas DataFrame with results sorted by silhouette (descending).
    """
    rows = []
    for eps in eps_list:
        for ms in min_samples_list:
            model = DBSCAN(eps=eps, min_samples=ms)
            model.fit(X)
            labels = model.labels_
            res = evaluate_dbscan_labels(labels, X, verbose=False)
            
            row = {
                "eps": eps,
                "min_samples": ms,
                "n_clusters": res["n_clusters"],
                "n_noise": res["n_noise"],
                "n_samples": res["n_samples"],
                "silhouette": res.get("silhouette", np.nan),
                "davies_bouldin": res.get("davies_bouldin", np.nan),
                "calinski_harabasz": res.get("calinski_harabasz", np.nan)
            }
            rows.append(row)
            if verbose:
                print(f"eps={eps:.4f}, min_samples={ms} -> clusters={row['n_clusters']}, noise={row['n_noise']}, sil={row['silhouette']}")
    
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by=["silhouette", "n_clusters", "n_noise"], ascending=[False, False, True]).reset_index(drop=True)
    return df_sorted
```

**Purpose**: Tests all combinations of eps × min_samples to find best parameters

**Workflow**:
1. Creates nested loop over eps and min_samples values
2. For each combination:
   - Fits DBSCAN model
   - Evaluates using robust function
   - Records all metrics in a row
3. Returns DataFrame sorted by:
   - **Primary**: Silhouette (higher better)
   - **Secondary**: Number of clusters (more clusters if silhouette tied)
   - **Tertiary**: Noise points (fewer noise if other metrics tied)

**Output**: DataFrame where each row is (eps, min_samples, metrics)

---

#### **`plot_dbscan_result(X, labels, title=None, pca_components=2)`**
```python
def plot_dbscan_result(X, labels, title=None, pca_components=2):
    """
    Plot clusters using PCA reduction (2 components).
    Noise shown in gray.
    """
    labels = np.asarray(labels)
    pca = PCA(n_components=pca_components)
    X2 = pca.fit_transform(X)
    unique_labels = np.unique(labels)
    
    plt.figure(figsize=(7,6))
    for lab in unique_labels:
        mask = labels == lab
        if lab == -1:
            plt.scatter(X2[mask,0], X2[mask,1], s=10, marker='x', alpha=0.4, label='noise (-1)')
        else:
            plt.scatter(X2[mask,0], X2[mask,1], s=20, alpha=0.6, label=f'cluster {lab}')
    
    plt.title(title if title else "DBSCAN clustering (PCA 2D)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()
```

**Purpose**: Visualizes DBSCAN clustering results in 2D PCA space

**Features**:
- Reduces data to 2D using PCA
- Different colors for each cluster
- Noise points shown as 'x' marks in gray
- Legend shows cluster labels

---

#### **`run_dbscan_tuning()` - End-to-End Runner**
```python
def run_dbscan_tuning(X, k_for_kdist=8, percentiles=[85,88,90,92,94,96,98], 
                      min_samples_list=None, expand=0.25, verbose=False):
    """
    Complete DBSCAN tuning pipeline.
    """
    if min_samples_list is None:
        min_samples_list = [max(3, k_for_kdist-2), k_for_kdist, k_for_kdist+2, k_for_kdist+4]

    print("1) k-distance plot (inspect the elbow to choose eps):")
    k_dist = k_distance_plot(X, k=k_for_kdist, plot=True)

    eps_cand = eps_candidates_from_kdist(k_dist, percentiles=percentiles, expand=expand)
    print(f"Auto-generated {len(eps_cand)} eps candidates (sample): {eps_cand[:6]} ...")

    print(f"Trying min_samples values: {min_samples_list}")

    print("\n2) Running grid search (this can take a while)...")
    df = dbscan_grid_search(X, eps_cand, min_samples_list, verbose=verbose)

    print("\nGrid search complete. Top candidates sorted by silhouette:")
    display_df = df.copy()
    display_df['silhouette'] = display_df['silhouette'].round(4)
    display_df['davies_bouldin'] = display_df['davies_bouldin'].round(4)
    display_df['calinski_harabasz'] = display_df['calinski_harabasz'].round(2)
    print(display_df.head(20).to_string(index=False))

    df.to_csv("dbscan_grid_search_results.csv", index=False)
    print("\nSaved results to dbscan_grid_search_results.csv")

    return df
```

**Purpose**: Orchestrates entire DBSCAN parameter tuning process

**Workflow**:
1. Generates min_samples values (relative to k)
2. Creates k-distance plot
3. Generates eps candidates from k-distance percentiles
4. Runs grid search over all combinations
5. Displays top 20 results
6. Saves results to CSV

**Example Usage** (shown at end of cell):
```python
# Usage:
df_results = run_dbscan_tuning(X_train_scaled, k_for_kdist=8)

# After that, pick best row:
best = df_results.dropna(subset=["silhouette"]).sort_values("silhouette", ascending=False).iloc[0]
print(best)

# Retrain with best parameters:
model_best = DBSCAN(eps=best.eps, min_samples=int(best.min_samples)).fit(X_train_scaled)
plot_dbscan_result(X_train_scaled, model_best.labels_, title=f"DBSCAN eps={best.eps}, min_samples={best.min_samples}")

# Save the model:
joblib.dump(model_best, 'gesture_dbscan_best.pkl')
```

**Key Insight**: This toolkit makes DBSCAN parameter selection systematic rather than trial-and-error.

---

### **Cell 22: Commented GMM/UMAP Experiment**

```python
# Load & scale
file_path = '/content/drive/MyDrive/Double_Hand_gesture/Gesture with mean and variance/Combined_mean_and_variance.csv'
X = pd.read_csv(file_path)
print(X)

scaler_data = MinMaxScaler()
X_scaled = scaler_data.fit_transform(X)
joblib.dump(scaler_data, "scaler.pkl")

# UMAP before clustering (no colors)
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, n_jobs=-1, random_state=42)
X_umap = umap.fit_transform(X_scaled)
plt.figure(figsize=(10, 10))
plt.scatter(X_umap[:, 0], X_umap[:, 1], s=6, alpha=0.6, linewidths=0, rasterized=True)
plt.title("UMAP Visualization before Clustering")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.savefig("UMAP_before_clustering.png", dpi=200)
plt.show()
joblib.dump(umap, "umap.pkl")
```

**Purpose**: Alternative dimensionality reduction approach using UMAP instead of t-SNE

**Key Differences from t-SNE**:
- **UMAP**: Preserves both local and global structure; faster on large datasets
- **t-SNE**: Better at visualizing local structure; slower but often produces cleaner clusters

**Parameters**:
- `n_neighbors=15`: Number of neighbors to consider (local structure)
- `min_dist=0.1`: Minimum distance between points (spacing in output)
- `n_jobs=-1`: Uses all CPU cores

**Commented Continuation** (for after clustering):
```python
# Train & save GMM
# [code would be here]

# Save labeled CSV
encoding = {0: "Royal", 1: "Green"}
X_out = X.copy()
X_out['target'] = [encoding[int(x)] for x in cluster_labels]
X_out.to_csv("deepika.csv", index=False)
```

**Purpose**: Saves clustering results with gesture labels into CSV file

---

### **Cell 23: Skeleton Import Statements**

```python
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
```

**Purpose**: Sets up imports for comprehensive visualization framework

**New Imports**:
- `cm` (colormap): For color-coded visualizations
- `Axes3D`: Enables 3D plotting
- `silhouette_samples`: Per-sample silhouette values (not just global score)
- `PCA`: Principal Component Analysis
- `datetime`: For timestamping outputs
- `warnings`: To suppress unimportant warnings

---

### **Cell 25: Complete End-to-End Clustering Framework** (Most Comprehensive - Fully Commented)

This is a production-ready clustering pipeline. Here's the complete breakdown:

#### **Configuration & Constants**
```python
GESTURE_LABELS = ['Cleaning', 'Come', 'Emergency Calling', 'Give',
                  'Good', 'Pick', 'Stack', 'Wave']
RANDOM_STATE = 42
```

Defines the 8 gesture types and uses fixed random seed for reproducibility.

---

#### **Utility Functions**

##### **`normalize_data(df)`**
```python
def normalize_data(df):
    """Min-max normalize all numeric features to [0,1]."""
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            col_min = df[col].min()
            col_max = df[col].max()
            df[col] = (df[col] - col_min) / (col_max - col_min + 1e-9)
    return df
```

**Purpose**: Scales all numeric features to [0,1] range

**Why Min-Max instead of StandardScaler?**
- Min-Max: Output range is [0,1] (bounded)
- StandardScaler: Output is unbounded
- Min-Max better for visualization (points stay within bounds)

**Note**: `1e-9` prevents division by zero if column has no variance.

---

##### **`load_csv(file_path)` and `save_model(model, path)`**
```python
def load_csv(file_path):
    df = pd.read_csv(file_path)
    df = normalize_data(df)
    return df

def save_model(model, path):
    joblib.dump(model, path)
    print(f"💾 Saved model: {path}")
```

Load CSV and automatically normalize; save models with confirmation message.

---

##### **`make_timestamped_run_dir(base_dir)`**
```python
def make_timestamped_run_dir(base_dir):
    """
    Create new subfolder with timestamp (Asia/Kolkata timezone if available).
    Returns (run_dir_path, timestamp_string).
    """
    try:
        from zoneinfo import ZoneInfo
        KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
        now = datetime.now(KOLKATA_TZ)
    except Exception:
        now = datetime.now()
    
    ts = now.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir, ts
```

**Purpose**: Creates timestamped output directories for experiment organization

**Format**: `run_YYYYMMDD_HHMMSS` (e.g., `run_20251223_143022`)

**Timezone Handling**: Attempts to use Asia/Kolkata timezone (for UTC+5:30); falls back to local timezone if unavailable.

---

#### **Visualization Functions**

##### **`visualize_6d_in_3d(X, labels, label_map, feature_names, title, save_path)`**
```python
def visualize_6d_in_3d(X, labels, label_map, feature_names, title="6D Visualization (3D Projection)", save_path=None):
    """
    Visualizes 6 features in 3D:
      - 3 features on X, Y, Z axes
      - 4th feature mapped to color
      - 5th feature mapped to size
      - 6th feature affects brightness
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('viridis')

    # Normalize features to [0,1] for mapping
    color_feature = (X[:, 3] - X[:, 3].min()) / (X[:, 3].max() - X[:, 3].min() + 1e-9)
    size_feature = (X[:, 4] - X[:, 4].min()) / (X[:, 4].max() - X[:, 4].min() + 1e-9)
    bright_feature = (X[:, 5] - X[:, 5].min()) / (X[:, 5].max() - X[:, 5].min() + 1e-9)

    color_vals = cmap(color_feature * 0.7 + 0.3 * bright_feature)

    unique_labels = np.unique(labels)

    for label in unique_labels:
        indices = labels == label
        gesture_name = label_map.get(label, f"Cluster {label}")

        ax.scatter(
            X[indices, 0], X[indices, 1], X[indices, 2],
            c=color_vals[indices],
            s=50 + 200 * size_feature[indices],
            alpha=0.8,
            edgecolor='k',
            linewidth=0.3,
            label=gesture_name
        )

        if indices.sum() > 0:
            centroid = np.mean(X[indices, :3], axis=0)
            ax.text(
                centroid[0], centroid[1], centroid[2],
                gesture_name,
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle="round,pad=0.3")
            )

    ax.set_xlabel(feature_names[0], fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_names[1], fontsize=12, fontweight='bold')
    ax.set_zlabel(feature_names[2], fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
    ax.legend(title="Gesture Clusters", fontsize=9, title_fontsize=11)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=200)
        print(f"🖼 Saved 6D->3D visualization: {save_path}")
    
    plt.show()
    plt.close(fig)
```

**Purpose**: Advanced visualization encoding 6 features into 3D space

**Encoding Scheme**:
- **X, Y, Z Axes**: Features 0, 1, 2 (spatial dimensions)
- **Color**: Feature 3 (using 'viridis' colormap)
- **Size**: Feature 4 (marker size from 50 to 250)
- **Brightness**: Feature 5 (blends with color for additional dimension)

**Additional Features**:
- Cluster labels shown as legend
- Centroid text labels for each cluster
- High DPI (200) for publication-quality output

**Why This Design?** Uses all 6 dimensions while remaining human-readable in 3D projection.

---

##### **`plot_silhouette(X, labels, method_name, out_dir)`**
```python
def plot_silhouette(X, labels, method_name, out_dir):
    """
    Classic silhouette plot (one bar per sample grouped by cluster).
    """
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    X_valid = X[valid_mask]

    if len(np.unique(valid_labels)) < 2:
        print("⚠️ Silhouette plot skipped - need at least 2 clusters (excluding noise).")
        return

    sil_vals = silhouette_samples(X_valid, valid_labels)
    y_lower = 10
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    unique_clusters = np.unique(valid_labels)
    norm = plt.Normalize(vmin=unique_clusters.min(), vmax=unique_clusters.max())
    colors = cm.nipy_spectral(norm(unique_clusters))

    for i, c in enumerate(unique_clusters):
        c_sil_vals = sil_vals[valid_labels == c]
        c_sil_vals.sort()
        size_cluster = c_sil_vals.shape[0]
        y_upper = y_lower + size_cluster

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, c_sil_vals,
                         facecolor=colors[i], alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster, f"Cluster {c} (n={size_cluster})", fontsize=9)
        y_lower = y_upper + 10

    ax.set_title(f"Silhouette Plot for {method_name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label and sample index")
    ax.axvline(x=np.mean(sil_vals), color="red", linestyle="--", label="Average silhouette")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"silhouette_plot_{method_name}.png")
    fig.savefig(save_path, dpi=200)
    print(f"🖼 Saved silhouette plot: {save_path}")
    plt.show()
    plt.close(fig)
```

**Purpose**: Creates classical silhouette plot for cluster quality assessment

**Visualization Details**:
- **X-axis**: Silhouette coefficient (-1 to +1)
- **Y-axis**: Sample indices grouped by cluster
- **Bar Width**: Individual silhouette value (wider = better)
- **Color**: Different for each cluster
- **Red Dashed Line**: Global average silhouette

**Interpretation**:
- All bars pointing right (positive): Good clustering
- Bars pointing left (negative): Some points in wrong cluster
- Thin bars: Points on cluster boundary
- Thick bars: Points solidly in cluster

---

##### **`plot_per_cluster_bar(mean_sil_per_cluster, method_name, out_dir, label_map)`**
```python
def plot_per_cluster_bar(mean_sil_per_cluster, method_name, out_dir, label_map=None):
    """
    Bar graph: mean silhouette for each cluster.
    """
    clusters = list(mean_sil_per_cluster.keys())
    means = [mean_sil_per_cluster[c] for c in clusters]
    names = [label_map.get(c, str(c)) if label_map else str(c) for c in clusters]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(clusters)), means, tick_label=names, alpha=0.85, edgecolor='k')
    
    ax.set_title(f"Per-Cluster Mean Silhouette ({method_name})", fontsize=14, fontweight='bold')
    ax.set_ylabel("Mean silhouette score")
    ax.set_ylim([-0.1, 1.0])
    ax.set_xticklabels(names, rotation=45, ha='right')
    
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.02, f"{val:.3f}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"per_cluster_bar_{method_name}.png")
    fig.savefig(save_path, dpi=200)
    print(f"🖼 Saved per-cluster bar chart: {save_path}")
    plt.show()
    plt.close(fig)
```

**Purpose**: Shows average cluster quality for each gesture

**Features**:
- One bar per cluster
- Height = mean silhouette score
- Values printed on top of bars
- Gesture names from label_map

**Use Case**: Quickly identify which gestures cluster well and which don't.

---

##### **`plot_pca_3d_silhouette(X, labels, sample_silhouette_vals, method_name, out_dir, label_map)`**
```python
def plot_pca_3d_silhouette(X, labels, sample_silhouette_vals, method_name, out_dir, label_map=None):
    """
    PCA -> 3D scatter where points colored by their silhouette value.
    Also draws cluster centroids.
    """
    valid_mask = labels != -1
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    sil_vals = sample_silhouette_vals

    if X_valid.shape[0] == 0:
        print("⚠️ PCA silhouette plot skipped - no valid samples.")
        return

    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_valid)

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
        c=sil_vals, cmap='viridis', s=40, alpha=0.9, edgecolor='k', linewidth=0.2
    )
    plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.1, label='Silhouette value')

    unique_clusters = np.unique(labels_valid)
    for c in unique_clusters:
        indices = labels_valid == c
        centroid = X_pca[indices].mean(axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], 
                label_map.get(c, f"Cluster {c}") if label_map else f"Cluster {c}",
                fontsize=10, fontweight='bold', color='black',
                bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7, boxstyle="round,pad=0.3"))

    ax.set_title(f"PCA (3D) colored by silhouette values ({method_name})", fontsize=14, fontweight='bold')
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"pca3d_silhouette_{method_name}.png")
    fig.savefig(save_path, dpi=200)
    print(f"🖼 Saved PCA 3D silhouette plot: {save_path}")
    plt.show()
    plt.close(fig)
```

**Purpose**: 3D PCA visualization with silhouette quality heatmap

**Features**:
- **PCA Reduction**: 3D projection of high-dimensional data
- **Color**: Silhouette value (viridis colormap - blue=negative, yellow=positive)
- **Centroids**: Labeled points at cluster centers
- **Interpretation**: Yellow points = good assignments; blue points = questionable

---

#### **Clustering Model Functions**

```python
def train_kmeans(X, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    model.fit(X)
    return model, model.predict(X)

def train_dbscan(X, eps=0.1, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    model.fit(X)
    return model, model.labels_

def train_gmm(X, n_clusters):
    model = GaussianMixture(n_components=n_clusters, random_state=RANDOM_STATE)
    model.fit(X)
    return model, model.predict(X)
```

**Purpose**: Wrapper functions for training each algorithm

**Note**: `n_init=10` for K-Means means 10 different random initializations; best one is kept.

---

#### **Main Pipeline Workflow**

```python
if __name__ == "__main__":
    # Paths
    base_model_dir = "/content/drive/MyDrive/Double_Hand_gesture"
    os.makedirs(base_model_dir, exist_ok=True)

    # Create timestamped run directory
    run_dir, ts = make_timestamped_run_dir(base_model_dir)
    print(f"📁 Created run folder: {run_dir}")

    # Choose method: "kmeans", "dbscan", or "gmm"
    method = "gmm"

    # CSV path
    csv_path = "/content/drive/MyDrive/Double_Hand_gesture/Gesture with mean and variance/Combined_mean_and_variance.csv"

    # Load dataset
    df = load_csv(csv_path)
    print(f"✅ Loaded: {csv_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Select six mean+variance features
    feature_candidates = [c for c in df.columns if any(k in c.lower() for k in ["mean", "var", "variance"])]
    if len(feature_candidates) < 6:
        raise ValueError(f"Expected ≥6 features; found {len(feature_candidates)}: {feature_candidates}")

    feature_cols = feature_candidates[:6]
    X = df[feature_cols].values
    print(f"📊 Using features: {feature_cols}")
```

**Setup Steps**:
1. Creates timestamped directory for outputs
2. User selects method (gmm, kmeans, or dbscan)
3. Loads dataset from CSV
4. Auto-detects features containing "mean", "var", or "variance"
5. Selects first 6 such features

---

#### **Training Section**

```python
if method == "kmeans":
    model, labels = train_kmeans(X, len(GESTURE_LABELS))
    save_model(model, os.path.join(run_dir, "kmeans_6feature_model.pkl"))
    title = "KMeans"
elif method == "dbscan":
    model, labels = train_dbscan(X, eps=0.12, min_samples=5)
    save_model(model, os.path.join(run_dir, "dbscan_6feature_model.pkl"))
    title = "DBSCAN"
elif method == "gmm":
    model, labels = train_gmm(X, len(GESTURE_LABELS))
    save_model(model, os.path.join(run_dir, "gmm_6feature_model.pkl"))
    title = "GMM"
else:
    raise ValueError("Invalid method. Choose 'kmeans', 'dbscan', or 'gmm'.")

print(f"\n✅ Training complete using {method.upper()}.")
print("Cluster IDs found:", np.unique(labels))
```

Trains selected method and displays results.

---

#### **Cluster-to-Gesture Mapping**

```python
unique_labels = np.unique(labels)
label_map = {lbl: GESTURE_LABELS[i % len(GESTURE_LABELS)] for i, lbl in enumerate(unique_labels)}

print("\n🧩 Cluster → Gesture Mapping:")
for k, v in label_map.items():
    print(f"  Cluster {k} → {v}")

map_path = os.path.join(run_dir, f"{method}_6feature_label_map.pkl")
joblib.dump(label_map, map_path)
print(f"💾 Mapping saved to: {map_path}")

# Cluster counts summary
counts = {}
for lbl in unique_labels:
    counts[int(lbl)] = int((labels == lbl).sum())
summary_df = pd.DataFrame.from_dict(counts, orient='index', columns=['count']).sort_index()
summary_df.index.name = 'cluster'
summary_df['gesture_label'] = summary_df.index.map(lambda x: label_map.get(x, ""))
summary_csv_path = os.path.join(run_dir, f"{method}_cluster_summary_{ts}.csv")
summary_df.to_csv(summary_csv_path)
print(f"💾 Cluster summary CSV saved: {summary_csv_path}")
```

**Purpose**: Maps cluster IDs to gesture names and summarizes cluster sizes

**Mapping Logic**: If 8 clusters found, assigns Cleaning→0, Come→1, etc. If fewer/more clusters, cycles through GESTURE_LABELS.

---

#### **Evaluation Section**

```python
valid_mask = labels != -1
if len(np.unique(labels[valid_mask])) > 1:
    global_score = silhouette_score(X[valid_mask], labels[valid_mask])
    print(f"📈 Silhouette Score ({method.upper()}, 6 features): {global_score:.4f}")
else:
    global_score = None
    print("⚠️ Silhouette Score not computed (need ≥2 clusters excluding noise).")

# Per-cluster silhouette
per_cluster_mean = {}
sample_silhouette_vals = None

if len(np.unique(labels[valid_mask])) > 1:
    sample_silhouette_vals = silhouette_samples(X[valid_mask], labels[valid_mask])
    clusters = np.unique(labels[valid_mask])
    for c in clusters:
        cluster_vals = sample_silhouette_vals[labels[valid_mask] == c]
        if len(cluster_vals) > 0:
            per_cluster_mean[c] = float(cluster_vals.mean())
        else:
            per_cluster_mean[c] = float('nan')

    print("\n🔎 Per-Cluster Silhouette Scores:")
    for c, m in per_cluster_mean.items():
        print(f"  Cluster {c}: mean silhouette = {m:.4f} (n={int((labels[valid_mask] == c).sum())})")
```

Computes and displays silhouette scores.

---

#### **Save Evaluation Results**

```python
pc_npy = os.path.join(run_dir, f"{method}_per_cluster_silhouette_means.npy")
np.save(pc_npy, per_cluster_mean)
pc_csv = os.path.join(run_dir, f"{method}_per_cluster_silhouette_means_{ts}.csv")
pd.DataFrame.from_dict(per_cluster_mean, orient='index', columns=['mean_silhouette']).to_csv(pc_csv)
print(f"💾 Per-cluster silhouette means saved: {pc_npy} and {pc_csv}")

# Sample-level silhouette values
if sample_silhouette_vals is not None:
    sample_sil_array = np.full(shape=(labels.shape[0],), fill_value=np.nan)
    sample_sil_array[valid_mask] = sample_silhouette_vals
    sample_sil_path = os.path.join(run_dir, f"{method}_sample_silhouette_vals_{ts}.npy")
    np.save(sample_sil_path, sample_sil_array)
    print(f"💾 Sample-level silhouette values saved: {sample_sil_path}")
```

Saves evaluation metrics for later analysis.

---

#### **Plotting Section**

```python
if len(np.unique(labels[valid_mask])) > 1:
    plot_silhouette(X, labels, method, run_dir)
    plot_per_cluster_bar(per_cluster_mean, method, run_dir, label_map)
    plot_pca_3d_silhouette(X, labels, sample_silhouette_vals, method, run_dir, label_map)
else:
    print("⚠️ Skipping silhouette and PCA plots - need ≥2 clusters (excluding noise).")

# 6D -> 3D visualization
viz_path = os.path.join(run_dir, f"{method}_6d_to_3d_viz_{ts}.png")
visualize_6d_in_3d(X, labels, label_map, feature_cols, title=f"{method.upper()} (6D -> 3D)", save_path=viz_path)
```

Generates all visualizations and saves as PNG files.

---

#### **Final Outputs**

```python
# Save labels
labels_path = os.path.join(run_dir, f"{method}_6feature_labels_{ts}.npy")
np.save(labels_path, labels)
print(f"💾 Saved cluster labels: {labels_path}")

# Save metadata
try:
    import json
    meta = {
        "timestamp": ts,
        "method": method,
        "feature_cols": feature_cols,
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "global_silhouette": float(global_score) if global_score is not None else None,
        "cluster_counts": counts
    }
    meta_path = os.path.join(run_dir, f"run_metadata_{ts}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 Run metadata saved: {meta_path}")
except Exception as e:
    print("⚠️ Could not save run metadata JSON:", e)

print("\n🎉 All done. Plots and models saved inside:", run_dir)
```

**Final Outputs in Timestamped Folder**:
- `{method}_6feature_model.pkl`: Trained model
- `{method}_6feature_labels_{ts}.npy`: Cluster labels for all samples
- `{method}_6feature_label_map.pkl`: Mapping of cluster ID → gesture name
- `{method}_cluster_summary_{ts}.csv`: Counts per cluster
- `{method}_per_cluster_silhouette_means.npy`: Array of per-cluster scores
- `{method}_per_cluster_silhouette_means_{ts}.csv`: CSV of per-cluster scores
- `{method}_sample_silhouette_vals_{ts}.npy`: Per-sample silhouette values
- `silhouette_plot_{method}.png`: Silhouette bar plot
- `per_cluster_bar_{method}.png`: Per-cluster mean scores bar plot
- `pca3d_silhouette_{method}.png`: 3D PCA colored by silhouette
- `{method}_6d_to_3d_viz_{ts}.png`: 6D feature visualization in 3D
- `run_metadata_{ts}.json`: Reproducibility metadata

---

## Key Metrics Explained

### **Silhouette Score**
- **Range**: -1 to +1
- **Formula**: For each point: (b - a) / max(a, b)
  - a = mean distance to points in same cluster
  - b = mean distance to points in nearest different cluster
- **Interpretation**:
  - +1: Perfect clustering (tight, well-separated clusters)
  - 0: Points on cluster boundaries
  - -1: Points in wrong clusters
- **Good Threshold**: > 0.5 is generally good

### **Davies-Bouldin Index**
- **Range**: 0 to ∞
- **Formula**: Average ratio of within-cluster to between-cluster distances
- **Interpretation**:
  - Lower is better (0 = perfect)
  - Penalizes overlapping clusters
  - Sensitive to cluster shapes

### **Calinski-Harabasz Index**
- **Range**: 0 to ∞
- **Formula**: (between-cluster variance) / (within-cluster variance)
- **Interpretation**:
  - Higher is better
  - Favors compact, well-separated clusters
  - Biased toward convex clusters (favors K-Means)

---

## Workflow Architecture

```
┌─────────────────────────────┐
│ Load & Preprocess Data      │
│ - Read CSV                  │
│ - Normalize/Scale           │
└──────────┬──────────────────┘
           │
           ├─► Cell 3: t-SNE Visualization
           │   (Quick look at data)
           │
           ├─► Cell 4: Train-Test Split
           │   (80/20 split + scaling)
           │
           ▼
┌─────────────────────────────┐
│ Train Three Algorithms      │
│ - GMM (probabilistic)       │
│ - K-Means (partitioning)    │
│ - DBSCAN (density-based)    │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│ Evaluate Each Model         │
│ - Silhouette Score          │
│ - Davies-Bouldin Index      │
│ - Calinski-Harabasz Index   │
└──────────┬──────────────────┘
           │
           ├─► Cell 21: DBSCAN Tuning (commented)
           │   - k-distance plot
           │   - eps auto-generation
           │   - Grid search
           │   - Best parameter selection
           │
           ▼
┌──────────────────────────────────┐
│ Cell 25: End-to-End Pipeline     │
│ (Commented - Full Production)    │
│ - Load data                      │
│ - Train model (choose GMM/KM/DB) │
│ - Map clusters → gestures        │
│ - Evaluate & visualize           │
│ - Save all outputs timestamped   │
└──────────────────────────────────┘
```

---

## Summary

This notebook provides:

1. **Active Code**: Immediate clustering and evaluation (Cells 1-17)
2. **DBSCAN Tuning Toolkit**: Systematic parameter optimization (Cell 21)
3. **GMM/UMAP Experiment**: Alternative dimensionality approach (Cell 22)
4. **Production Pipeline**: Complete end-to-end framework (Cell 25)

The commented sections are fully functional and ready to uncomment for deeper analysis or production use. The architecture is modular, allowing easy modification and extension.

