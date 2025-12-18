# IIT Internship - Unsupervised Learning for Hand Gesture Recognition

This repository contains the implementation of unsupervised learning algorithms (K-Means and DBSCAN) for clustering 3D hand landmark data extracted from video sequences.

## 📋 Project Overview

The project focuses on analyzing 3D hand pose data using unsupervised clustering techniques. The dataset consists of approximately 2 million rows of 3D coordinate data (X, Y, Z) representing hand landmarks extracted from video frames.

### Data Pipeline

1. **Video Input**: Video stream is decomposed into individual frames
2. **Hand Detection**: Hand recognition model detects and crops hand regions
3. **Landmark Extraction**: 21-point hand landmark model extracts (x, y) pixel coordinates
4. **3D Estimation**: CNN-based depth regression estimates z-depth for each landmark
5. **Output**: 63-dimensional vector (21 landmarks × 3 coordinates) per frame

## 📁 Project Structure

```
IIT_Internship/
├── Codes_Results/          # Main code and results
├── input_gesture_1/        # Raw gesture data
│   ├── Cleaning/           # Cleaning gesture samples
│   ├── Come/               # Come gesture samples
│   ├── Emergency_calling/  # Emergency calling gesture samples
│   ├── Give/               # Give gesture samples
│   ├── Good/               # Good gesture samples
│   ├── Pick/               # Pick gesture samples
│   ├── Stack/              # Stack gesture samples
│   └── Wave/               # Wave gesture samples
└── README.md               # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Required Packages

```bash
pip install numpy pandas scikit-learn matplotlib
```

### Optional (for large datasets)

For better performance on large datasets (>500k rows), install HDBSCAN:

```bash
pip install hdbscan
```

## 📊 Algorithms Implemented

### 1. K-Means Clustering

- **Clusters**: 8 clusters
- **Data**: Raw 3D coordinates (X, Y, Z) without scaling
- **Evaluation Metrics**:
  - Silhouette Score
  - Davies-Bouldin Index (DBI)
  - Calinski-Harabasz Index (CHI)

**Usage:**
```bash
cd Codes_Results
python K-means.py
```

### 2. DBSCAN Clustering

- **Algorithm**: Density-based clustering
- **Parameters**: 
  - `eps=0.5` (neighborhood radius)
  - `min_samples=10` (minimum points in neighborhood)
- **Data**: Standardized/scaled 3D coordinates
- **Features**:
  - Handles noise points (label = -1)
  - Memory-efficient chunked processing for large datasets
  - Automatic HDBSCAN fallback for datasets >500k rows
  - 3D visualization with multiple projections

**Usage:**
```bash
cd Codes_Results
python dbscan.py
```

## 📈 Evaluation Metrics

Both algorithms evaluate clustering quality using:

1. **Silhouette Score** (higher is better, range: -1 to 1)
   - Measures how similar an object is to its own cluster vs other clusters

2. **Davies-Bouldin Index (DBI)** (lower is better)
   - Average similarity ratio of each cluster with its most similar cluster

3. **Calinski-Harabasz Index (CHI)** (higher is better)
   - Ratio of between-clusters dispersion to within-cluster dispersion

## 📝 Documentation

- **Daily Status Updates**: [Google Docs](https://docs.google.com/document/d/1jIz3uZFPw5u9ZN1OFFs-1CXvdU_o5K5PhW5sWPdlgEA/edit?usp=sharing)
- **Research Paper Understanding**: [Google Slides](https://docs.google.com/presentation/d/1RE8Ny49hvOVgzPo7ZH6sdCAP0j7GWqqrd1zJg10WG58/edit?usp=sharing)

## 🔧 Technical Details

### Dataset Characteristics

- **Total Rows**: ~2,016,000 (2 million)
- **Features**: X, Y, Z coordinates
- **Data Type**: Continuous numerical values
- **Coordinate Ranges**:
  - X: -17 to 632
  - Y: 0 to 512
  - Z: 0 to 99

## 📊 Results

Results are saved in the `Codes_Results/` directory:
- Clustered datasets (CSV files)
- Evaluation metric visualizations (PNG files)
- 3D scatter plot visualizations


## 👤 Author

Adit Jain (IIT Internship Project - Imitation learning of robot manipulator by human demostrations)

---

**Note**: Make sure to update file paths in the Python scripts according to your local directory structure before running.

