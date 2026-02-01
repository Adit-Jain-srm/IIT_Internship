<p align="center">
  <img src="https://img.shields.io/badge/IIT-Internship-0066CC?style=for-the-badge" alt="IIT Internship" />
  <img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
</p>

<h1 align="center">IIT Internship — Unsupervised Learning & Imitation Learning</h1>
<p align="center">
  <strong>Imitation learning of robot manipulators by human demonstrations</strong>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-projects--models">Projects</a> •
  <a href="#-repository-structure">Structure</a> •
  <a href="#-getting-started">Getting Started</a> •
  <a href="#-documentation">Documentation</a>
</p>

---

## Overview

This repository contains implementations, trained models, and research from an **IIT internship** spanning:

| Domain | Data | Goal | Methods |
|--------|------|------|--------|
| **Hand gesture recognition** | 3D hand landmarks (MediaPipe), ~2M rows, 8 gestures | Unsupervised clustering / imitation learning | K-Means, DBSCAN, GMM, DTW-GC, Spectral Temporal Clustering, graph-based (GCN, T-GCN) |
| **Temperature classification** | 4-sensor readings, COLD / NORMAL / HOT | Unsupervised or semi-supervised 3-class classification | GMM (multiple variants), feature engineering, validation on collected data |

- **Gesture types (8):** Cleaning, Come, Emergency Calling, Give, Good, Pick, Stack, Wave  
- **Temperature classes:** COLD (15–25°C), NORMAL (45–60°C), HOT (60–70°C)

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language & runtime** | Python 3.7+ |
| **ML & clustering** | scikit-learn (K-Means, DBSCAN, GaussianMixture, PCA, StandardScaler, StratifiedKFold, metrics), optional: HDBSCAN, dtaidistance (DTW) |
| **Data & numerics** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Notebooks** | Jupyter |
| **Utilities** | SciPy, joblib, pickle, pathlib |
| **Graph / temporal** | NetworkX (spectral clustering, STC); documented: PyTorch, PyTorch Geometric (GCN, T-GCN, GAT) |

<details>
<summary><b>Core dependencies (pip)</b></summary>

```bash
numpy pandas scikit-learn matplotlib scipy joblib seaborn
```

</details>

<details>
<summary><b>Optional (by project)</b></summary>

| Package | Use case |
|---------|----------|
| `hdbscan` | Large-scale density clustering (DBSCAN fallback) |
| `dtaidistance` | DTW for DTW-GC gesture clustering |
| `networkx` | Graph construction for STC / spectral clustering |
| `PyTorch` / `torch-geometric` | GCN/T-GCN implementation (see GMM Hand Gesture docs) |

</details>

---

## Projects & Models

### Hand gesture recognition (3D landmarks)

- **Data:** Video → frames → hand detection → 21-point landmarks → depth → **63-D per frame** (21×3); optional 42 landmarks (2 hands), 126-D; sequences (e.g. 150 frames).
- **Datasets:** `combined.csv`-style (~2M rows) or 320 videos (40×8 gestures).

| Location | Model / method | Description |
|----------|----------------|-------------|
| **Codes_Results** | K-Means | 8 clusters, raw 3D coordinates; Silhouette, DBI, CHI |
| **Codes_Results** | DBSCAN | Density-based, chunked/HDBSCAN for large data |
| **Codes_Results** | GMM | 3-cluster, sequential, relative positions notebooks |
| **DTW_GC_Clustering** | DTW-GC | DTW → k-NN graph → spectral clustering; 8 gesture clusters |
| **Spectral_Temporal_Clustering** | STC | Spatial + temporal Laplacians; joint spectral clustering (α balance) |
| **GMM Hand Gesture** | Graph spectral / GCN / T-GCN | k-NN → Laplacian → spectral; documented GCN/T-GCN for imitation learning |

**Outputs:** `Codes_Results` (CSV, plots), `DTW_GC_Clustering/DTW_GC_Results` (labels, JSON), `Spectral_Temporal_Clustering/STC_Results` (labels, accuracy/alpha JSONs), `GMM Hand Gesture/GMM_Results` (spectral labels, figures).

---

### Temperature classification (sensor data)

- **Data:** 4 sensors per reading; CSVs by folder (COLD / NORMAL / HOT); per-file aggregation and feature engineering.

| Location | Model | Features | Performance | Artifacts |
|----------|--------|----------|-------------|-----------|
| **Improved** | GMM (9 components, tied) | 21 engineered | **48.92%** (5-fold CV); +8.6% over baseline | `gmm_temperature_classifier.pkl`, metadata |
| **Main_GMM** | GMM (3 clusters) | 4 sensors + scaling | **65.68%** on collected data (1,049 samples) | `predict_temperature.py`, VALIDATION_RESULTS.md |
| **Codes_Results** | GMM 3 clusters | 4 sensors | ~45.79% overall | Summaries, confusion matrices, PCA |
| **30 DECEMBER** | GMM (optimized) | 15 (v2_enhanced) | **40.86%** test; diag covariance | `gmm_model_best.pkl`, metadata |
| **Readings** | GMM binary | 12 (mean, std, min, max × 3 sensors) | ~60% test; Sensor 1 excluded | Notebook, `predict_temperature.py` |

**Evaluation:** 5-fold CV (Improved), hold-out test (30 DECEMBER, Readings), validation on separate `collect_data` (Main_GMM). Metrics: accuracy, F1, precision/recall, confusion matrix; clustering: Silhouette, DBI, CHI where applicable.

---

## Repository Structure

```
IIT_Internship/
├── Codes_Results/              # Core hand + temperature experiments (K-Means, DBSCAN, GMM)
├── Main_GMM/                   # Production GMM temperature classifier & validation
├── Improved/                   # Optimized GMM temperature (48.92% CV, 21 features, 9 components)
├── 30 DECEMBER/                # GMM optimization on Dec 30 sensor data (15 features)
├── Readings/                   # Binary temperature (Cold_normal vs Hot)
├── DTW_GC_Clustering/          # DTW + graph clustering for 8 gestures
├── Spectral_Temporal_Clustering/  # STC: spatial + temporal spectral clustering
├── GMM Hand Gesture/            # Graph-based methods & GCN/T-GCN implementation guides
├── GMM_vs_Recent_Soft_Clustering_Algorithms.md   # Research: ESM, SC-DEC, sERAL, etc.
├── Unsupervised Learning.docx
└── README.md
```

---

## Getting Started

### Prerequisites

- **Python** 3.7+
- **pip**

### Install

```bash
# Core
pip install numpy pandas scikit-learn matplotlib scipy joblib seaborn

# Optional: large-scale clustering
pip install hdbscan

# Optional: DTW-GC (gesture clustering)
pip install dtaidistance

# Optional: STC / graph methods
pip install networkx
```

### Run

| Task | Command |
|------|---------|
| K-Means clustering | `cd Codes_Results && python K-means.py` |
| DBSCAN clustering | `cd Codes_Results && python dbscan.py` |
| Temperature prediction (Main_GMM) | `cd Main_GMM && python predict_temperature.py` |
| Notebooks | Open the relevant `.ipynb` and run all cells (e.g. `Improved/GMM_Temperature_Classification_GroundTruth.ipynb`, `Spectral_Temporal_Clustering/STC_Clustering_Notebook.ipynb`) |

**Note:** Set data paths in scripts to match your layout (e.g. `input_gesture_1/`, `collect_data/`, `COLD/`, `HOT/`, `NORMAL/`).

---

## Documentation

| Area | Entry points |
|------|--------------|
| **Optimized GMM temperature** | `Improved/00_READ_ME_FIRST.md`, `Improved/INDEX.md` |
| **GMM improvements** | `Improved/GMM_IMPROVEMENTS_SUMMARY.md`, `QUICK_IMPROVEMENTS_REFERENCE.md` |
| **Gesture clustering (STC)** | `Spectral_Temporal_Clustering/README.md`, `STC_APPROACH_DOCUMENTATION.md` |
| **Gesture clustering (DTW-GC)** | `DTW_GC_Clustering/README.md` |
| **Graph-based / imitation learning** | `GMM Hand Gesture/INDEX.md`, `GRAPH_METHODS_IMPLEMENTATION_GUIDE.md` |
| **Soft clustering research** | `GMM_vs_Recent_Soft_Clustering_Algorithms.md` |
| **Validation (Main_GMM)** | `Main_GMM/VALIDATION_RESULTS.md` |

---

## Evaluation Metrics

- **Clustering:** Silhouette score, Davies–Bouldin index, Calinski–Harabasz index; BIC/AIC for GMM.
- **Classification:** Accuracy, precision, recall, F1 (macro/weighted), confusion matrix.
- **Validation:** 5-fold CV (Improved), hold-out test, validation on separate `collect_data` (Main_GMM).

---

## Author & Links

**Adit Jain** — IIT Internship (Imitation learning of robot manipulator by human demonstrations)

| Link | Description |
|------|-------------|
| [Google Docs](https://docs.google.com/document/d/1jIz3uZFPw5u9ZN1OFFs-1CXvdU_o5K5PhW5sWPdlgEA/edit?usp=sharing) | Daily status & write-up |
| [Google Slides](https://docs.google.com/presentation/d/1RE8Ny49hvOVgzPo7ZH6sdCAP0j7GWqqrd1zJg10WG58/edit?usp=sharing) | Research & slides |

---

<p align="center">
  <sub>Multiple trained GMMs (temperature) • Classical & graph-based clustering (gestures) • Research on soft clustering & imitation learning</sub>
</p>
