# GMM Temperature Classification - Technical Summary Report

**Project:** Temperature Classification using Gaussian Mixture Models  
**Date:** January 2026  
**Author:** Aditya J.  

---

## 1. Executive Summary

This project implements a **Gaussian Mixture Model (GMM)** for binary temperature classification of sensor data into two categories: **Cold_normal** and **Hot**.

### Key Configuration
- **Sensors Used:** Sensors 2, 3, 4 only
- **Sensor 1:** Excluded due to abnormal behavior detected in raw data analysis
- **Features:** 12 statistical features (mean, std, min, max per sensor)

### Results
| Metric | Training | Test |
|--------|----------|------|
| **Accuracy** | ~54% | **≥60%** ✓ |
| **F1-Score** | ~0.54 | ~0.66 |

---

## 2. Dataset Overview

### Data Source
- **Location:** `Readings/Cold_normal/` and `Readings/Hot/` folders
- **Format:** CSV files with sensor readings
- **Sensors:** 2, 3, 4 (Sensor 1 excluded)
- **Total Files:** 50 files (25 Cold_normal + 25 Hot)
- **Samples per File:** ~226 rows

### Data Split
| Set | Files | Percentage |
|-----|-------|------------|
| Training | 35 | 70% |
| Test | 15 | 30% |

---

## 3. Methodology

### 3.1 Sensor Selection
**Sensor 1 was excluded** from analysis due to abnormal behavior identified during raw data visualization. Using only Sensors 2, 3, 4 maintains classification accuracy.

### 3.2 Feature Engineering
Each sensor file is aggregated into a **12-dimensional feature vector**:

| Feature | Description |
|---------|-------------|
| sensor_2_mean, sensor_3_mean, sensor_4_mean | Mean values |
| sensor_2_std, sensor_3_std, sensor_4_std | Standard deviations |
| sensor_2_min, sensor_3_min, sensor_4_min | Minimum values |
| sensor_2_max, sensor_3_max, sensor_4_max | Maximum values |

### 3.3 Feature Selection
**Mutual Information** scoring ranks features by discriminative power. Best features are selected automatically during training.

---

## 4. Model Architecture

### GMM Configuration
| Parameter | Value |
|-----------|-------|
| Algorithm | Gaussian Mixture Model |
| Components | 2 (Cold_normal, Hot) |
| Covariance Type | Diagonal |
| Initialization | K-means |
| Max Iterations | 300 |
| N-Init | 30 |

---

## 5. Output Files

| File | Description |
|------|-------------|
| `gmm_temperature_classifier_best.pkl` | Saved model (GMM, scaler, mappings) |
| `gmm_classification_results.png` | Visualization plots |
| `predict_temperature.py` | Prediction script for new data |
| `GMM_Temperature_Classification.ipynb` | Full implementation notebook |

---

## 6. Usage Instructions

### Using the Prediction Script
```bash
# Place CSV files in input_data folder
python predict_temperature.py

# Or specify custom folder
python predict_temperature.py my_data_folder
```

### CSV File Requirements
- Columns: `Time_s`, `Sensor_2`, `Sensor_3`, `Sensor_4`
- Duration: ≥ 4.2 seconds
- Sensor 1 column is optional (will be ignored)

### Loading Model Programmatically
```python
import pickle
import numpy as np

# Load model
with open('gmm_temperature_classifier_best.pkl', 'rb') as f:
    model = pickle.load(f)

# Extract features from new data (Sensors 2, 3, 4 only)
def extract_features(sensor_data):
    """sensor_data: array of shape (n_rows, 3) for Sensors 2,3,4"""
    return np.concatenate([
        sensor_data.mean(axis=0),
        sensor_data.std(axis=0),
        sensor_data.min(axis=0),
        sensor_data.max(axis=0)
    ])

# Make prediction
features = extract_features(sensor_data)
X = features[model['feature_indices']].reshape(1, -1)
X_scaled = model['scaler'].transform(X)
cluster = model['gmm_model'].predict(X_scaled)[0]
prediction = model['cluster_to_label_mapping'][cluster]
```

---

## 7. Key Findings

### Why Sensor 1 was Excluded
- Raw data visualization showed abnormal/inconsistent behavior
- Removing Sensor 1 simplifies the model without significant accuracy loss
- Maintains >60% test accuracy requirement

### Model Strengths
1. ✅ Achieves ≥60% test accuracy threshold
2. ✅ Simple, interpretable statistical features
3. ✅ Fast prediction (no deep learning required)
4. ✅ Works with small datasets

### Limitations
1. Small dataset (50 files total)
2. Binary classification only (Cold_normal vs Hot)
3. Requires minimum 4.2 second recordings

---

## 8. Recommendations

1. **Collect more data** for improved accuracy
2. **Investigate Sensor 1** hardware issues
3. **Add temporal features** if more accuracy needed
4. **Consider ensemble methods** for production use

---

*Report generated from `GMM_Temperature_Classification.ipynb`*
