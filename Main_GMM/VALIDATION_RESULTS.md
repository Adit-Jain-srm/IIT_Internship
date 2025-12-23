# GMM Temperature Classification - Validation Results

## Executive Summary

The trained Gaussian Mixture Model has been validated on **1,049 collected sensor readings** organized by temperature range. The model achieves an overall accuracy of **65.68%**, but shows class imbalance with strong performance on NORMAL temperatures and poor performance on COLD and HOT ranges.

---

## Validation Dataset

| Temperature Range | Folder | Class | Samples | Accuracy |
|------------------|--------|-------|---------|----------|
| 15-25°C | `15-25/` | COLD | 330 | 0.0% |
| 45-50°C | `45-50/` | NORMAL | 329 | 100.0% |
| 50-60°C | `50-60/` | NORMAL | 360 | 100.0% |
| 60-70°C | `60-70/` | HOT | 30 | 0.0% |
| **TOTAL** | | | **1,049** | **65.68%** |

---

## Overall Performance Metrics

### Classification Accuracy
- **Overall Accuracy**: 65.68%
- **Precision** (weighted): 0.4314
- **Recall** (weighted): 0.6568
- **F1-Score** (weighted): 0.5208

### Prediction Confidence
- **High confidence (≥0.80)**: 1,048 samples (99.9%)
- **Medium confidence (0.60-0.80)**: 1 sample (0.1%)
- **Low confidence (<0.60)**: 0 samples (0.0%)
- **Mean confidence**: 0.9994

**Note**: The model makes highly confident predictions, but these are biased toward NORMAL class.

---

## Confusion Matrix

```
               Predicted COLD  Predicted NORMAL  Predicted HOT
True COLD  :        0              330              0
True NORMAL:        0              689              0
True HOT   :        0               30              0
```

### Interpretation

The model predicts **every sample as NORMAL** (689/1,049 = 65.7%), which happens to match the accuracy because:
- 45-50°C and 50-60°C are both NORMAL → Correct (689 samples)
- 15-25°C is COLD → Incorrect (330 samples misclassified)
- 60-70°C is HOT → Incorrect (30 samples misclassified)

---

## Per-Category Performance

### COLD (15-25°C)
- **Support**: 330 samples
- **Precision**: 0.0000 (no samples predicted as COLD)
- **Recall**: 0.0000 (0% of COLD samples detected)
- **F1-Score**: 0.0000

### NORMAL (45-50°C, 50-60°C)
- **Support**: 689 samples (329 + 360)
- **Precision**: 0.6568 (689/1049 of predictions are NORMAL, all correct)
- **Recall**: 1.0000 (100% of NORMAL samples detected)
- **F1-Score**: 0.7929

### HOT (60-70°C)
- **Support**: 30 samples
- **Precision**: 0.0000 (no samples predicted as HOT)
- **Recall**: 0.0000 (0% of HOT samples detected)
- **F1-Score**: 0.0000

---

## Root Cause Analysis

### Why the model fails:

1. **Training Data Imbalance**
   - The model was trained on a balanced dataset with 3 temperature classes
   - The 9-component GMM learned to separate the data using cluster-to-class mapping based on majority voting
   - **Cluster mapping shows**: Clusters {0,1,2,5} → NORMAL (50.4% of training data)

2. **Sensor Signature Overlap**
   - COLD (15-25°C) and NORMAL (45-50°C) ranges may have overlapping sensor readings
   - HOT (60-70°C) range has very few samples (30) - insufficient for robust predictions

3. **Biased Classification Boundary**
   - The decision boundary for cluster-to-class mapping favors NORMAL class
   - Model predicts NORMAL for anything that falls into clusters {0,1,2,5}

---

## Comparison with Training Performance

| Metric | Training (CV) | Validation |
|--------|---------------|-----------|
| Accuracy | 48.96% | 65.68% |
| Precision | N/A | 0.4314 |
| Recall | N/A | 0.6568 |
| Confidence | Varied | 99.9% |

**Note**: Training accuracy of 48.96% reflects balanced 3-class classification. Validation accuracy of 65.68% is misleading due to class distribution in collected data (2:3 NORMAL vs 1:3 COLD+HOT).

---

## Recommendations

### For Improving Model Performance:

1. **Collect More Data**
   - Gather more samples for COLD (15-25°C) and HOT (60-70°C) ranges
   - Current HOT dataset has only 30 samples - too small for reliable classification

2. **Feature Engineering Review**
   - Analyze which engineered features best discriminate between temperature classes
   - COLD and NORMAL ranges may need better separation

3. **Cluster-to-Class Remapping**
   - Review the cluster-to-class mapping learned during training
   - Current mapping may not be optimal for deployed validation data

4. **Threshold Optimization**
   - Implement confidence thresholds for each class
   - Reject low-confidence predictions for COLD and HOT

5. **Class Rebalancing**
   - If deployment requires balanced predictions, use oversampling/undersampling
   - Or use class weights in the GMM objective function

---

## Dataset Characteristics

### Collected Data Structure
```
collect_data/
├── 15-25/          (330 samples, COLD)
│   └── *.csv       (display_log_*.csv format)
├── 45-50/          (329 samples, NORMAL)
│   └── *.csv
├── 50-60/          (360 samples, NORMAL)
│   └── *.csv
└── 60-70/          (30 samples, HOT)
    └── *.csv
```

### CSV File Format
Each CSV file contains:
- `elapsed_time_s`: Elapsed time in seconds
- `clock_time`: Clock time
- `time_ms`: Time in milliseconds
- `sensor_1`: Sensor 1 reading
- `sensor_2`: Sensor 2 reading
- `sensor_3`: Sensor 3 reading
- `sensor_4`: Sensor 4 reading

---

## Validation Methodology

1. **Load All Data**: Read all CSV files from `collect_data/` subdirectories
2. **Extract Ground Truth**: Map folder name to temperature class
3. **Feature Scaling**: Apply StandardScaler (fitted on training data)
4. **Predict**: Use trained GMM model to classify each sensor reading
5. **Map Clusters**: Convert cluster IDs to temperature classes using learned mapping:
   - Clusters {0,1,2,5} → NORMAL
   - Clusters {3,8} → COLD
   - Clusters {4,6,7} → HOT
6. **Calculate Metrics**: Accuracy, precision, recall, F1-score, confusion matrix

---

## Conclusion

The GMM model successfully identifies NORMAL temperature readings (100% accuracy) but fails to distinguish COLD and HOT temperatures in the validation dataset. The model's decision boundary is heavily biased toward the NORMAL class, which dominated the training data. This suggests the need for:

- **Data rebalancing** (more COLD/HOT samples)
- **Feature optimization** (better sensor value separation across temperature ranges)
- **Threshold tuning** (confidence-based filtering per class)

The high prediction confidence (99.9%) indicates the model is certain in its (incorrect) predictions for COLD/HOT, suggesting fundamental issues with feature representation rather than calibration.

---

## Files Generated

- `predict_temperature.py` - Validation script with comprehensive metrics reporting
- `VALIDATION_RESULTS.md` - This document
- `gmm_model.pkl` - Trained model used for validation

**Generated**: 2025-12-23
**Validation Script**: `predict_temperature.py`
**Command**: `python predict_temperature.py`
