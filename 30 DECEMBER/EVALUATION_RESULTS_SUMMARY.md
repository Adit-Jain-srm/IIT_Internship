# GMM Models Evaluation on New Data (30 December)

## Overview
Evaluated 4 trained GMM models on new sensor data collected on December 30, 2024. The data consists of:
- **COLD**: 1,406 samples (7 files, ~11°C)
- **HOT**: 1,367 samples (6 files, 57-70°C)
- **NORMAL**: 1,628 samples (8 files, 27-42°C)
- **Total**: 4,401 sensor readings

Each reading contains 4 sensor values (sensor_1, sensor_2, sensor_3, sensor_4) extracted from VALID rows in raw sensor log CSV files.

## Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Mean Confidence | Clusters |
|-------|----------|-----------|--------|----------|-----------------|----------|
| **Improved_gmm_model_v1_original** | **35.33%** | **0.5557** | 0.3533 | 0.2468 | 0.9930 | 9 |
| **Improved_gmm_model** | 33.45% | 0.3147 | 0.3345 | 0.2737 | 0.9962 | 12 |
| **Improved_gmm_temperature_classifier** | 32.47% | 0.2939 | 0.3247 | 0.2352 | 0.9948 | 9 |
| **Main_GMM_gmm_model** | 26.63% | 0.1620 | 0.2663 | 0.1835 | 0.9827 | 3 |

### Best Performing Model: **Improved_gmm_model_v1_original**
- **Accuracy**: 35.33%
- **Precision**: 0.5557 (highest precision)
- **Best for**: NORMAL temperature classification (83.85% accuracy)

## Detailed Performance by Model

### 1. Improved_gmm_model_v1_original (BEST OVERALL)
**Configuration**: 9 clusters, 21 engineered features

**Overall Metrics:**
- Accuracy: 35.33%
- Precision: 0.5557
- Recall: 0.3533
- F1-Score: 0.2468

**Confusion Matrix:**
```
               Pred COLD  Pred NORMAL  Pred HOT
True COLD  :        6         1303        97
True NORMAL:        0         1365       263
True HOT   :        0         1183       184
```

**Per-Class Performance:**
- **COLD**: Precision=1.0000, Recall=0.0043, F1=0.0085 (very poor recall)
- **NORMAL**: Precision=0.3545, Recall=0.8385, F1=0.4983 (best class)
- **HOT**: Precision=0.3382, Recall=0.1346, F1=0.1926

**Accuracy by Folder:**
- COLD: 0.43% (very poor - almost never predicts COLD correctly)
- NORMAL: 83.85% (excellent)
- HOT: 13.46% (poor)

**Key Insight**: This model heavily biases toward NORMAL predictions (83.85% of NORMAL samples correctly classified), but struggles with COLD and HOT classes.

---

### 2. Improved_gmm_model
**Configuration**: 12 clusters, 21 engineered features

**Overall Metrics:**
- Accuracy: 33.45%
- Precision: 0.3147
- Recall: 0.3345
- F1-Score: 0.2737

**Confusion Matrix:**
```
               Pred COLD  Pred NORMAL  Pred HOT
True COLD  :       15          646       745
True NORMAL:       18          625       985
True HOT   :       24          511       832
```

**Per-Class Performance:**
- **COLD**: Precision=0.2632, Recall=0.0107, F1=0.0205
- **NORMAL**: Precision=0.3508, Recall=0.3839, F1=0.3665
- **HOT**: Precision=0.3244, Recall=0.6086, F1=0.4228

**Accuracy by Folder:**
- COLD: 1.07% (very poor)
- NORMAL: 38.39%
- HOT: 60.86% (best HOT performance among all models)

**Key Insight**: Best performance on HOT samples (60.86% accuracy), but overall balanced distribution across predictions.

---

### 3. Improved_gmm_temperature_classifier
**Configuration**: 9 clusters, 21 engineered features

**Overall Metrics:**
- Accuracy: 32.47%
- Precision: 0.2939
- Recall: 0.3247
- F1-Score: 0.2352

**Confusion Matrix:**
```
               Pred COLD  Pred NORMAL  Pred HOT
True COLD  :     1133          269         4
True NORMAL:     1328          293         7
True HOT   :     1055          309         3
```

**Per-Class Performance:**
- **COLD**: Precision=0.3222, Recall=0.8058, F1=0.4604 (best COLD recall)
- **NORMAL**: Precision=0.3364, Recall=0.1800, F1=0.2345
- **HOT**: Precision=0.2143, Recall=0.0022, F1=0.0043 (very poor HOT recall)

**Accuracy by Folder:**
- COLD: 80.58% (best COLD performance)
- NORMAL: 18.00% (poor)
- HOT: 0.22% (extremely poor - almost never predicts HOT correctly)

**Key Insight**: Best at detecting COLD samples (80.58% accuracy) but fails almost completely on HOT samples (0.22% accuracy).

---

### 4. Main_GMM_gmm_model
**Configuration**: 3 clusters, 4 raw sensor features

**Overall Metrics:**
- Accuracy: 26.63%
- Precision: 0.1620
- Recall: 0.2663
- F1-Score: 0.1835

**Confusion Matrix:**
```
               Pred COLD  Pred NORMAL  Pred HOT
True COLD  :      957            0       449
True NORMAL:     1400            0       228
True HOT   :     1152            0       215
```

**Per-Class Performance:**
- **COLD**: Precision=0.2727, Recall=0.6807, F1=0.3894
- **NORMAL**: Precision=0.0000, Recall=0.0000, F1=0.0000 (never predicts NORMAL)
- **HOT**: Precision=0.2410, Recall=0.1573, F1=0.1903

**Accuracy by Folder:**
- COLD: 68.07% (good COLD performance)
- NORMAL: 0.00% (completely fails - never predicts NORMAL)
- HOT: 15.73% (poor)

**Key Insight**: This simple 3-cluster model only predicts COLD or HOT, completely missing NORMAL class. Works reasonably well for COLD detection (68.07%) but fails on NORMAL and HOT.

---

## Key Findings

### 1. **Class Imbalance Issues**
All models show significant bias toward certain classes:
- **Improved_gmm_model_v1_original**: Biased toward NORMAL (83.85% accuracy on NORMAL, but only 0.43% on COLD)
- **Improved_gmm_temperature_classifier**: Biased toward COLD (80.58% accuracy on COLD, but only 0.22% on HOT)
- **Main_GMM_gmm_model**: Only predicts COLD or HOT, never NORMAL (0.00% NORMAL accuracy)

### 2. **Feature Engineering Impact**
- Models using 21 engineered features (Improved models) generally perform better than the simple 4-feature model (Main_GMM)
- However, the improvement is modest (26.63% → 33-35% accuracy)

### 3. **Model Confidence vs. Accuracy**
All models show very high confidence (mean > 0.98) but low accuracy, indicating:
- Overconfident predictions
- Potential distribution shift between training and test data
- Models are very certain but often wrong

### 4. **Best Use Cases**
- **For NORMAL detection**: Use `Improved_gmm_model_v1_original` (83.85% accuracy)
- **For COLD detection**: Use `Improved_gmm_temperature_classifier` (80.58% accuracy)
- **For HOT detection**: Use `Improved_gmm_model` (60.86% accuracy)

### 5. **Overall Performance**
None of the models achieve high overall accuracy (best: 35.33%). This suggests:
- Potential distribution shift between training data and new December 30 data
- Temperature ranges in new data (11°C COLD, 27-42°C NORMAL, 57-70°C HOT) may differ from training data
- Models may need retraining or fine-tuning on similar data

## Recommendations

1. **Model Selection**: Use `Improved_gmm_model_v1_original` for overall best performance, but be aware of its bias toward NORMAL predictions.

2. **Ensemble Approach**: Consider using different models for different temperature ranges:
   - COLD detection: `Improved_gmm_temperature_classifier`
   - NORMAL detection: `Improved_gmm_model_v1_original`
   - HOT detection: `Improved_gmm_model`

3. **Retraining**: Consider retraining models on data similar to the December 30 dataset to improve performance.

4. **Investigation**: Investigate why models show such high confidence but low accuracy - this suggests a distribution shift or calibration issue.

## Files Generated

1. **evaluation_summary.txt**: Text summary of all metrics
2. **{model_name}_evaluation_results.csv**: Detailed predictions for each model
3. **evaluate_models.py**: Evaluation script for future use

## Data Summary

- **Test Data Location**: `30 DECEMBER/COLD/`, `30 DECEMBER/HOT/`, `30 DECEMBER/NORMAL/`
- **Input Format**: 4 raw sensor readings (sensor_1, sensor_2, sensor_3, sensor_4) from VALID rows
- **Ground Truth**: Folder names (COLD, HOT, NORMAL)
- **Total Samples**: 4,401 readings across 21 CSV files

