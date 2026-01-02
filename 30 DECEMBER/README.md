# GMM Model Optimization - December 30 Data

This directory contains the optimized GMM model for temperature classification using December 30 sensor data.

## Files

- **`GMM_Optimization_Notebook.ipynb`**: Main notebook containing the complete pipeline:
  - Data loading from COLD, HOT, NORMAL folders
  - Statistical feature engineering (3 variants)
  - GMM model training and optimization (9 configurations)
  - Model evaluation and selection
  - Best model saving

- **`gmm_model_best.pkl`**: The best performing GMM model (pickled)
  - Features: v2_enhanced (15 features)
  - Covariance type: diag
  - Test accuracy: 40.86%

- **`model_metadata.json`**: Metadata containing:
  - Best model configuration
  - Performance metrics
  - All tested configurations

- **`model_comparison.png`**: Visualization comparing all model configurations

## Best Model Details

- **Feature Engineering**: v2_enhanced (15 features)
  - 4 raw sensor readings
  - Statistical features: sum, mean, std, var, median
  - Range features: max, min, range
  - Percentile features: q25, q75, IQR

- **Covariance Type**: diagonal

- **Performance**:
  - Test Accuracy: 40.86%
  - Test F1-Score: 0.296
  - Classes Predicted: 2/3 (COLD, NORMAL)
  - Per-Class Accuracy:
    - COLD: 19.57%
    - NORMAL: 93.56%
    - HOT: 0% (not predicted)

## Usage

1. Open `GMM_Optimization_Notebook.ipynb` in Jupyter
2. Run all cells to reproduce the optimization process
3. Use `gmm_model_best.pkl` for inference with new data

## Data Structure

The notebook expects the following folder structure:
```
30 DECEMBER/
├── COLD/
│   └── raw_sensor_log_*.csv
├── HOT/
│   └── raw_sensor_log_*.csv
└── NORMAL/
    └── raw_sensor_log_*.csv
```

Each CSV file should contain columns: `Time_s`, `Status`, `Sensor_1`, `Sensor_2`, `Sensor_3`, `Sensor_4`

