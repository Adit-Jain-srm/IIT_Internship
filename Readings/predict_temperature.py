"""
Temperature Classification Prediction Script
============================================
Predicts temperature class (Cold_normal or Hot) from sensor CSV files.

NOTE: Sensor 1 is excluded from analysis due to abnormal behavior.
      Only Sensors 2, 3, 4 are used for prediction.

Usage:
    python predict_temperature.py [input_folder]
    
Data Requirements:
    - CSV files with columns: Time_s, Sensor_2, Sensor_3, Sensor_4
    - Files with duration < 4.2 seconds will be skipped
    - Data will be resampled to 226 rows
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys

# Configuration
INPUT_FOLDER = "input_data"
MODEL_FILE = "gmm_temperature_classifier_best.pkl"
MIN_DURATION = 4.2
TARGET_ROWS = 226


def aggregate_file_to_features(file_sensor_data):
    """Extract statistical features from sensor data.
    Input: array of shape (n_rows, 3) - Sensors 2, 3, 4 only
    Output: 12 features (mean, std, min, max for each sensor)"""
    X = np.array(file_sensor_data)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.concatenate([
        X.mean(axis=0),  # 3 features
        X.std(axis=0),   # 3 features
        X.min(axis=0),   # 3 features
        X.max(axis=0)    # 3 features
    ])


def clean_and_validate_data(df, min_duration=MIN_DURATION, target_rows=TARGET_ROWS):
    """Clean and validate sensor data."""
    # Check required columns (Sensor 1 excluded)
    required_cols = ['Time_s', 'Sensor_2', 'Sensor_3', 'Sensor_4']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    
    # Filter IGNORED status
    if 'Status' in df.columns:
        df_valid = df[df['Status'] != 'IGNORED'].copy()
    else:
        df_valid = df.copy()
    
    if len(df_valid) == 0:
        return None, "No valid rows"
    
    # Check duration
    duration = df_valid['Time_s'].max() - df_valid['Time_s'].min()
    if duration < min_duration:
        return None, f"Duration {duration:.2f}s < {min_duration}s"
    
    # Resample to target rows
    if len(df_valid) > target_rows:
        df_valid = df_valid.iloc[:target_rows].copy()
    elif len(df_valid) < target_rows:
        last_row = df_valid.iloc[-1:].copy()
        n_needed = target_rows - len(df_valid)
        df_valid = pd.concat([df_valid] + [last_row] * n_needed, ignore_index=True)
    
    return df_valid, f"OK ({duration:.2f}s, {len(df_valid)} rows)"


def predict_files(input_folder=INPUT_FOLDER, model_file=MODEL_FILE):
    """Load CSV files and predict temperature class."""
    print("=" * 70)
    print("GMM TEMPERATURE CLASSIFICATION - PREDICTION")
    print("Note: Sensor 1 excluded, using Sensors 2, 3, 4 only")
    print("=" * 70)
    
    # Check input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"\nERROR: Folder '{input_folder}' not found!")
        input_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {input_path.absolute()}")
        return
    
    # Find CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        print(f"\nNo CSV files in '{input_folder}'")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s)")
    
    # Load model
    print(f"\nLoading model: {model_file}")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"  Features: {model['n_features']}")
        print(f"  Test accuracy: {model['test_accuracy']*100:.2f}%")
    except FileNotFoundError:
        print(f"ERROR: Model file not found!")
        return
    
    gmm = model['gmm_model']
    scaler = model['scaler']
    feature_indices = model['feature_indices']
    cluster_mapping = model['cluster_to_label_mapping']
    
    print(f"\n{'='*70}")
    print("PROCESSING")
    print("=" * 70)
    
    results = []
    
    for csv_file in csv_files:
        print(f"\n{csv_file.name}")
        
        try:
            df = pd.read_csv(csv_file)
            df_clean, status = clean_and_validate_data(df)
            
            if df_clean is None:
                print(f"  SKIPPED: {status}")
                continue
            
            print(f"  {status}")
            
            # Extract features (Sensor 1 excluded)
            sensor_data = df_clean[['Sensor_2', 'Sensor_3', 'Sensor_4']].values.astype(float)
            features = aggregate_file_to_features(sensor_data)
            
            # Select and scale features
            X = features[feature_indices].reshape(1, -1)
            X_scaled = scaler.transform(X)
            
            # Predict
            cluster = gmm.predict(X_scaled)[0]
            probs = gmm.predict_proba(X_scaled)[0]
            prediction = cluster_mapping[cluster]
            confidence = probs[cluster] * 100
            
            print(f"  PREDICTION: {prediction} ({confidence:.1f}%)")
            results.append({'file': csv_file.name, 'prediction': prediction, 'confidence': confidence})
            
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print("=" * 70)
    
    if results:
        print(f"\nProcessed: {len(results)} file(s)")
        cold = sum(1 for r in results if r['prediction'] == 'Cold_normal')
        hot = sum(1 for r in results if r['prediction'] == 'Hot')
        print(f"Cold_normal: {cold}, Hot: {hot}")
        
        # Save results
        pd.DataFrame(results).to_csv("prediction_results.csv", index=False)
        print(f"\nSaved: prediction_results.csv")
    
    print("\nDone!")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else INPUT_FOLDER
    predict_files(input_folder=folder)
