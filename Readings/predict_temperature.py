"""
Temperature Classification Prediction Script
============================================
This script loads CSV sensor files from the 'input_data' folder,
cleans the data, and predicts the temperature class (Cold_normal or Hot)
using the trained GMM model.

Usage:
    1. Place your CSV files in the 'input_data' folder
    2. Run: python predict_temperature.py
    
Data Requirements:
    - CSV files with columns: Time_s, Status, Sensor_1, Sensor_2, Sensor_3, Sensor_4
    - Files with duration < 4.2 seconds will be skipped
    - Data will be resampled to exactly 226 rows
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import sys


# ============================================================================
# CONFIGURATION
# ============================================================================
INPUT_FOLDER = "input_data"           # Folder containing CSV files to predict
MODEL_FILE = "gmm_temperature_classifier_best.pkl"
MIN_DURATION = 4.2                    # Minimum duration in seconds
TARGET_ROWS = 226                     # Target number of data rows (excluding header)


# ============================================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================================
def aggregate_file_to_features(file_sensor_data):
    """Aggregate all rows from a file into statistical features.
    Input: file_sensor_data - array of shape (n_rows, 4_sensors)
    Output: single feature vector with 29 statistical features"""
    X = np.array(file_sensor_data)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Per-sensor statistics (4 sensors × 4 stats = 16 features)
    sensor_means = X.mean(axis=0)
    sensor_stds = X.std(axis=0)
    sensor_maxs = X.max(axis=0)
    sensor_mins = X.min(axis=0)
    
    # Global statistics (13 features)
    all_values = X.flatten()
    global_mean = all_values.mean()
    global_std = all_values.std()
    global_var = all_values.var()
    global_max = all_values.max()
    global_min = all_values.min()
    global_range = global_max - global_min
    global_median = np.median(all_values)
    
    # Percentiles
    q25 = np.percentile(all_values, 25)
    q75 = np.percentile(all_values, 75)
    iqr = q75 - q25
    
    # Higher moments
    mean_centered = all_values - global_mean
    global_skew = (mean_centered ** 3).mean() / (global_std ** 3 + 1e-8)
    global_kurtosis = (mean_centered ** 4).mean() / (global_std ** 4 + 1e-8)
    cv = global_std / (global_mean + 1e-8)
    
    # Combine all features (16 + 13 = 29 features)
    features = np.concatenate([
        sensor_means,      # 4 features (indices 0-3)
        sensor_stds,       # 4 features (indices 4-7)
        sensor_maxs,       # 4 features (indices 8-11)
        sensor_mins,       # 4 features (indices 12-15)
        [global_mean, global_std, global_var, global_max, global_min, global_range, 
         global_median, q25, q75, iqr, cv, global_skew, global_kurtosis]  # 13 features (indices 16-28)
    ])
    
    return features


# ============================================================================
# DATA CLEANING FUNCTION
# ============================================================================
def clean_and_validate_data(df, filename, min_duration=MIN_DURATION, target_rows=TARGET_ROWS):
    """
    Clean and validate sensor data:
    1. Remove rows with Status='IGNORED'
    2. Check if duration >= min_duration
    3. Resample to exactly target_rows
    
    Returns:
        cleaned_df: DataFrame with exactly target_rows, or None if invalid
        status: Status message
    """
    # Check required columns
    required_cols = ['Time_s', 'Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return None, f"Missing columns: {missing_cols}"
    
    # Filter out IGNORED rows if Status column exists
    if 'Status' in df.columns:
        df_valid = df[df['Status'] != 'IGNORED'].copy()
    else:
        df_valid = df.copy()
    
    if len(df_valid) == 0:
        return None, "No valid rows after filtering IGNORED status"
    
    # Check duration
    duration = df_valid['Time_s'].max() - df_valid['Time_s'].min()
    if duration < min_duration:
        return None, f"Duration {duration:.2f}s < {min_duration}s minimum"
    
    # Resample to target_rows
    if len(df_valid) > target_rows:
        # Take first target_rows
        df_valid = df_valid.iloc[:target_rows].copy()
    elif len(df_valid) < target_rows:
        # Repeat last row to reach target_rows
        last_row = df_valid.iloc[-1:].copy()
        n_needed = target_rows - len(df_valid)
        additional_rows = pd.concat([last_row] * n_needed, ignore_index=True)
        df_valid = pd.concat([df_valid, additional_rows], ignore_index=True)
    
    return df_valid, f"OK (duration: {duration:.2f}s, rows: {len(df_valid)})"


# ============================================================================
# MAIN PREDICTION FUNCTION
# ============================================================================
def predict_files(input_folder=INPUT_FOLDER, model_file=MODEL_FILE):
    """
    Load all CSV files from input folder, clean data, and predict using GMM model.
    """
    print("=" * 80)
    print("GMM TEMPERATURE CLASSIFICATION - PREDICTION")
    print("=" * 80)
    
    # Check input folder
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"\n❌ ERROR: Input folder '{input_folder}' not found!")
        print(f"   Please create the folder and place your CSV files in it.")
        input_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ Created folder: {input_path.absolute()}")
        return
    
    # Find CSV files
    csv_files = sorted(input_path.glob("*.csv"))
    if not csv_files:
        print(f"\n⚠ No CSV files found in '{input_folder}'")
        print(f"   Please place your sensor CSV files in: {input_path.absolute()}")
        return
    
    print(f"\nInput folder: {input_path.absolute()}")
    print(f"Found {len(csv_files)} CSV file(s)")
    
    # Load model
    print(f"\nLoading model: {model_file}")
    try:
        with open(model_file, 'rb') as f:
            model = pickle.load(f)
        print(f"   ✓ Model loaded successfully")
        print(f"   - Features: {model['n_features']}")
        print(f"   - Training accuracy: {model['train_accuracy']*100:.2f}%")
        print(f"   - Test accuracy: {model['test_accuracy']*100:.2f}%")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Model file '{model_file}' not found!")
        print(f"   Please run the training notebook first.")
        return
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        return
    
    # Extract model components
    gmm = model['gmm_model']
    scaler = model['scaler']
    feature_indices = model['feature_indices']
    cluster_mapping = model['cluster_to_label_mapping']
    
    print(f"\n" + "=" * 80)
    print("PROCESSING FILES")
    print("=" * 80)
    
    results = []
    skipped = []
    
    for csv_file in csv_files:
        print(f"\n📁 {csv_file.name}")
        
        try:
            # Load CSV
            df = pd.read_csv(csv_file)
            print(f"   Loaded: {len(df)} rows")
            
            # Clean and validate
            df_clean, status = clean_and_validate_data(df, csv_file.name)
            
            if df_clean is None:
                print(f"   ❌ SKIPPED: {status}")
                skipped.append({'file': csv_file.name, 'reason': status})
                continue
            
            print(f"   Cleaned: {status}")
            
            # Extract sensor data
            sensor_data = df_clean[['Sensor_1', 'Sensor_2', 'Sensor_3', 'Sensor_4']].values.astype(float)
            
            # Extract features (29 features)
            features = aggregate_file_to_features(sensor_data)
            
            # Select only the trained features
            X = features[feature_indices].reshape(1, -1)
            
            # Scale and predict
            X_scaled = scaler.transform(X)
            cluster = gmm.predict(X_scaled)[0]
            probabilities = gmm.predict_proba(X_scaled)[0]
            
            # Map cluster to label
            prediction = cluster_mapping[cluster]
            confidence = probabilities[cluster] * 100
            
            print(f"   ✓ PREDICTION: {prediction} (confidence: {confidence:.1f}%)")
            
            results.append({
                'file': csv_file.name,
                'prediction': prediction,
                'cluster': cluster,
                'confidence': confidence,
                'prob_cluster_0': probabilities[0] * 100,
                'prob_cluster_1': probabilities[1] * 100
            })
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            skipped.append({'file': csv_file.name, 'reason': str(e)})
    
    # Print summary
    print(f"\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    
    if results:
        print(f"\n✓ Successfully processed: {len(results)} file(s)")
        print(f"\n{'File':<50} {'Prediction':<15} {'Confidence':>12}")
        print("-" * 80)
        for r in results:
            print(f"{r['file']:<50} {r['prediction']:<15} {r['confidence']:>10.1f}%")
        
        # Count predictions
        cold_count = sum(1 for r in results if r['prediction'] == 'Cold_normal')
        hot_count = sum(1 for r in results if r['prediction'] == 'Hot')
        print(f"\n📊 Class Distribution:")
        print(f"   Cold_normal: {cold_count} file(s)")
        print(f"   Hot: {hot_count} file(s)")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        output_file = "prediction_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    
    if skipped:
        print(f"\n⚠ Skipped: {len(skipped)} file(s)")
        for s in skipped:
            print(f"   - {s['file']}: {s['reason']}")
    
    print(f"\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    # Check for custom input folder from command line
    if len(sys.argv) > 1:
        input_folder = sys.argv[1]
    else:
        input_folder = INPUT_FOLDER
    
    predict_files(input_folder=input_folder)

