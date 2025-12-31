"""
Evaluate multiple GMM models on new data from 30 DECEMBER folder
Tests models against ground truth labels (COLD, HOT, NORMAL)
"""

import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Suppress sklearn version mismatch warnings
warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')
warnings.filterwarnings('ignore', category=UserWarning)


def load_model(model_path):
    """Load a GMM model and return model components."""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Handle different model structures
        if isinstance(model_data, dict):
            gmm_model = model_data.get('gmm_model') or model_data.get('model')
            scaler = model_data.get('scaler')
            cluster_mapping = model_data.get('cluster_to_temp_mapping') or model_data.get('cluster_to_temperature_mapping')
            
            # If mapping is dict with string keys, convert to int keys and normalize values
            if cluster_mapping and isinstance(cluster_mapping, dict):
                normalized_mapping = {}
                for k, v in cluster_mapping.items():
                    key = int(k) if isinstance(k, (str, np.integer)) else k
                    # Normalize temperature class names (Cold -> COLD, etc.)
                    val = str(v).upper()
                    if 'COLD' in val:
                        normalized_mapping[key] = 'COLD'
                    elif 'HOT' in val:
                        normalized_mapping[key] = 'HOT'
                    elif 'NORMAL' in val:
                        normalized_mapping[key] = 'NORMAL'
                    else:
                        normalized_mapping[key] = 'NORMAL'  # Default
                cluster_mapping = normalized_mapping
            else:
                # Default mapping for 3-cluster models
                cluster_mapping = {0: 'COLD', 1: 'NORMAL', 2: 'HOT'}
        else:
            # Assume it's just the GMM model itself
            gmm_model = model_data
            scaler = None
            cluster_mapping = {0: 'COLD', 1: 'NORMAL', 2: 'HOT'}
        
        return {
            'gmm_model': gmm_model,
            'scaler': scaler,
            'cluster_mapping': cluster_mapping
        }
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def engineer_features_from_4_sensors(sensor_values):
    """
    Convert 4 raw sensor readings into 21 engineered features.
    This matches the feature engineering used in the Improved models.
    """
    sensor_array = np.array(sensor_values).reshape(-1, 4)
    
    # Extract individual sensors
    sensor_1 = sensor_array[:, 0]
    sensor_2 = sensor_array[:, 1]
    sensor_3 = sensor_array[:, 2]
    sensor_4 = sensor_array[:, 3]
    
    # Layer 2: Ratio features
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    ratio_1_3 = sensor_1 / (sensor_3 + 1e-8)
    ratio_2_4 = sensor_2 / (sensor_4 + 1e-8)
    
    # Layer 3: Statistical features
    sensor_sum = sensor_array.sum(axis=1)
    sensor_mean = sensor_array.mean(axis=1)
    sensor_std = sensor_array.std(axis=1)
    sensor_var = sensor_array.var(axis=1)
    
    # Layer 4: Extrema features
    sensor_max = sensor_array.max(axis=1)
    sensor_min = sensor_array.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    # Layer 5: Polynomial features
    sensor_3_squared = sensor_3 ** 2
    sensor_mean_squared = sensor_mean ** 2
    
    # Layer 6: Interaction features
    sum_1_2 = sensor_1 + sensor_2
    sum_3_4 = sensor_3 + sensor_4
    product_1_3 = sensor_1 * sensor_3
    product_2_4 = sensor_2 * sensor_4
    
    # Combine all features (4 raw + 17 engineered = 21 total)
    features = np.column_stack([
        sensor_array,  # Layer 1: Original 4 features
        ratio_1_2, ratio_3_4, ratio_1_3, ratio_2_4,  # Layer 2: 4 ratio features
        sensor_sum, sensor_mean, sensor_std, sensor_var,  # Layer 3: 4 statistical features
        sensor_max, sensor_min, sensor_range,  # Layer 4: 3 extrema features
        sensor_3_squared, sensor_mean_squared,  # Layer 5: 2 polynomial features
        sum_1_2, sum_3_4, product_1_3, product_2_4  # Layer 6: 4 interaction features
    ])
    
    return features


def predict_with_model(sensor_values, model_info, expected_features=None):
    """Make prediction using model info."""
    gmm_model = model_info['gmm_model']
    scaler = model_info['scaler']
    cluster_mapping = model_info['cluster_mapping']
    
    # Prepare input - check if we need to engineer features
    sensor_array = np.array(sensor_values).reshape(1, -1)
    
    # If expected_features is specified and different from input, engineer features
    if expected_features is not None and expected_features != sensor_array.shape[1]:
        if expected_features == 21 and sensor_array.shape[1] == 4:
            # Need to engineer 21 features from 4 sensors
            sensor_array = engineer_features_from_4_sensors(sensor_array)
        else:
            print(f"Warning: Feature mismatch - expected {expected_features}, got {sensor_array.shape[1]}")
    
    # Scale if scaler available
    if scaler is not None:
        try:
            sensor_scaled = scaler.transform(sensor_array)
        except ValueError as e:
            # If scaling fails due to feature mismatch, try to detect expected features
            if "features" in str(e):
                if sensor_array.shape[1] == 4:
                    # Try engineering features
                    sensor_array = engineer_features_from_4_sensors(sensor_array)
                    sensor_scaled = scaler.transform(sensor_array)
                else:
                    raise
            else:
                raise
    else:
        sensor_scaled = sensor_array
    
    # Predict
    cluster_id = gmm_model.predict(sensor_scaled)[0]
    probabilities = gmm_model.predict_proba(sensor_scaled)[0]
    
    # Map cluster to class
    if cluster_mapping and cluster_id in cluster_mapping:
        predicted_class = cluster_mapping[cluster_id]
    else:
        # Fallback: use cluster ID directly (0=COLD, 1=NORMAL, 2=HOT)
        if cluster_id == 0:
            predicted_class = 'COLD'
        elif cluster_id == 1:
            predicted_class = 'NORMAL'
        elif cluster_id == 2:
            predicted_class = 'HOT'
        else:
            predicted_class = 'NORMAL'  # Default
    
    confidence = float(probabilities[cluster_id])
    
    return {
        'class': predicted_class,
        'cluster_id': int(cluster_id),
        'confidence': confidence,
        'all_probs': probabilities.tolist()
    }


def load_sensor_data_from_csv(csv_path):
    """Load sensor readings from raw sensor log CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check column names (may have different casing)
        columns = df.columns.str.lower()
        
        # Find sensor columns (handle different naming conventions)
        sensor_cols = []
        for i in range(1, 5):
            # Try different variations
            for pattern in [f'sensor_{i}', f'sensor-{i}', f'sensor {i}']:
                if pattern in columns:
                    sensor_cols.append(df.columns[columns == pattern][0])
                    break
        
        if len(sensor_cols) != 4:
            # Try to find columns containing 'sensor'
            sensor_cols = [col for col in df.columns if 'sensor' in col.lower() and any(str(i) in col for i in [1,2,3,4])]
            sensor_cols = sorted(sensor_cols)[:4]  # Take first 4
        
        if len(sensor_cols) != 4:
            print(f"Warning: Could not find 4 sensor columns in {csv_path}. Found: {df.columns.tolist()}")
            return None
        
        # Filter for VALID rows only (case-insensitive)
        status_col = None
        for col in df.columns:
            if 'status' in col.lower():
                status_col = col
                break
        
        if status_col:
            valid_mask = df[status_col].str.upper() == 'VALID'
            df_valid = df[valid_mask].copy()
        else:
            # If no status column, use all rows (skip header if it's IGNORED)
            df_valid = df.copy()
            # Remove rows where first sensor value looks like a header
            if len(df_valid) > 0:
                first_val = df_valid[sensor_cols[0]].iloc[0]
                if isinstance(first_val, str) and 'sensor' in str(first_val).lower():
                    df_valid = df_valid.iloc[1:]
        
        if len(df_valid) == 0:
            return None
        
        # Extract sensor values
        sensor_data = df_valid[sensor_cols].values.astype(float)
        
        return sensor_data
        
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None


def evaluate_models_on_data(data_dir='30 DECEMBER', model_paths=None):
    """Evaluate all models on the new data."""
    
    if model_paths is None:
        # Use absolute paths relative to project root
        base_dir = Path(__file__).parent.parent
        model_paths = [
            base_dir / r'Improved\gmm_model.pkl',
            base_dir / r'Improved\gmm_model_v1_original.pkl',
            base_dir / r'Improved\gmm_temperature_classifier.pkl',
            base_dir / r'Main_GMM\gmm_model.pkl'
        ]
        model_paths = [str(p) for p in model_paths]
    
    # Load all models
    print("=" * 80)
    print("LOADING MODELS")
    print("=" * 80)
    models = {}
    for model_path in model_paths:
        # Create unique model name from full path
        model_path_obj = Path(model_path)
        # Use parent folder name + model name for uniqueness
        if 'Improved' in str(model_path):
            model_name = f"Improved_{model_path_obj.stem}"
        elif 'Main_GMM' in str(model_path):
            model_name = f"Main_GMM_{model_path_obj.stem}"
        else:
            model_name = f"{model_path_obj.parent.name}_{model_path_obj.stem}"
        
        print(f"\nLoading {model_name} from {model_path}...")
        model_info = load_model(model_path)
        if model_info:
            models[model_name] = model_info
            print(f"  [OK] Loaded successfully")
            print(f"  - Clusters: {model_info['gmm_model'].n_components}")
            print(f"  - Has scaler: {model_info['scaler'] is not None}")
            print(f"  - Cluster mapping: {model_info['cluster_mapping']}")
        else:
            print(f"  [FAILED] Failed to load")
    
    if not models:
        print("No models loaded successfully!")
        return None
    
    # Collect data from all folders
    print("\n" + "=" * 80)
    print("LOADING TEST DATA")
    print("=" * 80)
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory {data_dir} not found!")
        return None
    
    # Ground truth mapping from folder names
    ground_truth_map = {
        'COLD': 'COLD',
        'HOT': 'HOT',
        'NORMAL': 'NORMAL'
    }
    
    all_results = {}
    
    # Process each folder (COLD, HOT, NORMAL)
    for folder_name in ['COLD', 'HOT', 'NORMAL']:
        folder_path = data_path / folder_name
        if not folder_path.exists():
            print(f"\nWarning: Folder {folder_name} not found, skipping...")
            continue
        
        ground_truth_class = ground_truth_map.get(folder_name, folder_name.upper())
        print(f"\nProcessing {folder_name} folder (Ground truth: {ground_truth_class})...")
        
        # Find all raw sensor log CSV files
        csv_files = sorted(folder_path.glob('raw_sensor_log*.csv'))
        print(f"  Found {len(csv_files)} raw sensor log files")
        
        # Collect all sensor readings from this folder
        folder_sensor_data = []
        for csv_file in csv_files:
            sensor_data = load_sensor_data_from_csv(csv_file)
            if sensor_data is not None and len(sensor_data) > 0:
                folder_sensor_data.append(sensor_data)
                print(f"    {csv_file.name}: {len(sensor_data)} valid sensor readings")
        
        if not folder_sensor_data:
            print(f"  No valid sensor data found in {folder_name} folder")
            continue
        
        # Combine all readings from this folder
        all_readings = np.vstack(folder_sensor_data)
        print(f"  Total valid readings: {len(all_readings)}")
        
        # Evaluate each model
        for model_name, model_info in models.items():
            if model_name not in all_results:
                all_results[model_name] = {
                    'predictions': [],
                    'ground_truth': [],
                    'confidences': [],
                    'clusters': [],
                    'files': []
                }
            
            # Make predictions for all readings
            # Detect expected features from scaler if available
            expected_features = None
            if model_info['scaler'] is not None:
                try:
                    # Try to get expected features from scaler
                    test_input = np.array([[100, 300, 500, 250]]).reshape(1, -1)
                    _ = model_info['scaler'].transform(test_input)
                    expected_features = 4
                except ValueError as e:
                    if "features" in str(e):
                        # Extract expected feature count from error message
                        import re
                        match = re.search(r'expecting (\d+)', str(e))
                        if match:
                            expected_features = int(match.group(1))
                        else:
                            # Default to 21 for Improved models
                            expected_features = 21
            
            for reading in all_readings:
                if len(reading) != 4:
                    continue
                
                try:
                    pred = predict_with_model(reading, model_info, expected_features=expected_features)
                    
                    all_results[model_name]['predictions'].append(pred['class'])
                    all_results[model_name]['ground_truth'].append(ground_truth_class)
                    all_results[model_name]['confidences'].append(pred['confidence'])
                    all_results[model_name]['clusters'].append(pred['cluster_id'])
                    all_results[model_name]['files'].append(folder_name)
                except Exception as e:
                    print(f"    Warning: Failed to predict with {model_name} for reading {reading}: {e}")
                    continue
    
    # Calculate metrics for each model
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    summary_results = {}
    
    for model_name, results in all_results.items():
        if len(results['predictions']) == 0:
            print(f"\n{model_name}: No predictions made")
            continue
        
        predictions = np.array(results['predictions'])
        ground_truth = np.array(results['ground_truth'])
        confidences = np.array(results['confidences'])
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
        recall = recall_score(ground_truth, predictions, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
        f1 = f1_score(ground_truth, predictions, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
        
        # Confusion matrix
        conf_matrix = confusion_matrix(ground_truth, predictions, labels=['COLD', 'NORMAL', 'HOT'])
        
        # Per-class metrics
        report = classification_report(
            ground_truth, predictions,
            labels=['COLD', 'NORMAL', 'HOT'],
            output_dict=True,
            zero_division=0
        )
        
        summary_results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'total_samples': len(predictions),
            'mean_confidence': np.mean(confidences),
            'predictions': predictions,
            'ground_truth': ground_truth,
            'confidences': confidences
        }
        
        # Print results
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        print(f"Total Samples: {len(predictions):,}")
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Mean Confidence: {np.mean(confidences):.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"               Pred COLD  Pred NORMAL  Pred HOT")
        labels = ['COLD', 'NORMAL', 'HOT']
        for i, label in enumerate(labels):
            print(f"True {label:6}: {conf_matrix[i, 0]:8}  {conf_matrix[i, 1]:11}  {conf_matrix[i, 2]:8}")
        
        print(f"\nPer-Class Performance:")
        for label in ['COLD', 'NORMAL', 'HOT']:
            if label in report:
                metrics = report[label]
                print(f"  {label}:")
                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall:    {metrics['recall']:.4f}")
                print(f"    F1-Score:  {metrics['f1-score']:.4f}")
                print(f"    Support:   {int(metrics['support'])}")
        
        # Breakdown by folder
        print(f"\nAccuracy by Folder:")
        for folder in ['COLD', 'HOT', 'NORMAL']:
            mask = np.array(results['files']) == folder
            if np.sum(mask) > 0:
                folder_preds = predictions[mask]
                folder_true = ground_truth[mask]
                folder_acc = accuracy_score(folder_true, folder_preds)
                print(f"  {folder}: {folder_acc:.4f} ({folder_acc*100:.2f}%) - {np.sum(mask)} samples")
    
    # Comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(f"{'Model':<40} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Mean Conf':<12}")
    print("-" * 80)
    for model_name, results in summary_results.items():
        print(f"{model_name:<40} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
              f"{results['recall']:<12.4f} {results['f1']:<12.4f} {results['mean_confidence']:<12.4f}")
    
    # Save detailed results to CSV
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_dir = Path(data_dir)
    for model_name, results in all_results.items():
        if len(results['predictions']) == 0:
            continue
        
        df_results = pd.DataFrame({
            'ground_truth': results['ground_truth'],
            'prediction': results['predictions'],
            'cluster_id': results['clusters'],
            'confidence': results['confidences'],
            'folder': results['files']
        })
        
        output_file = output_dir / f"{model_name}_evaluation_results.csv"
        df_results.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
    
    # Save summary report
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("GMM MODELS EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for model_name, results in summary_results.items():
            f.write(f"MODEL: {model_name}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Samples: {results['total_samples']:,}\n")
            f.write(f"Accuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall:    {results['recall']:.4f}\n")
            f.write(f"F1-Score:  {results['f1']:.4f}\n")
            f.write(f"Mean Confidence: {results['mean_confidence']:.4f}\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write("               Pred COLD  Pred NORMAL  Pred HOT\n")
            labels = ['COLD', 'NORMAL', 'HOT']
            conf_matrix = results['confusion_matrix']
            for i, label in enumerate(labels):
                f.write(f"True {label:6}: {conf_matrix[i, 0]:8}  {conf_matrix[i, 1]:11}  {conf_matrix[i, 2]:8}\n")
            f.write("\n")
    
    print(f"\nSaved summary: {summary_file}")
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE!")
    print("=" * 80)
    
    return summary_results


if __name__ == "__main__":
    # Run evaluation with absolute paths
    base_dir = Path(__file__).parent.parent
    results = evaluate_models_on_data(
        data_dir=str(Path(__file__).parent),
        model_paths=[
            str(base_dir / r'Improved\gmm_model.pkl'),
            str(base_dir / r'Improved\gmm_model_v1_original.pkl'),
            str(base_dir / r'Improved\gmm_temperature_classifier.pkl'),
            str(base_dir / r'Main_GMM\gmm_model.pkl')
        ]
    )

