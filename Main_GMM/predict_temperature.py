"""
GMM Temperature Prediction & Validation
Predicts: COLD, NORMAL, or HOT
Tests against sensor readings from collect_data folder
"""

import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Suppress sklearn version mismatch warnings
warnings.filterwarnings('ignore', message='.*Trying to unpickle estimator.*')


def predict_temperature(sensor_values, model_file='gmm_model.pkl'):
    """
    Predict temperature class from 4 sensor values.
    
    Parameters:
    -----------
    sensor_values : list or array
        [sensor_1, sensor_2, sensor_3, sensor_4]
        
    model_file : str
        Path to gmm_model.pkl
    
    Returns:
    --------
    dict with:
        - 'class': 'COLD', 'NORMAL', or 'HOT'
        - 'cluster_id': 0, 1, or 2
        - 'confidence': probability (0-1)
        - 'all_probs': [prob_0, prob_1, prob_2]
    """
    
    # Load model file (contains dict with model and scaler)
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    # Extract components
    gmm_model = model_data['gmm_model']
    scaler = model_data['scaler']
    
    # Prepare and scale input
    sensor_array = np.array(sensor_values).reshape(1, -1)
    sensor_scaled = scaler.transform(sensor_array)
    
    # Predict
    cluster_id = gmm_model.predict(sensor_scaled)[0]
    probabilities = gmm_model.predict_proba(sensor_scaled)[0]
    
    # Map cluster to temperature class
    class_mapping = {
        0: 'COLD',
        1: 'NORMAL',
        2: 'HOT'
    }
    
    return {
        'class': class_mapping[cluster_id],
        'cluster_id': int(cluster_id),
        'confidence': float(probabilities[cluster_id]),
        'all_probs': {
            'cold': float(probabilities[0]),
            'normal': float(probabilities[1]),
            'hot': float(probabilities[2])
        }
    }


def temperature_range_to_class(temp_range):
    """Map temperature range to class label."""
    range_val = int(temp_range.split('-')[0])
    if range_val < 45:
        return 'COLD'
    elif range_val < 60:
        return 'NORMAL'
    else:
        return 'HOT'


def load_and_predict(collect_data_path='collect_data', model_file='gmm_model.pkl'):
    """
    Load sensor readings from collect_data folder and make predictions.
    Calculate accuracy against ground truth (folder names).
    
    Parameters:
    -----------
    collect_data_path : str
        Path to collect_data folder
    
    model_file : str
        Path to gmm_model.pkl
    
    Returns:
    --------
    dict with accuracy metrics and detailed results
    """
    
    # First, load model once
    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)
    
    gmm_model = model_data['gmm_model']
    scaler = model_data['scaler']
    
    # Collect all predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    all_confidences = []
    all_sensors = []
    all_temp_ranges = []  # Track which temperature range each sample came from
    
    # Walk through temperature range folders
    collect_path = Path(collect_data_path)
    
    if not collect_path.exists():
        print(f"Error: {collect_data_path} not found!")
        return None
    
    for temp_folder in sorted(collect_path.iterdir()):
        if not temp_folder.is_dir():
            continue
        
        # Get ground truth from folder name
        ground_truth_class = temperature_range_to_class(temp_folder.name)
        
        # Load all CSV files in this folder
        csv_files = sorted(temp_folder.glob('*.csv'))
        
        for csv_file in csv_files:
            try:
                # Load sensor data
                df = pd.read_csv(csv_file)
                
                # Extract sensor columns (skip first row if ignored=1)
                sensor_cols = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
                sensor_data = df[sensor_cols].values
                
                # Make predictions for each row
                for sensors in sensor_data:
                    # Scale and predict
                    sensor_scaled = scaler.transform(sensors.reshape(1, -1))
                    cluster_id = gmm_model.predict(sensor_scaled)[0]
                    probabilities = gmm_model.predict_proba(sensor_scaled)[0]
                    
                    # Map cluster to class (based on majority voting in training)
                    # Clusters: 0,1,2,5 -> NORMAL; 3,8 -> COLD; 4,6,7 -> HOT
                    cluster_mapping = {
                        0: 'NORMAL', 1: 'NORMAL', 2: 'NORMAL', 3: 'COLD',
                        4: 'HOT', 5: 'NORMAL', 6: 'HOT', 7: 'HOT', 8: 'COLD'
                    }
                    predicted_class = cluster_mapping[cluster_id]
                    confidence = float(probabilities[cluster_id])
                    
                    # Store results
                    all_predictions.append(predicted_class)
                    all_ground_truth.append(ground_truth_class)
                    all_confidences.append(confidence)
                    all_sensors.append(sensors)
                    all_temp_ranges.append(temp_folder.name)  # Store folder name
                    
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
    
    # Calculate metrics
    if len(all_predictions) == 0:
        print("No data loaded!")
        return None
    
    all_predictions = np.array(all_predictions)
    all_ground_truth = np.array(all_ground_truth)
    all_confidences = np.array(all_confidences)
    
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    precision = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    f1 = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(all_ground_truth, all_predictions, labels=['COLD', 'NORMAL', 'HOT'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'total_samples': len(all_predictions),
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'confidences': all_confidences,
        'sensors': np.array(all_sensors),
        'temp_ranges': np.array(all_temp_ranges)  # Add temperature ranges
    }


if __name__ == "__main__":
    
    print("=" * 80)
    print("GMM TEMPERATURE PREDICTION - VALIDATION ON COLLECTED DATA")
    print("=" * 80)
    
    # Load and validate on collect_data
    print("\nLoading sensor readings from collect_data folder...")
    results = load_and_predict(
        collect_data_path='../collect_data',
        model_file='gmm_model.pkl'
    )
    
    if results:
        print("\n" + "=" * 80)
        print("OVERALL ACCURACY METRICS")
        print("=" * 80)
        print(f"Total Samples Tested: {results['total_samples']:,}")
        print(f"\nAccuracy:  {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        
        # Confusion matrix
        print("\n" + "=" * 80)
        print("CONFUSION MATRIX")
        print("=" * 80)
        print("               Pred COLD  Pred NORMAL  Pred HOT")
        labels = ['COLD', 'NORMAL', 'HOT']
        for i, label in enumerate(labels):
            print(f"True {label:6}: {results['confusion_matrix'][i, 0]:8}  {results['confusion_matrix'][i, 1]:11}  {results['confusion_matrix'][i, 2]:8}")
        
        # Per-category breakdown
        print("\n" + "=" * 80)
        print("PER-CATEGORY PERFORMANCE")
        print("=" * 80)
        
        report = classification_report(
            results['ground_truth'],
            results['predictions'],
            labels=['COLD', 'NORMAL', 'HOT'],
            output_dict=True
        )
        
        for label in ['COLD', 'NORMAL', 'HOT']:
            metrics = report[label]
            print(f"\n{label}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1-score']:.4f}")
            print(f"  Support:   {int(metrics['support'])} samples")
        
        # Confidence analysis
        print("\n" + "=" * 80)
        print("PREDICTION CONFIDENCE ANALYSIS")
        print("=" * 80)
        high_conf = np.sum(results['confidences'] >= 0.8)
        medium_conf = np.sum((results['confidences'] >= 0.6) & (results['confidences'] < 0.8))
        low_conf = np.sum(results['confidences'] < 0.6)
        
        print(f"High confidence (≥0.80):   {high_conf:6} ({high_conf/len(results['confidences'])*100:.1f}%)")
        print(f"Medium confidence (0.60-0.80): {medium_conf:6} ({medium_conf/len(results['confidences'])*100:.1f}%)")
        print(f"Low confidence (<0.60):    {low_conf:6} ({low_conf/len(results['confidences'])*100:.1f}%)")
        print(f"Mean confidence:           {np.mean(results['confidences']):.4f}")
        
        # Summary statistics by temperature range
        print("\n" + "=" * 80)
        print("ACCURACY BY TEMPERATURE RANGE")
        print("=" * 80)
        
        # Calculate accuracy per temperature range
        for temp_range in sorted(set(results['temp_ranges'])):
            mask = results['temp_ranges'] == temp_range
            expected_class = temperature_range_to_class(temp_range)
            range_preds = results['predictions'][mask]
            range_true = results['ground_truth'][mask]
            
            if len(range_preds) > 0:
                range_acc = np.mean(range_preds == range_true)
                print(f"{temp_range}°C ({expected_class:6}): {len(range_preds):5} samples, Accuracy: {range_acc:.4f} ({range_acc*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print("✓ Validation Complete!")
        print("=" * 80)
    else:
        print("Failed to load data!")