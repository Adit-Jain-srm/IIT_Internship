"""
Evaluate GMM temperature predictions on collect_data
Maps predicted labels to ground truth labels from folder names
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json

# Add parent directory to path to import predict_temperature
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_temperature import predict_temperature


def map_temperature_range_to_class(temp_range):
    """
    Map temperature range folder name to ground truth class.
    
    Parameters:
    -----------
    temp_range : str
        Folder name like '15-25', '45-50', '50-60', '60-70'
    
    Returns:
    --------
    str: 'COLD', 'NORMAL', or 'HOT'
    """
    # Extract min temperature from range
    temp_min = int(temp_range.split('-')[0])
    
    # Mapping based on temperature ranges
    if temp_min < 30:
        return 'COLD'
    elif temp_min < 50:
        return 'NORMAL'
    else:
        return 'HOT'


def evaluate_collect_data(data_dir='../collect_data', model_file='gmm_model.pkl'):
    """
    Evaluate predictions on all CSV files in collect_data directory.
    
    Parameters:
    -----------
    data_dir : str
        Path to collect_data directory
    model_file : str
        Path to gmm_model.pkl
    
    Returns:
    --------
    dict with evaluation results
    """
    
    data_dir = Path(data_dir)
    
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        print("Please ensure gmm_model.pkl is in the current directory")
        return None
    
    # Collect all predictions and ground truth labels
    all_predictions = []
    all_ground_truth = []
    all_results = []
    
    # Iterate through temperature range folders
    for temp_folder in sorted(data_dir.iterdir()):
        if not temp_folder.is_dir():
            continue
        
        temp_range = temp_folder.name
        ground_truth = map_temperature_range_to_class(temp_range)
        
        print(f"\n📁 Processing {temp_range} (Ground Truth: {ground_truth})")
        print("-" * 60)
        
        # Process each CSV file in the temperature folder
        csv_files = list(temp_folder.glob('*.csv'))
        print(f"   Found {len(csv_files)} files")
        
        correct = 0
        total = 0
        
        for csv_file in sorted(csv_files):
            try:
                # Read CSV
                df = pd.read_csv(csv_file)
                
                # Filter out rows with ignored=1 and extract sensor columns
                df_valid = df[df['ignored'] == 0][['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']]
                
                if len(df_valid) == 0:
                    print(f"   ⚠️  {csv_file.name}: No valid readings (all ignored)")
                    continue
                
                # Average the sensor values for this file
                avg_sensors = df_valid.mean().values.astype(float)
                
                # Predict
                result = predict_temperature(avg_sensors, model_file)
                predicted_class = result['class']
                confidence = result['confidence']
                
                # Check if correct
                is_correct = predicted_class == ground_truth
                if is_correct:
                    correct += 1
                
                total += 1
                
                # Store results
                all_predictions.append(predicted_class)
                all_ground_truth.append(ground_truth)
                all_results.append({
                    'file': csv_file.name,
                    'temp_range': temp_range,
                    'ground_truth': ground_truth,
                    'predicted': predicted_class,
                    'confidence': confidence,
                    'correct': is_correct,
                    'avg_sensors': avg_sensors.tolist()
                })
                
                # Visual indicator
                status = "✅" if is_correct else "❌"
                print(f"   {status} {csv_file.name}: {predicted_class} (conf: {confidence:.3f})")
                
            except Exception as e:
                print(f"   ⚠️  {csv_file.name}: Error - {str(e)}")
                continue
        
        if total > 0:
            accuracy = correct / total
            print(f"   📊 Accuracy for {temp_range}: {accuracy:.2%} ({correct}/{total})")
    
    
    # Calculate overall metrics
    print("\n" + "=" * 60)
    print("OVERALL EVALUATION RESULTS")
    print("=" * 60)
    
    if len(all_predictions) > 0:
        overall_accuracy = accuracy_score(all_ground_truth, all_predictions)
        
        print(f"\n📈 Overall Accuracy: {overall_accuracy:.2%} ({sum(np.array(all_predictions) == np.array(all_ground_truth))}/{len(all_predictions)})")
        
        print("\n📋 Classification Report:")
        print("-" * 60)
        print(classification_report(all_ground_truth, all_predictions, 
                                     labels=['COLD', 'NORMAL', 'HOT'],
                                     zero_division=0))
        
        print("\n🔀 Confusion Matrix:")
        print("-" * 60)
        cm = confusion_matrix(all_ground_truth, all_predictions, labels=['COLD', 'NORMAL', 'HOT'])
        print("                 Predicted COLD  NORMAL  HOT")
        labels = ['True COLD    ', 'True NORMAL  ', 'True HOT     ']
        for i, row in enumerate(cm):
            print(f"{labels[i]}: {row}")
        
        # Per-class accuracy
        print("\n📊 Per-Class Accuracy:")
        print("-" * 60)
        for label in ['COLD', 'NORMAL', 'HOT']:
            mask = np.array(all_ground_truth) == label
            if mask.sum() > 0:
                class_acc = np.array(all_predictions)[mask] == label
                acc = class_acc.sum() / mask.sum()
                print(f"{label:8s}: {acc:.2%} ({class_acc.sum()}/{mask.sum()})")
    else:
        print("❌ No valid predictions made")
        return None
    
    return {
        'overall_accuracy': overall_accuracy,
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'detailed_results': all_results,
        'confusion_matrix': cm.tolist(),
        'total_samples': len(all_predictions)
    }


if __name__ == "__main__":
    
    print("=" * 60)
    print("GMM TEMPERATURE PREDICTION - COLLECT_DATA EVALUATION")
    print("=" * 60)
    
    # Run evaluation
    results = evaluate_collect_data(data_dir='../collect_data', model_file='gmm_model.pkl')
    
    if results:
        # Save results to JSON
        output_file = 'evaluation_results_collect_data.json'
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            'overall_accuracy': float(results['overall_accuracy']),
            'total_samples': results['total_samples'],
            'detailed_results': results['detailed_results'],
            'confusion_matrix': results['confusion_matrix']
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
    
    print("\n✨ Evaluation complete!")
