"""
Simple GMM Temperature Prediction Function
Predicts: COLD, NORMAL, or HOT
"""

import pickle
import numpy as np
import warnings

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


# USAGE EXAMPLES

if __name__ == "__main__":
    
    print("=" * 60)
    print("GMM TEMPERATURE PREDICTION")
    print("=" * 60)

    sensors = [181,502,551,482]
    result = predict_temperature(sensors)
    
    print(f"Input: {sensors}")
    print(f"Predicted Class: {result['class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Cluster: {result['cluster_id']}")
    print(f"All Probabilities: {result['all_probs']}")
'''
    test_cases = [
        [180,477,540,481]
    ]
    
    for sensors in test_cases:
        result = predict_temperature(sensors)
        print(f"{sensors} → {result['class']} (confidence: {result['confidence']:.3f})")
    
    print("\n" + "=" * 60)
    print("Done!")
'''