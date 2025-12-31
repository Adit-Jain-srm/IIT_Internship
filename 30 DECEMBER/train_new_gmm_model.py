"""
Train a new GMM model on December 30 data
- Train-test split
- Simple feature engineering
- 3 clusters (COLD, NORMAL, HOT)
"""

import pickle
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')


def load_sensor_data_from_csv(csv_path):
    """Load sensor readings from raw sensor log CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check column names (may have different casing)
        columns = df.columns.str.lower()
        
        # Find sensor columns
        sensor_cols = []
        for i in range(1, 5):
            for pattern in [f'sensor_{i}', f'sensor-{i}', f'sensor {i}']:
                if pattern in columns:
                    sensor_cols.append(df.columns[columns == pattern][0])
                    break
        
        if len(sensor_cols) != 4:
            sensor_cols = [col for col in df.columns if 'sensor' in col.lower() and any(str(i) in col for i in [1,2,3,4])]
            sensor_cols = sorted(sensor_cols)[:4]
        
        if len(sensor_cols) != 4:
            return None
        
        # Filter for VALID rows only
        status_col = None
        for col in df.columns:
            if 'status' in col.lower():
                status_col = col
                break
        
        if status_col:
            valid_mask = df[status_col].str.upper() == 'VALID'
            df_valid = df[valid_mask].copy()
        else:
            df_valid = df.copy()
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


def simple_feature_engineering(X):
    """
    Simple feature engineering from 4 raw sensors.
    Creates additional features to help with classification.
    """
    X = np.array(X)
    
    # Ensure 2D array
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    # Extract individual sensors
    sensor_1 = X[:, 0]
    sensor_2 = X[:, 1]
    sensor_3 = X[:, 2]
    sensor_4 = X[:, 3]
    
    # Simple feature engineering
    # 1. Ratios (normalize sensor relationships)
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    
    # 2. Statistical features
    sensor_mean = X.mean(axis=1)
    sensor_std = X.std(axis=1)
    
    # 3. Range features
    sensor_max = X.max(axis=1)
    sensor_min = X.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    # Combine features: 4 raw + 7 engineered = 11 total
    X_engineered = np.column_stack([
        X,  # Original 4 sensors
        ratio_1_2, ratio_3_4,  # 2 ratios
        sensor_mean, sensor_std,  # 2 statistical
        sensor_max, sensor_min, sensor_range  # 3 range features
    ])
    
    return X_engineered


def load_all_data(data_dir=None):
    """Load all sensor data from COLD, HOT, NORMAL folders."""
    if data_dir is None:
        # Default to script's directory
        data_path = Path(__file__).parent
    else:
        data_path = Path(data_dir)
    
    X_all = []
    y_all = []
    
    # Ground truth mapping
    ground_truth_map = {
        'COLD': 'COLD',
        'HOT': 'HOT',
        'NORMAL': 'NORMAL'
    }
    
    for folder_name in ['COLD', 'HOT', 'NORMAL']:
        folder_path = data_path / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder {folder_name} not found, skipping...")
            continue
        
        ground_truth_class = ground_truth_map.get(folder_name, folder_name.upper())
        print(f"\nLoading {folder_name} folder (Ground truth: {ground_truth_class})...")
        
        # Find all raw sensor log CSV files
        csv_files = sorted(folder_path.glob('raw_sensor_log*.csv'))
        print(f"  Found {len(csv_files)} raw sensor log files")
        
        folder_readings = 0
        for csv_file in csv_files:
            sensor_data = load_sensor_data_from_csv(csv_file)
            if sensor_data is not None and len(sensor_data) > 0:
                X_all.append(sensor_data)
                y_all.extend([ground_truth_class] * len(sensor_data))
                folder_readings += len(sensor_data)
        
        print(f"  Total readings: {folder_readings}")
    
    if not X_all:
        print("No data loaded!")
        return None, None
    
    # Combine all readings
    X = np.vstack(X_all)
    y = np.array(y_all)
    
    print(f"\n{'='*80}")
    print(f"TOTAL DATA LOADED")
    print(f"{'='*80}")
    print(f"Total samples: {len(X):,}")
    print(f"Features per sample: {X.shape[1]} (raw sensors)")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt:,} samples ({cnt/len(y)*100:.1f}%)")
    
    return X, y


def map_clusters_to_temperature(clusters, ground_truth):
    """
    Map cluster IDs to temperature classes using majority voting.
    """
    cluster_to_temp = {}
    
    unique_clusters = np.unique(clusters)
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        temps_in_cluster = pd.Series(ground_truth[cluster_mask]).value_counts()
        
        if len(temps_in_cluster) > 0:
            dominant_temp = temps_in_cluster.idxmax()
            cluster_to_temp[int(cluster_id)] = dominant_temp
        else:
            cluster_to_temp[int(cluster_id)] = 'NORMAL'  # Default
    
    return cluster_to_temp


def train_gmm_model(X_train, y_train, n_components=3, random_state=42):
    """Train GMM model with feature engineering."""
    print(f"\n{'='*80}")
    print(f"TRAINING GMM MODEL")
    print(f"{'='*80}")
    
    # Feature engineering
    print(f"\nFeature Engineering:")
    print(f"  Input: {X_train.shape[1]} raw sensor features")
    X_train_engineered = simple_feature_engineering(X_train)
    print(f"  Output: {X_train_engineered.shape[1]} engineered features")
    print(f"    - 4 raw sensors")
    print(f"    - 2 ratios (sensor_1/sensor_2, sensor_3/sensor_4)")
    print(f"    - 2 statistical (mean, std)")
    print(f"    - 3 range (max, min, range)")
    
    # Standardize features
    print(f"\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_engineered)
    print(f"  Mean: {X_train_scaled.mean(axis=0)[:5]}")
    print(f"  Std: {X_train_scaled.std(axis=0)[:5]}")
    
    # Train GMM
    print(f"\nTraining GMM with {n_components} components...")
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type='full',
        random_state=random_state,
        max_iter=200,
        n_init=10
    )
    gmm.fit(X_train_scaled)
    print(f"  Convergence: {gmm.converged_}")
    print(f"  Iterations: {gmm.n_iter_}")
    print(f"  Log-likelihood: {gmm.score(X_train_scaled):.2f}")
    
    # Predict on training data
    clusters_train = gmm.predict(X_train_scaled)
    
    # Map clusters to temperature classes
    cluster_to_temp = map_clusters_to_temperature(clusters_train, y_train)
    print(f"\nCluster to Temperature Mapping (Majority Voting):")
    for cluster_id, temp_class in sorted(cluster_to_temp.items()):
        cluster_mask = clusters_train == cluster_id
        cluster_size = cluster_mask.sum()
        # Show distribution within cluster
        temps_in_cluster = pd.Series(y_train[cluster_mask]).value_counts()
        print(f"  Cluster {cluster_id} -> {temp_class} ({cluster_size:,} samples, {cluster_size/len(clusters_train)*100:.1f}%)")
        for temp, count in temps_in_cluster.items():
            print(f"    - {temp}: {count:,} ({count/cluster_size*100:.1f}%)")
    
    return gmm, scaler, cluster_to_temp


def evaluate_model(gmm, scaler, cluster_to_temp, X, y, set_name="Test"):
    """Evaluate model on a dataset."""
    print(f"\n{'='*80}")
    print(f"EVALUATION: {set_name.upper()} SET")
    print(f"{'='*80}")
    
    # Feature engineering
    X_engineered = simple_feature_engineering(X)
    X_scaled = scaler.transform(X_engineered)
    
    # Predict
    clusters = gmm.predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    # Map to temperature classes
    predictions = np.array([cluster_to_temp[c] for c in clusters])
    
    # Metrics
    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    recall = recall_score(y, predictions, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    f1 = f1_score(y, predictions, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Mean Confidence: {probabilities.max(axis=1).mean():.4f}")
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y, predictions, labels=['COLD', 'NORMAL', 'HOT'])
    print(f"\nConfusion Matrix:")
    print(f"               Pred COLD  Pred NORMAL  Pred HOT")
    labels = ['COLD', 'NORMAL', 'HOT']
    for i, label in enumerate(labels):
        print(f"True {label:6}: {conf_matrix[i, 0]:8}  {conf_matrix[i, 1]:11}  {conf_matrix[i, 2]:8}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    report = classification_report(y, predictions, labels=['COLD', 'NORMAL', 'HOT'], output_dict=True, zero_division=0)
    for label in ['COLD', 'NORMAL', 'HOT']:
        if label in report:
            metrics = report[label]
            print(f"  {label}:")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall:    {metrics['recall']:.4f}")
            print(f"    F1-Score:  {metrics['f1-score']:.4f}")
            print(f"    Support:   {int(metrics['support'])}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions,
        'confusion_matrix': conf_matrix,
        'mean_confidence': probabilities.max(axis=1).mean()
    }


def main():
    """Main training pipeline."""
    print("=" * 80)
    print("GMM MODEL TRAINING - DECEMBER 30 DATA")
    print("=" * 80)
    
    # Load data
    X, y = load_all_data()
    if X is None:
        print("Failed to load data!")
        return
    
    # Train-test split (80-20)
    print(f"\n{'='*80}")
    print(f"TRAIN-TEST SPLIT")
    print(f"{'='*80}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    print(f"Train set: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Test set:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model
    gmm, scaler, cluster_to_temp = train_gmm_model(X_train, y_train, n_components=3, random_state=42)
    
    # Evaluate on train set
    train_results = evaluate_model(gmm, scaler, cluster_to_temp, X_train, y_train, "Train")
    
    # Evaluate on test set
    test_results = evaluate_model(gmm, scaler, cluster_to_temp, X_test, y_test, "Test")
    
    # Save model
    print(f"\n{'='*80}")
    print(f"SAVING MODEL")
    print(f"{'='*80}")
    
    model_package = {
        'gmm_model': gmm,
        'scaler': scaler,
        'cluster_to_temp_mapping': cluster_to_temp,
        'n_clusters': 3,
        'n_features': 11,  # 4 raw + 7 engineered
        'feature_engineering': 'simple',  # Simple feature engineering applied
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'train_accuracy': train_results['accuracy'],
        'test_accuracy': test_results['accuracy']
    }
    
    output_dir = Path(__file__).parent
    output_file = output_dir / 'gmm_model_dec30.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Model saved to: {output_file}")
    
    # Save metadata
    metadata = {
        'model_name': 'GMM Temperature Classifier (Dec 30 Data)',
        'n_clusters': 3,
        'n_features': 11,
        'feature_engineering': 'simple',
        'train_accuracy': float(train_results['accuracy']),
        'test_accuracy': float(test_results['accuracy']),
        'train_precision': float(train_results['precision']),
        'train_recall': float(train_results['recall']),
        'train_f1': float(train_results['f1']),
        'test_precision': float(test_results['precision']),
        'test_recall': float(test_results['recall']),
        'test_f1': float(test_results['f1']),
        'cluster_to_temp_mapping': {str(k): v for k, v in cluster_to_temp.items()},
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    import json
    metadata_file = output_dir / 'gmm_model_dec30_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_file}")
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Test Accuracy: {test_results['accuracy']:.4f} ({test_results['accuracy']*100:.2f}%)")


if __name__ == "__main__":
    main()

