"""
Iterative GMM Model Optimization
Tests different feature engineering approaches and model configurations
to find the best performing model
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
import json
from datetime import datetime

warnings.filterwarnings('ignore')


def load_sensor_data_from_csv(csv_path):
    """Load sensor readings from raw sensor log CSV file."""
    try:
        df = pd.read_csv(csv_path)
        columns = df.columns.str.lower()
        
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
        
        sensor_data = df_valid[sensor_cols].values.astype(float)
        return sensor_data
        
    except Exception as e:
        return None


def load_all_data():
    """Load all sensor data from COLD, HOT, NORMAL folders."""
    data_path = Path(__file__).parent
    
    X_all = []
    y_all = []
    
    ground_truth_map = {'COLD': 'COLD', 'HOT': 'HOT', 'NORMAL': 'NORMAL'}
    
    for folder_name in ['COLD', 'HOT', 'NORMAL']:
        folder_path = data_path / folder_name
        if not folder_path.exists():
            continue
        
        ground_truth_class = ground_truth_map.get(folder_name, folder_name.upper())
        csv_files = sorted(folder_path.glob('raw_sensor_log*.csv'))
        
        for csv_file in csv_files:
            sensor_data = load_sensor_data_from_csv(csv_file)
            if sensor_data is not None and len(sensor_data) > 0:
                X_all.append(sensor_data)
                y_all.extend([ground_truth_class] * len(sensor_data))
    
    if not X_all:
        return None, None
    
    X = np.vstack(X_all)
    y = np.array(y_all)
    return X, y


def feature_engineering_simple(X):
    """Simple feature engineering: 11 features"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    sensor_1, sensor_2, sensor_3, sensor_4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    sensor_mean = X.mean(axis=1)
    sensor_std = X.std(axis=1)
    sensor_max = X.max(axis=1)
    sensor_min = X.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    return np.column_stack([X, ratio_1_2, ratio_3_4, sensor_mean, sensor_std, sensor_max, sensor_min, sensor_range])


def feature_engineering_advanced(X):
    """Advanced feature engineering: 21 features"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    sensor_1, sensor_2, sensor_3, sensor_4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # Ratios
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    ratio_1_3 = sensor_1 / (sensor_3 + 1e-8)
    ratio_2_4 = sensor_2 / (sensor_4 + 1e-8)
    
    # Statistical
    sensor_sum = X.sum(axis=1)
    sensor_mean = X.mean(axis=1)
    sensor_std = X.std(axis=1)
    sensor_var = X.var(axis=1)
    
    # Range
    sensor_max = X.max(axis=1)
    sensor_min = X.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    # Polynomial
    sensor_3_squared = sensor_3 ** 2
    sensor_mean_squared = sensor_mean ** 2
    
    # Interactions
    sum_1_2 = sensor_1 + sensor_2
    sum_3_4 = sensor_3 + sensor_4
    product_1_3 = sensor_1 * sensor_3
    product_2_4 = sensor_2 * sensor_4
    
    return np.column_stack([
        X, ratio_1_2, ratio_3_4, ratio_1_3, ratio_2_4,
        sensor_sum, sensor_mean, sensor_std, sensor_var,
        sensor_max, sensor_min, sensor_range,
        sensor_3_squared, sensor_mean_squared,
        sum_1_2, sum_3_4, product_1_3, product_2_4
    ])


def feature_engineering_ratios_focused(X):
    """Ratios-focused: 15 features (emphasizes cross-sensor relationships)"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    sensor_1, sensor_2, sensor_3, sensor_4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # All pairwise ratios
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_1_3 = sensor_1 / (sensor_3 + 1e-8)
    ratio_1_4 = sensor_1 / (sensor_4 + 1e-8)
    ratio_2_3 = sensor_2 / (sensor_3 + 1e-8)
    ratio_2_4 = sensor_2 / (sensor_4 + 1e-8)
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    
    # Statistical
    sensor_mean = X.mean(axis=1)
    sensor_std = X.std(axis=1)
    
    # Range
    sensor_max = X.max(axis=1)
    sensor_min = X.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    # Important interactions (sensor 3 seems important for temperature)
    product_1_3 = sensor_1 * sensor_3
    sensor_3_squared = sensor_3 ** 2
    
    return np.column_stack([
        X, ratio_1_2, ratio_1_3, ratio_1_4, ratio_2_3, ratio_2_4, ratio_3_4,
        sensor_mean, sensor_std, sensor_max, sensor_min, sensor_range,
        product_1_3, sensor_3_squared
    ])


def map_clusters_to_temperature(clusters, ground_truth):
    """Map cluster IDs to temperature classes using majority voting."""
    cluster_to_temp = {}
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        temps_in_cluster = pd.Series(ground_truth[cluster_mask]).value_counts()
        
        if len(temps_in_cluster) > 0:
            dominant_temp = temps_in_cluster.idxmax()
            cluster_to_temp[int(cluster_id)] = dominant_temp
        else:
            cluster_to_temp[int(cluster_id)] = 'NORMAL'
    
    return cluster_to_temp


def evaluate_configuration(X_train, y_train, X_test, y_test, feature_func, feature_name, 
                          covariance_type='full', random_state=42, n_init=10, init_params='kmeans'):
    """Evaluate a specific model configuration."""
    
    # Feature engineering
    X_train_feat = feature_func(X_train)
    X_test_feat = feature_func(X_test)
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    
    # Train GMM
    gmm = GaussianMixture(
        n_components=3,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
        n_init=n_init,
        init_params=init_params
    )
    gmm.fit(X_train_scaled)
    
    if not gmm.converged_:
        return None
    
    # Predict
    clusters_train = gmm.predict(X_train_scaled)
    clusters_test = gmm.predict(X_test_scaled)
    
    # Map clusters
    cluster_to_temp = map_clusters_to_temperature(clusters_train, y_train)
    
    # Evaluate
    pred_train = np.array([cluster_to_temp[c] for c in clusters_train])
    pred_test = np.array([cluster_to_temp[c] for c in clusters_test])
    
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1 = f1_score(y_test, pred_test, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    test_precision = precision_score(y_test, pred_test, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    test_recall = recall_score(y_test, pred_test, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    
    # Check if we have all 3 classes predicted
    unique_pred_test = set(pred_test)
    has_all_classes = len(unique_pred_test) == 3
    
    # Calculate per-class performance
    conf_matrix = confusion_matrix(y_test, pred_test, labels=['COLD', 'NORMAL', 'HOT'])
    per_class_acc = {}
    for i, label in enumerate(['COLD', 'NORMAL', 'HOT']):
        if conf_matrix[i].sum() > 0:
            per_class_acc[label] = conf_matrix[i, i] / conf_matrix[i].sum()
        else:
            per_class_acc[label] = 0.0
    
    results = {
        'feature_name': feature_name,
        'n_features': X_train_feat.shape[1],
        'covariance_type': covariance_type,
        'init_params': init_params,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'has_all_classes': has_all_classes,
        'per_class_accuracy': per_class_acc,
        'converged': gmm.converged_,
        'n_iter': int(gmm.n_iter_),
        'log_likelihood': float(gmm.score(X_train_scaled)),
        'gmm': gmm,
        'scaler': scaler,
        'cluster_mapping': cluster_to_temp,
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return results


def main():
    """Main optimization loop."""
    print("=" * 80)
    print("GMM MODEL OPTIMIZATION - ITERATIVE IMPROVEMENT")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    X, y = load_all_data()
    if X is None:
        print("Failed to load data!")
        return
    
    print(f"Total samples: {len(X):,}")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls}: {cnt:,} ({cnt/len(y)*100:.1f}%)")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Test configurations
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATIONS")
    print("=" * 80)
    
    # Try best configurations with different init methods
    configurations = [
        # (feature_func, feature_name, covariance_type, init_params)
        (feature_engineering_simple, 'simple', 'tied', 'kmeans'),
        (feature_engineering_simple, 'simple', 'tied', 'random'),
        (feature_engineering_ratios_focused, 'ratios_focused', 'full', 'kmeans'),
        (feature_engineering_ratios_focused, 'ratios_focused', 'full', 'random'),
        (feature_engineering_advanced, 'advanced', 'full', 'kmeans'),
        (feature_engineering_advanced, 'advanced', 'full', 'random'),
        (feature_engineering_simple, 'simple', 'full', 'kmeans'),
        (feature_engineering_simple, 'simple', 'diag', 'kmeans'),
    ]
    
    all_results = []
    
    for i, config in enumerate(configurations, 1):
        if len(config) == 4:
            feature_func, feat_name, cov_type, init_params = config
        else:
            feature_func, feat_name, cov_type = config
            init_params = 'kmeans'
        
        print(f"\n[{i}/{len(configurations)}] Testing: {feat_name} features, {cov_type} covariance, {init_params} init...")
        try:
            result = evaluate_configuration(
                X_train, y_train, X_test, y_test,
                feature_func, feat_name, covariance_type=cov_type, init_params=init_params
            )
            if result:
                all_results.append(result)
                print(f"  Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
                print(f"  Test F1: {result['test_f1']:.4f}")
                print(f"  Classes predicted: {len(set([result['cluster_mapping'][c] for c in range(3)]))}/3")
            else:
                print(f"  Failed: Did not converge")
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    if not all_results:
        print("\nNo successful configurations!")
        return
    
    # Find best model
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    # Sort by test accuracy
    all_results.sort(key=lambda x: x['test_accuracy'], reverse=True)
    
    print(f"\n{'Rank':<6} {'Features':<20} {'Cov Type':<12} {'Init':<10} {'Test Acc':<12} {'Test F1':<12} {'Classes':<8}")
    print("-" * 90)
    for i, r in enumerate(all_results[:10], 1):
        n_classes = len(set([r['cluster_mapping'][c] for c in range(3)]))
        init = r.get('init_params', 'kmeans')
        print(f"{i:<6} {r['feature_name']:<20} {r['covariance_type']:<12} {init:<10} "
              f"{r['test_accuracy']:<12.4f} {r['test_f1']:<12.4f} {n_classes:<8}")
    
    # Best model
    best_result = all_results[0]
    
    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Features: {best_result['feature_name']} ({best_result['n_features']} features)")
    print(f"Covariance Type: {best_result['covariance_type']}")
    print(f"Init Method: {best_result.get('init_params', 'kmeans')}")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {best_result['test_accuracy']:.4f} ({best_result['test_accuracy']*100:.2f}%)")
    print(f"  Precision: {best_result['test_precision']:.4f}")
    print(f"  Recall:    {best_result['test_recall']:.4f}")
    print(f"  F1-Score:  {best_result['test_f1']:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    for cls, acc in best_result['per_class_accuracy'].items():
        print(f"  {cls}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nCluster Mapping:")
    for cluster_id, temp_class in sorted(best_result['cluster_mapping'].items()):
        print(f"  Cluster {cluster_id} -> {temp_class}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    print(f"               Pred COLD  Pred NORMAL  Pred HOT")
    labels = ['COLD', 'NORMAL', 'HOT']
    conf_matrix = np.array(best_result['confusion_matrix'])
    for i, label in enumerate(labels):
        print(f"True {label:6}: {conf_matrix[i, 0]:8}  {conf_matrix[i, 1]:11}  {conf_matrix[i, 2]:8}")
    
    # Save best model
    print("\n" + "=" * 80)
    print("SAVING BEST MODEL")
    print("=" * 80)
    
    output_dir = Path(__file__).parent
    model_package = {
        'gmm_model': best_result['gmm'],
        'scaler': best_result['scaler'],
        'cluster_to_temp_mapping': best_result['cluster_mapping'],
        'n_clusters': 3,
        'n_features': best_result['n_features'],
        'feature_engineering': best_result['feature_name'],
        'covariance_type': best_result['covariance_type'],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': best_result['test_accuracy'],
        'test_precision': best_result['test_precision'],
        'test_recall': best_result['test_recall'],
        'test_f1': best_result['test_f1'],
        'feature_func_name': best_result['feature_name']
    }
    
    # Save model
    model_file = output_dir / 'gmm_model_optimized.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Model saved: {model_file}")
    
    # Save all results for comparison
    results_summary = {
        'timestamp': datetime.now().isoformat(),
        'best_model': {
            'feature_name': best_result['feature_name'],
            'n_features': best_result['n_features'],
            'covariance_type': best_result['covariance_type'],
            'test_accuracy': best_result['test_accuracy'],
            'test_f1': best_result['test_f1'],
            'per_class_accuracy': best_result['per_class_accuracy']
        },
        'all_configurations': [
            {
                'feature_name': r['feature_name'],
                'n_features': r['n_features'],
                'covariance_type': r['covariance_type'],
                'test_accuracy': r['test_accuracy'],
                'test_f1': r['test_f1'],
                'has_all_classes': r['has_all_classes'],
                'per_class_accuracy': r['per_class_accuracy']
            }
            for r in all_results
        ]
    }
    
    results_file = output_dir / 'optimization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"Results saved: {results_file}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

