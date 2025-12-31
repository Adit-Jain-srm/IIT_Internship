"""
Final GMM Optimization - Comprehensive search for best model
Balances accuracy with proper 3-class classification
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
    confusion_matrix
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


def feature_engineering_v1(X):
    """Simple: 11 features"""
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


def feature_engineering_v2(X):
    """Enhanced: 13 features - more ratios"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    sensor_1, sensor_2, sensor_3, sensor_4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # More ratios
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    ratio_1_3 = sensor_1 / (sensor_3 + 1e-8)
    ratio_2_4 = sensor_2 / (sensor_4 + 1e-8)
    
    # Statistical
    sensor_mean = X.mean(axis=1)
    sensor_std = X.std(axis=1)
    
    # Range
    sensor_max = X.max(axis=1)
    sensor_min = X.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    return np.column_stack([X, ratio_1_2, ratio_3_4, ratio_1_3, ratio_2_4, sensor_mean, sensor_std, sensor_max, sensor_min, sensor_range])


def feature_engineering_v3(X):
    """Best: 15 features - optimal selection"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    sensor_1, sensor_2, sensor_3, sensor_4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # Key ratios
    ratio_3_4 = sensor_3 / (sensor_4 + 1e-8)
    ratio_1_2 = sensor_1 / (sensor_2 + 1e-8)
    ratio_1_3 = sensor_1 / (sensor_3 + 1e-8)
    
    # Statistical
    sensor_mean = X.mean(axis=1)
    sensor_std = X.std(axis=1)
    sensor_sum = X.sum(axis=1)
    
    # Range
    sensor_max = X.max(axis=1)
    sensor_min = X.min(axis=1)
    sensor_range = sensor_max - sensor_min
    
    # Key interactions
    sensor_3_squared = sensor_3 ** 2
    sum_3_4 = sensor_3 + sensor_4
    
    return np.column_stack([
        X, ratio_3_4, ratio_1_2, ratio_1_3,
        sensor_mean, sensor_std, sensor_sum,
        sensor_max, sensor_min, sensor_range,
        sensor_3_squared, sum_3_4
    ])


def map_clusters_to_temperature(clusters, ground_truth):
    """Map clusters to temperature classes using majority voting."""
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


def evaluate_config(X_train, y_train, X_test, y_test, feature_func, feat_name, 
                   cov_type='tied', n_init=30, random_state=42):
    """Evaluate configuration."""
    
    X_train_feat = feature_func(X_train)
    X_test_feat = feature_func(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feat)
    X_test_scaled = scaler.transform(X_test_feat)
    
    gmm = GaussianMixture(
        n_components=3,
        covariance_type=cov_type,
        random_state=random_state,
        max_iter=300,
        n_init=n_init,
        init_params='kmeans'
    )
    gmm.fit(X_train_scaled)
    
    if not gmm.converged_:
        return None
    
    clusters_train = gmm.predict(X_train_scaled)
    clusters_test = gmm.predict(X_test_scaled)
    
    cluster_to_temp = map_clusters_to_temperature(clusters_train, y_train)
    
    pred_train = np.array([cluster_to_temp.get(c, 'NORMAL') for c in clusters_train])
    pred_test = np.array([cluster_to_temp.get(c, 'NORMAL') for c in clusters_test])
    
    train_acc = accuracy_score(y_train, pred_train)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1 = f1_score(y_test, pred_test, average='weighted', zero_division=0, labels=['COLD', 'NORMAL', 'HOT'])
    
    # Per-class accuracy
    conf_matrix = confusion_matrix(y_test, pred_test, labels=['COLD', 'NORMAL', 'HOT'])
    per_class_acc = {}
    for i, label in enumerate(['COLD', 'NORMAL', 'HOT']):
        if conf_matrix[i].sum() > 0:
            per_class_acc[label] = conf_matrix[i, i] / conf_matrix[i].sum()
        else:
            per_class_acc[label] = 0.0
    
    # Calculate balanced score (weight accuracy and class coverage)
    n_classes_predicted = len(set([cluster_to_temp.get(c, 'NORMAL') for c in range(3)]))
    balanced_score = test_acc * (0.7 + 0.3 * (n_classes_predicted / 3))
    
    return {
        'feature_name': feat_name,
        'n_features': X_train_feat.shape[1],
        'covariance_type': cov_type,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'test_f1': float(test_f1),
        'per_class_accuracy': per_class_acc,
        'n_classes_predicted': n_classes_predicted,
        'balanced_score': float(balanced_score),
        'gmm': gmm,
        'scaler': scaler,
        'cluster_mapping': cluster_to_temp,
        'confusion_matrix': conf_matrix.tolist()
    }


def main():
    """Main optimization."""
    print("=" * 80)
    print("FINAL GMM OPTIMIZATION")
    print("=" * 80)
    
    X, y = load_all_data()
    if X is None:
        print("Failed to load data!")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}\n")
    
    # Test all combinations
    configs = [
        (feature_engineering_v1, 'v1_simple', 'tied'),
        (feature_engineering_v1, 'v1_simple', 'full'),
        (feature_engineering_v1, 'v1_simple', 'diag'),
        (feature_engineering_v2, 'v2_enhanced', 'tied'),
        (feature_engineering_v2, 'v2_enhanced', 'full'),
        (feature_engineering_v2, 'v2_enhanced', 'diag'),
        (feature_engineering_v3, 'v3_optimal', 'tied'),
        (feature_engineering_v3, 'v3_optimal', 'full'),
        (feature_engineering_v3, 'v3_optimal', 'diag'),
    ]
    
    results = []
    for i, (feat_func, feat_name, cov_type) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {feat_name}, {cov_type}...", end=' ')
        try:
            result = evaluate_config(X_train, y_train, X_test, y_test, feat_func, feat_name, cov_type)
            if result:
                results.append(result)
                print(f"Acc: {result['test_accuracy']:.4f}, Classes: {result['n_classes_predicted']}/3")
            else:
                print("Failed")
        except Exception as e:
            print(f"Error: {e}")
    
    if not results:
        print("No results!")
        return
    
    # Sort by balanced score (accuracy + class coverage)
    results.sort(key=lambda x: x['balanced_score'], reverse=True)
    
    print("\n" + "=" * 80)
    print("TOP RESULTS (sorted by balanced score)")
    print("=" * 80)
    print(f"{'Rank':<6} {'Features':<20} {'Cov':<8} {'Acc':<8} {'F1':<8} {'Classes':<8} {'Score':<8}")
    print("-" * 80)
    for i, r in enumerate(results[:5], 1):
        print(f"{i:<6} {r['feature_name']:<20} {r['covariance_type']:<8} "
              f"{r['test_accuracy']:<8.4f} {r['test_f1']:<8.4f} "
              f"{r['n_classes_predicted']:<8} {r['balanced_score']:<8.4f}")
    
    best = results[0]
    
    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Features: {best['feature_name']} ({best['n_features']} features)")
    print(f"Covariance: {best['covariance_type']}")
    print(f"\nPerformance:")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f} ({best['test_accuracy']*100:.2f}%)")
    print(f"  Test F1: {best['test_f1']:.4f}")
    print(f"  Classes predicted: {best['n_classes_predicted']}/3")
    
    print(f"\nPer-Class Accuracy:")
    for cls, acc in best['per_class_accuracy'].items():
        print(f"  {cls}: {acc:.4f} ({acc*100:.2f}%)")
    
    print(f"\nCluster Mapping:")
    for cid, temp in sorted(best['cluster_mapping'].items()):
        print(f"  Cluster {cid} -> {temp}")
    
    conf_matrix = np.array(best['confusion_matrix'])
    print(f"\nConfusion Matrix:")
    print(f"               Pred COLD  Pred NORMAL  Pred HOT")
    for i, label in enumerate(['COLD', 'NORMAL', 'HOT']):
        print(f"True {label:6}: {conf_matrix[i, 0]:8}  {conf_matrix[i, 1]:11}  {conf_matrix[i, 2]:8}")
    
    # Save
    output_dir = Path(__file__).parent
    model_package = {
        'gmm_model': best['gmm'],
        'scaler': best['scaler'],
        'cluster_to_temp_mapping': best['cluster_mapping'],
        'n_clusters': 3,
        'n_features': best['n_features'],
        'feature_engineering': best['feature_name'],
        'covariance_type': best['covariance_type'],
        'test_accuracy': best['test_accuracy'],
        'test_f1': best['test_f1'],
        'per_class_accuracy': best['per_class_accuracy']
    }
    
    model_file = output_dir / 'gmm_model_final.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved: {model_file}")
    
    # Save all results
    summary_file = output_dir / 'final_optimization_results.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_model': {
                'feature_name': best['feature_name'],
                'n_features': best['n_features'],
                'covariance_type': best['covariance_type'],
                'test_accuracy': best['test_accuracy'],
                'test_f1': best['test_f1'],
                'n_classes_predicted': best['n_classes_predicted'],
                'per_class_accuracy': best['per_class_accuracy']
            },
            'all_results': [
                {
                    'feature_name': r['feature_name'],
                    'n_features': r['n_features'],
                    'covariance_type': r['covariance_type'],
                    'test_accuracy': r['test_accuracy'],
                    'test_f1': r['test_f1'],
                    'n_classes_predicted': r['n_classes_predicted'],
                    'balanced_score': r['balanced_score']
                }
                for r in results
            ]
        }, f, indent=2)
    
    print("Optimization complete!")


if __name__ == "__main__":
    main()

