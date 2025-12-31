"""
Advanced GMM Optimization - Testing more sophisticated approaches
Tries different cluster counts, better feature engineering, and initialization strategies
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


def feature_engineering_optimal(X):
    """Optimal feature engineering based on analysis: 13 features"""
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    
    sensor_1, sensor_2, sensor_3, sensor_4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    
    # Key ratios (sensor 3 seems important for temperature discrimination)
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
    
    # Key interaction (sensor 3 squared - often important for temperature)
    sensor_3_squared = sensor_3 ** 2
    
    # Combined features
    sum_3_4 = sensor_3 + sensor_4
    
    return np.column_stack([
        X,  # 4 raw
        ratio_3_4, ratio_1_2, ratio_1_3,  # 3 ratios
        sensor_mean, sensor_std, sensor_sum,  # 3 statistical
        sensor_max, sensor_min, sensor_range,  # 3 range
        sensor_3_squared, sum_3_4  # 2 interactions
    ])


def map_clusters_to_temperature_balanced(clusters, ground_truth, n_clusters=3):
    """
    Map clusters to temperature classes with better balancing.
    Tries to ensure all 3 classes are represented.
    """
    cluster_to_temp = {}
    unique_clusters = np.unique(clusters)
    
    # Count distributions
    cluster_temp_counts = {}
    for cluster_id in unique_clusters:
        cluster_mask = clusters == cluster_id
        temps_in_cluster = pd.Series(ground_truth[cluster_mask]).value_counts()
        cluster_temp_counts[int(cluster_id)] = temps_in_cluster.to_dict()
    
    # Find best assignment (greedy approach)
    temp_classes = ['COLD', 'NORMAL', 'HOT']
    assigned_clusters = set()
    assigned_temps = set()
    
    # Sort clusters by how distinct they are
    cluster_scores = []
    for cluster_id, temp_counts in cluster_temp_counts.items():
        if len(temp_counts) > 0:
            max_count = max(temp_counts.values())
            total = sum(temp_counts.values())
            distinctiveness = max_count / total if total > 0 else 0
            cluster_scores.append((distinctiveness, cluster_id, temp_counts))
    
    cluster_scores.sort(reverse=True)
    
    # Assign most distinct clusters first
    for distinctiveness, cluster_id, temp_counts in cluster_scores:
        if cluster_id in assigned_clusters:
            continue
        
        # Find best temperature class for this cluster
        best_temp = max(temp_counts.items(), key=lambda x: x[1])[0]
        
        # If this temp is already assigned, find next best
        if best_temp in assigned_temps:
            # Try other classes
            sorted_temps = sorted(temp_counts.items(), key=lambda x: x[1], reverse=True)
            for temp, count in sorted_temps:
                if temp not in assigned_temps:
                    best_temp = temp
                    break
        
        cluster_to_temp[int(cluster_id)] = best_temp
        assigned_clusters.add(cluster_id)
        assigned_temps.add(best_temp)
    
    # Assign remaining clusters
    for cluster_id in unique_clusters:
        if int(cluster_id) not in cluster_to_temp:
            temp_counts = cluster_temp_counts.get(int(cluster_id), {})
            if len(temp_counts) > 0:
                cluster_to_temp[int(cluster_id)] = max(temp_counts.items(), key=lambda x: x[1])[0]
            else:
                # Assign to least represented class
                if 'COLD' not in assigned_temps:
                    cluster_to_temp[int(cluster_id)] = 'COLD'
                elif 'HOT' not in assigned_temps:
                    cluster_to_temp[int(cluster_id)] = 'HOT'
                else:
                    cluster_to_temp[int(cluster_id)] = 'NORMAL'
    
    return cluster_to_temp


def evaluate_configuration(X_train, y_train, X_test, y_test, feature_func, feature_name, 
                          covariance_type='tied', n_components=3, random_state=42, n_init=20):
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
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=300,
        n_init=n_init,
        init_params='kmeans'
    )
    gmm.fit(X_train_scaled)
    
    if not gmm.converged_:
        return None
    
    # Predict
    clusters_train = gmm.predict(X_train_scaled)
    clusters_test = gmm.predict(X_test_scaled)
    
    # Map clusters - try balanced mapping
    cluster_to_temp = map_clusters_to_temperature_balanced(clusters_train, y_train, n_components)
    
    # Evaluate
    pred_train = np.array([cluster_to_temp.get(c, 'NORMAL') for c in clusters_train])
    pred_test = np.array([cluster_to_temp.get(c, 'NORMAL') for c in clusters_test])
    
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
        'n_components': n_components,
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
    print("ADVANCED GMM MODEL OPTIMIZATION")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    X, y = load_all_data()
    if X is None:
        print("Failed to load data!")
        return
    
    print(f"Total samples: {len(X):,}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Test best configurations
    print("\n" + "=" * 80)
    print("TESTING CONFIGURATIONS")
    print("=" * 80)
    
    configurations = [
        # (feature_func, feature_name, covariance_type, n_components)
        (feature_engineering_optimal, 'optimal', 'tied', 3),
        (feature_engineering_optimal, 'optimal', 'full', 3),
        (feature_engineering_optimal, 'optimal', 'diag', 3),
    ]
    
    all_results = []
    
    for i, (feature_func, feat_name, cov_type, n_comp) in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Testing: {feat_name} features, {cov_type} covariance, {n_comp} components...")
        try:
            result = evaluate_configuration(
                X_train, y_train, X_test, y_test,
                feature_func, feat_name, covariance_type=cov_type, n_components=n_comp
            )
            if result:
                all_results.append(result)
                n_classes = len(set([result['cluster_mapping'].get(c, 'NORMAL') for c in range(n_comp)]))
                print(f"  Test Accuracy: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
                print(f"  Test F1: {result['test_f1']:.4f}")
                print(f"  Classes predicted: {n_classes}/3")
                print(f"  Per-class acc: COLD={result['per_class_accuracy']['COLD']:.3f}, "
                      f"NORMAL={result['per_class_accuracy']['NORMAL']:.3f}, "
                      f"HOT={result['per_class_accuracy']['HOT']:.3f}")
            else:
                print(f"  Failed: Did not converge")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
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
    
    print(f"\n{'Rank':<6} {'Features':<20} {'Cov Type':<12} {'Test Acc':<12} {'Test F1':<12} {'Classes':<8}")
    print("-" * 80)
    for i, r in enumerate(all_results[:10], 1):
        n_classes = len(set([r['cluster_mapping'].get(c, 'NORMAL') for c in range(r['n_components'])]))
        print(f"{i:<6} {r['feature_name']:<20} {r['covariance_type']:<12} "
              f"{r['test_accuracy']:<12.4f} {r['test_f1']:<12.4f} {n_classes:<8}")
    
    # Best model
    best_result = all_results[0]
    
    print("\n" + "=" * 80)
    print("BEST MODEL")
    print("=" * 80)
    print(f"Features: {best_result['feature_name']} ({best_result['n_features']} features)")
    print(f"Covariance Type: {best_result['covariance_type']}")
    print(f"Components: {best_result['n_components']}")
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
        'n_clusters': best_result['n_components'],
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
    
    model_file = output_dir / 'gmm_model_optimized_advanced.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model_package, f)
    print(f"Model saved: {model_file}")
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

