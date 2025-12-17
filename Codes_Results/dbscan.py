# DBSCAN Clustering Algorithm
#
# This script runs DBSCAN on the 3D landmark dataset and reports clustering
# quality metrics when possible. Because DBSCAN can label points as noise
# (label = -1) or yield a single cluster, we only compute metrics when at
# least two non-noise clusters are present.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler
import gc  # Garbage collection for memory management

# Try to import HDBSCAN for large dataset support
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Note: HDBSCAN not available. Install with: pip install hdbscan")
    print("Falling back to standard DBSCAN with chunked processing...")


def main():
    # Load dataset
    print("Loading dataset...")
    dataset = pd.read_csv(
        "C:/Users/aditj/New Projects/IIT_Internship/Codes/combined.csv"
    )
    
    total_rows = len(dataset)
    print(f"Total rows in dataset: {total_rows:,}")

    # Extract X, Y, Z coordinates
    coordinates = dataset[["X", "Y", "Z"]].values
    
    # Convert to float32 to save memory (sufficient precision for coordinates)
    coordinates = coordinates.astype(np.float32)
    
    print(f"Processing entire dataset: {total_rows:,} points...")
    print("Scaling data for distance-based DBSCAN...")
    
    # Scale the data for distance-based DBSCAN
    scaler = StandardScaler()
    coordinates_scaled = scaler.fit_transform(coordinates)
    
    # Convert scaled data to float32 to save memory
    coordinates_scaled = coordinates_scaled.astype(np.float32)
    
    print("Running DBSCAN clustering on full dataset...")
    print("This may take several minutes for large datasets...")
    
    # For very large datasets, use HDBSCAN if available (more memory-efficient)
    # Otherwise use chunked DBSCAN approach
    if total_rows > 500000 and HDBSCAN_AVAILABLE:
        print("Using HDBSCAN for large dataset (more memory-efficient)...")
        # HDBSCAN is compatible with DBSCAN and handles large datasets better
        # min_cluster_size is equivalent to min_samples
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=10,
            cluster_selection_epsilon=0.5,  # Similar to DBSCAN eps
            core_dist_n_jobs=1  # Single-threaded to reduce memory
        )
        labels = clusterer.fit_predict(coordinates_scaled)
        # HDBSCAN uses -1 for noise, same as DBSCAN
    else:
        if total_rows > 500000:
            print("\n" + "="*60)
            print("WARNING: HDBSCAN not available!")
            print("="*60)
            print("For datasets >500k rows, HDBSCAN is strongly recommended.")
            print("Install with: pip install hdbscan")
            print("="*60)
            print("\nAttempting chunked DBSCAN (may be slow and memory-intensive)...")
            print("If you encounter memory errors, please install HDBSCAN.\n")
        
        # Use chunked DBSCAN for memory efficiency
        print("Using chunked DBSCAN processing...")
        try:
            labels = chunked_dbscan(coordinates_scaled, eps=0.5, min_samples=10)
        except MemoryError as e:
            print("\n" + "="*60)
            print("MEMORY ERROR: Cannot process full dataset with standard DBSCAN")
            print("="*60)
            print("Solutions:")
            print("1. Install HDBSCAN: pip install hdbscan (RECOMMENDED)")
            print("2. Use a representative sample of the data")
            print("3. Increase system RAM or use a machine with more memory")
            print("="*60)
            raise
    
    # Force garbage collection to free memory
    gc.collect()
    
    # Add cluster assignment to dataset
    dataset["cluster"] = labels

    # Summary of labels
    unique_labels = set(labels)
    noise_count = (labels == -1).sum()
    cluster_labels = [c for c in unique_labels if c != -1]
    num_clusters = len(cluster_labels)

    print("=" * 60)
    print("DBSCAN Clustering (on scaled data - FULL DATASET)")
    print("=" * 60)
    print(f"Total points processed: {total_rows:,}")
    print(f"Noise points: {noise_count:,} ({100*noise_count/total_rows:.2f}%)")
    print(f"Clusters found (excluding noise): {num_clusters}")

    # Only compute metrics when there are at least 2 clusters (excluding noise)
    if num_clusters >= 2:
        mask = labels != -1  # exclude noise for metrics
        coords_for_metrics = coordinates_scaled[mask]
        labels_for_metrics = labels[mask]

        print("\nCalculating evaluation metrics on full dataset...")
        print("Note: This may take time for large datasets...")
        
        # For very large datasets, sample for metrics calculation to avoid memory issues
        if len(coords_for_metrics) > 500000:
            print(f"Sampling 500k points for metrics calculation (out of {len(coords_for_metrics):,} clustered points)...")
            metric_sample_idx = np.random.choice(len(coords_for_metrics), 500000, replace=False)
            coords_for_metrics = coords_for_metrics[metric_sample_idx]
            labels_for_metrics = labels_for_metrics[metric_sample_idx]
        
        sil_score = silhouette_score(coords_for_metrics, labels_for_metrics)
        dbi_score = davies_bouldin_score(coords_for_metrics, labels_for_metrics)
        chi_score = calinski_harabasz_score(coords_for_metrics, labels_for_metrics)

        print(f"Silhouette Score: {sil_score:.6f} (higher is better, -1 to 1)")
        print(f"Davies-Bouldin Index (DBI): {dbi_score:.6f} (lower is better)")
        print(f"Calinski-Harabasz Index (CHI): {chi_score:.6f} (higher is better)")
    else:
        print(
            "Not enough clusters for metrics (need >= 2 clusters excluding noise)."
        )

    print("=" * 60)
    
    # Optionally save the clustered dataset
    print("\nSaving clustered dataset...")
    output_csv = 'combined_dbscan_clustered.csv'
    dataset.to_csv(output_csv, index=False)
    print(f"Clustered dataset saved as: {output_csv}")
    
    # Visualization (sample for visualization, but clustering is on full dataset)
    visualize_clusters(coordinates, labels, num_clusters, noise_count)


def chunked_dbscan(coordinates, eps=0.5, min_samples=10, chunk_size=50000):
    """
    Process DBSCAN in chunks to handle large datasets that don't fit in memory.
    Uses smaller chunks and brute force algorithm to minimize memory usage.
    This is an approximation - for exact results, use HDBSCAN or process full dataset.
    """
    n_samples = len(coordinates)
    labels = np.full(n_samples, -1, dtype=np.int32)
    
    # Use smaller chunks for very large datasets to avoid memory issues
    if n_samples > 1000000:
        chunk_size = 30000  # Even smaller chunks for 1M+ points
    
    # Process in chunks
    n_chunks = (n_samples + chunk_size - 1) // chunk_size
    current_cluster_id = 0
    
    print(f"Processing {n_chunks} chunks of ~{chunk_size:,} points each...")
    print("Using memory-efficient settings...")
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, n_samples)
        chunk_indices = np.arange(start_idx, end_idx, dtype=np.int32)
        chunk_data = coordinates[chunk_indices].copy()  # Explicit copy to avoid memory issues
        
        print(f"Processing chunk {i+1}/{n_chunks} (indices {start_idx:,} to {end_idx:,})...")
        
        # Run DBSCAN on chunk with memory-efficient settings
        # Using 'brute' algorithm for small chunks - more memory efficient
        # Using larger leaf_size to reduce tree construction overhead
        try:
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                n_jobs=1,  # Single-threaded
                algorithm='brute' if chunk_size < 100000 else 'ball_tree',  # Brute for small chunks
                leaf_size=50  # Larger leaf size to reduce memory
            )
            chunk_labels = dbscan.fit_predict(chunk_data)
        except MemoryError:
            # If still memory error, try even smaller chunk or brute force
            print(f"Memory error on chunk {i+1}, trying brute force with smaller subset...")
            # Process in sub-chunks
            sub_chunk_size = chunk_size // 2
            chunk_labels = np.full(len(chunk_data), -1, dtype=np.int32)
            sub_cluster_id = 0
            
            for j in range(0, len(chunk_data), sub_chunk_size):
                sub_end = min(j + sub_chunk_size, len(chunk_data))
                sub_data = chunk_data[j:sub_end]
                
                dbscan_sub = DBSCAN(
                    eps=eps,
                    min_samples=min_samples,
                    n_jobs=1,
                    algorithm='brute'
                )
                sub_labels = dbscan_sub.fit_predict(sub_data)
                
                # Remap labels
                for old_label in np.unique(sub_labels):
                    if old_label != -1:
                        mask = sub_labels == old_label
                        chunk_labels[j + np.where(mask)[0]] = sub_cluster_id
                        sub_cluster_id += 1
                    else:
                        mask = sub_labels == -1
                        chunk_labels[j + np.where(mask)[0]] = -1
        
        # Remap cluster IDs to avoid conflicts between chunks
        unique_chunk_labels = np.unique(chunk_labels)
        for old_label in unique_chunk_labels:
            if old_label != -1:  # Not noise
                mask = chunk_labels == old_label
                labels[chunk_indices[mask]] = current_cluster_id
                current_cluster_id += 1
            else:
                # Keep noise as -1
                mask = chunk_labels == -1
                labels[chunk_indices[mask]] = -1
        
        # Aggressive memory cleanup
        del chunk_data, chunk_labels, dbscan
        gc.collect()
    
    return labels


def visualize_clusters(coordinates, labels, num_clusters, noise_count):
    """
    Create visualizations of DBSCAN clustering results.
    Includes 3D plot and 2D projections (XY, XZ, YZ planes).
    """
    print("\nGenerating visualizations...")
    
    # Sample data for visualization (too many points slow down plotting)
    # Sample max 30k points for better performance
    max_vis_points = 30000
    if len(coordinates) > max_vis_points:
        print(f"Sampling {max_vis_points:,} points for visualization...")
        sample_indices = np.random.choice(len(coordinates), max_vis_points, replace=False)
        coords_sample = coordinates[sample_indices]
        labels_sample = labels[sample_indices]
    else:
        coords_sample = coordinates
        labels_sample = labels
    
    # Get unique cluster labels (including noise)
    unique_labels = np.unique(labels_sample)
    n_clusters_vis = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    # Create color map for clusters
    colors = plt.cm.tab20(np.linspace(0, 1, max(n_clusters_vis, 1)))
    color_map = {}
    color_idx = 0
    for label in unique_labels:
        if label == -1:
            color_map[label] = 'black'  # Noise points in black
        else:
            color_map[label] = colors[color_idx % len(colors)]
            color_idx += 1
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 6))
    
    # 1. 3D Scatter Plot
    ax1 = fig.add_subplot(131, projection='3d')
    for label in unique_labels:
        mask = labels_sample == label
        if label == -1:
            ax1.scatter(coords_sample[mask, 0], coords_sample[mask, 1], 
                       coords_sample[mask, 2], c=color_map[label], 
                       label=f'Noise ({mask.sum()} pts)', alpha=0.3, s=1)
        else:
            ax1.scatter(coords_sample[mask, 0], coords_sample[mask, 1], 
                       coords_sample[mask, 2], c=[color_map[label]], 
                       label=f'Cluster {label}', alpha=0.6, s=2)
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'DBSCAN Clustering - 3D View\n({n_clusters_vis} clusters, {noise_count} noise points)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # 2. XY Projection (2D)
    ax2 = fig.add_subplot(132)
    for label in unique_labels:
        mask = labels_sample == label
        if label == -1:
            ax2.scatter(coords_sample[mask, 0], coords_sample[mask, 1], 
                       c=color_map[label], label=f'Noise', alpha=0.3, s=1)
        else:
            ax2.scatter(coords_sample[mask, 0], coords_sample[mask, 1], 
                       c=[color_map[label]], label=f'Cluster {label}', alpha=0.6, s=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. XZ Projection (2D)
    ax3 = fig.add_subplot(133)
    for label in unique_labels:
        mask = labels_sample == label
        if label == -1:
            ax3.scatter(coords_sample[mask, 0], coords_sample[mask, 2], 
                       c=color_map[label], label=f'Noise', alpha=0.3, s=1)
        else:
            ax3.scatter(coords_sample[mask, 0], coords_sample[mask, 2], 
                       c=[color_map[label]], label=f'Cluster {label}', alpha=0.6, s=2)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_file = 'DBSCAN_clustering_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as: {output_file}")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

