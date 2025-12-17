#K-means Clustering Algorithm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

#Importing the dataset
dataset = pd.read_csv('C:/Users/aditj/New Projects/IIT_Internship/Codes/combined.csv')

# Extract X, Y, Z coordinates (raw data - no scaling)
coordinates = dataset[['X', 'Y', 'Z']].values

# Create the K-Means model with 8 clusters
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
# Fit the model and assign each data point to a cluster
cluster_labels = kmeans.fit_predict(coordinates)

# Add the cluster assignment to the dataset
dataset['cluster'] = cluster_labels

# Calculate evaluation metrics on raw data
sil_score = silhouette_score(coordinates, cluster_labels)
dbi_score = davies_bouldin_score(coordinates, cluster_labels)
chi_score = calinski_harabasz_score(coordinates, cluster_labels)

# Print all evaluation scores
print("=" * 60)
print("K-Means Clustering Evaluation (8 clusters on raw data)")
print("=" * 60)
print(f"Silhouette Score: {sil_score:.6f} (higher is better, range: -1 to 1)")
print(f"Davies-Bouldin Index (DBI): {dbi_score:.6f} (lower is better)")
print(f"Calinski-Harabasz Index (CHI): {chi_score:.6f} (higher is better)")
print("=" * 60)
