import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class SensitiveDivisionKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, sensitive_feature):
        X = check_array(X)
        sensitive_feature = np.array(sensitive_feature)

        if len(X) != len(sensitive_feature):
            raise ValueError("X and sensitive_feature must have the same length.")

        # Step 1: Separate data in groups by sensitive feature values
        print("Separating data by sensitive feature values...")
        unique_values = np.unique(sensitive_feature)
        group_data = {}
        group_indices = {}
        for val in unique_values:
            indices = np.where(sensitive_feature == val)[0]
            group_data[val] = X[indices]
            group_indices[val] = indices
            print(f"  Group '{val}' has {len(indices)} samples")

        # Step 2: Perform KMeans on each group separately
        group_kmeans = {}
        for val in unique_values:
            print(f"Running KMeans for group '{val}'")
            kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state, n_init='auto')
            kmeans.fit(group_data[val])
            group_kmeans[val] = kmeans

        # Step 3: Stack all centroids and match them optimally
        print("Matching clusters across all groups...")
        centroid_list = []
        centroid_map = []  # To track which centroid belongs to which group and local cluster
        labels_by_group = {}

        for val_idx, val in enumerate(unique_values):
            centers = group_kmeans[val].cluster_centers_
            centroid_list.extend(centers)
            centroid_map.extend([(val, i) for i in range(len(centers))])
            labels_by_group[val] = group_kmeans[val].labels_

        centroid_list = np.array(centroid_list)

        # Step 4: Cluster centroids into final clusters (n_clusters overall clusters)
        final_kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        final_kmeans.fit(centroid_list)
        print("Final centroid clustering complete. Mapping sub-clusters to global clusters...")

        # Step 5: Assign new global labels to each point
        labels = np.empty(X.shape[0], dtype=int)
        for idx, (group_val, local_cluster_idx) in enumerate(centroid_map):
            global_cluster_label = final_kmeans.labels_[idx]
            indices = group_indices[group_val][labels_by_group[group_val] == local_cluster_idx]
            labels[indices] = global_cluster_label
            print(f"  Group '{group_val}' cluster {local_cluster_idx} assigned to global cluster {global_cluster_label}")

        # Step 6: Calculate final cluster centroids based on full dataset
        print("Calculating final cluster centroids...")
        final_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

        self.labels_ = labels
        self.cluster_centers_ = final_centroids
        return self

    def predict(self, X):
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
