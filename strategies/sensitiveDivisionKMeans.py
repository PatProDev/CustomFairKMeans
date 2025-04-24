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

        # Step 1: Separate data by sensitive feature values
        unique_values = np.unique(sensitive_feature)
        group_data = {}
        group_indices = {}
        for val in unique_values:
            indices = np.where(sensitive_feature == val)[0]
            group_data[val] = X[indices]
            group_indices[val] = indices
            print(f"  Group '{val}' has {len(indices)} samples")

        group_kmeans = {}

        # Step 2: Perform KMeans on each group separately
        for val in unique_values:
            print(f"  Running KMeans for group '{val}'")
            kmeans = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter, tol=self.tol, random_state=self.random_state, n_init='auto')
            kmeans.fit(group_data[val])
            group_kmeans[val] = kmeans
            #print(f"    Group '{val}' centroids:\n{group_kmeans[val].cluster_centers_}")

        # Step 3: Combine centroids and match clusters between groups
        centroids_by_group = [group_kmeans[val].cluster_centers_ for val in unique_values]
        labels_by_group = [group_kmeans[val].labels_ for val in unique_values]

        # Compute distances between centroids of different groups and pair them
        centroid_pairs = []
        used = set()
        for i in range(self.n_clusters):
            best_pair = None
            best_dist = float('inf')
            for a_idx, a_centroid in enumerate(centroids_by_group[0]):
                if a_idx in used:
                    continue
                for b_idx, b_centroid in enumerate(centroids_by_group[1]):
                    dist = np.linalg.norm(a_centroid - b_centroid)
                    if dist < best_dist:
                        best_dist = dist
                        best_pair = (a_idx, b_idx)
            centroid_pairs.append(best_pair)
            used.add(best_pair[0])
            print(f"  Matched Group 0 cluster {best_pair[0]} with Group 1 cluster {best_pair[1]} (distance = {best_dist:.4f})")

        # Step 4: Assign new global labels based on closest centroid pairing
        labels = np.empty(X.shape[0], dtype=int)
        current_label = 0
        for pair in centroid_pairs:
            a_val, b_val = unique_values[0], unique_values[1]
            a_indices_group = group_indices[a_val][labels_by_group[0] == pair[0]]
            b_indices_group = group_indices[b_val][labels_by_group[1] == pair[1]]
            labels[a_indices_group] = current_label
            labels[b_indices_group] = current_label
            print(f"  Assigned new label {current_label} to pair ({pair[0]}, {pair[1]})")
            current_label += 1

        # Calculate final centroids for the merged clusters
        final_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        # print("\nFinal centroids after merging:")
        # print(final_centroids)

        self.labels_ = labels
        self.cluster_centers_ = final_centroids
        return self

    def predict(self, X):
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
