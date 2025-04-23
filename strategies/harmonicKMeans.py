import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

class HarmonicKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = 1e-4
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X, sensitive_feature):
        """
        Fits the K-Harmonic Means model to the data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).
        """
        # Ensures that X is a valid 2D array and that sensitive_feature is converted into a NumPy array for processing
        X = check_array(X)
        sensitive_feature = np.array(sensitive_feature)

        # Check for consistent lengths
        if len(X) != len(sensitive_feature):
            raise ValueError("X and sensitive_feature must have the same length.")

        # Initialize centroids - Randomly selects n_clusters points from X to serve as the initial centroids
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[indices]

        # Initialize cluster assignments:
        # - 'labels' stores the cluster assignment for each data point
        # - 'prev_labels' stores the previous cluster assignment for each data point (for debugging purposes)
        # - 'sensitive_counts' stores the counts of each sensitive feature value in each cluster
        labels = np.zeros(X.shape[0], dtype=int)
        prev_labels = labels.copy()
        sensitive_counts = {i: {val: 0 for val in np.unique(sensitive_feature)} for i in range(self.n_clusters)}

        for iteration in range(self.max_iters):
            print(f"\nITERATION {iteration + 1}")

            # Iterate over all data points and assign them to the closest cluster 
            for i, data_point in enumerate(X):
                distances = np.linalg.norm(self.centroids - data_point, axis=1)     # Compute the distance from data point to all centroids                
                sorted_clusters = np.argsort(distances)                             # Sort clusters by distance                   
                
                for cluster in sorted_clusters:
                    labels[i] = cluster
                    sensitive_counts[cluster][sensitive_feature[i]] += 1
                    break

            print(f"Sensitive feature distributions: {sensitive_counts}")

            # Update centroids using Harmonic Mean
            new_centroids = np.array([self._harmonic_mean(X[labels == i]) if np.any(labels == i) else self.centroids[i] for i in range(self.n_clusters)])

            # Difference between previous and current cluster assignments
            num_changes = np.sum(labels != prev_labels)
            print(f"Points Changing Clusters: {num_changes}")
            prev_labels = labels.copy()

            centroid_shift = np.linalg.norm(self.centroids - new_centroids)
            print(f"Centroid shift: {centroid_shift}")
            
            if centroid_shift < self.tol:
                print("\nConvergence reached!\n")
                break
            self.centroids = new_centroids

    def _harmonic_mean(self, points):
        if len(points) == 0:
            return np.zeros(points.shape[1])
        #print("points:", points)
        #print("points.shape:", points.shape)

        # Replace zeros with np.nan temporarily to avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            reciprocal = 1 / points
            reciprocal[points == 0] = np.nan  # avoid divide-by-zero issues
            harmonic = len(points) / np.nansum(reciprocal, axis=0)

        # Replace NaNs (in case all values were zero) with zero
        har_mean = np.nan_to_num(harmonic, nan=0.0)
        #har_mean = len(points) / np.sum(1 / points, axis=0)

        print("har_mean for each feature:", har_mean)
        return har_mean
    
    def predict(self, X):
        """
        Predicts the cluster labels for the data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).

        Returns:
        - Cluster labels, shape (n_samples,).
        """
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        distances = distances.T
        return np.argmin(distances, axis=1)