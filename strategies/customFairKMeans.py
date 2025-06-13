import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import silhouette_score, davies_bouldin_score

class FairKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters        # number of clusters to form   
        self.max_iter = max_iter            # maximum number of iterations to run the algorithm
        self.tol = tol                      # minimum change in centroids between iterations required for convergence
        self.random_state = random_state    # random seed for reproducibility
        self.fairness_tolerance = 1.5       # maximum allowed difference between the ratio of a sensitive group in a cluster and the global ratio

    def fit(self, X, sensitive_feature, global_ratios=None):
        """
        Fit the model to the data X with fairness constraints based on the sensitive feature.

        Parameters:
        X : array-like of shape (n_samples, n_features)
            Training data.
        sensitive_feature : array-like of shape (n_samples,)
            Sensitive feature values for each sample.
        """
        # Ensures that X is a valid 2D array and that sensitive_feature is converted into a NumPy array for processing
        X = check_array(X)
        sensitive_feature = np.array(sensitive_feature)
        n_samples, n_features = X.shape

        # Check for consistent lengths
        if len(X) != len(sensitive_feature):
            raise ValueError("X and sensitive_feature must have the same length.")

        # Initialize centroids - Randomly selects n_clusters points from X to serve as the initial centroids
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(len(X), self.n_clusters, replace=False)
        self.cluster_centers_ = X[indices]

        # Initialize cluster assignments:
        # - 'labels' stores the cluster assignment for each data point
        # - 'prev_labels' stores the previous cluster assignment for each data point (for debugging purposes)
        # - 'sensitive_counts' stores the counts of each sensitive feature value in each cluster
        labels = np.zeros(X.shape[0], dtype=int)
        prev_labels = labels.copy()
        sensitive_counts = {i: {val: 0 for val in np.unique(sensitive_feature)} for i in range(self.n_clusters)}

        for iteration in range(self.max_iter):
            print(f"\nITERATION {iteration + 1}")

            # Iterate over all data points and assign them to the closest cluster with fairness
            for i, sample in enumerate(X):
                distances = np.linalg.norm(self.cluster_centers_ - sample, axis=1)  # Compute the distance from data point to all centroids                
                sorted_clusters = np.argsort(distances)                             # Sort clusters by distance                   
                
                # Variables to track assignment and fairness violation
                assigned = False
                best_violation = float('inf')
                best_cluster = None

                for cluster in sorted_clusters:
                    total_in_cluster = sum(sensitive_counts[cluster].values())
                    # If the cluster is empty, automatically assign the point
                    if total_in_cluster == 0:
                        labels[i] = cluster
                        sensitive_counts[cluster][sensitive_feature[i]] += 1
                        assigned = True
                        break
                    # Check if assigning the point to this cluster maintains fairness
                    if self._check_fairness(cluster, sensitive_feature[i], sensitive_counts, total_in_cluster, global_ratios, iteration):
                        labels[i] = cluster
                        sensitive_counts[cluster][sensitive_feature[i]] += 1
                        assigned = True
                        break
                    else:
                        # Track least fairness violation in case we need a fallback
                        current_ratio = (sensitive_counts[cluster][sensitive_feature[i]] + 1) / (total_in_cluster + 1)
                        target_ratio = global_ratios[sensitive_feature[i]]
                        violation = abs(current_ratio - target_ratio)
                        if violation < best_violation:
                            best_violation = violation
                            best_cluster = cluster

                # Fallback: assign point to least unfair cluster
                if not assigned and best_cluster is not None:
                    labels[i] = best_cluster
                    sensitive_counts[best_cluster][sensitive_feature[i]] += 1
                    print(f"⚠️ Point {i} fallback-assigned to Cluster {best_cluster} with fairness violation {best_violation:.4f}")


            # Update centroids:
            # - compute the new centroids by taking the mean of all points assigned to each cluster
            # - if a cluster has no points assigned to it, keep the previous centroid
            new_centroids = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else self.cluster_centers_[i] \
                                      for i in range(self.n_clusters)])

            # Compute and print metrics:
            sample = rng.choice(n_samples, min(5000, n_samples), replace=False)
            sil_score = silhouette_score(X[sample], labels[sample])
            dbi_score = davies_bouldin_score(X[sample], labels[sample])
            print(f"  Silhouette (sampled): {sil_score:.4f} | DBI (sampled): {dbi_score:.4f}")

            # Difference between previous and current cluster assignments & centroid shift:
            num_changes = np.sum(labels != prev_labels)
            prev_labels = labels.copy()
            centroid_shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
            print(f"  Centroid Shift: {centroid_shift} | Points Changing Clusters: {num_changes}")           
    
            # Check for convergence:
            # - if the change in centroids is less than the tolerance (tol), the algorithm stops
            if centroid_shift < self.tol:
                print("\n========= Convergence reached! =========")
                break
            self.cluster_centers_ = new_centroids

        self.labels_ = labels
        return self

    def _check_fairness(self, cluster, sensitive_value, sensitive_counts, total_in_cluster, global_ratios, iteration):
        """Check if assigning a point to a cluster maintains fairness."""
        current_ratio = (sensitive_counts[cluster][sensitive_value] + 1) / (total_in_cluster + 1) if total_in_cluster > 0 else 1.0  # Default to 1 if the cluster is empty
        target_ratio = global_ratios[sensitive_value]
        
        # Penalize distance if fairness is violated
        ratio_difference = abs(current_ratio - target_ratio) 
        # Using dynamic tolerance: Starting relaxed and tighten as iterations progress
        dynamic_tolerance = self.fairness_tolerance / (iteration + 1) ** 0.5

        can_assign = ratio_difference <= dynamic_tolerance
        return can_assign

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to."""
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        # Assigns each point to the nearest centroid
        return np.argmin(distances, axis=1)    