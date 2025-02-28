import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans

class SilhouetteKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42, sample_fraction=0.1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.sample_fraction = sample_fraction  # Fraction of points to sample for silhouette computation

    def fit(self, X, sensitive_feature):
        X = check_array(X)
        sensitive_feature = np.array(sensitive_feature)

        if len(X) != len(sensitive_feature):
            raise ValueError("X and sensitive_feature must have the same length.")

        # Step 1: Initial clustering using standard KMeans
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        labels = kmeans.fit_predict(X)
        self.cluster_centers_ = kmeans.cluster_centers_
        
        # Step 2: Track sensitive feature distribution
        sensitive_counts = {i: {val: 0 for val in np.unique(sensitive_feature)} for i in range(self.n_clusters)}
        for i, label in enumerate(labels):
            sensitive_counts[label][sensitive_feature[i]] += 1

        print("Initial Sensitive Counts:", sensitive_counts)

        # Step 3: Iteratively improve clustering based on silhouette score
        for iteration in range(self.max_iter):
            sample_size = max(1, int(self.sample_fraction * len(X)))  # Ensure at least one point is sampled
            sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
            
            silhouette_vals = silhouette_samples(X, labels)
            low_score_indices = sample_indices[silhouette_vals[sample_indices] < np.percentile(silhouette_vals[sample_indices], 25)]
            
            num_changes = 0
            for i in low_score_indices:
                distances = np.linalg.norm(self.cluster_centers_ - X[i], axis=1)
                sorted_clusters = np.argsort(distances)
                current_cluster = labels[i]
                
                for cluster in sorted_clusters:
                    if cluster != current_cluster:
                        temp_labels = labels.copy()
                        temp_labels[i] = cluster
                        
                        sampled_indices = np.random.choice(len(X), size=sample_size, replace=False)
                        new_silhouette = silhouette_score(X[sampled_indices], temp_labels[sampled_indices])

                        if new_silhouette > silhouette_score(X[sampled_indices], labels[sampled_indices]):
                            labels[i] = cluster
                            sensitive_counts[current_cluster][sensitive_feature[i]] -= 1
                            sensitive_counts[cluster][sensitive_feature[i]] += 1
                            num_changes += 1
                            break
            
            print(f"Iteration {iteration + 1}, Points Reassigned: {num_changes}")

            if num_changes == 0:
                print("Convergence reached.")
                break

        self.labels_ = labels
        return self

    def predict(self, X):
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
