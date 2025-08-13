import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score
import time

class MetricDrivenKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42,
                 score_threshold=0.5, sample_size=10000, use_dbi=False):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        # A "good" silhouette score is higher: 
        #       - Scores between 0.25 and 0.5 are considered acceptable, while those above 0.5 suggest a good clustering result.
        # A "good" DBI score is lower:
        #       - Values below 0.5 indicating well-separated clusters.
        self.score_threshold = score_threshold
        # When calling sklearn silhouette_score, we sample a subset of the data to speed up the computation, especially for large datasets.
        self.sample_size = sample_size
        # Incidate whether to use silhouette score or Davies-Bouldin Index (DBI) for clustering quality assessment.
        self.use_dbi = use_dbi
    
    # ------------------------------------------------------------------
    #                           HELPER METHODS
    # ------------------------------------------------------------------
  
    # def _initialize_labels_with_dbi(self, X, d2c):
    #     # Fast: assign each point to its closest centroid (like KMeans step)
    #     labels = np.argmin(d2c, axis=1)
    #     return labels

    def _initialize_labels_with_dbi(self, X, d2c, sample_limit=500, top_n_centroids=2):
        """
        Semi-greedy DBI: instead of the classic assignment to the nearest centroid, 
        a few points are checked to see if it makes sense to move them relative to the DBI.

        Parameters:
        - X: ndarray, input data
        - d2c: ndarray, distances to centroids
        - sample_limit: int, max number of worst points to examine
        - top_n_centroids: int, how many candidate centroids to try per point

        Returns:
        - labels: ndarray, reassigned cluster labels
        """
        print("Initializing labels with semi-greedy DBI...")

        n_samples = X.shape[0]
        labels = np.argmin(d2c, axis=1)
        current_score = davies_bouldin_score(X, labels)

        # 1. Identify worst points based on distance to their assigned centroid
        cluster_dists = d2c[np.arange(n_samples), labels]
        worst_indices = np.argsort(cluster_dists)[-sample_limit:]

        print(f"Selected {len(worst_indices)} worst points for DBI reassignment")

        for idx in worst_indices:
            #print(f"  Point {idx}: Current label: {labels[idx]}, Distance to centroid: {cluster_dists[idx]:.4f}")
            current_cluster = labels[idx]
            sorted_clusters = np.argsort(d2c[idx])
            candidate_clusters = [c for c in sorted_clusters if c != current_cluster][:top_n_centroids]

            best_score = current_score
            best_label = current_cluster

            for new_cluster in candidate_clusters:
                temp_labels = labels.copy()
                temp_labels[idx] = new_cluster

                # Recalculate DBI
                try:
                    new_score = davies_bouldin_score(X, temp_labels)
                except:
                    continue  # Skip if invalid clustering (e.g., empty cluster)

                if new_score < best_score:
                    best_score = new_score
                    best_label = new_cluster

            labels[idx] = best_label

        return labels

    def _approx_silhouette_for_point(self, dists, label):
        a = dists[label]
        b = np.min(np.delete(dists, label))
        return 0.0 if max(a, b) == 0 else (b - a) / max(a, b)   
      
    def _initialize_labels_with_silhouette(self, X, labels, d2c):
        n_samples = X.shape[0]
        for i in range(n_samples):
            best_label = labels[i]
            best_silhouette = self._approx_silhouette_for_point(d2c[i], best_label)
            #print(f"  Point {i}: Initial silhouette: {best_silhouette:.4f}")
            for c in range(self.n_clusters):
                if c == labels[i]:
                    continue
                old_label = labels[i]
                labels[i] = c
                temp_silhouette = self._approx_silhouette_for_point(d2c[i], c)
                #print(f"  Point {i}: Testing cluster {c} gives silhouette: {temp_silhouette:.4f}")
                if temp_silhouette > best_silhouette:
                    best_silhouette = temp_silhouette
                    best_label = c
                labels[i] = old_label
            labels[i] = best_label
        return labels

    # ------------------------------------------------------------------
    #                           FIT METHOD
    # ------------------------------------------------------------------
    def fit(self, X):
        X = check_array(X)
        n_samples, _ = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1) random initial centroids
        self.cluster_centers_ = X[rng.choice(n_samples, self.n_clusters, replace=False)]
        # initial labels by nearest centroid
        d2c = pairwise_distances(X, self.cluster_centers_)
        labels = np.argmin(d2c, axis=1)

        for it in range(1, self.max_iter + 1):
            print(f"\nITERATION {it}")
            start_time = time.time()
            labels_prev = labels.copy()

            # 2) Initialization:
            #       - fast KMeans-like assignment if using DBI (it's a global score, and the fast version works perfectly well if we just re-assign to nearest centroid)
            #       - greedy per-point reassignment to improve Silhouette (improves Silhouette locally)
            if self.use_dbi:
                labels = self._initialize_labels_with_dbi(X, d2c)
                #labels = self._initialize_labels_with_dbi_cluster_level(X, labels, self.cluster_centers_)
            else:
                labels = self._initialize_labels_with_silhouette(X, labels, d2c)   

            # 3) Update centroids and point‑to‑centroid distances
            new_centroids = np.vstack([
                X[labels == c].mean(axis=0) if np.any(labels == c) else self.cluster_centers_[c]
                for c in range(self.n_clusters)
            ])
            centroid_shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
            self.cluster_centers_ = new_centroids
            d2c = pairwise_distances(X, self.cluster_centers_)

            # 4) Global silhouette or DBI (sampled for speed)
            if len(np.unique(labels)) > 1:
                sample = min(self.sample_size, n_samples)
                sample_idx = rng.choice(n_samples, sample, replace=False)
                if self.use_dbi:
                    score = davies_bouldin_score(X[sample_idx], labels[sample_idx])
                    print(f"  Centroid shift: {centroid_shift:.6f} | Global DBI (sampled): {score:.3f}")
                else:
                    score = silhouette_score(X[sample_idx], labels[sample_idx])
                    print(f"  Centroid shift: {centroid_shift:.6f} | Global Silhouette (sampled): {score:.3f}")
            else:
                score = float('inf') if self.use_dbi else 0.0

            # 5) Stopping criteria 1.0 for centroid stabilization and score threshold
            if centroid_shift < self.tol:
                if (self.use_dbi and score <= self.score_threshold) or \
                (not self.use_dbi and score >= self.score_threshold):
                    print("  \nConvergence reached! Centroids stabilized & score threshold met.")
                    break
                else:
                    print("  \nConvergence reached! Centroids stabilized, but score threshold not met...")
                    break
            else:
                if (self.use_dbi and score <= self.score_threshold) or \
                (not self.use_dbi and score >= self.score_threshold):
                    print("  Score threshold met but centroids still moving - continuing to refine...")
            
            # 5) Stopping criteria 2.0 for no label changes
            if np.array_equal(labels, labels_prev):
                if (self.use_dbi and score <= self.score_threshold) or \
                   (not self.use_dbi and score >= self.score_threshold):
                    print("  No label changes and score threshold met; stopping.")
                    break
                else:
                    print("  Warning: no label changes but score threshold not satisfied. Refining...")

            print("Time taken: %s seconds" % (time.time() - start_time))

        self.labels_ = labels
        return self
    
    
    def predict(self, X):
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        distances = pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)