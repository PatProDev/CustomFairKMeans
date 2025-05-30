import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import pairwise_distances, silhouette_score

class SilhouetteDbiKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42, silhouette_threshold=0.50, sample_size=10000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.silhouette_threshold = silhouette_threshold    # A "good" silhouette score typically falls above 0.5. Scores between 0.25 and 0.5 are considered acceptable, while those above 0.5 suggest a good clustering result
        self.sample_size = sample_size                      # When calling sklearn silhouette_score, we sample a subset of the data to speed up the computation, especially for large datasets.

    # ------------------------------------------------------------------
    #                           FIT METHOD
    # ------------------------------------------------------------------
    def fit(self, X, sensitive_feature=None):
        X = check_array(X)
        n_samples, _ = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1) random initial centroids
        self.cluster_centers_ = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        # initial labels by nearest centroid
        d2c = pairwise_distances(X, self.cluster_centers_)
        labels = np.argmin(d2c, axis=1)


        for it in range(1, self.max_iter + 1):
            print(f"ITERATION {it}")
            labels_prev = labels.copy()

            # 2) Greedy reassignment based on *approximate* silhouette value
            for i in range(n_samples):
                # distances of this point to all centroids
                dists_i = d2c[i]

                best_label = labels[i]
                best_s = self._approx_silhouette_for_point(dists_i, best_label)

                # Evaluate silhouette if moved to any other cluster
                for c in range(self.n_clusters):
                    if c == labels[i]:
                        continue
                    s_candidate = self._approx_silhouette_for_point(dists_i, c)
                    if s_candidate > best_s:
                        best_s = s_candidate
                        best_label = c

                labels[i] = best_label

            # 3) Update centroids and point‑to‑centroid distances
            new_centroids = np.vstack([
                X[labels == c].mean(axis=0) if np.any(labels == c) else self.cluster_centers_[c]
                for c in range(self.n_clusters)
            ])
            centroid_shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
            self.cluster_centers_ = new_centroids
            d2c = pairwise_distances(X, self.cluster_centers_)

            # 4) Global silhouette (sampled for speed)
            if len(np.unique(labels)) > 1:
                sample = min(self.sample_size, n_samples)
                sample_idx = rng.choice(n_samples, sample, replace=False)
                global_sil = silhouette_score(X[sample_idx], labels[sample_idx])
            else:
                global_sil = 0.0

            print(f"  Centroid shift: {centroid_shift:.6f} | Global silhouette (sampled): {global_sil:.3f}")

            # 5) Stopping criteria
            if centroid_shift < self.tol and global_sil >= self.silhouette_threshold:
                print("  \nConvergence reached! Stable centroids & silhouette threshold met!")
                break

            if np.array_equal(labels, labels_prev):
                if global_sil >= self.silhouette_threshold:
                    print("  No label changes and silhouette threshold met; stopping.")
                    break
                else:
                    print("  Warning: no label changes but silhouette below threshold. Continuing to refine centroids...")

        self.labels_ = labels
        return self

    # ------------------------------------------------------------------
    #                           HELPER METHOD
    # ------------------------------------------------------------------
    def _approx_silhouette_for_point(self, dists, label):
        """Return *approximate* silhouette for one point, given its distances
        to all centroids (vector `dists`) and its current label.

        We approximate *a* with the distance to its own centroid and *b* with
        the second‑smallest centroid distance (nearest other cluster).
        """
        a = dists[label]
        b = np.min(np.delete(dists, label))
        return 0.0 if max(a, b) == 0 else (b - a) / max(a, b)

    # ------------------------------------------------------------------
    #                         PREDICT METHOD
    # ------------------------------------------------------------------
    def predict(self, X):
        check_is_fitted(self, 'cluster_centers_')
        X = check_array(X)
        distances = pairwise_distances(X, self.cluster_centers_)
        return np.argmin(distances, axis=1)
