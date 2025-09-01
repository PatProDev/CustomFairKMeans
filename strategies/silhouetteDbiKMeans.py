import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

class SilhouetteDbiKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, use_dbi=False, random_state=None, score_threshold=0.5, sample_size=10000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.use_dbi = use_dbi
        self.random_state = random_state
        self.score_threshold = score_threshold
        self.sample_size = sample_size
        self.cluster_centers_ = None
        self.labels_ = None
        self.best_score_ = None

    def _approx_silhouette_for_point(self, dists, label):
        a = dists[label]
        b = np.min(np.delete(dists, label))
        return 0.0 if max(a, b) == 0 else (b - a) / max(a, b)   
      
    def _assign_labels_with_silhouette(self, X, labels, d2c):
        """
        Greedy per-point reassignment strategy to improve the Silhouette score.
        This method iterates through each data point and attempts to reassign it to a different cluster
        if doing so improves its individual (local) approximate silhouette score.
        Parameters:
        - X: ndarray, input data points.
        - labels: ndarray, current cluster labels for each point.
        - d2c: ndarray, distances from each data point to each centroid.
        
        Returns:
        - labels: ndarray, updated cluster labels after reassignment attempts.
        """
        n_samples = X.shape[0]
        for point in range(n_samples):
            # Initialize the best label and silhouette score for the current point
            best_label = labels[point]
            best_silhouette = self._approx_silhouette_for_point(d2c[point], best_label)
            # Check all other clusters as potential new assignments for the current point
            for c in range(self.n_clusters):
                if c == labels[point]:
                    continue
                old_label = labels[point]
                labels[point] = c

                # Calculate the approximate silhouette score for this temporary assignment
                temp_silhouette = self._approx_silhouette_for_point(d2c[point], c)

                if temp_silhouette > best_silhouette:
                    best_silhouette = temp_silhouette
                    best_label = c
                labels[point] = old_label
            
            # After evaluating all candidate clusters, assign the point to the best label found
            labels[point] = best_label
        return labels

    def _assign_labels_with_dbi_batch(self, X, labels, n_batch_iterations=5, sample_limit=500):
        """
        Performs batch reassignments of data points to optimize the Davies-Bouldin Index (DBI).
        This method identifies the most poorly separated cluster pair and reassigns a
        sample of their boundary points to directly improve the DBI score.

        Parameters:
        - X: ndarray, input data points of shape (n_samples, n_features).
        - d2c: ndarray, distances from each data point to each centroid, shape (n_samples, n_clusters).
        - n_batch_iterations: int, the maximum number of batch reassignment cycles.
        - sample_limit: int, the maximum number of boundary points to consider for reassignment.

        Returns:
        - labels: ndarray, the updated cluster labels after batch optimization.
        """
        n_samples, _ = X.shape

        if len(np.unique(labels)) < 2 or n_samples < 2:
            print("Warning: Insufficient clusters or samples for DBI-based optimization.")
            return labels

        try:
            best_score = davies_bouldin_score(X, labels)
        except ValueError:
            print("  Initial labels led to empty clusters, returning base labels.")
            return labels
        print(f"  Initial DBI score: {best_score:.4f}")

        rng = np.random.RandomState(self.random_state)
        current_labels = labels.copy()

        for i in range(n_batch_iterations):
            labels_prev = current_labels.copy()

            # Calculate centroids and intra-cluster scatter in one pass
            centroids = np.array([X[current_labels == c].mean(axis=0) if np.any(current_labels == c) else np.nan for c in range(self.n_clusters)])
            intra_scatter = np.array([np.mean(pairwise_distances(X[current_labels == c], [centroids[c]])) if np.any(current_labels == c) else np.nan for c in range(self.n_clusters)])

            inter_separation = pairwise_distances(centroids, centroids)
            with np.errstate(divide='ignore', invalid='ignore'):
                dbi_matrix = (intra_scatter[:, None] + intra_scatter[None, :]) / inter_separation
                np.fill_diagonal(dbi_matrix, -np.inf)

            # Find the pair of clusters with the highest DBI ratio
            c1, c2 = np.unravel_index(np.argmax(dbi_matrix), dbi_matrix.shape)

            # Get distances to the current centroids
            d2c_current = pairwise_distances(X, centroids)

            # Identify "boundary" points between the two worst-performing clusters (c1 and c2)
            # A point is a boundary point if it is assigned to one of these two clusters, but is closer to the other's centroid.
            boundary_indices = np.where(
                ((current_labels == c1) & (d2c_current[:, c1] > d2c_current[:, c2])) |
                ((current_labels == c2) & (d2c_current[:, c2] > d2c_current[:, c1]))
            )[0]
            
            if len(boundary_indices) == 0:
                print(f"  Batch {i+1}: No boundary points found. Stopping batch optimization.")
                break

            # Reassign a sample of boundary points
            sample_indices = rng.choice(boundary_indices, min(len(boundary_indices), sample_limit), replace=False)
            current_labels[sample_indices] = np.argmin(d2c_current[sample_indices], axis=1)
            
            # Check for convergence
            if np.array_equal(current_labels, labels_prev):
                print(f"  Batch {i+1}: No labels changed. Stopping batch optimization.")
                break

            try:
                new_score = davies_bouldin_score(X, current_labels)
                print(f"  Batch {i+1}: New DBI score: {new_score:.4f}")
            except ValueError:
                print(f"  Batch {i+1}: Invalid labels, reverting.")
                current_labels = labels_prev
                break

            if new_score < best_score:
                best_score = new_score
            else:
                print(f"  Batch {i+1}: Score did not improve, reverting and stopping.")
                current_labels = labels_prev
                break
                
        return current_labels


    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, _ = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1) random initial centroids
        self.cluster_centers_ = X[rng.choice(n_samples, self.n_clusters, replace=False)]
        d2c = pairwise_distances(X, self.cluster_centers_)
        labels = np.argmin(d2c, axis=1)
        best_score = float('inf') if self.use_dbi else -float('inf')
        best_score_counter = 0
        for it in range(1, self.max_iter + 1):
            print(f"\nITERATION {it}")

            # 2) Intelligent label initialization using either DBI or Silhouette
            if self.use_dbi:
                labels = self._assign_labels_with_dbi_batch(X, labels)
            else:
                labels = self._assign_labels_with_silhouette(X, labels, d2c)

            # 3) Update centroids and point-to-centroid distances
            new_centroids = np.vstack([
                X[labels == c].mean(axis=0) if np.any(labels == c) else self.cluster_centers_[c]
                for c in range(self.n_clusters)
            ])
            centroid_shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
            self.cluster_centers_ = new_centroids
            d2c = pairwise_distances(X, self.cluster_centers_)

            # 4) Global score evaluation (sampled for speed)
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

            # 5) Best score update
            if self.use_dbi:
                best_score = min(best_score, score)
                if score <= best_score:
                    if best_score_counter > 0:
                        best_score_counter = best_score_counter + 1
                    else:
                        best_score_counter = 1
                    self.best_score_ = score
                    self.labels_ = labels
                    print("  Best score saved. Labels set.")
            else:
                best_score = max(best_score, score)
                if score >= best_score:
                    if best_score_counter > 0:
                        best_score_counter = best_score_counter + 1
                    else:
                        best_score_counter = 1
                    self.best_score_ = score
                    self.labels_ = labels
                    print("  Best score saved. Labels set.")

            # 6) Stopping criteria
            is_dbi_met = self.use_dbi and score <= self.score_threshold
            is_sil_met = not self.use_dbi and score >= self.score_threshold
            
            if centroid_shift < self.tol:
                if is_dbi_met or is_sil_met:
                    print("  \nConvergence reached! Centroids stabilized & score threshold met. STOPPING.")
                    break
                else:
                    print("  \nConvergence reached! Centroids stabilized, but score threshold not met. STOPPING.")
                    break
            
            if best_score_counter >= 20:
                print("  Best score hasn't improved in 20 consecutive iterations. STOPPING.")
                break

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
