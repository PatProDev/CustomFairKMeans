import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

class SilhouetteDbiKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, use_dbi=False, random_state=None, score_threshold=None, sample_size=10000):
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
        if doing so improves its individual approximate silhouette score.
        
        Parameters:
        - X: ndarray, input data points.
        - labels: ndarray, current cluster labels for each point.
        - d2c: ndarray, distances from each data point to each centroid.
        
        Returns:
        - labels: ndarray, updated cluster labels after reassignment attempts.
        """
        n_samples = X.shape[0]
        for i in range(n_samples):
            # Initialize the best label and silhouette score for the current point
            best_label = labels[i]
            best_silhouette = self._approx_silhouette_for_point(d2c[i], best_label)
            # Check all other clusters as potential new assignments for the current point
            for c in range(self.n_clusters):
                if c == labels[i]:
                    continue
                old_label = labels[i]
                labels[i] = c

                # Calculate the approximate silhouette score for this temporary assignment
                temp_silhouette = self._approx_silhouette_for_point(d2c[i], c)

                if temp_silhouette > best_silhouette:
                    best_silhouette = temp_silhouette
                    best_label = c
                labels[i] = old_label
            
            # After evaluating all candidate clusters, assign the point to the best label found
            labels[i] = best_label
        return labels

    def _assign_labels_with_dbi_batch(self, X, d2c, n_batch_iterations=5, sample_limit=500):
        """
        Performs batch reassignments of data points to optimize the Davies-Bouldin Index (DBI).
        This method iteratively identifies the most poorly separated cluster pair and
        reassigns a sample of their boundary points to improve overall clustering quality.
        
        Parameters:
        - X: ndarray, input data points of shape (n_samples, n_features).
        - d2c: ndarray, distances from each data point to each centroid, shape (n_samples, n_clusters).
        - n_batch_iterations: int, the maximum number of batch reassignment cycles to perform.
        - sample_limit: int, the maximum total number of boundary points to consider for reassignment
                        across both clusters in a given batch.
        
        Returns:
        - labels: ndarray, the updated cluster labels after batch optimization.
        """
        print("Initializing labels with optimized batch DBI...")
        n_samples, _ = X.shape
        # Initial assignment of each point to its nearest centroid
        labels = np.argmin(d2c, axis=1)

        # Handle edge cases where DBI cannot be computed (e.g., less than 2 clusters or points)
        if len(np.unique(labels)) < 2 or n_samples < 2:
            print("Warning: Insufficient clusters or samples for DBI-based initialization.")
            return labels

        # Calculate the initial DBI score to track improvement
        try:
            best_score = davies_bouldin_score(X, labels)
        except ValueError:
            # If initial assignment leads to an empty cluster, DBI calculation will fail.
            # In such cases, return the base labels.
            print("Initial labels led to empty clusters, returning base labels.")
            return labels
        print(f"Initial DBI score: {best_score:.4f}")
        
        rng = np.random.RandomState(self.random_state) # Initialize random state for reproducibility
        
        # Iterate for a fixed number of batch reassignment cycles
        for i in range(n_batch_iterations):
            labels_prev = labels.copy() # Store labels from the previous iteration to check for convergence
            
            # Recalculate centroids and intra-cluster scatter for the current labels
            centroids = np.zeros((self.n_clusters, X.shape[1]))
            intra_scatter = np.zeros(self.n_clusters)
            
            for c in range(self.n_clusters):
                cluster_points = X[labels == c]
                if len(cluster_points) > 0:
                    centroids[c] = np.mean(cluster_points, axis=0) # Update centroid mean
                    # Calculate average distance of points to their centroid (intra-cluster scatter)
                    intra_scatter[c] = np.mean(pairwise_distances(cluster_points, [centroids[c]]))

            # Calculate inter-cluster separation (distances between centroids)
            inter_separation = pairwise_distances(centroids)
            
            # Compute the DBI matrix: (intra_scatter_i + intra_scatter_j) / inter_separation_ij
            # Ignore division by zero warnings and fill diagonal with -inf to avoid self-comparison
            with np.errstate(divide='ignore', invalid='ignore'):
                dbi_matrix = (intra_scatter[:, None] + intra_scatter[None, :]) / inter_separation
                np.fill_diagonal(dbi_matrix, -np.inf) # Set diagonal to -inf as clusters are not compared to themselves
                
            # Find the pair of clusters with the highest DBI ratio (most poorly separated/compact)
            c1, c2 = np.unravel_index(np.argmax(dbi_matrix), dbi_matrix.shape)
            
            # Identify and reassign points between the worst-performing clusters (c1 and c2)
            to_reassign = []
            # Check points from c1 to c2, and then from c2 to c1
            for c_from, c_to in [(c1, c2), (c2, c1)]:
                points_to_check = np.where(labels == c_from)[0]
                # Find points in the 'from' cluster that are actually closer to the 'to' cluster's centroid
                boundary_points = points_to_check[d2c[points_to_check, c_to] < d2c[points_to_check, c_from]]
                
                # Randomly sample a subset of these boundary points for reassignment
                sample = rng.choice(boundary_points, min(len(boundary_points), sample_limit // 2), replace=False)
                # Store the index and the new cluster label for these sampled points
                to_reassign.extend([(idx, c_to) for idx in sample])

            # Apply all identified reassignments in one go
            if to_reassign:
                reassign_indices, new_labels = zip(*to_reassign)
                labels[list(reassign_indices)] = list(new_labels)
            
            # Check for convergence: if no labels changed in this batch iteration, stop
            if np.array_equal(labels, labels_prev):
                print(f"Batch {i+1}: No labels changed. Stopping batch optimization.")
                break
            
            # Recalculate the global DBI score after the batch reassignment
            try:
                new_score = davies_bouldin_score(X, labels)
                print(f"Batch {i+1}: New DBI score: {new_score:.4f}")
            except ValueError:
                # If the new labels lead to an invalid clustering (e.g., empty cluster), revert and stop
                print(f"Batch {i+1}: Invalid labels, reverting.")
                labels = labels_prev
                break

            # Check if the new score improved; if not, revert to previous labels and stop
            if new_score < best_score:
                best_score = new_score
            else:
                print(f"Batch {i+1}: Score did not improve, reverting and stopping.")
                labels = labels_prev
                break
                
        return labels


    def fit(self, X, y=None):
        X = check_array(X)
        n_samples, _ = X.shape
        rng = np.random.RandomState(self.random_state)

        # 1) random initial centroids
        self.cluster_centers_ = X[rng.choice(n_samples, self.n_clusters, replace=False)]
        d2c = pairwise_distances(X, self.cluster_centers_)
        labels = np.argmin(d2c, axis=1)
        
        for it in range(1, self.max_iter + 1):
            print(f"\nITERATION {it}")
            labels_prev = labels.copy()

            # 2) Intelligent label initialization using either DBI or Silhouette
            if self.use_dbi:
                labels = self._assign_labels_with_dbi_batch(X, d2c)
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

            # 5) Stopping criteria
            is_dbi_met = self.use_dbi and score <= self.score_threshold
            is_sil_met = not self.use_dbi and score >= self.score_threshold
            
            if centroid_shift < self.tol:
                if is_dbi_met or is_sil_met:
                    print("  \nConvergence reached! Centroids stabilized & score threshold met.")
                    break
                else:
                    print("  \nConvergence reached! Centroids stabilized, but score threshold not met...")
                    break
            
            if np.array_equal(labels, labels_prev):
                if is_dbi_met or is_sil_met:
                    print("  No label changes and score threshold met; stopping.")
                    break
                else:
                    print("  Warning: no label changes but score threshold not satisfied. Refining...")

        self.labels_ = labels
        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
