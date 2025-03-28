import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin

class HarmonicKMeans(BaseEstimator, ClusterMixin):
    def __init__(self, n_clusters=3, max_iters=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = 1e-4
        self.random_state = random_state
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Fits the K-Harmonic Means model to the data.

        Parameters:
        - X: Input data, shape (n_samples, n_features).
        """
        # if self.random_state is not None:
        #     np.random.seed(self.random_state)

        # n_samples, n_features = X.shape
        # random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        # self.centroids = X[random_indices].astype(float)                    # make sure the centroids are floats
        # self.centroids += np.random.normal(0, 1e-6, self.centroids.shape)   # Add small noise
        # empty_cluster_counts = [0] * self.n_clusters                        # Track empty cluster counts

        # print("Initial Centroids:\n", self.centroids)   
        # previous_centroids = self.centroids.copy()      

        # for iteration in range(self.max_iters):
        #     print(f"\nIteration {iteration + 1}")
        #     distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        #     distances = distances.T

        #     # Add a small epsilon to avoid division by zero
        #     epsilon = 1e-10
        #     distances = np.where(distances == 0, epsilon, distances)

        #     harmonic_means = np.sum(1 / distances, axis=1)
        #     weights = (1 / distances) / harmonic_means[:, np.newaxis]

        #     # Normalize weights to avoid empty clusters (update: doesn't seem to work)
        #     weights = weights / np.sum(weights, axis=1, keepdims=True)

        #     new_centroids = np.zeros_like(self.centroids)
        #     for k in range(self.n_clusters):
        #         new_centroids[k] = np.sum(weights[:, k][:, np.newaxis] * X, axis=0) / np.sum(weights[:, k])

        #     # --- Track progress ---
        #     # distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in self.centroids])
        #     # distances = distances.T
        #     cluster_assignments = np.argmin(distances, axis=1)
        #     cluster_sizes = [np.sum(cluster_assignments == k) for k in range(self.n_clusters)]
        #     print("Cluster Sizes:", cluster_sizes)
            

        #     # Check for empty clusters and re-initialize (dillema: empty clusters OR not reaching convergence)
        #     # for k in range(self.n_clusters):
        #     #     if cluster_sizes[k] == 0:
        #     #         print(f"Cluster {k} is empty. Re-initializing.")
        #     #         random_index = np.random.choice(n_samples)
        #     #         new_centroids[k] = X[random_index]


        #     # Check for empty clusters and re-initialize conditionally (update: when re-initializing one cluster, the other gets empty)
        #     # for k in range(self.n_clusters):
        #     #     if cluster_sizes[k] == 0:
        #     #         empty_cluster_counts[k] += 1
        #     #         if empty_cluster_counts[k] > 5:  # Re-initialize after 5 consecutive empty iterations
        #     #             print(f"Cluster {k} is consistently empty. Re-initializing.")

        #     #             # Local re-initialization
        #     #             other_centroids = np.delete(new_centroids, k, axis=0)
        #     #             distances_to_others = np.linalg.norm(X[:, np.newaxis] - other_centroids, axis=2)
        #     #             min_distances = np.min(distances_to_others, axis=1)
        #     #             random_index = np.random.choice(np.argsort(min_distances)[:10]) #pick one of the 10 closest points
        #     #             new_centroids[k] = X[random_index]

        #     #             empty_cluster_counts[k] = 0  # Reset count
        #     #     else:
        #     #         empty_cluster_counts[k] = 0  # Reset count if cluster is not empty


        #     # Calculate centroid shifts
        #     centroid_shift = np.linalg.norm(new_centroids - previous_centroids, axis=1)
        #     print("Centroids Shift:", centroid_shift)
        #     previous_centroids = new_centroids.copy() #update previous centroids

        #     # Check if convergence is reached
        #     if np.all(centroid_shift < self.tol):
        #         print(f"Convergence reached after {iteration + 1} iterations.")
        #         break

        #     self.centroids = new_centroids

        # self.labels = self.predict(X)
        # return self




        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Randomly initialize centroids
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices].astype(float)
        self.centroids += np.random.normal(0, 1e-6, self.centroids.shape)

        # Controids copy for calculating centroid shift
        previous_centroids = self.centroids.copy()

        for iteration in range(self.max_iters):
            print(f"\nIteration {iteration + 1}")
            distances = np.array([np.sum((X - centroid)**2, axis=1) for centroid in self.centroids]) # the Squared Euclidean distances between each data point in X and each centroid
            distances = distances.T                                     # distances from the i-th point to the k-th centroid
            epsilon = 1e-10 
            distances = np.where(distances == 0, epsilon, distances)    # replaces any zero distances with epsilon

            # Calculate q_i_k using equation (7) from the paper 
            q_values = np.zeros((n_samples, self.n_clusters))   # Initialize an array to store the q_i,k values
            for i in range(n_samples):                          # Iterate over each data point
                d_i = distances[i]                              # Distances from the i-th data point to all centroids
                d_min_index = np.argmin(d_i)                    # Index of the closest centroid 
                d_min = d_i[d_min_index]                        # Distance to the closest centroid

                for k in range(self.n_clusters):                # Iterate over each centroid
                    ratios = d_min / d_i                        # Ratios of the minimum distance to all other distances 
                    #print("Ratios:", ratios)    
                    numerator = d_i[k]**4                       # Numerator of Equation 7 from the paper
                    denominator = (d_i[k]**3) * (1 + np.sum(ratios[ratios != ratios[k]])) * \
                                  (1 + np.sum(ratios**2))**2    # Denominator of Equation 7 from the paper
                    q_values[i, k] = numerator / denominator if denominator != 0 else 0
                    #print("q_i,k values:", q_values[i, k])

            # Calculate q_i and p_i_k using equations (6.3-6.5) from the paper 
            q_i = np.sum(q_values, axis=1)                      # Sum of q_i,k values for each data point
            p_values = q_values / q_i[:, np.newaxis]            # Calculate p_i,k values (weights) by dividing each q_i,k by the corresponding q_i value

            # Update centroids using equation (5) or (6.5) from the paper 
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                new_centroids[k] = np.sum(p_values[:, k][:, np.newaxis] * X, axis=0)    # Calculate a weighted sum of the data points, where the weights are the p_i,k values
                                                                                        # Each centroid is moved to the weighted "center" of the data points, 
                                                                                        # where the weights are determined by the harmonic averages of the distances.

            # --- Added for investigation ---
            # distances_calc = np.array([np.sum((X - centroid)**2, axis=1) for centroid in self.centroids])
            # distances_calc = distances_calc.T
            cluster_assignments = np.argmin(distances, axis=1)
            cluster_sizes = [np.sum(cluster_assignments == k) for k in range(self.n_clusters)]
            print("Cluster Sizes:", cluster_sizes)

            # Calculate and print centroid shift
            centroid_shift = np.linalg.norm(new_centroids - previous_centroids, axis=1)
            previous_centroids = new_centroids.copy()
            print("Centroid Shift:", centroid_shift)
            # --- End of added section ---

            # Check for convergence
            if np.all(centroid_shift < self.tol):
                print(f"Converged after {iteration + 1} iterations.")
                break

            self.centroids = new_centroids

        self.labels = self.predict(X)
        return self

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