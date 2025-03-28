import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

class SilhouetteKMeans(BaseEstimator, ClusterMixin):
    """
    K-Means clustering with Silhouette Coefficient for cluster selection.
    """

    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X, y=None):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training instances to cluster.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        n_samples, n_features = X.shape

        best_silhouette = -1
        best_centers = None
        best_labels = None
        best_inertia = float('inf')

        for iteration in range(self.max_iters):
            print(f"\nIteration {iteration + 1}")
            # Initialize centroids randomly
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centers = X[indices]

            for _ in range(100):  # Inner iterations for convergence
                # Assign points to the nearest centroid
                distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                labels = np.argmin(distances, axis=1)

                # Update centroids
                new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

                # Check for convergence
                if np.all(centers == new_centers):
                    break
                centers = new_centers

            # Calculate silhouette score
            if len(np.unique(labels)) > 1:
                silhouette = silhouette_score(X, labels)
            else:
                silhouette = -1 # handle case when only one cluster is found

            # calculate inertia
            distances = np.linalg.norm(X - centers[labels], axis=1)
            inertia = np.sum(distances**2)

            # Update best results if silhouette is better
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_centers = centers
                best_labels = labels
                best_inertia = inertia

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)
        X = check_array(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)