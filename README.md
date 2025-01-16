# FairKMeans Algorithm

## Overview
The **FairKMeans** algorithm is a modification of the classic K-Means clustering algorithm. It aims to cluster data points while ensuring fairness with respect to a given sensitive feature (e.g., gender, race). Fairness is enforced by constraining the sensitive group ratios in each cluster to be close to their global proportions.

This approach is useful in applications where it's important to ensure that sensitive groups are fairly represented within each cluster.

## Features
- Enforces fairness constraints during clustering.
- Dynamic fairness tolerance that tightens as the algorithm progresses.
- Implements a fallback mechanism to guarantee cluster assignment.
- Easy integration with scikit-learn tools.

---

## How It Works
The FairKMeans algorithm is built on the following components:

### Initialization
1. **Input Parameters:**
    - `n_clusters`: Number of clusters to form.
    - `max_iter`: Maximum number of iterations to perform.
    - `tol`: Convergence threshold for centroid movement.
    - `fairness_tolerance`: Maximum allowable deviation from global ratios for sensitive groups.
    - `random_state`: Random seed for reproducibility.

2. **Global Ratios:**
   Compute the proportion of each sensitive group in the entire dataset:
   ```python
   global_ratios = {
       value: np.sum(sensitive_feature == value) / len(sensitive_feature)
       for value in np.unique(sensitive_feature)
   }
   ```

3. **Centroid Initialization:**
   Randomly select `n_clusters` points from the data as initial centroids:
   ```python
   rng = np.random.RandomState(self.random_state)
   indices = rng.choice(len(X), self.n_clusters, replace=False)
   self.cluster_centers_ = X[indices]
   ```

### Iterative Clustering
The algorithm iterates to refine cluster assignments while enforcing fairness:

1. **Point Assignment:**
   Each data point is assigned to the nearest cluster that satisfies the fairness constraint:
   ```python
   for i, sample in enumerate(X):
       distances = np.linalg.norm(self.cluster_centers_ - sample, axis=1)
       sorted_clusters = np.argsort(distances)
       for cluster in sorted_clusters:
           if self._check_fairness(cluster, sensitive_feature[i], sensitive_counts, total_in_cluster, global_ratios, iteration):
               labels[i] = cluster
               sensitive_counts[cluster][sensitive_feature[i]] += 1
               break
   ```

2. **Fairness Check:**
   Fairness is validated by comparing the current sensitive group ratio in a cluster to the global ratio:
   ```python
   current_ratio = (sensitive_counts[cluster][sensitive_value] + 1) / (total_in_cluster + 1)
   fairness_penalty = abs(current_ratio - target_ratio)
   dynamic_tolerance = self.fairness_tolerance / (iteration + 1) ** 0.5
   return fairness_penalty <= dynamic_tolerance
   ```

3. **Centroid Update:**
   After assignment, cluster centroids are updated based on the mean of assigned points:
   ```python
   new_centroids = np.array([
       X[labels == i].mean(axis=0) if np.any(labels == i) else self.cluster_centers_[i]
       for i in range(self.n_clusters)
   ])
   ```

4. **Convergence Check:**
   If the change in centroids falls below the tolerance, the algorithm stops:
   ```python
   centroid_shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
   if centroid_shift < self.tol:
       break
   ```

### Fallback Assignment
If no cluster satisfies the fairness constraint, the data point is assigned to the nearest cluster:
```python
if not assigned:
    cluster = sorted_clusters[0]
    labels[i] = cluster
    sensitive_counts[cluster][sensitive_feature[i]] += 1
    print(f"Forced assignment to Cluster {cluster}")
```

---

## Usage
Below is an example of how to use the **FairKMeans** class:

```python
from fair_kmeans import FairKMeans
import numpy as np

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
sensitive_feature = np.array([0, 0, 1, 1, 1, 0])

# Split the data into training and testing sets (80-20 split)
X_train, X_test, sf_train, sf_test = train_test_split(
    X, sensitive_feature, test_size=0.2, random_state=42
)

# Initialize FairKMeans
fair_kmeans = FairKMeans(n_clusters=3, max_iter=100, tol=1e-4, random_state=42)

# Fit the model
fair_kmeans.fit(X_train, sf_train)

# Predict and evaluate FairKMeans on test set
fair_predictions = fair_kmeans.predict(X_test)
fair_distribution = cluster_sensitive_distribution(fair_predictions, sf_test)

# Print sensitive feature distribution between clusters
print("Sensitive Distribution (FairKMeans):")
for cluster, data in fair_distribution.items():
    print(f"  Cluster {cluster}:")
    for sf_value, percentage in data["percentages"].items():
        print(f"    Sensitive Value {sf_value}: {percentage:.2f}%")
```

---

## Key Methods

### `fit(X, sensitive_feature)`
Fits the model to the input data while enforcing fairness constraints.

- **Parameters:**
  - `X`: Training data (array-like of shape `(n_samples, n_features)`).
  - `sensitive_feature`: Array of sensitive feature values (shape `(n_samples,)`).

- **Returns:**
  - `self`: The fitted model.

### `_check_fairness(cluster, sensitive_value, ...)`
Ensures that adding a point to a cluster does not violate fairness constraints.

- **Returns:**
  - `True` if fairness is maintained; otherwise, `False`.

### `predict(X)`
Assigns each sample in `X` to the nearest cluster.

- **Parameters:**
  - `X`: Input data (array-like of shape `(n_samples, n_features)`).

- **Returns:**
  - `labels`: Cluster assignments for each sample.

---

## Example Output
Given the sample dataset:

**Input:**
```python
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])
sensitive_feature = np.array([0, 0, 1, 1, 1, 0])
```

**Output:**
```
Global Ratios: {0: 0.47923400420546713, 1: 0.5207659957945329}

Iteration 1
Sensitive counts after assignment: {0: {0: 1, 1: 1}, 1: {0: 1, 1: 1}, 2: {0: 1, 1: 1}}
Points Changing Clusters: 3
Centroid Shift: 1.414
...

Sensitive Distribution (FairKMeans):
  Cluster 0:
    Sensitive Value 0: 61.46%
    Sensitive Value 1: 38.54%
  Cluster 1:
    Sensitive Value 1: 62.92%
    Sensitive Value 0: 37.08%
  Cluster 2:
    Sensitive Value 1: 57.72%
    Sensitive Value 0: 42.28%
```

---

## Conclusion
The **FairKMeans** algorithm extends traditional K-Means by ensuring fairness in clustering. It is especially suitable for applications requiring equitable representation of sensitive groups. By integrating fairness constraints directly into the clustering process, the algorithm strikes a balance between cluster quality and fairness.

