# FairKMeans Algorithm

## Overview
The `FairKMeans` algorithm is a modification of the classic K-Means clustering algorithm. It aims to cluster data points while ensuring fairness with respect to a given sensitive feature (e.g., gender, race). Fairness is enforced by constraining the sensitive group ratios in each cluster to be close to their global proportions.

This approach is useful in applications where it's important to ensure that sensitive groups are fairly represented within each cluster.

---

## Key Features
1. **Fairness Constraint**: Ensures that the ratio of sensitive groups in each cluster adheres to a global ratio within a specified tolerance.
2. **Dynamic Tolerance**: Starts with a relaxed fairness constraint and tightens it as the algorithm progresses, facilitating convergence.
3. **Ensured Assignment**: Implements a fallback mechanism to guarantee cluster assignment.
4. **Flexibility**: Easy integration with scikit-learn tools.

---

## How It Works
The `FairKMeans` algorithm modifies the traditional K-Means clustering process with fairness-aware adjustments during the assignment step. The workflow is as follows:

1. **Initialization**:
   - Randomly select `n_clusters` data points as the initial centroids.
   - Compute the global ratios of each value in the sensitive feature.

2. **Cluster Assignment**:
   - Assign data points to clusters based on distance, subject to fairness constraints.
   - The fairness constraint ensures that the ratio of sensitive groups in each cluster stays within a dynamic tolerance of the global ratio.

3. **Centroid Update**:
   - Update cluster centroids based on the mean of assigned points.
   - Reassign points to clusters to ensure fairness after centroid updates.

4. **Convergence Check**:
   - Stop when the change in centroids falls below a specified tolerance or after a maximum number of iterations.

---

## Code Structure
### 1. Initialization
```python
def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=42):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.random_state = random_state
    self.fairness_tolerance = 0.07
```

**Input Parameters:**
  - `n_clusters`: Number of clusters to form.
  - `max_iter`: Maximum number of iterations to perform.
  - `tol`: Convergence threshold for centroid movement.
  - `random_state`: Random seed for reproducibility.
  - `fairness_tolerance`: Maximum allowable deviation from global ratios for sensitive groups.

### 2. Fitting the Model
```python
def fit(self, X, sensitive_feature):
    X = check_array(X)
    sensitive_feature = np.array(sensitive_feature)

    # Compute global ratios of the sensitive feature
    global_ratios = {
        value: np.sum(sensitive_feature == value) / len(sensitive_feature)
        for value in np.unique(sensitive_feature)
    }

    # Initialize centroids
    rng = np.random.RandomState(self.random_state)
    indices = rng.choice(len(X), self.n_clusters, replace=False)
    self.cluster_centers_ = X[indices]

    # Cluster Assignment Initialization
    labels = np.zeros(X.shape[0], dtype=int)
    prev_labels = labels.copy()
    sensitive_counts = {i: {val: 0 for val in np.unique(sensitive_feature)} for i in range(self.n_clusters)}
```

**Global Ratios:**
  - Compute the proportion of each sensitive group in the entire dataset.

**Centroid Initialization:**
  - Randomly select `n_clusters` points from the data as initial centroids.

**Cluster Assignment Initialization:**
  - `labels` stores the cluster assignment for each data point
  - `prev_labels` stores the previous cluster assignment for each data point (for debugging purposes)
  - `sensitive_counts` stores the counts of each sensitive feature value in each cluster

### 3. Cluster Assignment with Fairness
1. **Point assignment:**
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

2. **Fairness check:**
   Fairness is validated by comparing the current sensitive group ratio in a cluster to the global ratio:
    ```python
    def _check_fairness(self, cluster, sensitive_value, sensitive_counts, total_in_cluster, global_ratios, iteration):
        current_ratio = (sensitive_counts[cluster][sensitive_value] + 1) / (total_in_cluster + 1)
        target_ratio = global_ratios[sensitive_value]
        fairness_penalty = abs(current_ratio - target_ratio)
        dynamic_tolerance = self.fairness_tolerance / (iteration + 1) ** 0.5
        return fairness_penalty <= dynamic_tolerance
    ```

3. **Centroid update:**
   After assignment, cluster centroids are updated based on the mean of assigned points:
   ```python
   new_centroids = np.array([
       X[labels == i].mean(axis=0) if np.any(labels == i) else self.cluster_centers_[i]
       for i in range(self.n_clusters)
   ])
   ```

4. **Reassigning points after centroid update:**
    After centroids update, reassign points to clusters to enforce fairness under new conditions (new centroids): 
    ```python
    for i, sample in enumerate(X):
    distances = np.linalg.norm(new_centroids - sample, axis=1)
    sorted_clusters = np.argsort(distances)
    assigned = False
    for cluster in sorted_clusters:
        total_in_cluster = sum(sensitive_counts[cluster].values())
        if self._check_fairness(cluster, sensitive_feature[i], sensitive_counts, total_in_cluster, global_ratios, iteration):
            labels[i] = cluster
            sensitive_counts[cluster][sensitive_feature[i]] += 1
            assigned = True
            break
    ```

5. **Fallback assignment:**
    If no cluster satisfies the fairness constraint, the data point is assigned to the nearest cluster:
    ```python
    if not assigned:
        cluster = sorted_clusters[0]
        labels[i] = cluster
        sensitive_counts[cluster][sensitive_feature[i]] += 1
        print(f"Forced assignment to Cluster {cluster}")
    ```

6. **Convergence check:**
   If the change in centroids falls below the tolerance, the algorithm stops:
   ```python
   centroid_shift = np.linalg.norm(self.cluster_centers_ - new_centroids)
   if centroid_shift < self.tol:
       break
   ```

### 4. Prediction
  ```python
  def predict(self, X):
      """Predict the closest cluster each sample in X belongs to."""
      check_is_fitted(self, 'cluster_centers_')
      X = check_array(X)
      distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
      # Assigns each point to the nearest centroid
      return np.argmin(distances, axis=1)  
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

## Example Usage
### Dataset
We use the `fetch_acs_income` dataset from `fairlearn.datasets`. This dataset contains demographic and income information, including a sensitive feature such as `race` or `sex`.

### Code Example
```python
import numpy as np
from customFairKMeans import FairKMeans
from fairlearn.datasets import fetch_acs_income

# Fetch the ACSIncome dataset
data = fetch_acs_income()

# Convert the dataset to a DataFrame for easier column access
X = pd.DataFrame(data.data, columns=data.feature_names)
sensitive_feature = X["SEX"].values     # Extract the sensitive feature as a NumPy array
X = X.drop(columns=["SEX"]).values      # Drop the sensitive feature from the input data

# Preprocess the given dataset (Encode all class values in X and sensitive_feature with numeric values and handle missing values - included in main.py)
X, sensitive_feature = process_dataset(X, sensitive_feature)

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

### Example Output
```
Global Ratios: {0: 0.47923400420546713, 1: 0.5207659957945329}

Iteration 1
Sensitive counts after assignment: {0: {0: 10, 1: 6}, 1: {0: 12, 1: 9}, 2: {0: 8, 1: 5}}
Points Changing Clusters: 15
Centroid Shift: 1.243

Iteration 2
Sensitive counts after assignment: {0: {0: 11, 1: 7}, 1: {0: 13, 1: 9}, 2: {0: 6, 1: 4}}
Points Changing Clusters: 5
Centroid Shift: 0.542

...

Convergence reached.

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

## Results
The `FairKMeans` algorithm demonstrates a critical balance between clustering fairness and quality. By enforcing fairness constraints during the clustering process, the algorithm ensures that sensitive groups are equitably represented across clusters. However, this fairness comes with trade-offs:  

1. Fairness vs. Quality:  
- As the fairness constraints tighten, the cluster centroids may no longer perfectly reflect the natural groupings in the data, potentially impacting the quality of the clusters.  
- The algorithm ensures that sensitive groups are proportionately represented, but this can result in slightly higher within-cluster variance compared to traditional K-Means.  

2. Impact of the number of clusters:
- Increasing the number of clusters often makes it harder to maintain fairness. With more clusters, the ratios of sensitive groups within each cluster tend to deviate further from the global ratios.  
- This phenomenon arises because the available data points are distributed across more clusters, reducing the flexibility to balance sensitive group proportions in every cluster.  

3. Dynamic tolerance:
- The use of a dynamic tolerance helps manage the trade-off by gradually tightening fairness constraints. Early iterations focus on forming clusters, while later iterations emphasize fairness adjustments.  

By carefully tuning the number of clusters and fairness tolerance, the FairKMeans algorithm allows users to achieve a desirable balance between fairness and clustering quality, tailored to specific application needs.

