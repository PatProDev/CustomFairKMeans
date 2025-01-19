import numpy as np
import pandas as pd

from customFairKMeans import FairKMeans
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from fairlearn.datasets import fetch_acs_income
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

n_clusters = 4

# Preprocess the given dataset
def process_dataset(X, sensitive_feature):
    """
    Encode all class values in X and sensitive_feature with numeric values and handle missing values.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Input data.
    sensitive_feature : array-like of shape (n_samples,)
        Sensitive feature values for each sample.

    Returns:
    X_encoded : array-like of shape (n_samples, n_features)
        Encoded input data.
    sensitive_feature_encoded : array-like of shape (n_samples,)
        Encoded sensitive feature.
    """
    # Convert to DataFrame for easier handling
    X_df = pd.DataFrame(X)

    # Drop columns with ALL NaN values
    X_df = X_df.loc[:, X_df.notna().any(axis=0)]

    # Fill NaN values in X: numeric columns with mean, categorical columns with "unknown"
    X_df = X_df.apply(lambda col: col.fillna(col.mean()) if col.dtypes != 'object' else col.fillna("unknown"))
    # Encode categorical features in X
    X_encoded = X_df.apply(lambda col: pd.factorize(col)[0] if col.dtypes == 'object' else col).values

    # Fill NaN values and encode sensitive feature
    sensitive_feature = pd.Series(sensitive_feature).fillna("unknown")
    sensitive_feature_encoded = pd.factorize(sensitive_feature)[0]

    return X_encoded, sensitive_feature_encoded

# Get cluster-wise distribution of sensitive features
def cluster_sensitive_distribution(labels, sensitive_feature):
    """
    Compute the distribution of sensitive features within each cluster.

    Parameters:
    labels : array-like of shape (n_samples,)
        Cluster labels for each sample.
    sensitive_feature : array-like of shape (n_samples,)
        Sensitive feature values for each sample.

    Returns:
    dict : A dictionary where keys are cluster IDs, and values are dictionaries
           showing the count and percentage of each sensitive feature value.
    """

    distribution = {}
    sensitive_feature = np.array(sensitive_feature)
    labels = np.array(labels)

    for cluster in np.unique(labels):
        cluster_indices = labels == cluster
        total_in_cluster = np.sum(cluster_indices)

        # Count sensitive features in the cluster
        counts = Counter(sensitive_feature[cluster_indices])

        # Convert counts to percentages
        percentages = {key: (value / total_in_cluster) * 100 for key, value in counts.items()}

        # Store both counts and percentages in the distribution
        distribution[cluster] = {
            "counts": counts,
            "percentages": percentages
        }

    return distribution

# Bar chart to visualize the percentage distribution of sensitive groups within each cluster compared to the global proportion
def plot_sensitive_distribution(distribution, title, filename):
    clusters = sorted(distribution.keys())
    sensitive_values = sorted(next(iter(distribution.values()))["percentages"].keys())
    values_matrix = np.zeros((len(clusters), len(sensitive_values)))

    for cluster in clusters:
        for i, sf_value in enumerate(sensitive_values):
            values_matrix[cluster, i] = distribution[cluster]["percentages"][sf_value]

    x = np.arange(len(clusters))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, sf_value in enumerate(sensitive_values):
        ax.bar(x + i * width / len(sensitive_values), values_matrix[:, i], width / len(sensitive_values),
                label=f"Sensitive Value {sf_value}")

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Percentage")
    ax.set_title(title)
    ax.set_xticks(x + width / (2 * len(sensitive_values)))
    ax.set_xticklabels([f"Cluster {c}" for c in clusters])
    ax.legend()
    plt.savefig(filename)
    plt.close()

# Line chart helper function for calculated fairness penalties for different numbers of clusters for both algorithms
def compute_fairness_penalty(distribution, global_ratios):
    penalty = 0
    for cluster_data in distribution.values():
        for sf_value, percentage in cluster_data["percentages"].items():
            penalty += abs(percentage / 100 - global_ratios[sf_value])
    return penalty / len(distribution)


if __name__ == "__main__":
    # Fetch the ACSIncome dataset
    data = fetch_acs_income()

    # Convert the dataset to a DataFrame for easier column access
    X = pd.DataFrame(data.data, columns=data.feature_names)
    sensitive_feature = X["SEX"].values     # Extract the sensitive feature as a NumPy array
    X = X.drop(columns=["SEX"]).values      # Drop the sensitive feature from the input data

    X, sensitive_feature = process_dataset(X, sensitive_feature)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, sf_train, sf_test = train_test_split(
        X, sensitive_feature, test_size=0.2, random_state=42
    )


    # Instantiate and fit the FairKMeans
    fair_kmeans = FairKMeans(n_clusters, random_state=42)
    fair_kmeans.fit(X_train, sf_train)

    # Predict and evaluate FairKMeans on test set
    fair_predictions = fair_kmeans.predict(X_test)
    fair_distribution = cluster_sensitive_distribution(fair_predictions, sf_test)


    # Instantiate and fit the regular KMeans
    regular_kmeans = KMeans(n_clusters, random_state=42)
    regular_kmeans.fit(X_train)

    # Predict and evaluate KMeans on test set
    regular_predictions = regular_kmeans.predict(X_test)
    regular_distribution = cluster_sensitive_distribution(regular_predictions, sf_test)

    print("Sensitive Distribution (FairKMeans):")
    for cluster, data in fair_distribution.items():
        print(f"  Cluster {cluster}:")
        for sf_value, percentage in data["percentages"].items():
            print(f"    Sensitive Value {sf_value}: {percentage:.2f}%")

    print("\nSensitive Distribution (Regular KMeans):")
    for cluster, data in regular_distribution.items():
        print(f"  Cluster {cluster}:")
        for sf_value, percentage in data["percentages"].items():
            print(f"    Sensitive Value {sf_value}: {percentage:.2f}%")

    # 1. Scatter Plot (Plotted test points colored by cluster assignment for both algorithms)
    # pca = PCA(n_components=2)
    # X_test_2D = pca.fit_transform(X_test)

    # plt.figure(figsize=(10, 6))
    # plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=fair_predictions, cmap="viridis", alpha=0.7, edgecolor="k")
    # plt.title("FairKMeans Clustering: Distribution Across Reduced Dimensions")
    # plt.xlabel("PCA Reduced Feature 1")
    # plt.ylabel("PCA Reduced Feature 2")
    # plt.colorbar(label="Assigned Cluster")
    # plt.savefig(f'graphs/fair_kmeans_scatter_{n_clusters}.png')
    # plt.close()

    # plt.figure(figsize=(10, 6))
    # plt.scatter(X_test_2D[:, 0], X_test_2D[:, 1], c=regular_predictions, cmap="viridis", alpha=0.7, edgecolor="k")
    # plt.title("Regular KMeans Clustering: Distribution Across Reduced Dimensions")
    # plt.xlabel("PCA Reduced Feature 1")
    # plt.ylabel("PCA Reduced Feature 2")
    # plt.colorbar(label="Assigned Cluster")
    # plt.savefig(f'graphs/regular_kmeans_scatter_{n_clusters}.png')
    # plt.close()

    # 2. Bar Chart (Sensitive Distribution)
    plot_sensitive_distribution(fair_distribution, "FairKMeans Sensitive Group Distribution", f'graphs/fair_kmeans_bar_{n_clusters}.png')
    plot_sensitive_distribution(regular_distribution, "Regular KMeans Sensitive Group Distribution", f'graphs/regular_kmeans_bar_{n_clusters}.png')

    # # 3. Line Chart (Fairness Penalty)
    # global_ratios = {value: np.mean(sf_test == value) for value in np.unique(sf_test)}
    # n_clusters_range = range(2, 10)
    # fair_penalties = []
    # regular_penalties = []

    # for n_clusters in n_clusters_range:
    #     fair_kmeans = FairKMeans(n_clusters, random_state=42)
    #     fair_kmeans.fit(X_train, sf_train)
    #     fair_predictions = fair_kmeans.predict(X_test)
    #     fair_distribution = cluster_sensitive_distribution(fair_predictions, sf_test)
    #     fair_penalties.append(compute_fairness_penalty(fair_distribution, global_ratios))

    #     regular_kmeans = KMeans(n_clusters, random_state=42)
    #     regular_kmeans.fit(X_train)
    #     regular_predictions = regular_kmeans.predict(X_test)
    #     regular_distribution = cluster_sensitive_distribution(regular_predictions, sf_test)
    #     regular_penalties.append(compute_fairness_penalty(regular_distribution, global_ratios))

    # plt.figure(figsize=(10, 6))
    # plt.plot(n_clusters_range, fair_penalties, label="FairKMeans", marker="o")
    # plt.plot(n_clusters_range, regular_penalties, label="Regular KMeans", marker="o")
    # plt.title("Fairness Penalty vs Number of Clusters")
    # plt.xlabel("Number of Clusters")
    # plt.ylabel("Fairness Penalty")
    # plt.legend()
    # plt.savefig("fairness_penalty_line_chart.png")
    # plt.close()