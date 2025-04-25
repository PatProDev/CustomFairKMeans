import pandas as pd
from sklearn.model_selection import train_test_split
from fairlearn.datasets import fetch_acs_income
from sklearn.cluster import KMeans
import numpy as np
from strategies.sensitiveDivisionKMeans import SensitiveDivisionKMeans
from collections import Counter

n_clusters = 4  
sensitive_feature_name = "SEX"

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

if __name__ == "__main__":
    # Fetch the ACSIncome dataset
    data = fetch_acs_income()
    # Convert the dataset to a DataFrame for easier column access
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Ensure 'SEX' column is present.
    if sensitive_feature_name in X.columns:
        sensitive_feature = X[sensitive_feature_name].values    # Extract the sensitive feature as a NumPy array
        X = X.drop(columns=[sensitive_feature_name]).values     # Drop the sensitive feature from the input data
    else:
        raise ValueError("The 'SEX' column is not found in the dataset.")

    X, sensitive_feature = process_dataset(X, sensitive_feature)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, sf_train, sf_test = train_test_split(
        X, sensitive_feature, test_size=0.2, random_state=42
    )

    # Instantiate and fit the SensitiveDivisionKMeans model
    sensitive_divison_kmeans = SensitiveDivisionKMeans(n_clusters, random_state=42)
    sensitive_divison_kmeans.fit(X_train, sf_train)

    # Predict and evaluate FairKMeans on test set
    fair_predictions = sensitive_divison_kmeans.predict(X_test)
    fair_distribution = cluster_sensitive_distribution(fair_predictions, sf_test)


    # Instantiate and fit the regular KMeans
    regular_kmeans = KMeans(n_clusters, n_init='auto', random_state=42)
    regular_kmeans.fit(X_train)

    # Predict and evaluate KMeans on test set
    regular_predictions = regular_kmeans.predict(X_test)
    regular_distribution = cluster_sensitive_distribution(regular_predictions, sf_test)

    print("Sensitive Distribution (Sensitive Division KMeans):")
    for cluster, data in fair_distribution.items():
        print(f"  Cluster {cluster}:")
        for sf_value, percentage in data["percentages"].items():
            print(f"    Sensitive Value {sf_value}: {percentage:.2f}%")

    print("\nSensitive Distribution (Regular KMeans):")
    for cluster, data in regular_distribution.items():
        print(f"  Cluster {cluster}:")
        for sf_value, percentage in data["percentages"].items():
            print(f"    Sensitive Value {sf_value}: {percentage:.2f}%")