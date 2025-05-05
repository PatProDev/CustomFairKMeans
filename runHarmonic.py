import pandas as pd
from sklearn.model_selection import train_test_split
from fairlearn.datasets import fetch_acs_income
from sklearn.cluster import KMeans
import numpy as np
from strategies.harmonicKMeans import HarmonicKMeans
from collections import Counter

n_clusters = 3  
sensitive_feature_name = "SEX"  # Optional values:
                                # COW - Class of Worker
                                # SCHL - School Enrollment
                                # MAR - Marital Status                                
                                # SEX - Gender
                                # RAC1P - Race      

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

def print_cluster_sensitive_distribution(distribution, clustring_method="Regular KMeans"):
    """
    Pretty-print the sensitive feature distribution as a table.
    """
    # Get all unique sensitive feature values across all clusters
    all_sensitive_values = set()
    for data in distribution.values():
        all_sensitive_values.update(data["percentages"].keys())
    all_sensitive_values = sorted(all_sensitive_values)

    # Create a DataFrame with clusters as rows and sensitive feature values as columns
    table_data = []
    for cluster in sorted(distribution.keys()):
        row = {"Cluster": cluster}
        for val in all_sensitive_values:
            row[val] = f"{distribution[cluster]['percentages'].get(val, 0):.2f}%"
        table_data.append(row)

    df = pd.DataFrame(table_data)
    df.set_index("Cluster", inplace=True)
    print(f'\nFeature Distribution ({clustring_method}):')
    print(df.to_string())

if __name__ == "__main__":
    # Fetch the ACSIncome dataset
    data = fetch_acs_income()
    # Convert the dataset to a DataFrame for easier column access
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Ensure 'sensitive_feature_name' column is present
    if sensitive_feature_name in X.columns:
        sensitive_feature = X[sensitive_feature_name].values  # Extract the sensitive feature as a NumPy array
        X = X.drop(columns=[sensitive_feature_name]).values  # Drop the sensitive feature from the input data
    else:
        raise ValueError(f"The '{sensitive_feature_name}' column is not found in the dataset.")

    X, sensitive_feature = process_dataset(X, sensitive_feature)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, sf_train, sf_test = train_test_split(
        X, sensitive_feature, test_size=0.2, random_state=42
    )

    # Instantiate and fit the HarmonicKMeans model
    harmonic_kmeans = HarmonicKMeans(n_clusters, random_state=42)
    harmonic_kmeans.fit(X_train, sf_train)

    # Predict and evaluate FairKMeans on test set
    fair_predictions = harmonic_kmeans.predict(X_test)
    fair_distribution = cluster_sensitive_distribution(fair_predictions, sf_test)


    # Instantiate and fit the regular KMeans
    regular_kmeans = KMeans(n_clusters, n_init='auto', random_state=42)
    regular_kmeans.fit(X_train)

    # Predict and evaluate KMeans on test set
    regular_predictions = regular_kmeans.predict(X_test)
    regular_distribution = cluster_sensitive_distribution(regular_predictions, sf_test)

    print_cluster_sensitive_distribution(fair_distribution, "Harmonic KMeans")
    print_cluster_sensitive_distribution(regular_distribution, "Regular KMeans")