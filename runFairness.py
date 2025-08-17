# Run script: python runFairness.py <dataset> <strategy> <n_clusters> <sensitive_feature_name>
# Example usage: python runFairness.py Asc Sil 3 SEX
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from fairlearn.datasets import fetch_acs_income
from collections import Counter
from strategies.customFairKMeans import FairKMeans
from strategies.harmonicKMeans import HarmonicKMeans
from strategies.sensitiveDivisionKMeans import SensitiveDivisionKMeans
#from strategies.metricDrivenKMeans import MetricDrivenKMeans
from strategies.silhouetteDbiKMeans import SilhouetteDbiKMeans
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from datetime import datetime
import os
import openpyxl

# Default values for the parameters
strategy_name = "Sensitive Division KMeans"
kmeans_strategy = "Sen"         # Options: "Cus", "Har", "Sen", "Sil", "Dbi"
n_clusters = 5
dataset = "Asc"                 # Options: "Asc", "Sin"
sensitive_feature_name = "SEX"  # Options:
                                # COW - Class of Worker
                                # SCHL - School Enrollment
                                # MAR - Marital Status                                
                                # SEX - Gender
                                # RAC1P - Race                    

def print_usage():
    print("USAGE: python runFairness.py <dataset> <strategy> <num_of_clusters> <sensitive_feature_name>")
    print("Dataset options:")
    print("    Asc - ACS Income Dataset")
    print("    Sin - Synthetic Dataset")
    print("Strategy options:")
    print("    Cus - Custom KMeans")
    print("    Har - Harmonic KMeans")
    print("    Sen - Sensitive Division KMeans")
    print("    Sil - Silhouette Score KMeans")
    print("    Dbi - Davis-Bouldin Index KMeans")
    print("Sensitive feature options:")
    print("    COW - Class of Worker")
    print("    SCHL - School Enrollment")
    print("    MAR - Marital Status")
    print("    SEX - Gender")
    print("    RAC1P - Race")   

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

def get_global_ratios(sensitive_feature):
    """
    Calculate the global ratio of each sensitive feature value.

    Parameters:
    sensitive_feature : array-like of shape (n_samples,)
        Sensitive feature values for each sample.

    Returns:
    dict : A dictionary where keys are unique sensitive feature values,
           and values are their global ratios.
    """
    sensitive_feature = np.array(sensitive_feature)
    total_samples = len(sensitive_feature)
    
    return {
        value: np.sum(sensitive_feature == value) / total_samples
        for value in np.unique(sensitive_feature)
    }

def print_global_ratios(global_ratios, sensitive_feature_name):
    """
    Pretty-print global sensitive feature ratios as a single-row table.
    """
    # Format ratios as percentages with 2 decimals
    row = {key: f"{value * 100:.2f}%" for key, value in global_ratios.items()}
    
    # Pad or truncate sensitive_feature_name to 7 characters
    fixed_width_label = f"{sensitive_feature_name:<7}"

    # Wrap into DataFrame with one row
    df = pd.DataFrame([row], index=[fixed_width_label])
    
    print("\nGlobal Sensitive Feature Ratios:")
    print(df.to_string())

def run_strategy(kmeans_strategy):
    match kmeans_strategy:
        case "Cus":
            strategy_name = "Custom KMeans"
            # Instantiate and fit the FairKMeans
            fair_kmeans = FairKMeans(n_clusters, random_state=42)
            fair_kmeans.fit(X_train, sf_train, global_ratios)

            # Predict and evaluate FairKMeans on test set
            fair_predictions = fair_kmeans.predict(X_test)
            fair_distribution = calculate_cluster_sensitive_distribution(fair_predictions, sf_test)

        case "Har":
            strategy_name = "Harmonic KMeans"
            # Instantiate and fit the HarmonicKMeans model
            fair_kmeans = HarmonicKMeans(n_clusters, random_state=42)
            fair_kmeans.fit(X_train, sf_train)

            # Predict and evaluate HarmonicKMeans on test set
            fair_predictions = fair_kmeans.predict(X_test)
            fair_distribution = calculate_cluster_sensitive_distribution(fair_predictions, sf_test)

        case "Sen":
            strategy_name = "Sensitive Division KMeans"
            # Instantiate and fit the SensitiveDivisionKMeans model
            fair_kmeans = SensitiveDivisionKMeans(n_clusters, random_state=42)
            fair_kmeans.fit(X_train, sf_train)

            # Predict and evaluate SensitiveDivisionKMeans on test set
            fair_predictions = fair_kmeans.predict(X_test)
            fair_distribution = calculate_cluster_sensitive_distribution(fair_predictions, sf_test)
        
        case "Sil":
            strategy_name = "Silhouette Score KMeans"    
            # Instantiate and fit the SilhouetteDbiKMeans model
            # silhouette_kmeans = MetricDrivenKMeans(n_clusters, use_dbi=False, score_threshold=0.5, random_state=42)
            fair_kmeans = SilhouetteDbiKMeans(n_clusters, use_dbi=False, score_threshold=0.6, random_state=42)
            fair_kmeans.fit(X_train)

            # Predict and evaluate SilhouetteDbiKMeans on test set
            fair_predictions = fair_kmeans.predict(X_test)
            fair_distribution = calculate_cluster_sensitive_distribution(fair_predictions, sf_test)

        case "Dbi":
            strategy_name = "Davies-Bouldin Index KMeans"    
            # Instantiate and fit the SilhouetteDbiKMeans model
            #dbi_kmeans = MetricDrivenKMeans(n_clusters, use_dbi=True, score_threshold=0.5, random_state=42)
            fair_kmeans = SilhouetteDbiKMeans(n_clusters, use_dbi=True, score_threshold=0.5, random_state=42)
            fair_kmeans.fit(X_train)

            # Predict and evaluate SilhouetteDbiKMeans on test set
            fair_predictions = fair_kmeans.predict(X_test)
            fair_distribution = calculate_cluster_sensitive_distribution(fair_predictions, sf_test)   
    
    return fair_distribution, fair_predictions, strategy_name, fair_kmeans

# Get cluster-wise distribution of sensitive features
def calculate_cluster_sensitive_distribution(labels, sensitive_feature):
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

    # Compute span (max - min) for each column
    span_row = {}
    span_sum = 0
    for col in all_sensitive_values:
        values = df[col].str.rstrip('%').astype(float)
        span = values.max() - values.min()
        span_row[col] = f"{span:.2f}%"
        span_sum += span

    # Add a separator row of '=' characters
    separator_row = {col: '=' * len(span_row[col]) for col in all_sensitive_values}
    df.loc["======="] = separator_row

    # Append span row
    df.loc["Span"] = span_row

    # Compute and append average span (only one cell filled)
    avg_span = span_sum / len(all_sensitive_values) if all_sensitive_values else 0
    avg_row = {col: "" for col in all_sensitive_values}  # empty row
    avg_row[all_sensitive_values[0]] = f"{avg_span:.2f}%"  # put value in first column
    df.loc["Average Span"] = avg_row

    print(f'\nFeature Distribution ({clustring_method}):')
    print(df.to_string())
    return avg_span

# Silhouette and Davies-Bouldin evaluation
def evaluate_clustering(X, labels, name="Clustering", sample_size=10000, random_state=42, sample_dbi=False):
    """
    Fast evaluation of clustering quality.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    labels : ndarray, shape (n_samples,)
    name : str
        Label for print-out.
    sample_size : int
        Max points to use for Silhouette (and DBI if sample_dbi=True).
    sample_dbi : bool
        If True, compute DBI on the same sample; otherwise on all points.

    Notes
    -----
    Silhouette is expensive (O(N²)).  A random 10-20k sample provides a
    very good estimate for large datasets.
    """
    uniq = np.unique(labels)
    if len(uniq) < 2:
        print(f"{name}: Only one cluster present (not enough clusters); scores undefined.")
        return

    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)

    # -------- Silhouette on a sample -----------------------------------
    if n_samples > sample_size:
        idx = rng.choice(n_samples, sample_size, replace=False)
        sil = silhouette_score(X[idx], labels[idx])
    else:
        sil = silhouette_score(X, labels)

    # -------- DBI -------------------------------------------------------
    if sample_dbi and n_samples > sample_size:
        idx_dbi = rng.choice(n_samples, sample_size, replace=False)
        dbi = davies_bouldin_score(X[idx_dbi], labels[idx_dbi])
    else:
        dbi = davies_bouldin_score(X, labels)

    print(f"\n{name} Evaluation")
    print(f"  Silhouette score (sampled): {sil:.4f}")
    print(f"  Davies-Bouldin index      : {dbi:.4f}")
    return sil, dbi

def create_scatter_plot(X, labels, strategy_name, num_clusters, centroids=None, sensitive_feature=None, save_dir="scatter_plots"):
    """
    Creates and saves a scatter plot for one clustering result.

    Parameters:
    - X: array-like, shape (n_samples, n_features)
        Feature matrix.
    - labels: array-like, shape (n_samples,)
        Cluster labels for each point.
    - strategy_name: str
        Name of the algorithm/strategy (used for subdirectory and filename).
    - num_clusters: int
        Number of clusters (used in filename).
    - centroids: array-like, shape (n_clusters, n_features), optional
        Centroid positions to mark on the plot.
    - sensitive_feature: array-like, shape (n_samples,), optional
        Sensitive attribute values for marker shape differentiation.
    - save_dir: str
        Base directory to save plots in.
    """

    # Reduce to 2D for visualization
    if X.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        X_vis = pca.fit_transform(X)
        centroids_vis = pca.transform(centroids) if centroids is not None else None
    else:
        X_vis = X
        centroids_vis = centroids

    # Marker shapes for sensitive feature values
    marker_styles = ['o', 's', '^', 'D', 'P', 'X']  # extend if needed
    if sensitive_feature is not None:
        unique_sf = np.unique(sensitive_feature)
        sf_to_marker = {sf_val: marker_styles[i % len(marker_styles)] for i, sf_val in enumerate(unique_sf)}
    else:
        sf_to_marker = {None: 'o'}

    # Plot each sensitive feature group separately so markers differ
    plt.figure(figsize=(10, 6))
    for sf_val, marker in sf_to_marker.items():
        mask = (sensitive_feature == sf_val) if sensitive_feature is not None else np.ones(len(X_vis), dtype=bool)
        plt.scatter(
            X_vis[mask, 0],
            X_vis[mask, 1],
            c=labels[mask],
            cmap='tab10',
            marker=marker,
            s=20,
            edgecolor='k',
            linewidth=0.3,
            label=f"Sen. Value ={sf_val}" if sf_val is not None else None
        )

    # Plot centroids if provided
    if centroids_vis is not None:
        plt.scatter(
            centroids_vis[:, 0],
            centroids_vis[:, 1],
            marker='*',
            c=range(len(centroids_vis)),
            cmap='tab10',
            s=300,
            edgecolor='black',
            linewidth=1.2,
            label="Centroids"
        )

    plt.title(f"{strategy_name} Results")
    plt.xlabel("Clusters")
    plt.ylabel("Sensitive Feature")
    plt.grid(True)
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0,
        frameon=True
    )
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # leave space on the right for legend

    # Prepare directory path
    strategy_dir = os.path.join(save_dir, strategy_name)
    os.makedirs(strategy_dir, exist_ok=True)

    # Prepare file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sensitive_feature_name}_{num_clusters}_{dataset}_scatter_{timestamp}.png"
    filepath = os.path.join(strategy_dir, filename)

    # Save and close figure
    plt.savefig(filepath, dpi=300)
    plt.close()

    print(f"Scatter plot saved to: {filepath}")

def save_numeric_results(strategy_name, sensitive_feature_name, n_clusters, dataset, avg_span, sil, dbi, time_taken):
    # Prepare directory path
    results_dir = os.path.join("numeric_results", strategy_name)
    os.makedirs(results_dir, exist_ok=True)

    # File name
    filename = f"{sensitive_feature_name}_{dataset}_results.csv"
    filepath = os.path.join(results_dir, filename)

    # Round values
    avg_span = round(avg_span, 3)
    sil = round(sil, 4)
    dbi = round(dbi, 4)
    time_taken = round(time_taken, 2)

    # New result row
    new_row = {
        "Strategy": strategy_name,
        "Clusters": n_clusters, 
        "Average Span (%)": avg_span,
        "Silhouette Score": sil,
        "Davies-Bouldin Index": dbi,
        "Time Taken (s)": time_taken
    }

    # If file exists, append/update; otherwise create new
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        # Replace row if cluster count already exists, else append
        df = df[df["Clusters"] != n_clusters]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df = df.sort_values(by="Clusters")
    else:
        df = pd.DataFrame([new_row])

    # Save back to CSV
    df.to_csv(filepath, index=False)
    print(f"\nNumeric results saved to: {filepath}")


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print_usage()
        sys.exit(1)

    # Read command line arguments
    dataset = sys.argv[1]
    kmeans_strategy = sys.argv[2]
    n_clusters = int(sys.argv[3])
    sensitive_feature_name = sys.argv[4]

    if dataset == "Asc":
        data = fetch_acs_income()
        df = pd.DataFrame(data.data, columns=data.feature_names)
    elif dataset == "Sin":
        df = pd.read_csv("synthetic_data/synthetic_equal_odds(in).csv")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Ensure 'sensitive_feature_name' column is present
    if sensitive_feature_name not in df.columns:
        raise ValueError(f"The '{sensitive_feature_name}' column is not found in the dataset.")

    # Extract sensitive feature and remove it from features
    sensitive_feature = df[sensitive_feature_name].values
    X = df.drop(columns=[sensitive_feature_name]).values

    X, sensitive_feature = process_dataset(X, sensitive_feature)

    # Get the global ratio of each sensitive feature value
    global_ratios = get_global_ratios(sensitive_feature)

    # Split the data into training and testing sets (80-20 split)
    X_train, X_test, sf_train, sf_test = train_test_split(
        X, sensitive_feature, test_size=0.2, random_state=42
    )

    # Instantiate and fit the regular KMeans
    regular_kmeans = KMeans(n_clusters, n_init='auto', random_state=42)
    start_time = time.time()
    regular_kmeans.fit(X_train)
    # Predict and evaluate KMeans on test set
    regular_predictions = regular_kmeans.predict(X_test)
    time_taken2 = time.time() - start_time
    regular_distribution = calculate_cluster_sensitive_distribution(regular_predictions, sf_test)

    start_time = time.time()                    
    fair_distribution, fair_predictions, strategy_name, fair_kmeans_model = run_strategy(kmeans_strategy)
    time_taken1 = time.time() - start_time
    print("Time taken: %s seconds" % (time_taken1))

    print_global_ratios(global_ratios, sensitive_feature_name)
    
    avg_span1 = print_cluster_sensitive_distribution(fair_distribution, strategy_name)
    avg_span2 = print_cluster_sensitive_distribution(regular_distribution, "Regular KMeans")

    sil1, dbi1 = evaluate_clustering(X_test, fair_predictions, strategy_name)
    sil2, dbi2 = evaluate_clustering(X_test, regular_predictions, "Regular KMeans")

    #create_scatter_plot(X_test, fair_predictions, strategy_name, n_clusters, centroids=fair_kmeans_model.cluster_centers_, sensitive_feature=sf_test)
    #create_scatter_plot(X_test, regular_predictions, "Regular KMeans", n_clusters, centroids=regular_kmeans.cluster_centers_, sensitive_feature=sf_test)

    save_numeric_results(strategy_name, sensitive_feature_name, n_clusters, dataset, avg_span1, sil1, dbi1, time_taken1)
    save_numeric_results("Regular KMeans", sensitive_feature_name, n_clusters, dataset, avg_span2, sil2, dbi2, time_taken2)