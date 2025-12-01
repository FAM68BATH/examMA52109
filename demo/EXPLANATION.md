# Task 2: Analysis and Fix of cluster_plot.py

## 1. What Was Wrong with the Original Script

The original `cluster_plot.py` script contained a critical bug on line 52. The script was designed to demonstrate k-means clustering for different values of k (2, 3, 4, 5), but it had an erroneous parameter assignment:

```python
k = min(k, 3),
```

This line was inside the call to `run_clustering()`, and it was incorrectly capping the maximum value of k to 3. As a result:

- When the loop variable `k = 4`, the function received `min(4, 3) = 3`
- When the loop variable `k = 5`, the function received `min(5, 3) = 3`

This created a misleading situation where:
- Output files named `demo_data_k4.png` and `demo_data_clustered_k4.csv` actually contained results for k=3 clustering
- Output files named `demo_data_k5.png` and `demo_data_clustered_k5.csv` also contained results for k=3 clustering
- The metrics summary would show incorrect comparisons between different k values

The bug was particularly subtle because:
1. The script appeared to run without errors
2. It generated all expected output files
3. The console output correctly showed "Running k-means with k = 4" and "Running k-means with k = 5"
4. Only by examining the actual content of the output files could one discover that k=4 and k=5 were actually k=3

## 2. How I Fixed It

The fix was simple and surgical. I changed line 52 from:

```python
k = min(k, 3),
```

To:

```python
k = k,
```

This ensures that the actual value of k from the loop (2, 3, 4, or 5) is passed correctly to the `run_clustering()` function without any artificial limitation.

**Verification of the fix:**
After applying the fix, I ran the script and verified that:
- `demo_data_clustered_k4.csv` now contains 4 unique cluster labels (0, 1, 2, 3)
- `demo_data_clustered_k5.csv` now contains 5 unique cluster labels (0, 1, 2, 3, 4)
- The cluster plots for k=4 and k=5 show the correct number of clusters
- The metrics file shows distinct inertia values for each k value

## 3. What the Corrected Demo Script Does

The corrected `cluster_plot.py` script now performs as intended:

1. **Data Loading**: Reads the input CSV file (`demo_data.csv`) and identifies the first two numeric columns to use as features.

2. **Clustering Loop**: Iterates through k values 2, 3, 4, and 5, running k-means clustering for each.

3. **Output Generation**:
   - **Labeled Data**: Saves CSV files with an added "cluster" column (`demo_data_clustered_k2.csv`, `demo_data_clustered_k3.csv`, `demo_data_clustered_k4.csv`, `demo_data_clustered_k5.csv`)
   - **Visualizations**: Creates 2D scatter plots showing data points colored by cluster assignment, with centroids marked (`demo_data_k2.png`, `demo_data_k3.png`, `demo_data_k4.png`, `demo_data_k5.png`)
   - **Metrics**: Computes and saves clustering evaluation metrics (inertia and silhouette score) to `demo_data_metrics.csv`
   - **Comparison Plot**: Generates a bar chart comparing silhouette scores across different k values (`demo_data_silhouette.png`)

4. **Directory Management**: All outputs are organized in the `demo_output/` directory for easy access.

The script provides a complete demonstration of how to use the `cluster_maker` package for clustering analysis, including parameter exploration and result visualization. It showcases:
- The full clustering workflow from data loading to visualization
- How to evaluate clustering quality using multiple metrics
- The effect of different k values on clustering results
- Proper organization of analysis outputs

## 4. Overview of the `cluster_maker` Package

`cluster_maker` is an educational Python package designed for teaching clustering concepts through practical implementation. The package follows a modular architecture with clear separation of concerns:

### **Data Handling Modules**

- **`dataframe_builder.py`**: Creates synthetic clustered datasets for testing and demonstration. Key functions:
  - `define_dataframe_structure()`: Defines cluster centers
  - `simulate_data()`: Generates data points around specified centers

- **`data_analyser.py`**: Provides basic data analysis utilities:
  - `calculate_descriptive_statistics()`: Summary statistics for numeric columns
  - `calculate_correlation()`: Correlation matrix computation

- **`data_exporter.py`**: Handles data export:
  - `export_to_csv()`: Standard CSV export
  - `export_formatted()`: Formatted text table export

### **Preprocessing Module**

- **`preprocessing.py`**: Data preparation functions:
  - `select_features()`: Validates and extracts numeric feature columns with comprehensive error checking
  - `standardise_features()`: Standardizes data to zero mean and unit variance using scikit-learn's StandardScaler

### **Clustering Algorithms**

- **`algorithms.py`**: Core clustering implementations:
  - `kmeans()`: Manual k-means implementation for educational understanding
  - `sklearn_kmeans()`: Production-ready wrapper around scikit-learn's KMeans
  - Helper functions: `init_centroids()`, `assign_clusters()`, `update_centroids()` that demonstrate the algorithmic steps

### **Evaluation Module**

- **`evaluation.py`**: Clustering quality assessment:
  - `compute_inertia()`: Computes within-cluster sum of squared distances
  - `silhouette_score_sklearn()`: Calculates silhouette coefficient using scikit-learn
  - `elbow_curve()`: Generates inertia values across multiple k values for elbow method analysis

### **Visualization Module**

- **`plotting_clustered.py`**: Data visualization:
  - `plot_clusters_2d()`: Creates 2D scatter plots of clustered data with centroids marked
  - `plot_elbow()`: Generates elbow curves for determining optimal k

### **High-level Interface**

- **`interface.py`**: The `run_clustering()` function provides a unified interface that:
  1. Loads and preprocesses data
  2. Executes the chosen clustering algorithm
  3. Computes evaluation metrics
  4. Generates visualizations
  5. Exports results

This function encapsulates the complete clustering workflow, making it easy for users to perform end-to-end analysis with minimal code.


### **Typical Workflow**

A typical clustering analysis using `cluster_maker` follows these steps:
1. Generate or load data using the data handling modules
2. Preprocess data using `select_features()` and `standardise_features()`
3. Run clustering using either the manual `kmeans()` or `sklearn_kmeans()`
4. Evaluate results using `compute_inertia()` and `silhouette_score_sklearn()`
5. Visualize clusters with `plot_clusters_2d()` and elbow curves with `plot_elbow()`
6. Export results using `export_to_csv()`

For quick analysis, users can use the high-level `run_clustering()` function that handles all these steps automatically.

The package is particularly valuable in educational settings where students need to understand both the theoretical concepts behind clustering algorithms and their practical implementation. By providing clear, well-documented code alongside production-ready wrappers, `cluster_maker` bridges the gap between learning and application.
```
