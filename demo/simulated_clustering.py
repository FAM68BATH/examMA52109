###
## simulated_clustering.py
## Analysis of simulated_data.csv using cluster_maker
## Student: Fawaz Ahmed Mohideen
## Date: December 2025
###

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import essential functions from cluster_maker
from cluster_maker import (
    run_clustering,
    elbow_curve,
    plot_elbow,
    plot_clusters_2d,
    calculate_descriptive_statistics,
    standardise_features
)


def main():
    """Main clustering analysis for simulated_data.csv"""
    
    # Create output directory
    os.makedirs("demo_output", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("CLUSTERING ANALYSIS: simulated_data.csv")
    print("=" * 60)
    
    # 1. Load and explore data
    print("\n1. LOADING AND EXPLORING DATA")
    print("-" * 40)
    
    df = pd.read_csv("data/simulated_data.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    
    # Quick stats using cluster_maker
    print("\nBasic statistics:")
    stats = calculate_descriptive_statistics(df)
    print(stats.round(2))
    
    # 2. Determine plausible number of clusters
    print("\n\n2. DETERMINING PLAUSIBLE NUMBER OF CLUSTERS")
    print("-" * 40)
    
    # Standardize data
    X = df.values.astype(float)
    X_scaled = standardise_features(X)
    
    # Elbow method
    print("\nUsing elbow method to find optimal k...")
    k_values = list(range(1, 9))
    inertias = elbow_curve(X_scaled, k_values, random_state=42, use_sklearn=True)
    
    # Plot elbow curve
    fig_elbow, ax_elbow = plot_elbow(
        k_values, 
        [inertias[k] for k in k_values],
        title="Elbow Method: Inertia vs Number of Clusters"
    )
    
    # Identify potential elbow points (where improvement slows)
    improvements = []
    for i in range(1, len(k_values)-1):
        improvement = (inertias[k_values[i]] - inertias[k_values[i+1]]) / inertias[k_values[i]]
        improvements.append(improvement * 100)
    
    # Suggest plausible k values based on elbow
    plausible_k_values = []
    if improvements:
        avg_improvement = np.mean(improvements)
        for i, imp in enumerate(improvements):
            if imp > avg_improvement:
                plausible_k_values.append(k_values[i+1])
    
    # If no clear elbow, use common defaults
    if not plausible_k_values:
        plausible_k_values = [3, 4, 5]
    
    print(f"\nPlausible k values based on elbow method: {plausible_k_values}")
    
    plt.savefig('demo_output/elbow_plot.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. Test plausible clusterings
    print("\n\n3. TESTING PLAUSIBLE CLUSTERINGS")
    print("-" * 40)
    
    results = {}
    
    for k in plausible_k_values[:3]:  # Test up to 3 most plausible
        print(f"\n--- Testing k = {k} ---")
        
        # Run clustering
        result = run_clustering(
            input_path="data/simulated_data.csv",
            feature_cols=list(df.columns),
            algorithm="sklearn_kmeans",
            k=k,
            standardise=True,
            output_path=f"demo_output/simulated_k{k}_clustered.csv",
            random_state=42,
            compute_elbow=False
        )
        
        results[k] = result
        
        # Display key metrics
        print(f"  Inertia: {result['metrics']['inertia']:.2f}")
        if result['metrics']['silhouette']:
            print(f"  Silhouette score: {result['metrics']['silhouette']:.3f}")
        
        # Cluster sizes
        labels = result['labels']
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique, counts))}")
        
        # Create visualization
        fig, ax = plot_clusters_2d(
            X_scaled[:, :2],  # First two features for 2D plot
            labels,
            centroids=result['centroids'][:, :2],
            title=f"K-means Clustering with k={k}"
        )
        plt.savefig(f'demo_output/simulated_k{k}_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: simulated_k{k}_clustered.csv")
        print(f"  Saved: simulated_k{k}_clusters.png")
    
    # 4. Make final recommendation
    print("\n\n4. FINAL RECOMMENDATION")
    print("-" * 40)
    
    # Choose best based on silhouette score if available
    best_k = None
    best_score = -1
    
    for k, result in results.items():
        score = result['metrics']['silhouette']
        if score is not None and score > best_score:
            best_score = score
            best_k = k
    
    if best_k is not None:
        print(f"\nRECOMMENDED: k = {best_k}")
        print(f"    Reason: Highest silhouette score ({best_score:.3f})")
        
        if best_score > 0.5:
            print(f"  Interpretation: Strong cluster structure")
        elif best_score > 0.25:
            print(f"  Interpretation: Moderate cluster structure")
        else:
            print(f"  Interpretation: Weak but discernible structure")
        
        # Show cluster characteristics
        print(f"\n  Cluster distribution for k={best_k}:")
        best_result = results[best_k]
        labels = best_result['labels']
        unique, counts = np.unique(labels, return_counts=True)
        
        for cluster_id, count in zip(unique, counts):
            percentage = 100 * count / len(labels)
            print(f"    Cluster {cluster_id}: {count} points ({percentage:.1f}%)")
    
    else:
        # Fallback to lowest inertia
        best_k = min(results.keys(), key=lambda k: results[k]['metrics']['inertia'])
        print(f"\nRECOMMENDED: k = {best_k}")
        print(f"    Reason: Lowest inertia ({results[best_k]['metrics']['inertia']:.2f})")
    
    # 5. Create comparison visualization
    print("\n\n5. COMPARISON VISUALIZATION")
    print("-" * 40)
    
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    
    if len(results) == 1:
        axes = [axes]
    
    for idx, (k, result) in enumerate(results.items()):
        scatter = axes[idx].scatter(X_scaled[:, 0], X_scaled[:, 1], 
                                   c=result['labels'], cmap='tab10', alpha=0.7, s=30)
        axes[idx].scatter(result['centroids'][:, 0], result['centroids'][:, 1],
                         marker='*', s=200, c='red', edgecolor='black', linewidth=1.5)
        
        title = f'k={k}'
        if result['metrics']['silhouette']:
            title += f'\nSilhouette: {result["metrics"]["silhouette"]:.3f}'
        
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].set_xlabel('Feature 1 (standardized)')
        axes[idx].set_ylabel('Feature 2 (standardized)')
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Comparison of Plausible Clusterings', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('demo_output/clustering_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nOutput files saved to 'demo_output/':")
    print("  - elbow_plot.png - Elbow method analysis")
    for k in results.keys():
        print(f"  - simulated_k{k}_clustered.csv - Data with cluster labels")
        print(f"  - simulated_k{k}_clusters.png - Cluster visualization")
    print("  - clustering_comparison.png - Side-by-side comparison")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())