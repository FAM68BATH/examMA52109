###
## cluster_maker - Agglomerative Clustering Module
## Student: Fawaz Ahmed Mohideen
## Date: December 2025
###

from __future__ import annotations

from typing import Tuple, Optional
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def agglomerative_clustering(
    X: np.ndarray,
    k: int = 2,  # Changed from n_clusters to match kmeans()
    linkage: str = "ward",
    affinity: str = "euclidean",
    distance_threshold: Optional[float] = None,
    random_state: Optional[int] = None,  # Added for API compatibility (ignored for agglomerative)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative hierarchical clustering.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    k : int, default 2
        Number of clusters to find.
    linkage : {"ward", "complete", "average", "single"}, default "ward"
        Linkage criterion.
    affinity : str or callable, default "euclidean"
        Metric used to compute the linkage.
    distance_threshold : float, default None
        The linkage distance threshold above which clusters will not be merged.
    random_state : int or None, default None
        Included for API compatibility with kmeans functions. Ignored for
        agglomerative clustering as it's deterministic.
    
    Returns
    -------
    labels : ndarray of shape (n_samples,)
        Cluster labels for each point.
    centroids : ndarray of shape (k, n_features)
        Mean of points in each cluster.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    
    # Validate linkage method
    valid_linkages = ["ward", "complete", "average", "single"]
    if linkage not in valid_linkages:
        raise ValueError(f"linkage must be one of {valid_linkages}, got '{linkage}'")
    
    # Ward linkage requires Euclidean distance
    if linkage == "ward" and affinity != "euclidean":
        raise ValueError("Ward linkage requires Euclidean distance (affinity='euclidean')")
    
    # Validate k vs distance_threshold
    if distance_threshold is not None:
        raise ValueError("distance_threshold is not supported when k is specified. "
                        "Use distance_threshold only when k is None.")
    
    # Create and fit the model
    model = AgglomerativeClustering(
        n_clusters=k,
        metric=affinity,
        linkage=linkage,
        distance_threshold=distance_threshold,
    )
    
    model.fit(X)
    labels = model.labels_
    
    # Compute centroids as mean of each cluster (for compatibility with kmeans functions)
    n_features = X.shape[1]
    centroids = np.zeros((k, n_features), dtype=float)
    
    for cluster_id in range(k):
        mask = labels == cluster_id
        if np.any(mask):
            centroids[cluster_id] = X[mask].mean(axis=0)
        else:
            centroids[cluster_id] = np.mean(X, axis=0)
    
    return labels, centroids