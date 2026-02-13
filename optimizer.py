import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def project_to_fair_territory(point):
    """Projects (x,y) to Fair Territory: y >= |x| and y >= 0."""
    x, y = point
    if y >= abs(x) and y >= 0: return point
    if y < 0: return np.array([0, 0])
    if x > 0: 
        val = (x + y) / 2
        return np.array([val, val])
    else: 
        new_x = (x - y) / 2
        new_y = (y - x) / 2
        return np.array([new_x, new_y])

def constrained_kmeans(data, fixed_centroids, n_variable, max_iter=100):
    """K-Means with fixed battery points and fair territory constraints."""
    kmeans_init = KMeans(n_clusters=n_variable, n_init=10, random_state=42)
    kmeans_init.fit(data)
    variable_centroids = kmeans_init.cluster_centers_
    fixed_centroids = np.array(fixed_centroids)
    
    for _ in range(max_iter):
        all_centroids = np.vstack([fixed_centroids, variable_centroids]) if len(fixed_centroids) > 0 else variable_centroids
        distances = np.linalg.norm(data[:, np.newaxis] - all_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_variable_centroids = []
        for k in range(n_variable):
            cluster_idx = len(fixed_centroids) + k
            points_in_cluster = data[labels == cluster_idx]
            proposed = points_in_cluster.mean(axis=0) if len(points_in_cluster) > 0 else variable_centroids[k]
            new_variable_centroids.append(project_to_fair_territory(proposed))
            
        new_variable_centroids = np.array(new_variable_centroids)
        if np.allclose(variable_centroids, new_variable_centroids, atol=1e-4): break
        variable_centroids = new_variable_centroids

    final_centroids = np.vstack([fixed_centroids, variable_centroids]) if len(fixed_centroids) > 0 else variable_centroids
    
    # Final Calculation for labels and residual
    distances = np.linalg.norm(data[:, np.newaxis] - final_centroids, axis=2)
    final_labels = np.argmin(distances, axis=1)
    total_residual = np.sum(np.min(distances, axis=1))
    
    # RETURNS 3 VALUES
    return final_centroids, final_labels, total_residual

def transform_coordinates(df):
    """Converts Statcast hc_x/y to field feet."""
    df = df.copy()
    df['x'] = (df['hc_x'] - 125) * 2.43
    df['y'] = (204.5 - df['hc_y']) * 2.43
    df['dist'] = np.sqrt(df['x']**2 + df['y']**2)
    return df