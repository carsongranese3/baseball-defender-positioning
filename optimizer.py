import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def project_to_fair_territory(point):
    """
    Ensures optimal fielder coordinates stay within the foul lines.
    Mathematically projects (x,y) to the region: y >= |x| and y >= 0.
    """
    x, y = point
    # Already in fair territory
    if y >= abs(x) and y >= 0:
        return point
    
    # Behind home plate
    if y < 0:
        return np.array([0, 0])
    
    # Foul territory (Right Field side)
    if x > 0: 
        val = (x + y) / 2
        return np.array([val, val])
    
    # Foul territory (Left Field side)
    else: 
        new_x = (x - y) / 2
        new_y = (y - x) / 2
        return np.array([new_x, new_y])

def weighted_constrained_kmeans(data, weights, fixed_centroids, n_variable, max_iter=100):
    """
    K-Means algorithm that accounts for hit relevance (weights).
    
    - data: (N, 2) array of hit coordinates.
    - weights: (N,) array of importance scores (frequency * velocity similarity).
    - fixed_centroids: List of points that do not move (e.g., [[0, 60.5], [0, -2]]).
    - n_variable: Number of fielders to optimize (e.g., 4 for IF, 3 for OF).
    """
    
    # 1. Handle Empty Data Case
    if len(data) == 0:
        # Return default positions if no data points exist
        return np.vstack([fixed_centroids, np.zeros((n_variable, 2))])

    # 2. Initialize Variable Centroids
    kmeans_init = KMeans(n_clusters=n_variable, n_init=10, random_state=42)
    kmeans_init.fit(data)
    variable_centroids = kmeans_init.cluster_centers_
    
    # 3. Shape Handling for Fixed Centroids (The NumPy Fix)
    if len(fixed_centroids) == 0:
        fixed_centroids = np.empty((0, 2))
    else:
        fixed_centroids = np.array(fixed_centroids)
    
    # 4. Iterative Optimization
    for _ in range(max_iter):
        # Combine fixed (P/C) and variable (Fielders) points
        all_centroids = np.vstack([fixed_centroids, variable_centroids])
        
        # Calculate distance from every hit to every centroid
        distances = np.linalg.norm(data[:, np.newaxis] - all_centroids, axis=2)
        
        # Assign each hit to the nearest fielder
        labels = np.argmin(distances, axis=1)
        
        new_variable_centroids = []
        for k in range(n_variable):
            cluster_idx = len(fixed_centroids) + k
            cluster_mask = (labels == cluster_idx)
            
            if np.any(cluster_mask):
                # Calculate the Weighted Mean for the cluster
                # This pulls the fielder toward the "most relevant" hits
                points_in_cluster = data[cluster_mask]
                weights_in_cluster = weights[cluster_mask][:, np.newaxis]
                
                weighted_sum = np.sum(points_in_cluster * weights_in_cluster, axis=0)
                total_weight = np.sum(weights_in_cluster)
                
                proposed_centroid = weighted_sum / total_weight
                
                # Apply the Fair Territory Constraint
                new_variable_centroids.append(project_to_fair_territory(proposed_centroid))
            else:
                # If no hits assigned, keep the previous position
                new_variable_centroids.append(variable_centroids[k])
            
        new_variable_centroids = np.array(new_variable_centroids)
        
        # Exit if centroids have stabilized
        if np.allclose(variable_centroids, new_variable_centroids, atol=1e-4):
            break
        variable_centroids = new_variable_centroids

    return np.vstack([fixed_centroids, variable_centroids])

def transform_coordinates(df):
    """
    Converts Statcast raw hc_x/y (pixels) into actual field feet.
    Standard Statcast conversion: 2.43 feet per unit.
    """
    df = df.copy()
    # 125 and 204.5 are the standard Statcast origin offsets for Home Plate
    df['x'] = (df['hc_x'] - 125) * 2.43
    df['y'] = (204.5 - df['hc_y']) * 2.43
    df['dist'] = np.sqrt(df['x']**2 + df['y']**2)
    return df