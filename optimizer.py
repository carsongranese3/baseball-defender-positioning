import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def project_to_fair_territory(point):
    """Ensures optimal fielder coordinates stay within foul lines."""
    x, y = point
    if y >= abs(x) and y >= 0:
        return point
    if y < 0:
        return np.array([0, 0])
    if x > 0: 
        val = (x + y) / 2
        return np.array([val, val])
    else: 
        new_x = (x - y) / 2
        new_y = (y - x) / 2
        return np.array([new_x, new_y])

def project_infielder(point, fielder_index):
    """
    Enforces the MLB Anti-Shift Rules:
    1. Two fielders strictly on the left side of 2B, two on the right.
    2. All fielders must be on the infield dirt (<= 95ft from the mound).
    """
    x, y = point
    
    # RULE 1: Left/Right Boundary (X-Axis Split)
    # Indices 0 and 1 (3B, SS) must stay on the Left (Negative X)
    if fielder_index in [0, 1]:
        x = min(x, -1) # 1 foot buffer so they don't cross the exact centerline
    # Indices 2 and 3 (2B, 1B) must stay on the Right (Positive X)
    elif fielder_index in [2, 3]:
        x = max(x, 1)
        
    # RULE 2: The Dirt Boundary
    # The dirt arc is 95 feet from the mound (0, 60.5).
    dx = x - 0
    dy = y - 60.5
    dist_from_mound = np.sqrt(dx**2 + dy**2)
    
    if dist_from_mound > 95:
        # Scale the vector back so the player is touching the 95ft grass line
        ratio = 95 / dist_from_mound
        x = dx * ratio
        y = 60.5 + dy * ratio
        
    # Finally, ensure they didn't get pushed into foul territory down the lines
    return project_to_fair_territory([x, y])

def weighted_constrained_kmeans(data, weights, fixed_centroids, n_variable, situational_gravity=None, max_iter=100, is_infield=False):
    if len(data) == 0:
        return np.vstack([fixed_centroids, np.zeros((n_variable, 2))])

    # 1. Initial clustering based on hit density
    kmeans_init = KMeans(n_clusters=n_variable, n_init=10, random_state=42)
    kmeans_init.fit(data)
    variable_centroids = kmeans_init.cluster_centers_
    
    # 2. Geographic Sorting (Left to Right)
    sorted_indices = np.argsort(variable_centroids[:, 0])
    variable_centroids = variable_centroids[sorted_indices]
    
    if len(fixed_centroids) == 0:
        fixed_centroids = np.empty((0, 2))
    else:
        fixed_centroids = np.array(fixed_centroids)
    
    # 3. Iterative optimization
    for _ in range(max_iter):
        all_centroids = np.vstack([fixed_centroids, variable_centroids])
        distances = np.linalg.norm(data[:, np.newaxis] - all_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_variable_centroids = []
        for k in range(n_variable):
            cluster_idx = len(fixed_centroids) + k
            cluster_mask = (labels == cluster_idx)
            
            if np.any(cluster_mask):
                p_in_cluster = data[cluster_mask]
                w_in_cluster = weights[cluster_mask][:, np.newaxis]
                
                # Center of gravity for hits
                proposed = np.sum(p_in_cluster * w_in_cluster, axis=0) / np.sum(w_in_cluster)
                
                # Apply Situational Gravity (e.g., pulling 1B to the bag)
                if situational_gravity and k in situational_gravity:
                    target, strength = situational_gravity[k]
                    proposed = (proposed * (1 - strength)) + (np.array(target) * strength)
                
                # APPLY LEGAL BOUNDARIES
                if is_infield:
                    new_variable_centroids.append(project_infielder(proposed, k))
                else:
                    new_variable_centroids.append(project_to_fair_territory(proposed))
            else:
                new_variable_centroids.append(variable_centroids[k])
            
        new_variable_centroids = np.array(new_variable_centroids)
        if np.allclose(variable_centroids, new_variable_centroids, atol=1e-4):
            break
        variable_centroids = new_variable_centroids

    return np.vstack([fixed_centroids, variable_centroids])

def transform_coordinates(df):
    df = df.copy()
    df['x'] = (df['hc_x'] - 125) * 2.43
    df['y'] = (204.5 - df['hc_y']) * 2.43
    df['dist'] = np.sqrt(df['x']**2 + df['y']**2)
    return df