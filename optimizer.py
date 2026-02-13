import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

# --- 1. Helper Functions ---

def project_to_fair_territory(point):
    """
    Projects a point (x, y) onto the nearest location in Fair Territory.
    Fair Territory rule: y >= |x| and y >= 0.
    """
    x, y = point
    
    # 1. Check if already fair
    if y >= abs(x) and y >= 0:
        return point
        
    # 2. If behind home plate (y < 0), clamp to Home (0,0)
    if y < 0:
        return np.array([0, 0])
        
    # 3. If Foul (y < |x|), project to nearest foul line
    if x > 0: 
        # Right Field Line (y = x)
        val = (x + y) / 2
        return np.array([val, val])
    else: 
        # Left Field Line (y = -x)
        new_x = (x - y) / 2
        new_y = (y - x) / 2
        return np.array([new_x, new_y])

def constrained_kmeans(data, fixed_centroids, n_variable, max_iter=100):
    """
    Runs K-Means with constraints. Robust to empty fixed_centroids.
    """
    # Initialize Variable Centroids
    kmeans_init = KMeans(n_clusters=n_variable, n_init=10, random_state=42)
    kmeans_init.fit(data)
    variable_centroids = kmeans_init.cluster_centers_
    
    fixed_centroids = np.array(fixed_centroids)
    
    for i in range(max_iter):
        # Combine Fixed and Variable (Handle empty fixed list safely)
        if len(fixed_centroids) > 0:
            all_centroids = np.vstack([fixed_centroids, variable_centroids])
        else:
            all_centroids = variable_centroids

        # Assignment Step
        distances = np.linalg.norm(data[:, np.newaxis] - all_centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update Step
        new_variable_centroids = []
        for k in range(n_variable):
            cluster_idx = len(fixed_centroids) + k
            points_in_cluster = data[labels == cluster_idx]
            
            if len(points_in_cluster) > 0:
                proposed_center = points_in_cluster.mean(axis=0)
            else:
                proposed_center = variable_centroids[k]
            
            # Apply Fair Territory Constraint
            valid_center = project_to_fair_territory(proposed_center)
            new_variable_centroids.append(valid_center)
            
        new_variable_centroids = np.array(new_variable_centroids)
        
        if np.allclose(variable_centroids, new_variable_centroids, atol=1e-4):
            break
            
        variable_centroids = new_variable_centroids

    # Final Calculation
    if len(fixed_centroids) > 0:
        all_centroids = np.vstack([fixed_centroids, variable_centroids])
    else:
        all_centroids = variable_centroids

    distances = np.linalg.norm(data[:, np.newaxis] - all_centroids, axis=2)
    final_labels = np.argmin(distances, axis=1)
    total_residual = np.sum(np.min(distances, axis=1))
    
    return all_centroids, final_labels, total_residual

# --- 2. Main Execution ---

def main():
    print("--- Defensive Optimization (Fair Territory Only) ---")
    
    # User Input
    player_name = input("Enter Player Name (e.g. 'Shohei Ohtani'): ").strip()
    season = input("Enter Season (e.g. '2025'): ").strip()
    
    safe_name = player_name.lower().replace(" ", "_")
    filename = f"{safe_name}_{season}_batted_balls.csv"
    
    if not os.path.exists(filename):
        print(f"\n[Error] File not found: {filename}")
        print("Please run 'get_player_data.py' first.")
        return

    print(f"Loading {filename}...")
    df = pd.read_csv(filename)
    df = df[df['events'] != 'home_run'].dropna(subset=['hc_x', 'hc_y']).copy()

    # Transform to Feet
    df['x'] = (df['hc_x'] - 125) * 2.43
    df['y'] = (204.5 - df['hc_y']) * 2.43
    df['dist'] = np.sqrt(df['x']**2 + df['y']**2)

    # Split Infield vs Outfield
    infield_data = df[df['dist'] < 225][['x', 'y']].values
    outfield_data = df[df['dist'] >= 225][['x', 'y']].values
    
    # Optimize Infield (Fixed Pitcher/Catcher)
    fixed_pos = [[0, 60.5], [0, -2]] 
    if_centroids, if_labels, if_resid = constrained_kmeans(infield_data, fixed_pos, n_variable=4)
    
    # Optimize Outfield (No Fixed Points)
    of_centroids, of_labels, of_resid = constrained_kmeans(outfield_data, [], n_variable=3)

    total_resid = if_resid + of_resid
    print(f"Total Residuals: {int(total_resid)} ft")

    # Visualization
    plt.figure(figsize=(10, 10))
    
    # Field Geometry
    plt.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color='black', lw=2)
    plt.plot([0, 250], [0, 250], color='black', alpha=0.3)
    plt.plot([0, -250], [0, 250], color='black', alpha=0.3)
    
    theta = np.linspace(np.radians(-38), np.radians(218), 100)
    x_arc = 95 * np.cos(theta)
    y_arc = 60.5 + 95 * np.sin(theta)
    valid_arc = y_arc >= np.abs(x_arc)
    plt.plot(x_arc[valid_arc], y_arc[valid_arc], color='brown', linestyle='--', lw=2)

    # Plot Infield Results
    plt.scatter(if_centroids[:2, 0], if_centroids[:2, 1], c='black', s=150, marker='s', label='Fixed (P/C)', zorder=10)
    plt.scatter(if_centroids[2:, 0], if_centroids[2:, 1], c='red', s=250, marker='*', edgecolors='white', label='Infielders', zorder=10)
    plt.scatter(infield_data[:, 0], infield_data[:, 1], c=if_labels, cmap='tab10', alpha=0.3, s=20)
    
    # Plot Outfield Results
    plt.scatter(of_centroids[:, 0], of_centroids[:, 1], c='blue', s=250, marker='*', edgecolors='white', label='Outfielders', zorder=10)
    plt.scatter(outfield_data[:, 0], outfield_data[:, 1], c='blue', alpha=0.1, s=20)

    plt.title(f"Optimal Defense (Fair Territory Constraint): {player_name}", fontsize=14)
    plt.xlim(-250, 250)
    plt.ylim(-20, 450)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()