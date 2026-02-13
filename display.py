import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimizer import transform_coordinates, weighted_constrained_kmeans

# --- 1. UI & Navigation Helpers ---

def list_folders(directory):
    """Lists directories (batters or pitchers) alphabetically."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        return []
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def draw_field(ax):
    """Draws professional diamond geometry, bases, and outfield grass line."""
    # Foul Lines & Diamond
    ax.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color='black', lw=2) 
    ax.plot([0, 450], [0, 450], color='black', alpha=0.3)
    ax.plot([0, -450], [0, 450], color='black', alpha=0.3)
    
    # Infield Grass Arc
    theta = np.linspace(0, np.pi, 100)
    x_arc, y_arc = 95 * np.cos(theta), 60.5 + 95 * np.sin(theta)
    valid = y_arc >= np.abs(x_arc)
    ax.plot(x_arc[valid], y_arc[valid], color='brown', ls='--', lw=2)
    
    # Bases (Red Diamonds)
    ax.scatter([0, 63.6, 0, -63.6], [0, 63.6, 127.3, 63.6], 
               c='red', s=80, marker='D', edgecolors='black', zorder=5)

# --- 2. Main Matchup Logic ---

def main():
    # A. Selection UI
    batters = list_folders("Batters")
    pitchers = list_folders("Pitchers")
    
    if not batters or not pitchers:
        return print("Error: Ensure data exists in /Batters and /Pitchers.")

    print("\n--- Available Batters ---")
    for i, b in enumerate(batters): print(f" [{i}] {b.replace('_', ' ').title()}")
    b_idx = int(input("Select Batter Number: "))
    selected_batter = batters[b_idx]

    print("\n--- Available Pitchers ---")
    for i, p in enumerate(pitchers): print(f" [{i}] {p.replace('_', ' ').title()}")
    p_idx = int(input("Select Pitcher Number: "))
    selected_pitcher = pitchers[p_idx]

    # B. Load Pitcher Arsenal (for weighting)
    p_path = os.path.join("Pitchers", selected_pitcher)
    arsenal_path = os.path.join(p_path, "pitches.csv")
    
    if not os.path.exists(arsenal_path):
        return print(f"Error: Missing arsenal summary for {selected_pitcher}. Run get_pitcher_data.py.")
    
    arsenal_df = pd.read_csv(arsenal_path)
    arsenal_dict = arsenal_df.set_index('pitch_type').to_dict('index')

    # C. Load Batter Data
    b_path = os.path.join("Batters", selected_batter)
    b_files = [f for f in os.listdir(b_path) if f.endswith('.csv')]
    raw_df = pd.concat([pd.read_csv(os.path.join(b_path, f)) for f in b_files], ignore_index=True)
    df = transform_coordinates(raw_df.dropna(subset=['hc_x', 'hc_y']))

    # D. Probabilistic Weighting System
    # 1. Look for same pitch types
    # 2. Weight by Pitcher Usage %
    # 3. Weight by Velocity Similarity (Gaussian Decay)
    def calculate_weight(row):
        p_type = row['pitch_type']
        if p_type not in arsenal_dict: return 0 
        
        freq_w = arsenal_dict[p_type]['Usage_%'] / 100
        target_vel = arsenal_dict[p_type]['Avg_Vel']
        actual_vel = row['release_speed']
        
        # Gaussian Bell Curve (Standard Deviation = 4mph)
        vel_w = np.exp(-( (target_vel - actual_vel)**2 ) / (2 * 4**2))
        return freq_w * vel_w

    df['matchup_weight'] = df.apply(calculate_weight, axis=1)
    df_weighted = df[df['matchup_weight'] > 0.05].copy() # Filter low-relevance hits

    if df_weighted.empty:
        return print("No relevant matchup data found between this batter and pitcher's arsenal.")

    # E. Segmentation & Weighted K-Means
    if_df = df_weighted[(df_weighted['dist'] < 220) & (df_weighted['dist'] > 45)]
    of_df = df_weighted[df_weighted['dist'] >= 220]

    # Infield (Fixed Battery Points: Pitcher/Catcher)
    if len(if_df) > 4:
        if_centroids = weighted_constrained_kmeans(
            if_df[['x', 'y']].values, if_df['matchup_weight'].values, [[0, 60.5], [0, -2]], 4
        )
    else:
        # Fallback if IF data is sparse
        if_centroids = np.array([[0, 60.5], [0, -2], [-30, 150], [30, 150], [-60, 110], [60, 110]])

    # Outfield (No Fixed Points)
    if len(of_df) >= 3:
        of_centroids = weighted_constrained_kmeans(
            of_df[['x', 'y']].values, of_df['matchup_weight'].values, [], 3
        )
    else:
        # Standard OF positioning if data is sparse
        of_centroids = np.array([[150, 300], [0, 350], [-150, 300]])

    # F. Final Visualization
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal')
    draw_field(ax)
    
    # Cursor distance measurement tool
    def format_coord(x, y):
        dist = np.sqrt(x**2 + y**2)
        return f'x={x:.1f}, y={y:.1f} | Dist: {dist:.1f} ft'
    ax.format_coord = format_coord

    # Plot hits with transparency reflecting their weighting importance
    # Hits that match the pitcher's arsenal better will appear more solid
    ax.scatter(if_df['x'], if_df['y'], c='green', alpha=if_df['matchup_weight'].clip(0, 1) * 0.4, s=20, label='IF Potential')
    ax.scatter(of_df['x'], of_df['y'], c='blue', alpha=of_df['matchup_weight'].clip(0, 1) * 0.2, s=25, label='OF Potential')

    # Draw Optimal Stars
    # Red for Infielders, Dark Blue for Outfielders
    ax.scatter(if_centroids[2:, 0], if_centroids[2:, 1], c='red', s=250, marker='*', 
               label='Weighted IF', edgecolors='white', zorder=10)
    ax.scatter(of_centroids[:, 0], of_centroids[:, 1], c='darkblue', s=250, marker='*', 
               label='Weighted OF', edgecolors='white', zorder=10)
    
    # Static Battery Markers (Squares)
    ax.scatter(if_centroids[:2, 0], if_centroids[:2, 1], c='black', s=60, marker='s', label='P/C', zorder=20)

    # UI Finishing Touches
    plt.title(f"Weighted Probabilistic Matchup:\n{selected_batter.title()} vs {selected_pitcher.title()}", fontsize=12)
    plt.xlim(-250, 250); plt.ylim(-20, 450)
    plt.legend(loc='upper right', fontsize='x-small', framealpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()