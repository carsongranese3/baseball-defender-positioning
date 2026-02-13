import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimizer import transform_coordinates, constrained_kmeans

# --- 1. Helper UI Functions ---

def list_players(folder="Batters"):
    """Lists only directories inside the Batters folder."""
    if not os.path.exists(folder):
        print(f"Directory '{folder}' not found. Please run the downloader first.")
        return []
    players = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
    print("\n--- Available Players (Career Folders) ---")
    for i, p in enumerate(players):
        print(f" [{i}] {p.replace('_', ' ').title()}")
    return players

def draw_field(ax):
    """Draws the professional diamond geometry and bases."""
    # Diamond & Foul Lines
    ax.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color='black', lw=2) 
    ax.plot([0, 450], [0, 450], color='black', alpha=0.3)
    ax.plot([0, -450], [0, 450], color='black', alpha=0.3)
    
    # Infield Arc (Grass Line)
    theta = np.linspace(0, np.pi, 100)
    x_arc, y_arc = 95 * np.cos(theta), 60.5 + 95 * np.sin(theta)
    valid = y_arc >= np.abs(x_arc)
    ax.plot(x_arc[valid], y_arc[valid], color='brown', ls='--', lw=2)
    
    # Bases (Red Diamonds) - zorder 5
    ax.scatter([0, 63.6, 0, -63.6], [0, 63.6, 127.3, 63.6], 
               c='red', s=80, marker='D', edgecolors='black', zorder=5, label='Bases')

# --- 2. Main Execution ---

def main():
    players = list_players()
    if not players: return
    
    try:
        p_idx = int(input("\nSelect Player Number: "))
        selected_player = players[p_idx]
    except (ValueError, IndexError):
        return print("Invalid selection.")

    player_path = os.path.join("Batters", selected_player)

    # 1. Automatically Load and Combine All Data
    csv_files = [f for f in os.listdir(player_path) if f.endswith('.csv')]
    data_frames = [pd.read_csv(os.path.join(player_path, f)) for f in csv_files]
    
    if not data_frames: 
        return print("No data found in player folder.")
    
    df = pd.concat(data_frames, ignore_index=True).dropna(subset=['hc_x', 'hc_y'])
    
    # 2. Filtering Logic
    print(f"Loaded {len(df)} total hits for {selected_player.replace('_', ' ').title()}.")
    p_hand = input("\nFilter P-Hand? (R/L/All): ").strip().upper()
    if p_hand in ['R', 'L']: df = df[df['p_throws'] == p_hand]

    # Convert to feet before segmenting
    df = transform_coordinates(df)

    # --- REALITY CHECK FILTER ---
    # Exclude Home Runs and impossible outliers (like 490ft fly outs)
    # This prevents the stars from being pulled way out of position
    df_clean = df[(df['events'] != 'home_run') & (df['dist'] < 440)].copy()
    
    # 3. Data Segmentation for Optimization
    of_data = df_clean[df_clean['dist'] >= 220][['x', 'y']].values
    if_playable = df_clean[(df_clean['dist'] < 220) & (df_clean['dist'] > 45)][['x', 'y']].values
    battery_hits = df_clean[df_clean['dist'] <= 45][['x', 'y']].values

    # 4. Optimization Math
    # Returns: (centroids, labels, total_residual)
    if_centroids, if_labels, _ = constrained_kmeans(if_playable, [[0, 60.5], [0, -2]], 4)
    of_centroids, _, _ = constrained_kmeans(of_data, [], 3)

    # 5. Visualization (Shrunken for Laptop Compatibility)
    fig, ax = plt.subplots(figsize=(8, 7)) 
    ax.set_aspect('equal')
    draw_field(ax)
    
    # LIVE DISTANCE TRACKER: Returns distance from (0,0) [Home Plate] to mouse
    def format_coord(x, y):
        dist = np.sqrt(x**2 + y**2)
        return f'x={x:.1f}, y={y:.1f}, Dist={dist:.1f} ft'
    ax.format_coord = format_coord
    
    # Plot Results
    # Battery (Blue Squares)
    ax.scatter(if_centroids[:2, 0], if_centroids[:2, 1], c='blue', s=60, marker='s', label='Battery', zorder=20)
    
    # Infielders (Red Stars)
    ax.scatter(if_centroids[2:, 0], if_centroids[2:, 1], c='red', s=250, marker='*', label='Optimal IF', edgecolors='white', zorder=10)
    ax.scatter(if_playable[:, 0], if_playable[:, 1], c=if_labels, cmap='tab10', alpha=0.3, s=20)
    
    # Outfielders (Blue Stars)
    ax.scatter(of_centroids[:, 0], of_centroids[:, 1], c='blue', s=250, marker='*', label='Optimal OF', edgecolors='white', zorder=10)
    ax.scatter(of_data[:, 0], of_data[:, 1], c='blue', alpha=0.1, s=20)
    
    # Plot ignored Battery Zone (Grey hits)
    if len(battery_hits) > 0:
        ax.scatter(battery_hits[:, 0], battery_hits[:, 1], c='grey', alpha=0.2, s=10)

    plt.title(f"Optimal Defense: {selected_player.replace('_', ' ').title()}\n(Career Summary - No HRs)", fontsize=12)
    plt.xlim(-250, 250); plt.ylim(-20, 450)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()