import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimizer import transform_coordinates, constrained_kmeans

# --- 1. Helper Logic ---

def list_folders(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        return []
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def draw_field(ax):
    """Draws the professional diamond geometry and bases."""
    ax.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color='black', lw=2) 
    ax.plot([0, 450], [0, 450], color='black', alpha=0.3)
    ax.plot([0, -450], [0, 450], color='black', alpha=0.3)
    
    theta = np.linspace(0, np.pi, 100)
    x_arc, y_arc = 95 * np.cos(theta), 60.5 + 95 * np.sin(theta)
    valid = y_arc >= np.abs(x_arc)
    ax.plot(x_arc[valid], y_arc[valid], color='brown', ls='--', lw=2)
    
    ax.scatter([0, 63.6, 0, -63.6], [0, 63.6, 127.3, 63.6], 
               c='red', s=80, marker='D', edgecolors='black', zorder=5)

# --- 2. Main Execution ---

def main():
    pitchers = list_folders("Pitchers")
    if not pitchers: return print("No pitchers found in /Pitchers.")

    print("\n--- Available Pitchers ---")
    for i, p in enumerate(pitchers): print(f" [{i}] {p.replace('_', ' ').title()}")
    p_idx = int(input("\nSelect Pitcher: "))
    selected_pitcher = pitchers[p_idx]

    # Load Data
    path = os.path.join("Pitchers", selected_pitcher)
    files = [f for f in os.listdir(path) if f.endswith('_pitching_data.csv')]
    df = pd.concat([pd.read_csv(os.path.join(path, f)) for f in files], ignore_index=True)

    # Filter Option
    mode = input("\nShow matchup vs (L)efties, (R)ighties, or (B)oth? ").strip().upper()
    
    if mode == 'B':
        df_filtered = df.dropna(subset=['hc_x', 'hc_y']).copy()
    else:
        df_filtered = df[df['stand'] == mode].dropna(subset=['hc_x', 'hc_y']).copy()
    
    if df_filtered.empty: return print("No data found for this selection.")

    # Clean and Transform
    df_filtered = transform_coordinates(df_filtered)
    df_clean = df_filtered[(df_filtered['events'] != 'home_run') & (df_filtered['dist'] < 440)]

    # Math: Optimal Positions (Calculated on the selected subset)
    of_data = df_clean[df_clean['dist'] >= 220][['x', 'y']].values
    if_playable = df_clean[(df_clean['dist'] < 220) & (df_clean['dist'] > 45)][['x', 'y']].values

    if_centroids, _, _ = constrained_kmeans(if_playable, [[0, 60.5], [0, -2]], 4)
    of_centroids, _, _ = constrained_kmeans(of_data, [], 3)

    # 3. Visualization
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal')
    draw_field(ax)
    
    def format_coord(x, y):
        dist = np.sqrt(x**2 + y**2)
        return f'x={x:.1f}, y={y:.1f}, Dist={dist:.1f} ft'
    ax.format_coord = format_coord

    # Layer 1: Plotting Balls in Play
    if mode == 'B':
        # Split the clean dataframe for color coding
        rhb = df_clean[df_clean['stand'] == 'R']
        lhb = df_clean[df_clean['stand'] == 'L']
        
        ax.scatter(rhb['x'], rhb['y'], c='orange', alpha=0.15, s=15, label='vs Righties')
        ax.scatter(lhb['x'], lhb['y'], c='cyan', alpha=0.15, s=15, label='vs Lefties')
    else:
        color = 'orange' if mode == 'R' else 'cyan'
        label = f'vs {"Righties" if mode == "R" else "Lefties"}'
        ax.scatter(df_clean['x'], df_clean['y'], c=color, alpha=0.15, s=15, label=label)

    # Layer 2: Optimal Defensive Stars
    ax.scatter(if_centroids[2:, 0], if_centroids[2:, 1], c='red', s=280, marker='*', 
               label='Optimal IF', edgecolors='white', zorder=10)
    ax.scatter(of_centroids[:, 0], of_centroids[:, 1], c='blue', s=280, marker='*', 
               label='Optimal OF', edgecolors='white', zorder=10)
    ax.scatter(if_centroids[:2, 0], if_centroids[:2, 1], c='black', s=70, marker='s', label='Battery', zorder=20)

    title_str = f"Pitcher Profile: {selected_pitcher.replace('_', ' ').title()}"
    title_str += f" (vs {mode})" if mode != 'B' else " (Entire Career)"
    plt.title(title_str, fontsize=12)
    plt.xlim(-250, 250); plt.ylim(-20, 450)
    plt.legend(loc='upper right', fontsize='x-small', framealpha=0.8)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()