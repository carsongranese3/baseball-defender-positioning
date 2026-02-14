import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimizer import transform_coordinates, weighted_constrained_kmeans

def list_folders(directory):
    if not os.path.exists(directory): os.makedirs(directory)
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def draw_field(ax):
    """Draws diamond, foul lines, and infield grass arc."""
    ax.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color='black', lw=2) 
    ax.plot([0, 450], [0, 450], color='black', alpha=0.3)
    ax.plot([0, -450], [0, 450], color='black', alpha=0.3)
    
    # Infield Arc (95ft from the mound)
    theta = np.linspace(0, np.pi, 100)
    x_arc, y_arc = 95 * np.cos(theta), 60.5 + 95 * np.sin(theta)
    valid = y_arc >= np.abs(x_arc)
    ax.plot(x_arc[valid], y_arc[valid], color='brown', ls='--', lw=2)
    
    ax.scatter([0, 63.6, 0, -63.6], [0, 63.6, 127.3, 63.6], c='red', s=80, marker='D', edgecolors='black', zorder=5)

def main():
    batters, pitchers = list_folders("Batters"), list_folders("Pitchers")
    if not batters or not pitchers: return print("Missing data in /Batters or /Pitchers.")

    print("\n--- Matchup Selection ---")
    for i, b in enumerate(batters): print(f" [{i}] {b.replace('_', ' ').title()}")
    selected_batter = batters[int(input("Select Batter #: "))]

    for i, p in enumerate(pitchers): print(f" [{i}] {p.replace('_', ' ').title()}")
    selected_pitcher = pitchers[int(input("Select Pitcher #: "))]

    print("\n--- Game State Configuration ---")
    outs = int(input("Outs (0, 1, 2): "))
    runners = input("Runners on base (None, 1, 2, 3, 12, 13, 23, 123): ").strip()
    inning = int(input("Inning: "))

    p_path = os.path.join("Pitchers", selected_pitcher)
    arsenal_dict = pd.read_csv(os.path.join(p_path, "pitches.csv")).set_index('pitch_type').to_dict('index')

    b_path = os.path.join("Batters", selected_batter)
    b_files = [f for f in os.listdir(b_path) if f.endswith('.csv')]
    df = pd.concat([pd.read_csv(os.path.join(b_path, f)) for f in b_files], ignore_index=True)
    df = transform_coordinates(df.dropna(subset=['hc_x', 'hc_y']))

    def calculate_weight(row):
        p_type = row['pitch_type']
        if p_type not in arsenal_dict: return 0 
        freq_w = arsenal_dict[p_type]['Usage_%'] / 100
        vel_w = np.exp(-( (arsenal_dict[p_type]['Avg_Vel'] - row['release_speed'])**2 ) / (2 * 4**2))
        return freq_w * vel_w

    df['matchup_weight'] = df.apply(calculate_weight, axis=1)
    df_weighted = df[df['matchup_weight'] > 0.05].copy()

    if_gravity = {}
    if_depth_multiplier = 1.0
    of_depth_boost = 0
    first_base_bag = [63.6, 63.6]
    
    if '1' in runners:
        print(">> STRATEGY: Holding Runner - 1B pinned to bag.")
        if_gravity[3] = (first_base_bag, 0.95)
    else:
        if_gravity[3] = (first_base_bag, 0.25)

    if '1' in runners and outs < 2:
        print(">> STRATEGY: Double Play Depth - Mid-Infield pinching.")
        if_gravity[1] = ([0, 127.3], 0.35) 
        if_gravity[2] = ([0, 127.3], 0.35)

    if '3' in runners and outs < 2 and inning >= 7:
        print(">> STRATEGY: Infield IN - Protecting Home.")
        if_depth_multiplier = 0.82

    if outs == 2 and inning >= 7:
        print(">> STRATEGY: No Doubles - Deep Outfield.")
        of_depth_boost = 25

    if_raw = df_weighted[(df_weighted['dist'] < 220) & (df_weighted['dist'] > 45)]
    if_df = if_raw[(if_raw['events'] != 'pop_out') & (if_raw['launch_angle'] < 50)].copy()
    of_df = df_weighted[df_weighted['dist'] >= 220]

    # --- UPDATED OPTIMIZATION CALL (Notice is_infield=True) ---
    if len(if_df) > 4:
        if_centroids = weighted_constrained_kmeans(
            if_df[['x', 'y']].values, 
            if_df['matchup_weight'].values, 
            [[0, 60.5], [0, -2]], 
            4, 
            situational_gravity=if_gravity,
            is_infield=True   # <--- THE SHIFT BAN TOGGLE
        )
        if_centroids[2:] *= if_depth_multiplier
    else:
        # Fallback legal positions
        if_centroids = np.array([[0, 60.5], [0, -2], [-40, 110], [-15, 120], [15, 120], [40, 110]])
    
    if len(of_df) >= 3:
        of_centroids = weighted_constrained_kmeans(of_df[['x', 'y']].values, of_df['matchup_weight'].values, [], 3)
        of_centroids[:, 1] += of_depth_boost
    else:
        of_centroids = np.array([[150, 300], [0, 350], [-150, 300]])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal'); draw_field(ax)
    
    ax.scatter(if_df['x'], if_df['y'], c='green', alpha=if_df['matchup_weight'].clip(0, 1) * 0.4, s=20)
    ax.scatter(of_df['x'], of_df['y'], c='blue', alpha=of_df['matchup_weight'].clip(0, 1) * 0.2, s=25)
    
    ax.scatter(if_centroids[2:, 0], if_centroids[2:, 1], c='red', s=250, marker='*', edgecolors='white', zorder=10, label="Optimal IF")
    ax.scatter(of_centroids[:, 0], of_centroids[:, 1], c='darkblue', s=250, marker='*', edgecolors='white', zorder=10, label="Optimal OF")
    ax.scatter(if_centroids[:2, 0], if_centroids[:2, 1], c='black', s=60, marker='s', zorder=20, label="P/C")

    plt.title(f"{selected_batter.title()} vs {selected_pitcher.title()}\nMLB Legal Shifts | Runners {runners} | Outs {outs}", fontsize=11)
    plt.legend(loc='upper right', fontsize='x-small', framealpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()