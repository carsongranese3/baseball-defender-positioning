import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from optimizer import (transform_coordinates, weighted_constrained_kmeans,
                       project_infielder, project_to_fair_territory)

FIRST_BASE_BAG = [63.6, 63.6]
# A 1B who can't beat the runner to the bag isn't playing first base. Holding a
# runner means a foot on it; otherwise he can roam, but only this far. 30ft also
# keeps him on the dirt for free - the bag is 63.7ft from the mound, so the
# leash tops out at 93.7ft, inside the 95ft grass line.
FIRST_BASE_HOLD_LEASH = 3.0
FIRST_BASE_COVER_LEASH = 30.0

def list_folders(directory):
    if not os.path.exists(directory): os.makedirs(directory)
    return sorted([d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))])

def ask(prompt):
    """input() that exits cleanly on Ctrl-C / Ctrl-D instead of a traceback."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        sys.exit(0)

def ask_choice(prompt, options):
    """Re-prompts until the user picks a valid index. isdigit() also rejects
    negatives, which would otherwise index from the end of the list silently."""
    while True:
        raw = ask(prompt)
        if raw.isdigit() and int(raw) < len(options):
            return options[int(raw)]
        print(f"  [!] Enter a number from 0 to {len(options) - 1}.")

def ask_int(prompt, low, high):
    while True:
        raw = ask(prompt)
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        print(f"  [!] Enter a whole number from {low} to {high}.")

def ask_runners(prompt):
    """Accepts None/empty or any combination of 1/2/3, in any order."""
    while True:
        raw = ask(prompt)
        if raw.lower() in ("", "none", "0"):
            return "None"
        if all(c in "123" for c in raw):
            return "".join(sorted(set(raw)))
        print("  [!] Enter None, or any combination of 1/2/3 (e.g. 13).")

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
    selected_batter = ask_choice("Select Batter #: ", batters)

    for i, p in enumerate(pitchers): print(f" [{i}] {p.replace('_', ' ').title()}")
    selected_pitcher = ask_choice("Select Pitcher #: ", pitchers)

    print("\n--- Game State Configuration ---")
    outs = ask_int("Outs (0, 1, 2): ", 0, 2)
    runners = ask_runners("Runners on base (None, 1, 2, 3, 12, 13, 23, 123): ")
    inning = ask_int("Inning: ", 1, 30)

    p_path = os.path.join("Pitchers", selected_pitcher)
    arsenal_path = os.path.join(p_path, "pitches.csv")
    if not os.path.exists(arsenal_path):
        return print(f"[!] No pitches.csv for {selected_pitcher}. Re-run the pitcher "
                     f"downloader (menu option 2) to build the arsenal summary.")
    arsenal_dict = pd.read_csv(arsenal_path).set_index('pitch_type').to_dict('index')

    b_path = os.path.join("Batters", selected_batter)
    b_files = [f for f in os.listdir(b_path) if f.endswith('.csv')]
    if not b_files:
        return print(f"[!] No season data for {selected_batter}. Re-run the batter "
                     f"downloader (menu option 1).")
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
    first_base_bag = FIRST_BASE_BAG
    if_leash = {}

    if '1' in runners:
        print(">> STRATEGY: Holding Runner - 1B pinned to bag.")
        if_gravity[3] = (first_base_bag, 0.95)
        if_leash[3] = (first_base_bag, FIRST_BASE_HOLD_LEASH)
    else:
        if_gravity[3] = (first_base_bag, 0.25)
        if_leash[3] = (first_base_bag, FIRST_BASE_COVER_LEASH)

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
    # Statcast has no 'pop_out' event - popups come through as field_out with
    # bb_type 'popup' - so the launch angle cut is what actually excludes them.
    if_df = if_raw[if_raw['launch_angle'] < 50].copy()
    of_df = df_weighted[df_weighted['dist'] >= 220]

    # --- UPDATED OPTIMIZATION CALL (Notice is_infield=True) ---
    if len(if_df) > 4:
        if_centroids = weighted_constrained_kmeans(
            if_df[['x', 'y']].values, 
            if_df['matchup_weight'].values, 
            [[0, 60.5], [0, -2]], 
            4, 
            situational_gravity=if_gravity,
            is_infield=True,   # <--- THE SHIFT BAN TOGGLE
            situational_leash=if_leash
        )
    else:
        # Fallback legal positions
        if_centroids = np.array([[0, 60.5], [0, -2], [-40, 110], [-15, 120], [15, 120], [40, 110]], dtype=float)

    # Depth is the y-axis only; scaling x too would drag the middle infielders
    # toward the centerline. Applied to BOTH branches, so infield-in still
    # moves people when the matchup has too little data to cluster.
    if if_depth_multiplier != 1.0:
        if_centroids[2:, 1] *= if_depth_multiplier

    # Final legality + bag-coverage pass. The depth multiplier runs after the
    # optimizer and the fallback positions are hardcoded, so neither is
    # guaranteed legal or in coverage on its own.
    for k in range(4):
        if_centroids[2 + k] = project_infielder(if_centroids[2 + k], k, if_leash.get(k))

    if len(of_df) >= 3:
        of_centroids = weighted_constrained_kmeans(of_df[['x', 'y']].values, of_df['matchup_weight'].values, [], 3)
    else:
        of_centroids = np.array([[150, 300], [0, 350], [-150, 300]], dtype=float)

    # Same as the infield: the no-doubles boost applies to both branches.
    of_centroids[:, 1] += of_depth_boost
    for k in range(len(of_centroids)):
        of_centroids[k] = project_to_fair_territory(of_centroids[k])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_aspect('equal'); draw_field(ax)

    # Highlight any base that currently has a runner on it
    base_coords = {'1': [63.6, 63.6], '2': [0, 127.3], '3': [-63.6, 63.6]}
    occupied = [(num, xy) for num, xy in base_coords.items() if num in runners]
    if occupied:
        rx = [xy[0] for _, xy in occupied]
        ry = [xy[1] for _, xy in occupied]
        # Hollow ring so a fielder pinned to the same bag (e.g. 1B holding a
        # runner) stays visible through the center instead of being painted over.
        ax.scatter(rx, ry, facecolors='none', s=520, marker='o', edgecolors='gold',
                   linewidths=2.5, zorder=22, label="Runner On Base")
        for num, (bx, by) in occupied:
            ax.annotate(f"R{num}", (bx, by + 14), color='darkgoldenrod', fontsize=7,
                        fontweight='bold', ha='center', va='center', zorder=23)

    ax.scatter(if_df['x'], if_df['y'], c='green', alpha=if_df['matchup_weight'].clip(0, 1) * 0.4, s=20)
    ax.scatter(of_df['x'], of_df['y'], c='blue', alpha=of_df['matchup_weight'].clip(0, 1) * 0.2, s=25)
    
    ax.scatter(if_centroids[2:, 0], if_centroids[2:, 1], c='red', s=250, marker='*', edgecolors='white', zorder=10, label="Optimal IF")
    ax.scatter(of_centroids[:, 0], of_centroids[:, 1], c='darkblue', s=250, marker='*', edgecolors='white', zorder=10, label="Optimal OF")
    ax.scatter(if_centroids[:2, 0], if_centroids[:2, 1], c='black', s=60, marker='s', zorder=20, label="P/C")

    batter_name = selected_batter.replace('_', ' ').title()
    pitcher_name = selected_pitcher.replace('_', ' ').title()
    plt.title(f"{batter_name} vs {pitcher_name}\nMLB Legal Shifts | Runners {runners} | Outs {outs}", fontsize=11)
    plt.legend(loc='upper right', fontsize='x-small', framealpha=0.7,
               markerscale=0.6, scatterpoints=1)

    enable_scroll_zoom(ax)
    plt.show()

def enable_scroll_zoom(ax, base_scale=1.2):
    """Scroll wheel zooms in/out, centered on the cursor."""
    def on_scroll(event):
        if event.inaxes is not ax:
            return
        scale = 1 / base_scale if event.button == 'up' else base_scale
        cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (cur_xlim[1] - cur_xlim[0]) * scale
        new_h = (cur_ylim[1] - cur_ylim[0]) * scale
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        ax.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
        ax.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
        ax.figure.canvas.draw_idle()

    ax.figure.canvas.mpl_connect('scroll_event', on_scroll)

if __name__ == "__main__":
    main()