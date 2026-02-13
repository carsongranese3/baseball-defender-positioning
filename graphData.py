import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the Ohtani dataset
try:
    df = pd.read_csv("ohtani_2025_batted_balls.csv")
except FileNotFoundError:
    print("CSV not found. Run your data generation script first!")
    exit()

# 2. Filter: Keep Hits, Outs, and Errors; Ignore Home Runs
# 'field_error' is the Statcast event for reaching on an error
df_playable = df[df['events'] != 'home_run'].copy()

# 3. Transform Coordinates to Feet
df_playable['x_ft'] = (df_playable['hc_x'] - 125) * 2.43
df_playable['y_ft'] = (204.5 - df_playable['hc_y']) * 2.43

# 4. Set up Plotting
plt.figure(figsize=(10, 10))

# Define categories to plot
# Note: 'field_out' and 'error' are both "successes" for positioning
categories = {
    'Outs': (['field_out', 'force_out', 'grounded_into_double_play'], 'gray', 'x', 0.4),
    'Errors': (['field_error'], 'purple', 'D', 0.9), # 'D' for Diamond shape
    'Hits': (['single', 'double', 'triple'], 'orange', 'o', 0.7)
}

for label, (events, color, marker, alpha) in categories.items():
    subset = df_playable[df_playable['events'].isin(events)]
    plt.scatter(subset['x_ft'], subset['y_ft'], 
                label=f"{label} ({len(subset)})", 
                c=color, marker=marker, alpha=alpha, edgecolors='k' if marker == 'o' else None)

# 5. Field Geometry
plt.plot([0, 63.6, 0, -63.6, 0], [0, 63.6, 127.3, 63.6, 0], color='black', lw=2) # Infield
plt.axvline(0, color='black', alpha=0.1) # Center Field Line

# 6. Final Polish
plt.title("Ohtani 2025: Playable Balls (Hits, Outs, & Errors)", fontsize=15)
plt.xlabel("Feet from Home (Left < 0 > Right)")
plt.ylabel("Depth in Feet")
plt.xlim(-250, 250)
plt.ylim(-50, 450)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.show()