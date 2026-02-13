import pandas as pd
from pybaseball import playerid_lookup, statcast_batter, cache

# Enable caching to speed up the player lookup
cache.enable()

def save_ohtani_at_bats():
    # 1. Lookup ID
    print("Looking up ID...")
    id_results = playerid_lookup('ohtani', 'shohei')
    ohtani_id = id_results['key_mlbam'].values[0]

    # 2. Fetch the full 2025 season
    # Covers from the Tokyo Series through the World Series
    print("Fetching Ohtani 2025 data (this may take a minute)...")
    data = statcast_batter('2025-03-18', '2025-11-01', player_id=ohtani_id)

    # 3. Filter for balls put in play
    # We drop rows without coordinates (strikeouts/walks) 
    # but KEEP hits and outs (anything with an hc_x/hc_y)
    batted_balls = data.dropna(subset=['hc_x', 'hc_y']).copy()

    # 4. Save to CSV in your GitHub project folder
    filename = "ohtani_2025_batted_balls.csv"
    batted_balls.to_csv(filename, index=False)
    
    print(f"--- Process Complete ---")
    print(f"File Saved: {filename}")
    print(f"Total Batted Balls: {len(batted_balls)}")
    print(f"Events included: {batted_balls['events'].unique()}")

if __name__ == "__main__":
    save_ohtani_at_bats()