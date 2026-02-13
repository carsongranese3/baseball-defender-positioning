import pandas as pd
from pybaseball import playerid_lookup, statcast_batter, cache
import os
from datetime import datetime

cache.enable()

def main():
    print("--- Automatic Career Downloader ---")
    name_input = input("Enter player name: ").strip()
    if " " not in name_input: return print("Enter first and last name.")
    first, last = name_input.split(" ", 1)

    # 1. Lookup ID and Debut Year
    id_results = playerid_lookup(last, first, fuzzy=True)
    if id_results.empty: return print("Player not found.")
    
    p_id = id_results['key_mlbam'].values[0]
    debut_year = int(id_results['mlb_played_first'].values[0])
    current_year = datetime.now().year
    
    player_slug = f"{first.lower()}_{last.lower()}"
    player_dir = os.path.join("Batters", player_slug)
    
    if not os.path.exists(player_dir):
        os.makedirs(player_dir)
        print(f"Created folder for {player_slug.replace('_', ' ').title()}")

    # 2. Loop from Debut to Present
    print(f"Beginning download from debut season ({debut_year})...")
    for year in range(debut_year, current_year + 1):
        file_path = os.path.join(player_dir, f"{year}_data.csv")
        
        # Skip if we already have the data for past years
        if os.path.exists(file_path) and year < current_year:
            print(f"  - Skipping {year} (Already downloaded)")
            continue
            
        print(f"  - Fetching {year}...")
        try:
            data = statcast_batter(f'{year}-03-15', f'{year}-11-15', player_id=p_id)
            if not data.empty:
                df = data.dropna(subset=['hc_x', 'hc_y'])
                df.to_csv(file_path, index=False)
                print(f"    -> Saved {len(df)} hits")
            else:
                print(f"    -> No batted balls found.")
        except Exception as e:
            print(f"    -> Error: {e}")

if __name__ == "__main__":
    main()