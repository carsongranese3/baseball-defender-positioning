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

    # 1. Lookup ID and Debut Year. Exact match only - a fuzzy hit would happily
    # download the wrong player's whole career into a folder named after the
    # one you asked for, with nothing to tell you it happened.
    id_results = playerid_lookup(last, first, fuzzy=False)
    if id_results.empty: return print(f"No exact match for '{name_input}'.")

    match = id_results[
        (id_results['name_first'].str.lower() == first.lower()) &
        (id_results['name_last'].str.lower() == last.lower())
    ]
    if match.empty: return print(f"No exact match for '{name_input}'.")

    if len(match) > 1:
        print(f"[!] {len(match)} players named '{name_input}'. Pick one:")
        for i, row in enumerate(match.itertuples()):
            print(f"  [{i}] MLBAM {row.key_mlbam}, debut {row.mlb_played_first}")
        while True:
            try:
                raw = input("Select #: ").strip()
            except (EOFError, KeyboardInterrupt):
                return print("\nCancelled.")
            if raw.isdigit() and int(raw) < len(match): break
            print(f"  [!] Enter a number from 0 to {len(match) - 1}.")
        match = match.iloc[[int(raw)]]

    if match['mlb_played_first'].isna().values[0]:
        return print(f"No MLB debut year on record for '{name_input}'.")

    p_id = match['key_mlbam'].values[0]
    debut_year = int(match['mlb_played_first'].values[0])
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