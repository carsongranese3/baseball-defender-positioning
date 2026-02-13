import pandas as pd
from pybaseball import playerid_lookup, statcast_batter, cache

# Enable caching to avoid re-downloading the same data
cache.enable()

def save_player_data(first_name, last_name, season=2025):
    """
    Fetches Statcast data for a specific player and season, 
    filters for balls in play, and saves to a CSV.
    """
    # 1. Lookup Player ID
    print(f"Looking up ID for {first_name} {last_name}...")
    try:
        # fuzzy=True helps if you misspell names slightly
        id_results = playerid_lookup(last_name, first_name, fuzzy=True)
        
        if id_results.empty:
            print(f"Error: Could not find player '{first_name} {last_name}'.")
            return

        # Get the MLBAM ID (the one Statcast uses)
        player_id = id_results['key_mlbam'].values[0]
        full_name_str = f"{id_results['name_first'].values[0]} {id_results['name_last'].values[0]}"
        print(f"Found: {full_name_str} (ID: {player_id})")

    except IndexError:
        print("Error: Player found but ID lookup failed. Check spelling.")
        return

    # 2. Fetch Season Data
    # Extending date range to cover potential postseason
    start_date = f'{season}-03-15'
    end_date = f'{season}-11-05'
    
    print(f"Fetching {season} Statcast data...")
    data = statcast_batter(start_date, end_date, player_id=player_id)

    if data.empty:
        print(f"No data found for {full_name_str} in {season}.")
        return

    # 3. Filter for balls put in play (Valid Coordinates)
    # We drop rows where hc_x OR hc_y are NaN (strikeouts, walks, etc.)
    batted_balls = data.dropna(subset=['hc_x', 'hc_y']).copy()

    if batted_balls.empty:
        print("Player has stats but no batted ball coordinates (only Ks/BBs?).")
        return

    # 4. Generate Dynamic Filename
    # clean up name for file (e.g., "Aaron Judge" -> "aaron_judge_2025.csv")
    safe_name = f"{first_name}_{last_name}".lower().replace(" ", "_")
    filename = f"{safe_name}_{season}_batted_balls.csv"
    
    # Save to CSV
    batted_balls.to_csv(filename, index=False)
    
    print(f"--- Process Complete ---")
    print(f"Saved: {filename}")
    print(f"Total Batted Balls: {len(batted_balls)}")
    print(f"Events: {batted_balls['events'].unique()[:5]} ...") # Show first 5 event types

def main():
    # User Input Loop
    print("--- Statcast Data Downloader ---")
    user_input = input("Enter player name (e.g., 'Aaron Judge'): ").strip()
    
    if " " in user_input:
        first, last = user_input.split(" ", 1) # Split on first space only
        save_player_data(first, last)
    else:
        print("Please enter both a first and last name.")

if __name__ == "__main__":
    main()