import pandas as pd
from pybaseball import playerid_lookup, statcast_pitcher, cache
import os
import re
from datetime import datetime

cache.enable()

# Standard Statcast Pitch Type Mapping
PITCH_MAP = {
    '4-Seam Fastball': 'FF',
    'Slider': 'SL',
    'Sinker': 'SI',
    'Changeup': 'CH',
    'Cutter': 'FC',
    'Curveball': 'CU',
    'Sweeper': 'ST',
    'Knuckle Curve': 'KC',
    'Split-Finger': 'FS',
    'Slurve': 'SV',
    'Other': 'OT',
    'Forkball': 'FO',
    'Knuckleball': 'KN',
    'Eephus': 'EP',
    'Screwball': 'SC',
    'Slow Curve': 'CS',
    'Pitch Out': 'PO'
}

def generate_pitch_arsenal(pitcher_dir):
    """Analyzes the latest season and maps names to abbreviations (FF, SL, etc.)."""
    all_files = [f for f in os.listdir(pitcher_dir) if f.endswith('_pitching_data.csv')]
    if not all_files: return

    years = []
    for f in all_files:
        match = re.match(r"(\d{4})", f)
        if match: years.append(int(match.group(1)))
    
    if not years: return
    latest_year = max(years)
    latest_file = os.path.join(pitcher_dir, f"{latest_year}_pitching_data.csv")
    
    df = pd.read_csv(latest_file)

    if 'pitch_name' in df.columns:
        # Group by the full name first to get the metrics
        stats = df.groupby('pitch_name').agg(
            Count=('pitch_name', 'count'),
            Avg_Vel=('release_speed', 'mean'),
            Max_Vel=('release_speed', 'max')
        ).reset_index()

        # Create the 'pitch_type' column using the PITCH_MAP
        stats['pitch_type'] = stats['pitch_name'].map(PITCH_MAP).fillna(stats['pitch_name'])

        total_pitches = stats['Count'].sum()
        stats['Usage_%'] = (stats['Count'] / total_pitches * 100).round(1)
        stats['Avg_Vel'] = stats['Avg_Vel'].round(1)
        
        # Reorder columns to put the abbreviation front and center
        stats = stats[['pitch_type', 'pitch_name', 'Count', 'Avg_Vel', 'Max_Vel', 'Usage_%']]
        stats = stats.sort_values(by='Usage_%', ascending=False)

        summary_path = os.path.join(pitcher_dir, 'pitches.csv')
        stats.to_csv(summary_path, index=False)
        print(f"  -> Generated {latest_year} arsenal summary with abbreviations.")

def main():
    print("--- Strict Pitcher Downloader (Abbreviation Sync) ---")
    name_input = input("Enter pitcher name: ").strip()
    if " " not in name_input: return print("Error: Enter first and last name.")
    first, last = name_input.split(" ", 1)

    id_results = playerid_lookup(last, first, fuzzy=False)
    if id_results.empty: return print(f"No exact match for '{name_input}'.")

    match = id_results[
        (id_results['name_first'].str.lower() == first.lower()) & 
        (id_results['name_last'].str.lower() == last.lower())
    ]
    if match.empty: return print(f"No exact match for '{name_input}'.")

    p_id = match['key_mlbam'].values[0]
    debut_year = int(match['mlb_played_first'].values[0])
    current_year = datetime.now().year
    
    pitcher_slug = f"{first.lower()}_{last.lower()}"
    pitcher_dir = os.path.join("Pitchers", pitcher_slug)
    if not os.path.exists(pitcher_dir): os.makedirs(pitcher_dir)

    for year in range(debut_year, current_year + 1):
        file_path = os.path.join(pitcher_dir, f"{year}_pitching_data.csv")
        if os.path.exists(file_path) and year < current_year: continue
            
        print(f"  - Fetching {year} data...")
        try:
            data = statcast_pitcher(f'{year}-03-15', f'{year}-11-15', player_id=p_id)
            if not data.empty:
                data.to_csv(file_path, index=False)
                print(f"    -> Saved {len(data)} pitches")
            else:
                print(f"    -> No pitches found.")
        except Exception as e:
            print(f"    -> Error fetching {year}: {e}")

    generate_pitch_arsenal(pitcher_dir)

if __name__ == "__main__":
    main()