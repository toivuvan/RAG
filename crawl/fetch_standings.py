import soccerdata as sd
import pandas as pd
import os
import traceback

# =========================================================
# FETCH LEAGUE STANDINGS - EPL 2023-24
# =========================================================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "standings.csv")

def flatten_columns(df):
    """Flatten multi-level columns from FBref"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
    return df

def calculate_standings_from_schedule():
    """
    Tính standings từ schedule.csv
    W: Win (3 pts), D: Draw (1 pt), L: Loss (0 pts)
    """
    print("Calculating standings from schedule.csv...")
    
    schedule_file = os.path.join(OUTPUT_DIR, "schedule.csv")
    if not os.path.exists(schedule_file):
        print(f"❌ Error: {schedule_file} not found!")
        return None
    
    # Load schedule
    df_schedule = pd.read_csv(schedule_file)
    print(f"Loaded {len(df_schedule)} matches")
    
    # Parse score: "0–3" → 0, 3
    df_schedule[['home_goals', 'away_goals']] = df_schedule['score'].str.split('–', expand=True).astype(int)
    
    # Initialize standings
    standings_dict = {}
    
    # Process each match
    for idx, row in df_schedule.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        home_goals = row['home_goals']
        away_goals = row['away_goals']
        
        # Initialize teams
        for team in [home_team, away_team]:
            if team not in standings_dict:
                standings_dict[team] = {
                    'P': 0, 'W': 0, 'D': 0, 'L': 0,
                    'GF': 0, 'GA': 0, 'Pts': 0
                }
        
        # Update stats
        standings_dict[home_team]['P'] += 1
        standings_dict[away_team]['P'] += 1
        standings_dict[home_team]['GF'] += home_goals
        standings_dict[home_team]['GA'] += away_goals
        standings_dict[away_team]['GF'] += away_goals
        standings_dict[away_team]['GA'] += home_goals
        
        # Results
        if home_goals > away_goals:
            standings_dict[home_team]['W'] += 1
            standings_dict[home_team]['Pts'] += 3
            standings_dict[away_team]['L'] += 1
        elif home_goals < away_goals:
            standings_dict[away_team]['W'] += 1
            standings_dict[away_team]['Pts'] += 3
            standings_dict[home_team]['L'] += 1
        else:
            standings_dict[home_team]['D'] += 1
            standings_dict[home_team]['Pts'] += 1
            standings_dict[away_team]['D'] += 1
            standings_dict[away_team]['Pts'] += 1
    
    # Convert to DataFrame
    df_standings = pd.DataFrame.from_dict(standings_dict, orient='index')
    df_standings.reset_index(inplace=True)
    df_standings.rename(columns={'index': 'Team'}, inplace=True)
    
    # Calculate GD
    df_standings['GD'] = df_standings['GF'] - df_standings['GA']
    
    # Sort
    df_standings = df_standings.sort_values(
        by=['Pts', 'GD', 'GF'], 
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    # Rank
    df_standings.insert(0, 'Rank', range(1, len(df_standings) + 1))
    df_standings = df_standings[['Rank', 'Team', 'P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts']]
    
    return df_standings

def fetch_standings():
    print("=" * 70)
    print("FETCH LEAGUE STANDINGS - EPL 2023-24")
    print("=" * 70 + "\n")
    
    try:
        standings = calculate_standings_from_schedule()
        
        if standings is None:
            print("❌ Could not generate standings!")
            return
        
        standings.reset_index(inplace=True, drop=True)
        standings.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"\n✓ Saved to: {OUTPUT_FILE}\n")
        print("League Standings:")
        print(standings.to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    fetch_standings()