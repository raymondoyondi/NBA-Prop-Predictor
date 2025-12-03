from nba_api.stats.endpoints import playergamelog
from nba_api.stats.library.parameters import SeasonAll # Import SeasonAll
import pandas as pd
import os # Import os for path handling

def get_lebron_gamelogs(player_id=2544):
    """Fetches LeBron James's game logs from the NBA API for all seasons."""
    # Use SeasonAll.all to get all career game logs
    gamelogs = playergamelog.PlayerGameLog(player_id=player_id, season=SeasonAll.all)
    df = gamelogs.get_data_frames()[0]
    return df

if __name__ == '__main__':
    lebron_df = get_lebron_gamelogs()
    # Ensure the directory exists
    output_dir = '../data/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'lebron_stats.csv')
    lebron_df.to_csv(output_path, index=False)
    print(f"LeBron James game logs saved to {output_path}")