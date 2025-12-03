from nba_api.stats.endpoints import playergamelog, teamgamelogs, commonplayerinfo
from nba_api.stats.static import players, teams
import pandas as pd
import logging
import sys

# Setup logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Function to get Player ID by Partial Name (Active Players Only)
def get_player_id_by_partial_name(player_name):
    all_players = players.get_active_players()
    matching_players = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
    if not matching_players:
        raise ValueError(f"No active player found with the name '{player_name}'.")
    if len(matching_players) > 12:
        print("Too many results found. Showing the first 12 matches:")
        matching_players = matching_players[:12]
    if len(matching_players) > 1:
        print("\nMultiple active players found:")
        for idx, p in enumerate(matching_players, 1):
            print(f"{idx}. {p['full_name']}")
        choice = input("Enter the number of the player you want: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(matching_players):
            selected = matching_players[int(choice) - 1]
            return selected['id'], selected['full_name']
        else:
            raise ValueError("Invalid selection. Please try again.")
    else:
        return matching_players[0]['id'], matching_players[0]['full_name']

# Function to fetch Player Game Logs
def fetch_player_data_by_name(player_name, season="2023-24"):
    player_id, full_name = get_player_id_by_partial_name(player_name)
    logging.info(f"\nFetching game logs for: {full_name}")
    gamelog = playergamelog.PlayerGameLog(
        player_id=player_id, 
        season=season, 
        season_type_all_star="Regular Season"  # Correct parameter for PlayerGameLog
    )
    df = gamelog.get_data_frames()[0]
    return df, full_name, player_id

# Function to map Team Abbreviation to Team ID
def get_team_id_by_abbreviation(abbreviation):
    nba_teams = teams.get_teams()
    team = next((team for team in nba_teams if team['abbreviation'] == abbreviation), None)
    return team['id'] if team else None

# Function to find Team Name by Team ID
def find_team_name_by_id(team_id):
    nba_teams = teams.get_teams()
    team = next((team for team in nba_teams if team['id'] == team_id), None)
    return team['full_name'] if team else None

# Function to get Opponent Team ID from the Game Logs
def get_opponent_team_id(df):
    # Extract opponent team abbreviations from MATCHUP column
    opponent_abbreviations = df['MATCHUP'].apply(lambda x: x.split(' ')[-1]).unique()
    opponent_team_ids = []
    for abbrev in opponent_abbreviations:
        team_id = get_team_id_by_abbreviation(abbrev)
        if team_id:
            opponent_team_ids.append(team_id)
    if not opponent_team_ids:
        print("No opponent teams found in the game logs.")
        return None
    print("\nRecent Opponent Teams:")
    unique_opponents = list(set(opponent_team_ids))
    for idx, team_id in enumerate(unique_opponents, 1):
        team_name = find_team_name_by_id(team_id)
        print(f"{idx}. {team_name} ({team_id})")
    choice = input("Choose an opponent team number for strength analysis (or press Enter to skip): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(unique_opponents):
        return unique_opponents[int(choice) - 1]  # Return Team ID
    return None

# Function to get Opponent Team Name from Team ID
def get_team_name_by_id(team_id):
    team_name = find_team_name_by_id(team_id)
    return team_name if team_name else "Unknown Team"

# Function to fetch recent opponent team stats
def get_opponent_team_stats(team_id, rankings, season="2023-24", num_games=5):
    # Fetch recent team game logs
    logs = teamgamelogs.TeamGameLogs(
        team_id_nullable=team_id, 
        season_nullable=season, 
        season_type_nullable="Regular Season"  # Correct parameter for TeamGameLogs
    )
    df = logs.get_data_frames()[0]
    
    if df.empty:
        return None

    # Filter only numeric columns for calculations
    numeric_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    team_games = df[numeric_stats].head(num_games).apply(pd.to_numeric, errors='coerce')
    
    # Calculate averages for last 'num_games'
    team_avg = team_games.mean()
    
    # Add rankings
    team_rankings = {stat: rankings[stat].get(team_id, 'N/A') for stat in numeric_stats}
    
    team_avg = team_avg.round(1).to_dict()
    team_avg.update({f"{stat}_RANK": rank for stat, rank in team_rankings.items()})
    
    return team_avg

# Function to calculate league-wide team rankings
def get_league_team_rankings(season="2023-24"):
    logging.info("\nFetching league-wide team statistics for rankings...")
    logs = teamgamelogs.TeamGameLogs(
        season_nullable=season, 
        season_type_nullable="Regular Season"  # Correct parameter for TeamGameLogs
    )
    df = logs.get_data_frames()[0]
    
    # Aggregate stats per team
    aggregated = df.groupby('TEAM_ID').agg({
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'STL': 'mean',
        'BLK': 'mean'
    }).reset_index()
    
    # Calculate rankings
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    rankings = {}
    for stat in stats:
        aggregated[f'{stat}_RANK'] = aggregated[stat].rank(ascending=False, method='min')
        rankings[stat] = dict(zip(aggregated['TEAM_ID'], aggregated[f'{stat}_RANK']))
    
    logging.info("League-wide team rankings calculated.")
    return rankings

# Function to calculate weighted averages of player stats
def calculate_averages(df, num_games=5):
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
    stats = [stat for stat in stats if stat in df.columns]
    df[stats + ['MIN']] = df[stats + ['MIN']].apply(pd.to_numeric, errors='coerce')
    avg_minutes = df['MIN'].mean()
    
    def weight_game(row):
        if row['MIN'] < 0.7 * avg_minutes:
            return 0.5
        elif row['MIN'] > 0.85 * avg_minutes:
            return 1.5
        return 1.0
    
    df['WEIGHT'] = df.apply(weight_game, axis=1)
    recent_games = df.head(num_games)
    weighted_avg = (recent_games[stats].multiply(recent_games['WEIGHT'], axis=0)).sum() / recent_games['WEIGHT'].sum()
    p = weighted_avg.get('PTS', 0)
    r = weighted_avg.get('REB', 0)
    a = weighted_avg.get('AST', 0)
    p_r = p + r
    p_a = p + a
    r_a = r + a
    p_r_a = p + r + a
    return {
        'PTS': round(p, 1), 'REB': round(r, 1), 'AST': round(a, 1),
        'STL': round(weighted_avg.get('STL', 0), 1), 'BLK': round(weighted_avg.get('BLK', 0), 1),
        'P+R': round(p_r, 1), 'P+A': round(p_a, 1), 'R+A': round(r_a, 1), 'P+R+A': round(p_r_a, 1)
    }

from nba_api.stats.endpoints import commonteamroster

# Function to get player position with standardized codes
def get_player_position(player_id, season="2023-24"):
    try:
        info = commonplayerinfo.CommonPlayerInfo(player_id=player_id)
        data = info.get_data_frames()[0]
        position_full = data['POSITION'][0].strip()
        
        # Mapping full position names to shorthand codes
        position_map = {
            'G': 'G',
            'SG': 'SG',
            'Shooting Guard': 'SG',
            'SF': 'SF',
            'Shooting Forward': 'SF',
            'PF': 'PF',
            'Power Forward': 'PF',
            'C': 'C',
            'Center': 'C',
            '': '',
            None: ''
        }
        
        # Handle cases where 'POSITION' might be in full name or shorthand
        position_shorthand = position_map.get(position_full, '')
        if not position_shorthand:
            # Attempt to derive shorthand from first letter if possible
            if position_full.upper() in ['G', 'SG', 'SF', 'PF', 'C']:
                position_shorthand = position_full.upper()
            else:
                position_shorthand = ''
        
        return position_shorthand
    except Exception as e:
        logging.error(f"Error fetching position for player ID {player_id}: {e}")
        return None

# Function to calculate matchup deltas
def get_matchup_deltas(opponent_team_id, position, season="2023-24"):
    logging.info(f"\nCalculating matchup deltas for position: {position} against {get_team_name_by_id(opponent_team_id)}...")
    all_players = players.get_active_players()
    # Safely access 'position' key and compare
    same_position_players = [p for p in all_players if p.get('position', '') == position]
    
    deltas = {'PTS': [], 'REB': [], 'AST': [], 'STL': [], 'BLK': []}
    
    opponent_team_name = get_team_name_by_id(opponent_team_id)
    if opponent_team_name is None:
        logging.error("Opponent team name could not be determined.")
        return deltas
    
    for idx, player in enumerate(same_position_players, 1):
        player_id = player['id']
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season, 
                season_type_all_star="Regular Season"  # Correct parameter for PlayerGameLog
            )
            df = gamelog.get_data_frames()[0]
            if df.empty:
                continue
            # Games against the opponent
            matchup_games = df[df['MATCHUP'].str.contains(f"@ {opponent_team_name}|vs. {opponent_team_name}")]
            if matchup_games.empty:
                continue
            # Calculate average stats in matchup games
            matchup_avg = matchup_games[['PTS', 'REB', 'AST', 'STL', 'BLK']].mean()
            # Calculate season averages
            season_avg = df[['PTS', 'REB', 'AST', 'STL', 'BLK']].mean()
            # Calculate deltas
            delta = matchup_avg - season_avg
            for stat in deltas.keys():
                deltas[stat].append(delta.get(stat, 0))
        except Exception as e:
            logging.error(f"Error processing player {player['full_name']}: {e}")
            continue
    
    # Calculate average deltas
    avg_deltas = {stat: round(pd.Series(values).mean(), 2) if values else 0 for stat, values in deltas.items()}
    logging.info("Matchup deltas calculated.")
    return avg_deltas

# Function to adjust projected stats based on matchup deltas
def calculate_projected_stats(projected_line, matchup_deltas):
    projected = {}
    for stat, value in projected_line.items():
        # Only adjust base stats, not combined stats like P+R
        if stat in matchup_deltas:
            projected[stat] = round(value + matchup_deltas[stat], 1)
        else:
            projected[stat] = value
    return projected

# Main Program
if __name__ == "__main__":
    # Fetch league-wide team rankings once to avoid redundant API calls
    try:
        rankings = get_league_team_rankings(season="2023-24")
    except Exception as e:
        logging.error(f"Failed to fetch league team rankings: {e}")
        sys.exit(1)
    
    while True:
        player_name = input("Enter the player's name or last name (or type 'quit' to exit): ").strip()
        if player_name.lower() in ["quit", "exit"]:
            print("Exiting the program.")
            break

        try:
            num_games_input = input("Enter the number of recent games to analyze (default is 5): ").strip()
            num_games = int(num_games_input) if num_games_input.isdigit() else 5

            season = "2023-24"
            df, full_name, player_id = fetch_player_data_by_name(player_name, season)

            if df.empty:
                print(f"No game logs found for {full_name} in the {season} season.")
                continue

            opponent_team = get_opponent_team_id(df)
            matchup_deltas = {}
            position = None

            if opponent_team:
                opponent_stats = get_opponent_team_stats(opponent_team, rankings, season=season, num_games=num_games)
                if opponent_stats is not None:
                    print("\nOpponent Team Recent Averages and Rankings:")
                    for stat, value in opponent_stats.items():
                        print(f"{stat}: {value}")
                else:
                    print("Could not fetch opponent team stats.")
                
                # Get player's position
                position = get_player_position(player_id, season=season)
                if position:
                    matchup_deltas = get_matchup_deltas(opponent_team, position, season=season)
                    print("\nMatchup Deltas:")
                    for stat, delta in matchup_deltas.items():
                        print(f"{stat}: {delta}")
                else:
                    print("Could not determine player position.")
            else:
                print("No opponent team selected for strength analysis.")

            # Calculate projected averages
            projected_line = calculate_averages(df, num_games=num_games)
            
            # Adjust projections based on matchup deltas
            if opponent_team and position and matchup_deltas:
                projected_line = calculate_projected_stats(projected_line, matchup_deltas)
            
            print(f"\nProjected line for {full_name} based on last {num_games} games and matchup:")
            for stat, value in projected_line.items():
                print(f"{stat}: {value}")
            
            # Save projections to CSV
            file_name = f"{full_name.replace(' ', '_')}_projected_stats.csv"
            pd.DataFrame([projected_line]).to_csv(file_name, index=False)
            print(f"Projected stats saved to '{file_name}'.")
        
        except ValueError as e:
            print(e)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        print("\n--------------------------------------\n")