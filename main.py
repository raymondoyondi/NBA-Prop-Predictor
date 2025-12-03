import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from nba_api.stats.endpoints import playergamelogs, playergamelog
import warnings
# Suppress FutureWarnings (optional, as in original code)
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_lebron_gamelogs_fixed():
    """
    Fetches LeBron James' game logs for all seasons by iterating through years
    and returns a pandas DataFrame.
    """
    # LeBron James' Player ID is 2544
    player_id = '2544'
    all_logs_df = pd.DataFrame()
    
    # Define seasons to loop through (LeBron's career started in 2003)
    # Adjust the end range as the current year changes (e.g., to 2026 for the 2025-26 season)
    seasons = [f'{year}-{str(year+1)[2:]}' for year in range(2003, 2025)] # Range up to the 2024-25 season

    print(f"Fetching data for {len(seasons)} seasons...")

    for season in seasons:
        try:
            # Use the singular PlayerGameLog endpoint which is more reliable for individual season calls
            logs = playergamelog.PlayerGameLog(player_id=player_id, season=season).get_data_frames()[0]
            all_logs_df = pd.concat([all_logs_df, logs], ignore_index=True)
        except Exception as e:
            print(f"Error fetching data for season {season}: {e}")
            continue
            
    if all_logs_df.empty:
        print("No data fetched.")
        return pd.DataFrame()

    # Sort by game date to ensure correct rolling average calculation
    all_logs_df['GAME_DATE'] = pd.to_datetime(all_logs_df['GAME_DATE'])
    all_logs_df = all_logs_df.sort_values(by='GAME_DATE', ascending=True).reset_index(drop=True)
    
    return all_logs_df

def main():
    # 1. Fetch Data
    df = get_lebron_gamelogs_fixed() # Use the fixed function
    
    if df.empty:
        print("Could not proceed without data. Check connection or data source.")
        return

    # Filter for relevant columns and ensure numeric types
    # Columns from the singular endpoint (playergamelog) use slightly different names (lowercase PTS)
    df = df[['PTS', 'FGA', 'FGM', 'FG3A', 'AST', 'REB', 'MIN']].copy()
    df = df.apply(pd.to_numeric, errors='coerce').dropna()

    # 2. Feature Engineering (Example: Rolling Averages)
    # Using a simple rolling average of the last 5 games as a predictor
    df['PTS_AVG_L5'] = df['PTS'].shift(1).rolling(window=5).mean()
    df = df.dropna() # Drop rows with NaN values created by rolling average

    # 3. Define Features (X) and Target (y)
    X = df[['PTS_AVG_L5']]
    y = df['PTS']

    # 4. Model Training
    # We use a simple linear regression model for demonstration
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 5. Model Evaluation
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R-squared: {r2:.2f}")

    # 6. Make a Prediction for the Next Game
    # The last recorded game's point average is used to predict the next one
    latest_avg = df['PTS_AVG_L5'].iloc[-1]
    # Reshape for the model
    next_game_prediction = model.predict(np.array([[latest_avg]]))
    print(f"\nPrediction for LeBron James' next game points:")
    print(f"Using the last 5 games average of {latest_avg:.2f} points.")
    print(f"Projected Points: {next_game_prediction[0]:.2f}")

if __name__ == "__main__":
    main()