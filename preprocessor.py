import pandas as pd
import numpy as np

def preprocess_data(df):
    """
    Cleans and preprocesses the NBA game log data.
    """
    # Convert relevant columns to numeric
    numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'MIN']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Handle missing values (e.g., fill with median or mean)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Feature Engineering: Add rolling averages for key stats
    df = df.sort_values(by='GAME_DATE')
    for stat in ['PTS', 'REB', 'AST']:
        df[f'LAST_5_GAME_AVG_{stat}'] = df[stat].shift(1).rolling(window=5).mean()
        df[f'LAST_10_GAME_AVG_{stat}'] = df[stat].shift(1).rolling(window=10).mean()

    # Drop rows with NaN values introduced by rolling averages
    df = df.dropna()

    return df

if __name__ == "__main__":
    raw_df = pd.read_csv("../data/lebron_stats.csv")
    processed_df = preprocess_data(raw_df)
    processed_df.to_csv("../data/lebron_processed_stats.csv", index=False)
    print("Processed LeBron James stats saved to data/lebron_processed_stats.csv")