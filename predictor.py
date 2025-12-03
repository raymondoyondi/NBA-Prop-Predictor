import pandas as pd
import joblib
import numpy as np

def predict_props(player_id, last_n_games_df, model_path):
    """
    Predicts a player's prop based on their last N games and a trained model.
    """
    model = joblib.load(model_path)

    # Ensure the input DataFrame has the same features as the training data
    # Create features based on the last N games (e.g., rolling averages)
    features = [col for col in last_n_games_df.columns if 'LAST_' in col or col in ['MIN', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    
    # Calculate the features for prediction based on the last N games
    prediction_input = pd.DataFrame(columns=features)
    for stat in ['PTS', 'REB', 'AST']:
        prediction_input[f'LAST_5_GAME_AVG_{stat}'] = [last_n_games_df[stat].tail(5).mean()]
        prediction_input[f'LAST_10_GAME_AVG_{stat}'] = [last_n_games_df[stat].tail(10).mean()]
    
    # Add other relevant features (e.g., MIN from the last game played)
    for col in ['MIN', 'FG_PCT', 'FT_PCT', 'FG3_PCT']:
        if col in last_n_games_df.columns:
            prediction_input[col] = [last_n_games_df[col].iloc[-1]] # Use last game's value


    prediction = model.predict(prediction_input)
    return prediction[0]

if __name__ == "__main__":
    # Example usage: Load some recent data for LeBron
    processed_df = pd.read_csv("../data/lebron_processed_stats.csv")
    lebron_recent_games = processed_df.tail(10) # Using last 10 games for prediction features

    predicted_pts = predict_props("2544", lebron_recent_games, "../models/lebron_PTS_predictor.joblib")
    predicted_reb = predict_props("2544", lebron_recent_games, "../models/lebron_REB_predictor.joblib")
    predicted_ast = predict_props("2544", lebron_recent_games, "../models/lebron_AST_predictor.joblib")

    print(f"Predicted Points for LeBron James: {predicted_pts:.2f}")
    print(f"Predicted Rebounds for LeBron James: {predicted_reb:.2f}")
    print(f"Predicted Assists for LeBron James: {predicted_ast:.2f}")