import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

def train_model(df, target_col='PTS'):
    """
    Trains a RandomForestRegressor model to predict the target_col.
    """
    features = [col for col in df.columns if 'LAST_' in col or col in ['MIN', 'FG_PCT', 'FT_PCT', 'FG3_PCT']]
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error for {target_col}: {mse}")

    joblib.dump(model, f"../models/lebron_{target_col}_predictor.joblib")
    print(f"Model saved to models/lebron_{target_col}_predictor.joblib")
    return model

if __name__ == "__main__":
    processed_df = pd.read_csv("../data/lebron_processed_stats.csv")
    _ = train_model(processed_df, target_col='PTS')
    _ = train_model(processed_df, target_col='REB')
    _ = train_model(processed_df, target_col='AST')