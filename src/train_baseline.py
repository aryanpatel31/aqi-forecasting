"""
Train and evaluate an XGBoost baseline for next-day AQI forecasting,
compared against a naive persistence baseline.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_PATH = BASE_DIR / "data/processed/features.csv"

FEATURE_COLS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure", "precipitation",
    "aqi_lag_1", "aqi_lag_2", "aqi_lag_3",
    "aqi_roll_3", "aqi_roll_7",
    "month", "day_of_week", "day_of_year",
]
TARGET_COL = "target_aqi_next_day"


def load_split_data():
    df = pd.read_csv(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"Train: {len(train)} rows ({train['date'].min().date()} to {train['date'].max().date()})")
    print(f"Test:  {len(test)} rows ({test['date'].min().date()} to {test['date'].max().date()})")

    return train, test


def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label:20s} MAE: {mae:.2f}  RMSE: {rmse:.2f}")
    return mae, rmse


def persistence_baseline(test):
    y_true = test[TARGET_COL]
    y_pred = test["aqi_lag_1"]  # today's AQI as tomorrow's prediction
    return evaluate(y_true, y_pred, "Persistence baseline")


def train_xgboost(train, test):
    X_train, y_train = train[FEATURE_COLS], train[TARGET_COL]
    X_test, y_test = test[FEATURE_COLS], test[TARGET_COL]

    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    evaluate(y_test, preds, "XGBoost")

    return model, preds


if __name__ == "__main__":
    train, test = load_split_data()
    persistence_baseline(test)
    model, preds = train_xgboost(train, test)