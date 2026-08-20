"""
Feature engineering for AQI forecasting: lag features, rolling averages,
and time encodings. Builds the target variable (next-day AQI).
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_PATH = BASE_DIR / "data/processed/merged_daily.csv"
FEATURES_PATH = BASE_DIR / "data/processed/features.csv"


def build_features():
    df = pd.read_csv(MERGED_PATH)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Lag features
    df["aqi_lag_1"] = df["AQI"].shift(1)
    df["aqi_lag_2"] = df["AQI"].shift(2)
    df["aqi_lag_3"] = df["AQI"].shift(3)

    # Rolling averages
    df["aqi_roll_3"] = df["AQI"].shift(1).rolling(window=3).mean()
    df["aqi_roll_7"] = df["AQI"].shift(1).rolling(window=7).mean()

    # Time encodings
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # Target: next day's AQI
    df["target_aqi_next_day"] = df["AQI"].shift(-1)

    # Drop rows with NaNs from lag/rolling/target at the edges
    df = df.dropna().reset_index(drop=True)

    print(f"Feature rows after dropping NaNs: {len(df)}")
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Saved features to {FEATURES_PATH}")

    return df


if __name__ == "__main__":
    build_features()